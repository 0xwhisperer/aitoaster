"""Dense whole-track CNN prototype.

This module is deliberately separate from the production optimizer.  It
reuses the existing differentiable CQT surrogate, runs the split CNN trunk
once, and applies a sliding global-average-pool over the trunk's time axis.
It does not replace production verification: the exact librosa/ONNX scan is
still the certificate and is exposed here only as a diagnostic/benchmark.

The 2x2 max-pools reduce the 512-sample CQT hop by 8, so one trunk cell is
4,096 samples (0.256 s).  A 10-second input has 313 CQT frames and 39 trunk
cells after the three pools; 39 is therefore the model-aligned sliding pool
width for this fixed production model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import resource
import time
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .cnn_differentiable_v2 import (
    CQT_CFG,
    N_COEFFS,
    SEGMENT_SAMPLES,
    SR,
    convolutional_trunk_from_cepstrum,
    differentiable_cepstrum,
    forward_logit_differentiable,
    get_real_logit_segment,
    mlp_head_from_pooled,
)


GRID_SAMPLES = CQT_CFG["hop_length"] * 8
GRID_SECONDS = GRID_SAMPLES / SR
TRUNK_POOL_FACTOR = 8


def cqt_frame_count(n_samples: int) -> int:
    """nnAudio's frame count for the current centered CQT configuration."""
    return n_samples // CQT_CFG["hop_length"] + 1


def trunk_time_count_from_cqt_frames(n_frames: int) -> int:
    """Time length after the three floor-mode 2x2 max-pools."""
    for _ in range(3):
        n_frames //= 2
    return n_frames


def model_pool_width(segment_samples: int = SEGMENT_SAMPLES) -> int:
    """Number of last-trunk cells represented by one 10-second model input."""
    return trunk_time_count_from_cqt_frames(cqt_frame_count(segment_samples))


MODEL_POOL_WIDTH = model_pool_width()


def dense_grid_positions(n_samples: int, include_partial_boundary: bool = True) -> np.ndarray:
    """Return model-aligned starts for the dense sliding score grid.

    The whole-track trunk can produce one extra right-edge cell because the
    centered CQT and floor pooling do not express a strict waveform-window
    boundary.  It is useful diagnostically, so it is included by default;
    callers comparing only complete standalone 10-second windows can filter
    positions to ``p + SEGMENT_SAMPLES <= n_samples``.
    """
    if n_samples < SEGMENT_SAMPLES:
        return np.asarray([], dtype=np.int64)
    whole_trunk_cells = trunk_time_count_from_cqt_frames(cqt_frame_count(n_samples))
    n_dense = max(0, whole_trunk_cells - MODEL_POOL_WIDTH + 1)
    positions = np.arange(n_dense, dtype=np.int64) * GRID_SAMPLES
    if not include_partial_boundary:
        positions = positions[positions + SEGMENT_SAMPLES <= n_samples]
    return positions


def dense_logit_grid_from_cepstrum(cepstrum: torch.Tensor) -> torch.Tensor:
    """Score every sliding model-width span of a precomputed cepstrum.

    Input is [B, 24, CQT-time]; output is [B, dense-grid-cells].  The CQT and
    cepstrum are intentionally outside this function so an optimizer can
    compute them once and reuse the graph for repeated dense scoring.
    """
    trunk = convolutional_trunk_from_cepstrum(cepstrum)
    if trunk.shape[-1] < MODEL_POOL_WIDTH:
        raise ValueError(
            f"whole-track trunk has {trunk.shape[-1]} cells, fewer than the "
            f"{MODEL_POOL_WIDTH}-cell model window"
        )
    pooled = F.avg_pool2d(
        trunk,
        kernel_size=(trunk.shape[-2], MODEL_POOL_WIDTH),
        stride=(1, 1),
    ).squeeze(2).transpose(1, 2)
    return mlp_head_from_pooled(pooled)


def forward_dense_logit_grid(audio_1d: torch.Tensor) -> torch.Tensor:
    """Compute whole-track CQT/cepstrum once and return dense logits."""
    if audio_1d.ndim != 2:
        raise ValueError(f"audio_1d must be [batch, samples], got {tuple(audio_1d.shape)}")
    return dense_logit_grid_from_cepstrum(differentiable_cepstrum(audio_1d))


def _pad_segment(audio: np.ndarray, start: int, segment_samples: int) -> np.ndarray:
    segment = np.asarray(audio[start:start + segment_samples], dtype=np.float32)
    if len(segment) == segment_samples:
        return segment
    padded = np.zeros(segment_samples, dtype=np.float32)
    padded[:len(segment)] = segment
    return padded


def exact_standalone_logits(
    audio: np.ndarray,
    positions: Sequence[int],
    pad_partial: bool = True,
) -> np.ndarray:
    """Score positions with the exact librosa/ONNX 10-second scorer.

    This is diagnostic only.  It intentionally calls the same exact helper
    used by the existing certificate path and never feeds its result into the
    differentiable optimizer.
    """
    logits = []
    for position in positions:
        if position < 0 or position >= len(audio):
            raise ValueError(f"invalid standalone position {position}")
        if position + SEGMENT_SAMPLES > len(audio) and not pad_partial:
            continue
        segment = _pad_segment(audio, int(position), SEGMENT_SAMPLES)
        logits.append(get_real_logit_segment(segment))
    return np.asarray(logits, dtype=np.float64)


def differentiable_standalone_logits(
    audio: np.ndarray,
    positions: Sequence[int],
    pad_partial: bool = True,
) -> np.ndarray:
    """Run the existing nnAudio/ONNX-converted 10-second surrogate per window.

    Comparing this with the dense result isolates whole-track reuse, pooling
    alignment, and segment-boundary effects from the separate librosa versus
    nnAudio implementation difference in the exact scorer.
    """
    audio = np.asarray(audio, dtype=np.float32)
    logits = []
    with torch.no_grad():
        for position in positions:
            if position < 0 or position >= len(audio):
                raise ValueError(f"invalid standalone position {position}")
            if position + SEGMENT_SAMPLES > len(audio) and not pad_partial:
                continue
            segment = _pad_segment(audio, int(position), SEGMENT_SAMPLES)
            logits.append(
                float(forward_logit_differentiable(torch.from_numpy(segment).unsqueeze(0)))
            )
    return np.asarray(logits, dtype=np.float64)


def compare_dense_with_differentiable_standalone(
    audio: np.ndarray,
    dense_logits: np.ndarray | None = None,
    trim_cells: Sequence[int] = (0, 1, 4, 8, 16, 32, 64),
) -> dict:
    """Quantify the dense-vs-local-surrogate boundary/pooling discrepancy."""
    positions = dense_grid_positions(len(audio), include_partial_boundary=True)
    if dense_logits is None:
        with torch.no_grad():
            dense_logits = forward_dense_logit_grid(
                torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
            )[0].cpu().numpy()
    dense_logits = np.asarray(dense_logits, dtype=np.float64)
    local = differentiable_standalone_logits(audio, positions, pad_partial=True)
    complete = positions + SEGMENT_SAMPLES <= len(audio)
    summary = {
        "all": {
            "count": int(len(dense_logits)),
            "correlation": _safe_correlation(dense_logits, local),
            "max_abs_error": float(np.max(np.abs(dense_logits - local))),
            "mean_abs_error": float(np.mean(np.abs(dense_logits - local))),
        },
        "complete": {
            "count": int(np.sum(complete)),
            "correlation": _safe_correlation(dense_logits[complete], local[complete]),
            "max_abs_error": float(np.max(np.abs(dense_logits[complete] - local[complete]))),
            "mean_abs_error": float(np.mean(np.abs(dense_logits[complete] - local[complete]))),
        },
        "trimmed": {},
    }
    for trim in trim_cells:
        mask = complete.copy()
        if trim:
            if 2 * trim >= int(np.sum(complete)):
                continue
            mask[:trim] = False
            mask[-trim:] = False
        summary["trimmed"][str(trim)] = {
            "count": int(np.sum(mask)),
            "correlation": _safe_correlation(dense_logits[mask], local[mask]),
            "max_abs_error": float(np.max(np.abs(dense_logits[mask] - local[mask]))),
            "mean_abs_error": float(np.mean(np.abs(dense_logits[mask] - local[mask]))),
        }
    return summary


def exact_certificate_scan(
    audio: np.ndarray,
    hop_samples: int = SR // 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the unchanged exact 0.5-second scan used for final certification.

    The last complete 10-second start is appended when it is not on the
    0.5-second lattice, preserving whole-track coverage without changing the
    production detector or its thresholds.
    """
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")
    last = len(audio) - SEGMENT_SAMPLES
    if last < 0:
        positions = np.asarray([0], dtype=np.int64)
    else:
        positions = np.arange(0, last + 1, hop_samples, dtype=np.int64)
        if len(positions) == 0 or positions[-1] != last:
            positions = np.append(positions, last)
    return positions, exact_standalone_logits(
        audio, positions, pad_partial=(last < 0)
    )


@dataclass
class DenseExactComparison:
    positions: np.ndarray
    dense_logits: np.ndarray
    exact_logits: np.ndarray
    valid_complete: np.ndarray
    correlation: float
    max_abs_error: float
    mean_abs_error: float
    best_offset_samples: int | None = None
    best_offset_correlation: float | None = None
    best_offset_max_abs_error: float | None = None


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_dense_with_exact(
    audio: np.ndarray,
    dense_logits: np.ndarray | None = None,
    alignment_offset_samples: int = 0,
) -> DenseExactComparison:
    """Compare dense logits against exact standalone logits at grid starts."""
    positions = dense_grid_positions(len(audio), include_partial_boundary=True)
    if dense_logits is None:
        with torch.no_grad():
            dense_logits = forward_dense_logit_grid(
                torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
            )[0].cpu().numpy()
    dense_logits = np.asarray(dense_logits, dtype=np.float64)
    if len(dense_logits) != len(positions):
        raise ValueError(f"expected {len(positions)} dense logits, got {len(dense_logits)}")

    exact_positions = positions + int(alignment_offset_samples)
    exact_logits = exact_standalone_logits(audio, exact_positions, pad_partial=True)
    errors = np.abs(dense_logits - exact_logits)
    complete = positions + SEGMENT_SAMPLES <= len(audio)
    return DenseExactComparison(
        positions=positions,
        dense_logits=dense_logits,
        exact_logits=exact_logits,
        valid_complete=complete,
        correlation=_safe_correlation(dense_logits[complete], exact_logits[complete]),
        max_abs_error=float(np.max(errors[complete])) if np.any(complete) else float("nan"),
        mean_abs_error=float(np.mean(errors[complete])) if np.any(complete) else float("nan"),
    )


def probe_alignment_offsets(
    audio: np.ndarray,
    dense_logits: np.ndarray,
    offsets_samples: Iterable[int] = (-8192, -4096, -2048, 0, 2048, 4096, 8192),
    sample_count: int = 24,
) -> dict:
    """Probe pooling/CQT alignment using a small exact standalone sample.

    This avoids multiplying the expensive exact scan by every candidate
    offset.  It is a diagnostic for whether a constant boundary correction is
    useful, not a replacement for the exact certificate.
    """
    positions = dense_grid_positions(len(audio), include_partial_boundary=True)
    valid = positions + SEGMENT_SAMPLES <= len(audio)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) > sample_count:
        indices = np.unique(np.linspace(0, len(valid_indices) - 1, sample_count).astype(int))
        indices = valid_indices[indices]
    else:
        indices = valid_indices
    results = []
    for offset in offsets_samples:
        probe_positions = positions[indices] + int(offset)
        in_range = (probe_positions >= 0) & (probe_positions < len(audio))
        if not np.any(in_range):
            continue
        probe_positions = probe_positions[in_range]
        probe_dense = np.asarray(dense_logits)[indices][in_range]
        exact = exact_standalone_logits(audio, probe_positions, pad_partial=True)
        results.append({
            "offset_samples": int(offset),
            "offset_seconds": float(offset / SR),
            "correlation": _safe_correlation(probe_dense, exact),
            "max_abs_error": float(np.max(np.abs(probe_dense - exact))),
            "mean_abs_error": float(np.mean(np.abs(probe_dense - exact))),
            "sample_count": int(len(probe_positions)),
        })
    if not results:
        return {"sample_count": 0, "results": []}
    best = max(results, key=lambda x: (-math.inf if np.isnan(x["correlation"]) else x["correlation"]))
    return {"sample_count": int(len(indices)), "results": results, "best": best}


def benchmark_forward_backward(
    audio: np.ndarray,
    repeats: int = 3,
    baseline_windows: int = 1,
) -> dict:
    """Measure dense full-track and standalone-window surrogate cost.

    The baseline is one existing 10-second differentiable scorer forward plus
    backward.  Repeating that cost for every dense grid cell gives a
    conservative estimate for the old per-window whole-track loop; the dense
    path computes all cells in one graph.  The exact certificate timing is
    reported separately because it is not differentiable.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < SEGMENT_SAMPLES:
        raise ValueError("benchmark audio must be at least 10 seconds")
    positions = dense_grid_positions(len(audio), include_partial_boundary=False)
    if len(positions) == 0:
        raise ValueError("audio produced no complete dense windows")
    sample_positions = positions[:max(1, baseline_windows)]

    def timed(fn):
        values = []
        peaks = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            values.append(time.perf_counter() - start)
            peaks.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {
            "median_seconds": float(np.median(values)),
            "min_seconds": float(np.min(values)),
            "all_seconds": values,
            "peak_rss_mb": float(max(peaks) / (1024 * 1024)),
        }

    def dense_step():
        x = torch.tensor(audio, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        logits = forward_dense_logit_grid(x)
        logits.sum().backward()

    def baseline_step():
        x = torch.tensor(audio, dtype=torch.float32, requires_grad=True)
        losses = []
        for position in sample_positions:
            losses.append(
                forward_logit_differentiable(
                    x[position:position + SEGMENT_SAMPLES].unsqueeze(0)
                )
            )
        torch.stack(losses).sum().backward()

    dense = timed(dense_step)
    baseline = timed(baseline_step)
    per_window = baseline["median_seconds"] / len(sample_positions)
    estimated_old = per_window * len(positions)
    dense["dense_outputs"] = int(len(dense_grid_positions(len(audio), True)))
    dense["complete_grid_cells"] = int(len(positions))
    dense["grid_seconds"] = GRID_SECONDS
    baseline["sampled_windows"] = int(len(sample_positions))
    return {
        "dense_forward_backward": dense,
        "baseline_forward_backward": baseline,
        "estimated_old_all_grid_seconds": float(estimated_old),
        "estimated_speedup": float(estimated_old / dense["median_seconds"]),
    }


def benchmark_exact_certificate(audio: np.ndarray) -> dict:
    """Time the exact 0.5-second librosa/ONNX scan kept as certificate."""
    start = time.perf_counter()
    positions, logits = exact_certificate_scan(np.asarray(audio, dtype=np.float32))
    elapsed = time.perf_counter() - start
    return {
        "seconds": float(elapsed),
        "windows": int(len(positions)),
        "max_logit": float(np.max(logits)) if len(logits) else float("nan"),
    }
