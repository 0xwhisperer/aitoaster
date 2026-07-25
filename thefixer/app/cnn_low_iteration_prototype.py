"""Low-iteration whole-track CNN optimizer prototype.

This file is benchmark-only.  It never enters production routing.  Each
update computes the whole-track CQT/cepstrum and split CNN trunk once, then
scores every model-aligned dense cell with one sliding global-average pool.
EOT is represented by randomly shifting the whole-track alignment on each
update.  Exact certification uses the logical union of the production dense
0.5-second starts and every required +/-0.5-second EOT shift, deduplicated by
absolute segment start before the parallel librosa/ONNX scan.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .cnn_differentiable_v2 import (
    SEGMENT_SAMPLES,
    SR,
    get_real_score_segment,
    load_audio_mono,
)
from .cnn_gradient_optimizer_v2 import (
    apply_silence_guard_to_delta,
    band_limit_penalty,
    perceptual_penalty,
    tonality_penalty,
)
from .cnn_real_scanner import ParallelRealScoreScanner
from .cnn_wholetrack_dense_prototype import (
    dense_grid_positions,
    forward_dense_logit_grid,
)
from .cnn_wholetrack_optimizer_v2 import (
    build_sliding_windows,
    optimize_whole_track_verified,
)


DEFAULT_REAL_TARGET = 0.08
DEFAULT_EOT_JITTER_SEC = 0.5
DEFAULT_EOT_STEP_SEC = 0.1
DEFAULT_LR = 0.00002
DEFAULT_LAMBDA_PERCEPTUAL = 2000.0
DEFAULT_LAMBDA_BAND = 5000.0
DEFAULT_LAMBDA_TONALITY = 50.0
EOT_OFFSETS = tuple(range(-8000, 8001, 1600))
MAX_CQT_BOUNDARY_GRAD_SEC = 1.25


@dataclass
class Certification:
    dense_worst_score: float
    certified_worst_score: float
    failing_window_count: int
    dense_window_count: int
    union_window_count: int
    eot_shift_count: int
    passed: bool


@dataclass
class BenchmarkResult:
    audio: str
    method: str
    steps: int
    runtime_sec: float
    peak_delta: float
    snr_db: float
    max_rss_mb: float
    tracemalloc_peak_mb: float
    dense_worst_score: float
    certified_worst_score: float
    failing_window_count: int
    dense_window_count: int
    union_window_count: int
    eot_shift_count: int
    passed: bool
    error: str | None = None


def _pad_for_detector(audio_np: np.ndarray) -> tuple[np.ndarray, int]:
    original_length = len(audio_np)
    if original_length >= SEGMENT_SAMPLES:
        return np.asarray(audio_np, dtype=np.float32), original_length
    padded = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
    padded[:original_length] = audio_np
    return padded, original_length


def dense_window_positions(audio_np: np.ndarray, hop_sec: float = 0.5) -> tuple[list[int], int]:
    """Use the production thorough 0.5-second dense-window geometry."""
    padded, _ = _pad_for_detector(audio_np)
    positions, seg_len = build_sliding_windows(len(padded), hop_sec=hop_sec)
    return [int(position) for position in positions], int(seg_len)


def eot_offsets(
    jitter_sec: float = DEFAULT_EOT_JITTER_SEC,
    step_sec: float = DEFAULT_EOT_STEP_SEC,
    sr: int = SR,
) -> list[int]:
    if jitter_sec < 0 or step_sec <= 0:
        raise ValueError("EOT jitter must be non-negative and EOT step must be positive")
    radius = int(round(jitter_sec * sr))
    step = max(1, int(round(step_sec * sr)))
    return list(range(-radius, radius + 1, step))


def logical_union_positions(
    audio_length: int,
    dense_positions: Iterable[int],
    seg_len: int,
    *,
    jitter_sec: float = DEFAULT_EOT_JITTER_SEC,
    step_sec: float = DEFAULT_EOT_STEP_SEC,
) -> list[int]:
    """Return every valid dense/EOT absolute start exactly once."""
    starts = set()
    for center in dense_positions:
        for offset in eot_offsets(jitter_sec, step_sec):
            position = int(center) + int(offset)
            if 0 <= position and position + seg_len <= audio_length:
                starts.add(position)
    return sorted(starts)


def _score_exact_positions(
    audio_np: np.ndarray,
    positions: list[int],
    seg_len: int,
    *,
    scanner: ParallelRealScoreScanner | None = None,
    score_fn: Callable[[np.ndarray], float] = get_real_score_segment,
) -> list[float]:
    if scanner is not None:
        return scanner.scan(audio_np, positions, seg_len)
    # The injectable serial path keeps tests independent from multiprocessing.
    with ThreadPoolExecutor(max_workers=8) as pool:
        segments = [audio_np[pos:pos + seg_len] for pos in positions]
        return [float(value) for value in pool.map(score_fn, segments)]


def certify_delta(
    audio_np: np.ndarray,
    delta_np: np.ndarray,
    *,
    real_target: float = DEFAULT_REAL_TARGET,
    jitter_sec: float = DEFAULT_EOT_JITTER_SEC,
    step_sec: float = DEFAULT_EOT_STEP_SEC,
    scanner: ParallelRealScoreScanner | None = None,
    score_fn: Callable[[np.ndarray], float] = get_real_score_segment,
) -> tuple[np.ndarray, Certification]:
    """Guard, deduplicate, and exact-certify the actual candidate delta."""
    padded_audio, original_length = _pad_for_detector(audio_np)
    padded_delta = np.zeros_like(padded_audio)
    padded_delta[:len(delta_np)] = np.asarray(delta_np, dtype=np.float32)[:len(padded_audio)]
    guarded = apply_silence_guard_to_delta(
        torch.from_numpy(padded_delta), torch.from_numpy(padded_audio)
    ).numpy()
    dense_positions, seg_len = dense_window_positions(padded_audio)
    union_positions = logical_union_positions(
        len(padded_audio), dense_positions, seg_len,
        jitter_sec=jitter_sec, step_sec=step_sec,
    )
    scores = _score_exact_positions(
        padded_audio + guarded, union_positions, seg_len,
        scanner=scanner, score_fn=score_fn,
    )
    by_position = dict(zip(union_positions, scores))
    dense_scores = [by_position[pos] for pos in dense_positions if pos in by_position]
    failing = sum(score > real_target for score in scores)
    result = Certification(
        dense_worst_score=max(dense_scores, default=1.0),
        certified_worst_score=max(scores, default=1.0),
        failing_window_count=failing,
        dense_window_count=len(dense_scores),
        union_window_count=len(union_positions),
        eot_shift_count=len(eot_offsets(jitter_sec, step_sec)),
        passed=all(score <= real_target for score in scores),
    )
    return guarded[:original_length], result


def _shift_with_zeros(signal: torch.Tensor, shift_samples: int) -> torch.Tensor:
    """Shift alignment without circularly wrapping audio around the track."""
    if shift_samples == 0:
        return signal
    if abs(shift_samples) >= signal.shape[-1]:
        return torch.zeros_like(signal)
    if shift_samples > 0:
        return F.pad(signal[:-shift_samples], (shift_samples, 0))
    amount = -shift_samples
    return F.pad(signal[amount:], (0, amount))


def _objective(
    audio: torch.Tensor,
    delta: torch.Tensor,
    alignment_shift: int,
    *,
    logit_target: torch.Tensor,
    lambda_perceptual: float,
    lambda_band: float,
    lambda_tonality: float,
) -> torch.Tensor:
    """One whole-track forward graph plus the complete production penalties."""
    shifted = _shift_with_zeros(audio + delta, alignment_shift)
    dense_logits = forward_dense_logit_grid(shifted.unsqueeze(0))[0]
    logit_loss = F.leaky_relu(
        dense_logits - logit_target + 1.0, negative_slope=0.02
    ).mean()
    return (
        logit_loss
        + lambda_perceptual * perceptual_penalty(delta, audio)
        + lambda_band * band_limit_penalty(delta, lo_hz=400, hi_hz=8000, sr=SR)
        + lambda_tonality * tonality_penalty(delta)
    )


def _sanitize_boundary_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """Neutralize only measured CQT endpoint singularities.

    The whole-track CQT can produce finite logits but undefined derivatives in
    narrow endpoint bands at the literal waveform boundaries.  Alignment
    shifts can produce more than one run inside an endpoint band, so the
    contract is spatial rather than a contiguous-run assumption.  We refuse
    to sanitize any interior non-finite value or any endpoint band wider than
    the measured allowance.
    """
    finite = torch.isfinite(gradient)
    if finite.all():
        return gradient
    bad = torch.where(~finite)[0]
    boundary_start = int(bad[0])
    max_boundary = int(MAX_CQT_BOUNDARY_GRAD_SEC * SR)
    allowed = (bad < max_boundary) | (bad >= gradient.numel() - max_boundary)
    if not torch.all(allowed):
        raise ValueError("non-finite gradient is not confined to measured CQT boundary bands")
    sanitized = gradient.clone()
    boundary_mask = torch.zeros_like(finite)
    boundary_mask[:max_boundary] = True
    boundary_mask[-max_boundary:] = True
    sanitized[boundary_mask] = torch.nan_to_num(
        sanitized[boundary_mask], nan=0.0, posinf=0.0, neginf=0.0,
    )
    return sanitized


def optimize_low_iteration(
    audio_np: np.ndarray,
    *,
    method: str,
    steps: int,
    lr: float = DEFAULT_LR,
    target: float = 0.05,
    rng_seed: int = 1234,
    line_search_points: int = 4,
) -> np.ndarray:
    """Run one whole-track candidate with the requested update budget."""
    if method not in {"fgsm", "adam", "pgd", "lbfgs"}:
        raise ValueError(f"unknown prototype method: {method}")
    if steps <= 0:
        raise ValueError("steps must be positive")
    padded_audio, original_length = _pad_for_detector(audio_np)
    audio = torch.from_numpy(padded_audio)
    delta = torch.zeros_like(audio, requires_grad=True)
    objective_kwargs = dict(
        logit_target=torch.logit(torch.tensor(target), eps=1e-6),
        lambda_perceptual=DEFAULT_LAMBDA_PERCEPTUAL,
        lambda_band=DEFAULT_LAMBDA_BAND,
        lambda_tonality=DEFAULT_LAMBDA_TONALITY,
    )
    rng = np.random.default_rng(rng_seed)
    shifts = list(EOT_OFFSETS)

    def backward_once() -> torch.Tensor:
        delta.grad = None
        alignment = int(rng.choice(shifts))
        loss = _objective(audio, delta, alignment, **objective_kwargs)
        loss.backward()
        # nnAudio can return undefined derivatives at the literal end of a
        # long whole-track CQT even though the forward logits are finite.  No
        # dense logit is removed from the objective or certificate; only a
        # measured contiguous endpoint suffix is neutralized before update.
        if delta.grad is not None:
            delta.grad = _sanitize_boundary_gradient(delta.grad)
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise ValueError("non-finite gradient in whole-track CNN prototype")
        return loss.detach()

    if method == "fgsm":
        backward_once()
        direction = delta.grad.detach().sign()
        # Exact scalar line search over the certified union.  This is still a
        # one-gradient candidate; only the scalar amplitude is searched.
        candidates = [direction * float(scale) for scale in np.geomspace(
            max(lr, 1e-6), max(lr, 1e-6) * 1000, line_search_points
        )]
        with ParallelRealScoreScanner() as scanner:
            scored = []
            for candidate in candidates:
                _, cert = certify_delta(
                    padded_audio, candidate.detach().numpy(), scanner=scanner,
                )
                scored.append((cert.certified_worst_score, float(candidate.norm()), candidate))
        return min(scored, key=lambda item: (item[0], item[1]))[2].numpy()[:original_length]

    if method == "adam":
        optimizer = torch.optim.Adam([delta], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            backward_once()
            optimizer.step()
    elif method == "pgd":
        for _ in range(steps):
            backward_once()
            with torch.no_grad():
                delta -= lr * delta.grad.sign()
    else:
        # max_iter=1 makes the public ``steps`` budget equal to one
        # whole-track closure per update; no hidden line-search closures.
        optimizer = torch.optim.LBFGS(
            [delta], lr=1.0, max_iter=1, history_size=min(10, steps),
            line_search_fn=None,
        )

        def closure():
            optimizer.zero_grad()
            return backward_once()

        for _ in range(steps):
            optimizer.step(closure)

    guarded = apply_silence_guard_to_delta(delta.detach(), audio)
    return guarded.numpy()[:original_length]


def _max_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024.0


def _snr_db(audio: np.ndarray, delta: np.ndarray) -> float:
    n = min(len(audio), len(delta))
    return float(20 * np.log10(np.linalg.norm(audio[:n]) / (np.linalg.norm(delta[:n]) + 1e-12)))


def run_one(
    audio_path: str,
    *,
    method: str,
    steps: int,
    baseline: bool = False,
) -> BenchmarkResult:
    tracemalloc.start()
    started = time.perf_counter()
    audio = load_audio_mono(audio_path)
    scanner = None
    try:
        if baseline:
            delta, _, _, _ = optimize_whole_track_verified(
                audio, max_steps=steps, min_steps=steps, hop_sec=0.5,
                real_check_interval=10, verbose=False, mode="thorough",
            )
        else:
            delta = optimize_low_iteration(audio, method=method, steps=steps)
        scanner = ParallelRealScoreScanner()
        guarded, cert = certify_delta(audio, delta, scanner=scanner)
        result = BenchmarkResult(
            audio=Path(audio_path).name,
            method="current_dense" if baseline else method,
            steps=steps,
            runtime_sec=time.perf_counter() - started,
            peak_delta=float(np.abs(guarded).max(initial=0.0)),
            snr_db=_snr_db(audio, guarded),
            max_rss_mb=_max_rss_mb(),
            tracemalloc_peak_mb=tracemalloc.get_traced_memory()[1] / (1024 * 1024),
            dense_worst_score=cert.dense_worst_score,
            certified_worst_score=cert.certified_worst_score,
            failing_window_count=cert.failing_window_count,
            dense_window_count=cert.dense_window_count,
            union_window_count=cert.union_window_count,
            eot_shift_count=cert.eot_shift_count,
            passed=cert.passed,
        )
    except Exception as exc:
        result = BenchmarkResult(
            audio=Path(audio_path).name,
            method="current_dense" if baseline else method,
            steps=steps,
            runtime_sec=time.perf_counter() - started,
            peak_delta=0.0,
            snr_db=float("nan"),
            max_rss_mb=_max_rss_mb(),
            tracemalloc_peak_mb=tracemalloc.get_traced_memory()[1] / (1024 * 1024),
            dense_worst_score=float("nan"),
            certified_worst_score=float("nan"),
            failing_window_count=-1,
            dense_window_count=0,
            union_window_count=0,
            eot_shift_count=0,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if scanner is not None:
            scanner.close()
        tracemalloc.stop()
    return result


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", help="audio files to benchmark")
    parser.add_argument("--method", choices=["current_dense", "fgsm", "adam", "pgd", "lbfgs"], required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    for audio_path in args.audio:
        result = run_one(
            audio_path, method=args.method, steps=args.steps,
            baseline=args.method == "current_dense",
        )
        print(json.dumps(asdict(result), sort_keys=True) if args.json else json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    _cli()
