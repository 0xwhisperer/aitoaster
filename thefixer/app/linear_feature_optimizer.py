"""Fast feature-domain optimizer for the linear fakeprint detector.

The detector discards phase and time before classification: it averages the
log-power spectrum over time, extracts a 3,585-bin fakeprint, and applies a
fixed logistic-regression classifier.  Optimizing millions of waveform samples
through that same STFT hundreds of times is therefore unnecessary for the
first-stage solve.

This module optimizes a per-frequency dB correction against the detector's
exact feature algebra, applies it once to the original complex STFT (preserving
phase), reconstructs the waveform, and checks the reconstructed result with the
real numpy/ONNX detector.  It is deliberately standalone while its perceptual
tradeoffs are evaluated; ``linear_fix`` can retain the waveform optimizer as a
fallback.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from .detector import LinearDetector
from .linear_differentiable import (
    BIAS,
    FREQ_MASK_IDX,
    HULL_AREA,
    MAX_DB,
    MIN_DB,
    N_FFT,
    SAMPLE_RATE,
    WEIGHTS,
)


@dataclass
class FeatureOptimizationResult:
    audio: np.ndarray
    score: float
    surrogate_score: float
    snr_db: float
    elapsed_sec: float
    gain_rms_db: float
    gain_peak_db: float
    regularization: float
    passed: bool


def _minimum_filter(x: torch.Tensor) -> torch.Tensor:
    pad_left = HULL_AREA // 2
    pad_right = HULL_AREA - 1 - pad_left
    padded = F.pad(
        x.unsqueeze(0).unsqueeze(0),
        (pad_left, pad_right),
        mode="replicate",
    )
    return -F.max_pool1d(-padded, kernel_size=HULL_AREA, stride=1)[0, 0]


def fakeprint_from_mean_db(mean_db: torch.Tensor) -> torch.Tensor:
    """Apply the detector's exact post-STFT fakeprint algebra."""
    spectrum = mean_db[FREQ_MASK_IDX]
    hull = torch.clamp(_minimum_filter(spectrum), min=MIN_DB)
    residue = torch.clamp(spectrum - hull, min=0.0, max=MAX_DB)
    return residue / (residue.max() + 1e-6)


def score_from_mean_db(mean_db: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.dot(WEIGHTS, fakeprint_from_mean_db(mean_db)) + BIAS)


def _reconstruct(
    original_stft: torch.Tensor,
    gain_db: torch.Tensor,
    window: torch.Tensor,
    length: int,
) -> torch.Tensor:
    gain = torch.pow(10.0, gain_db.unsqueeze(1) / 20.0)
    return torch.istft(
        original_stft * gain,
        n_fft=N_FFT,
        hop_length=N_FFT // 4,
        win_length=N_FFT,
        window=window,
        center=True,
        length=length,
    )


def _snr_db(original: np.ndarray, candidate: np.ndarray) -> float:
    delta = candidate - original
    return float(
        20
        * np.log10(
            np.linalg.norm(original) / (np.linalg.norm(delta) + 1e-12)
        )
    )


def optimize_feature_eq(
    audio_orig: np.ndarray,
    *,
    target_score: float = 5e-5,
    logit_target: float = -10.0,
    iterations: int = 100,
    learning_rate: float = 0.03,
    regularization_values: tuple[float, ...] = (8.0, 4.0, 2.0, 1.0, 0.3),
    smoothness_ratio: float = 10.0,
    max_gain_db: float = 3.0,
    detector: LinearDetector | None = None,
) -> FeatureOptimizationResult:
    """Find the highest-SNR reconstructed candidate that clears ``target_score``.

    The regularization sweep is ordered from gentlest to strongest correction.
    It stops at the first candidate that passes the reconstructed-waveform
    detector check, so easy files avoid the more aggressive frontier points.
    """
    if audio_orig.ndim != 1:
        raise ValueError("audio_orig must be a mono 1-D array")
    if len(audio_orig) <= N_FFT // 2:
        raise ValueError(f"audio must contain more than {N_FFT // 2} samples")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if not regularization_values:
        raise ValueError("regularization_values must not be empty")

    started = perf_counter()
    detector = detector or LinearDetector()
    original_np = np.asarray(audio_orig, dtype=np.float32)
    original = torch.tensor(original_np, dtype=torch.float32)
    window = torch.hann_window(N_FFT, periodic=True)

    original_stft = torch.stft(
        original,
        n_fft=N_FFT,
        hop_length=N_FFT // 4,
        win_length=N_FFT,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    original_db = 10.0 * torch.log10(
        torch.clamp(original_stft.abs().square(), min=1e-10, max=1e6)
    )
    mean_db = original_db.mean(dim=1).detach()

    baseline_fp = detector.compute_fakeprint(original_np, SAMPLE_RATE)
    baseline_output = detector.session.run(
        None, {detector.input_name: baseline_fp.reshape(1, -1)}
    )
    baseline_score = float(baseline_output[0][0, 0])
    if baseline_score < target_score:
        return FeatureOptimizationResult(
            audio=original_np.copy(),
            score=baseline_score,
            surrogate_score=baseline_score,
            snr_db=float("inf"),
            elapsed_sec=perf_counter() - started,
            gain_rms_db=0.0,
            gain_peak_db=0.0,
            regularization=0.0,
            passed=True,
        )

    best_result = None
    for regularization in regularization_values:
        gain_db = torch.zeros_like(mean_db, requires_grad=True)
        optimizer = torch.optim.Adam([gain_db], lr=learning_rate)

        for _ in range(iterations):
            optimizer.zero_grad()
            fakeprint = fakeprint_from_mean_db(mean_db + gain_db)
            logit = torch.dot(WEIGHTS, fakeprint) + BIAS
            magnitude_cost = gain_db.square().mean()
            smoothness_cost = (gain_db[1:] - gain_db[:-1]).square().mean()
            constraint = F.relu(logit - logit_target).square()
            loss = constraint + regularization * (
                magnitude_cost + smoothness_ratio * smoothness_cost
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                gain_db.clamp_(-max_gain_db, max_gain_db)

        with torch.no_grad():
            candidate_t = _reconstruct(
                original_stft, gain_db, window, len(original_np)
            )
            candidate = candidate_t.numpy().astype(np.float32, copy=False)
            surrogate_score = float(score_from_mean_db(mean_db + gain_db))
            gain_rms_db = float(gain_db.square().mean().sqrt())
            gain_peak_db = float(gain_db.abs().max())

        candidate_fp = detector.compute_fakeprint(candidate, SAMPLE_RATE)
        candidate_output = detector.session.run(
            None, {detector.input_name: candidate_fp.reshape(1, -1)}
        )
        candidate_score = float(candidate_output[0][0, 0])
        result = FeatureOptimizationResult(
            audio=candidate.copy(),
            score=candidate_score,
            surrogate_score=surrogate_score,
            snr_db=_snr_db(original_np, candidate),
            elapsed_sec=perf_counter() - started,
            gain_rms_db=gain_rms_db,
            gain_peak_db=gain_peak_db,
            regularization=float(regularization),
            passed=candidate_score < target_score,
        )

        if best_result is None:
            best_result = result
        elif result.passed and (
            not best_result.passed or result.snr_db > best_result.snr_db
        ):
            best_result = result
        elif not best_result.passed and result.score < best_result.score:
            best_result = result

        if result.passed:
            break

    assert best_result is not None
    best_result.elapsed_sec = perf_counter() - started
    return best_result
