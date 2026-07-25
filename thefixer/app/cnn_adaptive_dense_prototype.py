"""Adaptive whole-track CNN optimizer prototype.

This keeps the fast one-graph whole-track surrogate, but periodically closes
the surrogate/real-model gap with the unchanged exact librosa/ONNX scanner.
After the dense proposal, only exact failing starts are used for short repair
rounds.  No certificate window, EOT shift, silence guard, or quality penalty is
removed.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .cnn_differentiable_v2 import (
    SEGMENT_SAMPLES,
    forward_logit_differentiable,
    get_real_evaluator_segments,
    load_audio_mono,
)
from .cnn_gradient_optimizer_v2 import (
    apply_silence_guard_to_delta,
    band_limit_penalty,
    perceptual_penalty,
    tonality_penalty,
)
from .cnn_low_iteration_prototype import (
    DEFAULT_LAMBDA_BAND,
    DEFAULT_LAMBDA_PERCEPTUAL,
    DEFAULT_LAMBDA_TONALITY,
    DEFAULT_LR,
    DEFAULT_REAL_TARGET,
    EOT_OFFSETS,
    _pad_for_detector,
    _sanitize_boundary_gradient,
    _score_exact_positions,
    _shift_with_zeros,
    dense_window_positions,
)
from .cnn_real_scanner import ParallelRealScoreScanner
from .cnn_wholetrack_dense_prototype import forward_dense_logit_grid


@dataclass
class ExactScan:
    positions: list[int]
    scores: list[float]
    failing_positions: list[int]
    worst_score: float
    passed: bool


@dataclass
class AdaptiveResult:
    audio: str
    base_steps: int
    repair_rounds_run: int
    repair_steps_run: int
    runtime_sec: float
    optimization_sec: float
    certificate_sec: float
    certified_worst_score: float
    failing_window_count: int
    union_window_count: int
    snr_db: float
    peak_delta: float
    max_rss_mb: float
    passed: bool
    error: str | None = None


def _quality_loss(
    delta: torch.Tensor,
    audio: torch.Tensor,
    *,
    lambda_perceptual: float = DEFAULT_LAMBDA_PERCEPTUAL,
    lambda_band: float = DEFAULT_LAMBDA_BAND,
    lambda_tonality: float = DEFAULT_LAMBDA_TONALITY,
) -> torch.Tensor:
    return (
        lambda_perceptual * perceptual_penalty(delta, audio)
        + lambda_band
        * band_limit_penalty(delta, lo_hz=400, hi_hz=8000, sr=16000)
        + lambda_tonality * tonality_penalty(delta)
    )


def _check_gradient(delta: torch.Tensor, *, whole_track: bool) -> None:
    if delta.grad is None:
        raise ValueError("CNN optimizer produced no gradient")
    if whole_track:
        delta.grad = _sanitize_boundary_gradient(delta.grad)
    if not torch.isfinite(delta.grad).all():
        raise ValueError("non-finite gradient in adaptive CNN optimizer")


def _dense_step(
    audio: torch.Tensor,
    delta: torch.Tensor,
    alignment_shift: int,
    logit_target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    shifted = _shift_with_zeros(audio + delta, alignment_shift)
    logits = forward_dense_logit_grid(shifted.unsqueeze(0))[0]
    # Preserve pressure on every dense cell while giving the worst cells
    # enough influence not to disappear inside a 500-window mean.
    margins = F.leaky_relu(
        logits - logit_target + 1.0, negative_slope=0.02
    )
    hardest = margins.topk(min(32, margins.numel())).values.mean()
    detector_loss = 0.5 * margins.mean() + 0.5 * hardest
    return detector_loss + _quality_loss(delta, audio), float(
        torch.sigmoid(logits.detach()).max()
    )


def _repair_step(
    audio: torch.Tensor,
    delta: torch.Tensor,
    failing_positions: list[int],
    failing_scores: dict[int, float],
    logit_target: torch.Tensor,
    *,
    max_active: int = 16,
) -> tuple[torch.Tensor, float]:
    ordered = sorted(
        failing_positions,
        key=lambda position: failing_scores[position],
        reverse=True,
    )[:max_active]
    perturbed = audio + delta
    terms = []
    max_score = 0.0
    for position in ordered:
        segment = perturbed[position : position + SEGMENT_SAMPLES]
        logit = forward_logit_differentiable(segment.unsqueeze(0))
        weight = 1.0 + 19.0 * max(
            0.0,
            (failing_scores[position] - DEFAULT_REAL_TARGET)
            / (1.0 - DEFAULT_REAL_TARGET),
        )
        terms.append(
            weight
            * F.leaky_relu(
                logit - logit_target + 1.0, negative_slope=0.02
            )
        )
        max_score = max(max_score, float(torch.sigmoid(logit.detach())))
    detector_loss = torch.stack(terms).mean()
    return detector_loss + _quality_loss(delta, audio), max_score


def exact_union_scan(
    audio: torch.Tensor,
    delta: torch.Tensor,
    scanner: ParallelRealScoreScanner,
    *,
    real_target: float = DEFAULT_REAL_TARGET,
) -> tuple[torch.Tensor, ExactScan]:
    guarded = apply_silence_guard_to_delta(delta.detach(), audio)
    audio_np = audio.detach().numpy()
    candidate = audio_np + guarded.numpy()
    positions, seg_len = required_exact_positions(audio_np)
    scores = _score_exact_positions(
        candidate, positions, seg_len, scanner=scanner
    )
    failing = [
        position
        for position, score in zip(positions, scores)
        if score > real_target
    ]
    return guarded, ExactScan(
        positions=positions,
        scores=scores,
        failing_positions=failing,
        worst_score=max(scores, default=1.0),
        passed=not failing,
    )


def required_exact_positions(
    audio_np: np.ndarray,
) -> tuple[list[int], int]:
    """Return every start that optimization and acceptance must protect."""
    dense_positions, seg_len = dense_window_positions(audio_np)
    # Preserve every production Thorough-mode 0.5-second start.
    positions = set(dense_positions)
    # The detector's five starts are generally fractional relative to both
    # the 0.5-second production grid and the 0.1-second EOT lattice (for
    # example 69.13s).  A real delivered file demonstrated that a candidate
    # could pass every lattice point yet score 99.9% at one of these omitted
    # starts.  Include the deployed starts and their complete EOT
    # neighborhoods explicitly; never infer that a nearby grid point is
    # equivalent for this position-brittle adversarial correction.  Applying
    # all 11 shifts to every one of the 500+ dense starts would be a different
    # 0.1-second whole-track product, not functionality production Thorough
    # currently provides.  The union below is therefore strictly stronger
    # than production while remaining tied to actual delivery behavior.
    evaluator_positions = get_real_evaluator_segments(audio_np, n_segments=5)
    for center in evaluator_positions:
        for offset in EOT_OFFSETS:
            position = int(center) + int(offset)
            if 0 <= position and position + seg_len <= len(audio_np):
                positions.add(position)
    return sorted(positions), seg_len


def exact_subset_scan(
    audio: torch.Tensor,
    delta: torch.Tensor,
    scanner: ParallelRealScoreScanner,
    positions: list[int],
    *,
    real_target: float = DEFAULT_REAL_TARGET,
) -> tuple[torch.Tensor, ExactScan]:
    """Recheck known failures between repair rounds.

    This is never an acceptance certificate.  It only avoids rescanning more
    than 2,700 already-passing starts while known failures are still bad.  A
    complete ``exact_union_scan`` always runs once those failures clear and
    again before returning if the repair budget ends.
    """
    guarded = apply_silence_guard_to_delta(delta.detach(), audio)
    candidate = audio.detach().numpy() + guarded.numpy()
    scores = _score_exact_positions(
        candidate,
        positions,
        SEGMENT_SAMPLES,
        scanner=scanner,
    )
    failing = [
        position
        for position, score in zip(positions, scores)
        if score > real_target
    ]
    return guarded, ExactScan(
        positions=list(positions),
        scores=scores,
        failing_positions=failing,
        worst_score=max(scores, default=1.0),
        passed=not failing,
    )


def optimize_adaptive_dense(
    audio_np: np.ndarray,
    *,
    base_steps: int = 120,
    repair_steps: int = 12,
    max_repair_rounds: int = 3,
    lr: float = DEFAULT_LR,
    target: float = 0.05,
    scanner: ParallelRealScoreScanner | None = None,
    progress_cb=None,
) -> tuple[np.ndarray, ExactScan, dict]:
    if base_steps < 1 or repair_steps < 1 or max_repair_rounds < 0:
        raise ValueError("adaptive CNN step counts are invalid")
    padded_audio, original_length = _pad_for_detector(audio_np)
    audio = torch.from_numpy(padded_audio)
    delta = torch.zeros_like(audio, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    logit_target = torch.logit(torch.tensor(target), eps=1e-6)
    own_scanner = scanner is None
    scanner = scanner or ParallelRealScoreScanner()
    optimization_sec = 0.0
    certificate_sec = 0.0
    repair_rounds_run = 0
    repair_steps_run = 0
    try:
        for step in range(base_steps):
            started = time.perf_counter()
            optimizer.zero_grad()
            # Cycle rather than randomly sample so every requested EOT
            # alignment receives the same number of whole-track updates.
            shift = EOT_OFFSETS[step % len(EOT_OFFSETS)]
            loss, estimate = _dense_step(
                audio, delta, shift, logit_target
            )
            loss.backward()
            _check_gradient(delta, whole_track=True)
            optimizer.step()
            optimization_sec += time.perf_counter() - started
            if progress_cb is not None:
                progress_cb("base", step + 1, base_steps, estimate, None)

        started = time.perf_counter()
        guarded, scan = exact_union_scan(audio, delta, scanner)
        certificate_sec += time.perf_counter() - started
        with torch.no_grad():
            delta.copy_(guarded)
        scan_is_full = True

        for repair_round in range(max_repair_rounds):
            if scan.passed:
                break
            repair_rounds_run += 1
            previously_failing = list(scan.failing_positions)
            score_by_position = dict(zip(scan.positions, scan.scores))
            for step in range(repair_steps):
                started = time.perf_counter()
                optimizer.zero_grad()
                loss, estimate = _repair_step(
                    audio,
                    delta,
                    scan.failing_positions,
                    score_by_position,
                    logit_target,
                )
                loss.backward()
                _check_gradient(delta, whole_track=False)
                optimizer.step()
                optimization_sec += time.perf_counter() - started
                repair_steps_run += 1
                if progress_cb is not None:
                    progress_cb(
                        "repair",
                        step + 1,
                        repair_steps,
                        estimate,
                        len(scan.failing_positions),
                    )
            started = time.perf_counter()
            guarded, scan = exact_subset_scan(
                audio, delta, scanner, previously_failing
            )
            certificate_sec += time.perf_counter() - started
            with torch.no_grad():
                delta.copy_(guarded)
            scan_is_full = False
            if scan.passed:
                # Only a complete union scan can accept a candidate or reveal
                # a newly regressed start.
                started = time.perf_counter()
                guarded, scan = exact_union_scan(audio, delta, scanner)
                certificate_sec += time.perf_counter() - started
                with torch.no_grad():
                    delta.copy_(guarded)
                scan_is_full = True

        if not scan_is_full:
            started = time.perf_counter()
            guarded, scan = exact_union_scan(audio, delta, scanner)
            certificate_sec += time.perf_counter() - started
            with torch.no_grad():
                delta.copy_(guarded)
        return guarded.numpy()[:original_length], scan, {
            "optimization_sec": optimization_sec,
            "certificate_sec": certificate_sec,
            "repair_rounds_run": repair_rounds_run,
            "repair_steps_run": repair_steps_run,
        }
    finally:
        if own_scanner:
            scanner.close()


def _snr_db(audio: np.ndarray, delta: np.ndarray) -> float:
    n = min(len(audio), len(delta))
    return float(
        20
        * np.log10(
            np.linalg.norm(audio[:n]) / (np.linalg.norm(delta[:n]) + 1e-12)
        )
    )


def _max_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def run_one(
    audio_path: str,
    *,
    base_steps: int,
    repair_steps: int,
    max_repair_rounds: int,
) -> AdaptiveResult:
    started = time.perf_counter()
    audio = load_audio_mono(audio_path)
    try:
        delta, scan, timing = optimize_adaptive_dense(
            audio,
            base_steps=base_steps,
            repair_steps=repair_steps,
            max_repair_rounds=max_repair_rounds,
        )
        return AdaptiveResult(
            audio=Path(audio_path).name,
            base_steps=base_steps,
            repair_rounds_run=timing["repair_rounds_run"],
            repair_steps_run=timing["repair_steps_run"],
            runtime_sec=time.perf_counter() - started,
            optimization_sec=timing["optimization_sec"],
            certificate_sec=timing["certificate_sec"],
            certified_worst_score=scan.worst_score,
            failing_window_count=len(scan.failing_positions),
            union_window_count=len(scan.positions),
            snr_db=_snr_db(audio, delta),
            peak_delta=float(np.abs(delta).max(initial=0)),
            max_rss_mb=_max_rss_mb(),
            passed=scan.passed,
        )
    except Exception as exc:
        return AdaptiveResult(
            audio=Path(audio_path).name,
            base_steps=base_steps,
            repair_rounds_run=0,
            repair_steps_run=0,
            runtime_sec=time.perf_counter() - started,
            optimization_sec=0,
            certificate_sec=0,
            certified_worst_score=float("nan"),
            failing_window_count=-1,
            union_window_count=0,
            snr_db=float("nan"),
            peak_delta=0,
            max_rss_mb=_max_rss_mb(),
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--base-steps", type=int, default=120)
    parser.add_argument("--repair-steps", type=int, default=12)
    parser.add_argument("--repair-rounds", type=int, default=3)
    args = parser.parse_args()
    result = run_one(
        args.audio,
        base_steps=args.base_steps,
        repair_steps=args.repair_steps,
        max_repair_rounds=args.repair_rounds,
    )
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    _cli()
