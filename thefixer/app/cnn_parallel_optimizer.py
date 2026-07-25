"""Parallel active-set optimizer for exact local-window CNN correction."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .cnn_adaptive_dense_prototype import (
    ExactScan,
    _check_gradient,
    _quality_loss,
    _repair_step,
    exact_subset_scan,
    exact_union_scan,
    required_exact_positions,
)
from .cnn_differentiable_v2 import load_audio_mono
from .cnn_gradient_optimizer_v2 import apply_silence_guard_to_delta
from .cnn_local_gradient_pool import LocalGradientPool
from .cnn_low_iteration_prototype import (
    DEFAULT_LAMBDA_BAND,
    DEFAULT_LAMBDA_PERCEPTUAL,
    DEFAULT_LAMBDA_TONALITY,
    DEFAULT_LR,
    DEFAULT_REAL_TARGET,
    _pad_for_detector,
    dense_window_positions,
)
from .cnn_quality_context import CachedCNNQualityPenalty
from .cnn_real_scanner import ParallelRealScoreScanner


def _initial_weights(
    positions: list[int],
    scores: list[float],
    real_target: float,
) -> dict[int, float]:
    weights = {}
    for position, score in zip(positions, scores):
        weights[position] = (
            min(
                20.0,
                1.0
                + 19.0
                * max(0.0, score - real_target)
                / max(1e-6, 1.0 - real_target),
            )
            if score > real_target
            else 1.0
        )
    total = sum(weights.values())
    if total:
        normalization = len(positions) / total
        weights = {
            position: weight * normalization
            for position, weight in weights.items()
        }
    return weights


def _active_positions(
    positions: list[int],
    scores: dict[int, float],
    *,
    real_target: float,
    minimum: int = 32,
) -> list[int]:
    # Keep all failures plus a real-score safety margin.  If few remain,
    # retain the worst passing starts as sentinels.  Every position is still
    # rescanned exactly at each check and is re-added immediately if it
    # regresses.
    active = [
        position
        for position in positions
        if scores[position] > real_target * 0.5
    ]
    if len(active) < min(minimum, len(positions)):
        active = sorted(
            positions, key=lambda position: scores[position], reverse=True
        )[:minimum]
    return sorted(set(active))


def _sentinel_positions(
    positions: list[int],
    *,
    count: int = 32,
) -> list[int]:
    """Choose deterministic whole-track sentinels for partial rescans."""
    if not positions or count <= 0:
        return []
    indices = np.linspace(
        0, len(positions) - 1, min(count, len(positions)), dtype=int
    )
    return [positions[int(index)] for index in indices]


def optimize_parallel_active(
    audio_np: np.ndarray,
    *,
    max_steps: int = 140,
    min_steps: int = 80,
    real_check_interval: int = 10,
    full_check_interval: int = 30,
    lr: float = DEFAULT_LR,
    target: float = 0.05,
    real_target: float = DEFAULT_REAL_TARGET,
    repair_steps: int = 12,
    max_repair_rounds: int = 3,
    gradient_workers: int = 5,
    gradient_threads: int = 2,
    delivery_transform=None,
    delivery_check_steps: tuple[int, ...] = (50, 60),
    progress_cb=None,
) -> tuple[np.ndarray, object, dict]:
    if (
        max_steps < 1
        or min_steps < 0
        or real_check_interval < 1
        or full_check_interval < real_check_interval
        or full_check_interval % real_check_interval
    ):
        raise ValueError("invalid parallel CNN check/step schedule")
    padded_audio, original_length = _pad_for_detector(audio_np)
    audio = torch.from_numpy(padded_audio)
    quality_context = CachedCNNQualityPenalty(audio)
    delta = torch.zeros_like(audio, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    logit_target = torch.logit(torch.tensor(target), eps=1e-6)
    # Optimize the exact delivered-file positions from the beginning instead
    # of discovering them only during final certification.
    positions, seg_len = required_exact_positions(padded_audio)
    sentinels = _sentinel_positions(positions)

    timing = {
        "gradient_sec": 0.0,
        "quality_sec": 0.0,
        "certificate_sec": 0.0,
        "repair_sec": 0.0,
        "steps_run": 0,
        "repair_rounds_run": 0,
        "repair_steps_run": 0,
        "active_counts": [],
        "delivery_checks": 0,
        "accepted_delivery": False,
    }
    with (
        LocalGradientPool(
            padded_audio,
            workers=gradient_workers,
            torch_threads=gradient_threads,
            target=target,
        ) as gradient_pool,
        ParallelRealScoreScanner() as scanner,
    ):
        started = time.perf_counter()
        initial_scores = scanner.scan(padded_audio, positions, seg_len)
        timing["certificate_sec"] += time.perf_counter() - started
        score_by_position = dict(zip(positions, initial_scores))
        weights = _initial_weights(positions, initial_scores, real_target)
        active = _active_positions(
            positions, score_by_position, real_target=real_target
        )
        best_delta = delta.detach().clone()
        best_real_max = max(initial_scores, default=1.0)

        for step in range(max_steps):
            optimizer.zero_grad()
            started = time.perf_counter()
            cnn_gradient, estimate, _ = gradient_pool.gradient(
                delta.detach().numpy(),
                active,
                [weights[position] for position in active],
            )
            timing["gradient_sec"] += time.perf_counter() - started

            started = time.perf_counter()
            quality = quality_context.loss(
                delta,
                lambda_perceptual=DEFAULT_LAMBDA_PERCEPTUAL,
                lambda_band=DEFAULT_LAMBDA_BAND,
                lambda_tonality=DEFAULT_LAMBDA_TONALITY,
            )
            quality.backward()
            _check_gradient(delta, whole_track=False)
            delta.grad.add_(torch.from_numpy(cnn_gradient))
            if not torch.isfinite(delta.grad).all():
                raise ValueError("combined parallel CNN gradient is non-finite")
            optimizer.step()
            timing["quality_sec"] += time.perf_counter() - started
            timing["steps_run"] = step + 1
            timing["active_counts"].append(len(active))
            if progress_cb is not None:
                progress_cb(
                    "active",
                    step + 1,
                    max_steps,
                    estimate,
                    len(active),
                )

            if (step + 1) % real_check_interval:
                continue
            started = time.perf_counter()
            candidate = (audio + delta.detach()).numpy()
            full_check = (
                (step + 1) % full_check_interval == 0
                or step + 1 == max_steps
            )
            check_positions = (
                positions
                if full_check
                else sorted(set(active).union(sentinels))
            )
            real_scores = scanner.scan(
                candidate, check_positions, seg_len
            )
            timing["certificate_sec"] += time.perf_counter() - started
            score_by_position.update(
                zip(check_positions, real_scores)
            )
            real_max = max(real_scores, default=1.0)
            if full_check and real_max < best_real_max:
                best_real_max = real_max
                best_delta = delta.detach().clone()
            for position, score in zip(check_positions, real_scores):
                if score > real_target:
                    weights[position] = min(
                        weights[position] * 1.5, 20.0
                    )
                else:
                    weights[position] = max(
                        weights[position] * 0.9, 1.0
                    )
            active = _active_positions(
                positions, score_by_position, real_target=real_target
            )
            comfortably_clear = real_max < real_target * 0.5
            pre_transfer_clear = (
                full_check
                and real_max < real_target
                and (step + 1 >= min_steps or comfortably_clear)
            )
            should_check_delivery = (
                delivery_transform is not None
                and (
                    pre_transfer_clear
                    or step + 1 in delivery_check_steps
                    or step + 1 == max_steps
                )
            )
            if should_check_delivery:
                started = time.perf_counter()
                guarded_candidate = apply_silence_guard_to_delta(
                    delta.detach(), audio
                )
                delivered = np.asarray(
                    delivery_transform(
                        guarded_candidate.numpy()[:original_length]
                    ),
                    dtype=np.float32,
                )
                if delivered.ndim != 1:
                    raise ValueError(
                        "CNN delivery transform must return mono audio"
                    )
                delivery_positions, delivery_seg_len = (
                    required_exact_positions(delivered)
                )
                delivery_scores = scanner.scan(
                    delivered, delivery_positions, delivery_seg_len
                )
                timing["certificate_sec"] += (
                    time.perf_counter() - started
                )
                timing["delivery_checks"] += 1
                delivery_failures = [
                    position
                    for position, score in zip(
                        delivery_positions, delivery_scores
                    )
                    if score > real_target
                ]
                delivery_scan = ExactScan(
                    positions=delivery_positions,
                    scores=delivery_scores,
                    failing_positions=delivery_failures,
                    worst_score=max(delivery_scores, default=1.0),
                    passed=not delivery_failures,
                )
                if progress_cb is not None:
                    progress_cb(
                        "delivery",
                        step + 1,
                        max_steps,
                        delivery_scan.worst_score,
                        len(delivery_failures),
                    )
                if delivery_scan.passed:
                    timing["accepted_delivery"] = True
                    return (
                        guarded_candidate.numpy()[:original_length],
                        delivery_scan,
                        timing,
                    )
                # Native transfer failures are the authoritative active set.
                # Keep the current Adam moments and correction, then focus
                # subsequent exact gradients on those starts.
                for position, score in zip(
                    delivery_positions, delivery_scores
                ):
                    if position not in score_by_position:
                        continue
                    score_by_position[position] = score
                    if score > real_target:
                        weights[position] = min(
                            weights[position] * 1.5, 20.0
                        )
                active = sorted(
                    set(active).union(
                        position
                        for position in delivery_failures
                        if position in score_by_position
                    )
                )
            elif pre_transfer_clear:
                break

        with torch.no_grad():
            delta.copy_(best_delta)
        started = time.perf_counter()
        guarded, scan = exact_union_scan(audio, delta, scanner)
        timing["certificate_sec"] += time.perf_counter() - started
        with torch.no_grad():
            delta.copy_(guarded)
        scan_is_full = True

        for _ in range(max_repair_rounds):
            if scan.passed:
                break
            timing["repair_rounds_run"] += 1
            previous_failures = list(scan.failing_positions)
            repair_scores = dict(zip(scan.positions, scan.scores))
            for _ in range(repair_steps):
                started = time.perf_counter()
                optimizer.zero_grad()
                loss, estimate = _repair_step(
                    audio,
                    delta,
                    scan.failing_positions,
                    repair_scores,
                    logit_target,
                )
                loss.backward()
                _check_gradient(delta, whole_track=False)
                optimizer.step()
                timing["repair_sec"] += time.perf_counter() - started
                timing["repair_steps_run"] += 1
                if progress_cb is not None:
                    progress_cb(
                        "repair",
                        timing["repair_steps_run"],
                        repair_steps * max_repair_rounds,
                        estimate,
                        len(scan.failing_positions),
                    )
            started = time.perf_counter()
            guarded, scan = exact_subset_scan(
                audio, delta, scanner, previous_failures
            )
            timing["certificate_sec"] += time.perf_counter() - started
            with torch.no_grad():
                delta.copy_(guarded)
            scan_is_full = False
            if scan.passed:
                started = time.perf_counter()
                guarded, scan = exact_union_scan(audio, delta, scanner)
                timing["certificate_sec"] += time.perf_counter() - started
                with torch.no_grad():
                    delta.copy_(guarded)
                scan_is_full = True

        if not scan_is_full:
            started = time.perf_counter()
            guarded, scan = exact_union_scan(audio, delta, scanner)
            timing["certificate_sec"] += time.perf_counter() - started
            with torch.no_grad():
                delta.copy_(guarded)
        return guarded.numpy()[:original_length], scan, timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--min-steps", type=int, default=80)
    args = parser.parse_args()
    audio = load_audio_mono(args.audio)
    started = time.perf_counter()
    delta, scan, timing = optimize_parallel_active(
        audio, max_steps=args.max_steps, min_steps=args.min_steps
    )
    snr = 20 * np.log10(
        np.linalg.norm(audio) / (np.linalg.norm(delta) + 1e-12)
    )
    print(
        json.dumps(
            {
                "runtime_sec": time.perf_counter() - started,
                "passed": scan.passed,
                "worst_score": scan.worst_score,
                "failures": len(scan.failing_positions),
                "union_windows": len(scan.positions),
                "snr_db": float(snr),
                "peak_delta": float(np.abs(delta).max(initial=0)),
                **timing,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
