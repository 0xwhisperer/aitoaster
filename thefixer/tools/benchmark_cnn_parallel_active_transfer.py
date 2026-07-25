"""Benchmark the exact parallel active optimizer through native transfer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cnn_adaptive_dense_prototype import exact_union_scan
from app.cnn_differentiable_v2 import load_audio_mono
from app.cnn_fix import CNN_SR, _resample_mono
from app.cnn_parallel_optimizer import optimize_parallel_active
from app.cnn_real_scanner import ParallelRealScoreScanner
from app.detector import CNNDetector
from tools.benchmark_cnn_adaptive_transfer import (
    load_native_stereo,
    transfer_delta,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--min-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--repair-rounds", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()

    original_mono = load_audio_mono(args.audio)
    stereo, native_sr = load_native_stereo(args.audio)

    def delivery_transform(delta_16k):
        candidate = transfer_delta(stereo, native_sr, delta_16k)
        mono = candidate.mean(axis=1)
        return (
            _resample_mono(mono, native_sr, CNN_SR)
            if native_sr != CNN_SR
            else mono
        )

    started = time.perf_counter()
    delta, pre_scan, timing = optimize_parallel_active(
        original_mono,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        lr=args.lr,
        max_repair_rounds=args.repair_rounds,
        delivery_transform=delivery_transform,
    )
    transferred = transfer_delta(stereo, native_sr, delta)
    output = args.output or "/tmp/cnn_parallel_active_transfer.wav"
    sf.write(output, transferred, native_sr, subtype="FLOAT")

    native_mono = load_audio_mono(output)
    with ParallelRealScoreScanner() as scanner:
        _, post_scan = exact_union_scan(
            torch.from_numpy(native_mono),
            torch.zeros_like(torch.from_numpy(native_mono)),
            scanner,
        )
    deployed_score = CNNDetector().predict(output)["probability"]
    n = min(len(original_mono), len(native_mono))
    native_delta = native_mono[:n] - original_mono[:n]
    native_snr = 20 * np.log10(
        np.linalg.norm(original_mono[:n])
        / (np.linalg.norm(native_delta) + 1e-12)
    )
    print(
        json.dumps(
            {
                "runtime_sec": time.perf_counter() - started,
                "output": output,
                "pre_transfer_passed": pre_scan.passed,
                "pre_transfer_worst": pre_scan.worst_score,
                "pre_transfer_failures": len(pre_scan.failing_positions),
                "pre_transfer_windows": len(pre_scan.positions),
                "post_transfer_passed": post_scan.passed,
                "post_transfer_worst": post_scan.worst_score,
                "post_transfer_failures": len(post_scan.failing_positions),
                "post_transfer_windows": len(post_scan.positions),
                "deployed_score": deployed_score,
                "native_snr_db": float(native_snr),
                "native_peak_delta": float(
                    np.abs(native_delta).max(initial=0)
                ),
                **timing,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
