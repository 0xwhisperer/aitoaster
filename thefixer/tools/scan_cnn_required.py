"""Scan production Thorough starts plus deployed-position EOT starts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cnn_adaptive_dense_prototype import exact_union_scan
from app.cnn_differentiable_v2 import load_audio_mono
from app.cnn_low_iteration_prototype import dense_window_positions
from app.cnn_real_scanner import ParallelRealScoreScanner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()
    audio = torch.from_numpy(load_audio_mono(args.audio))
    with ParallelRealScoreScanner() as scanner:
        _, scan = exact_union_scan(
            audio, torch.zeros_like(audio), scanner
        )
    result = {
        "windows": len(scan.positions),
        "failures": len(scan.failing_positions),
        "worst": scan.worst_score,
        "passed": scan.passed,
    }
    if args.show_failures:
        dense, _ = dense_window_positions(audio.numpy())
        dense_set = set(dense)
        result["failure_details"] = [
            {
                "seconds": position / 16000,
                "score": score,
                "kind": "dense" if position in dense_set else "fractional",
            }
            for position, score in zip(scan.positions, scan.scores)
            if score > 0.08
        ]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
