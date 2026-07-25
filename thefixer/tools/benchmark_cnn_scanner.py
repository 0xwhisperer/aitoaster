"""Benchmark the exact sequential and reusable parallel CNN certificates."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cnn_differentiable_v2 import get_real_score_segment, load_audio_mono
from app.cnn_real_scanner import ParallelRealScoreScanner
from app.cnn_wholetrack_optimizer_v2 import build_sliding_windows


FILES = [
    Path("/Users/daniel/Desktop/audio/thefixer/uploads/4d363ababbc6.m4a"),
    Path("/Users/daniel/Desktop/audio/thefixer/uploads/4bf180897cfb.wav"),
]


def timed(call):
    started = time.perf_counter()
    value = call()
    return value, time.perf_counter() - started


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    onnx_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for path in FILES:
        audio = load_audio_mono(str(path))
        positions, seg_len = build_sliding_windows(len(audio), hop_sec=0.5)
        print(f"{path.name}: samples={len(audio)} windows={len(positions)}", flush=True)

        sequential, sequential_seconds = timed(lambda: [
            get_real_score_segment(audio[pos:pos + seg_len])
            for pos in positions
        ])

        scanner = ParallelRealScoreScanner(workers=workers, onnx_threads=onnx_threads)
        try:
            cold, cold_seconds = timed(lambda: scanner.scan(audio, positions, seg_len))
            warm, warm_seconds = timed(lambda: scanner.scan(audio, positions, seg_len))
        finally:
            scanner.close()

        max_diff = max(
            np.max(np.abs(np.asarray(sequential) - np.asarray(cold))),
            np.max(np.abs(np.asarray(sequential) - np.asarray(warm))),
        )
        print(
            f"  sequential={sequential_seconds:.3f}s "
            f"parallel_cold={cold_seconds:.3f}s "
            f"parallel_warm={warm_seconds:.3f}s "
            f"speedup_warm={sequential_seconds / warm_seconds:.2f}x "
            f"max_abs_diff={max_diff:.3g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
