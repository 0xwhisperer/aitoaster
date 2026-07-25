"""Benchmark adaptive CNN optimization through the real delivery transfer."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.cnn_adaptive_dense_prototype import (
    _snr_db,
    exact_union_scan,
    optimize_adaptive_dense,
)
from app.cnn_differentiable_v2 import load_audio_mono
from app.cnn_fix import CNN_SR, _resample_mono
from app.cnn_real_scanner import ParallelRealScoreScanner
from app.detector import CNNDetector


def load_native_stereo(path: str) -> tuple[np.ndarray, int]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    sr = int(probe.stdout.strip())
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "quiet",
            "-i",
            path,
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            str(sr),
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(decoded.stdout, dtype=np.float32).copy().reshape(-1, 2), sr


def transfer_delta(stereo: np.ndarray, sr: int, delta_16k: np.ndarray) -> np.ndarray:
    peak = float(np.abs(delta_16k).max(initial=0))
    if peak < 1e-9:
        return stereo.copy()
    scale = 0.9 / peak
    normalized = delta_16k * scale
    native_normalized = (
        _resample_mono(normalized, CNN_SR, sr)
        if sr != CNN_SR
        else normalized
    )
    native_delta = native_normalized / scale
    output = stereo.copy()
    n = min(len(output), len(native_delta))
    output[:n, 0] += native_delta[:n]
    output[:n, 1] += native_delta[:n]
    output_peak = float(np.abs(output).max(initial=0))
    if output_peak > 0.97:
        output *= 0.97 / output_peak
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("output")
    parser.add_argument("--base-steps", type=int, default=100)
    parser.add_argument("--repair-steps", type=int, default=12)
    parser.add_argument("--repair-rounds", type=int, default=3)
    args = parser.parse_args()

    started = time.perf_counter()
    stereo, sr = load_native_stereo(args.audio)
    model_audio = load_audio_mono(args.audio)
    with ParallelRealScoreScanner() as scanner:
        delta, pre_scan, timing = optimize_adaptive_dense(
            model_audio,
            base_steps=args.base_steps,
            repair_steps=args.repair_steps,
            max_repair_rounds=args.repair_rounds,
            scanner=scanner,
        )
        delivered = transfer_delta(stereo, sr, delta)
        sf.write(args.output, delivered, sr, subtype="PCM_16")
        transferred_model_audio = load_audio_mono(args.output)
        import torch

        _, post_scan = exact_union_scan(
            torch.from_numpy(transferred_model_audio),
            torch.zeros_like(torch.from_numpy(transferred_model_audio)),
            scanner,
        )

    deployed = CNNDetector().predict(args.output)
    native_snr = _snr_db(
        stereo.reshape(-1), (delivered - stereo).reshape(-1)
    )
    print(
        json.dumps(
            {
                "audio": Path(args.audio).name,
                "output": str(Path(args.output)),
                "runtime_sec": time.perf_counter() - started,
                "optimization_sec": timing["optimization_sec"],
                "certificate_sec": timing["certificate_sec"],
                "repair_rounds_run": timing["repair_rounds_run"],
                "repair_steps_run": timing["repair_steps_run"],
                "pre_transfer_worst": pre_scan.worst_score,
                "pre_transfer_failures": len(pre_scan.failing_positions),
                "post_transfer_worst": post_scan.worst_score,
                "post_transfer_failures": len(post_scan.failing_positions),
                "union_windows": len(post_scan.positions),
                "deployed_probability": deployed["probability"],
                "deployed_segment_probs": deployed["segment_probs"],
                "native_snr_db": native_snr,
                "model_snr_db": _snr_db(model_audio, delta),
                "peak_delta_16k": float(np.abs(delta).max(initial=0)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
