"""Run the production Thorough CNN entry point on one audio file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cnn_fix import fix_cnn
from app.detector import CNNDetector
from tools.benchmark_cnn_adaptive_transfer import load_native_stereo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--output", default="/tmp/fix_cnn_thorough.wav")
    parser.add_argument("--lr", type=float, default=0.00004)
    args = parser.parse_args()
    stereo, sr = load_native_stereo(args.audio)
    progress = []
    started = time.perf_counter()
    fixed, info = fix_cnn(
        stereo,
        sr,
        max_steps=80,
        min_steps=40,
        mode="thorough",
        parallel_lr=args.lr,
        progress_cb=progress.append,
    )
    sf.write(args.output, fixed, sr, subtype="FLOAT")
    delta = fixed - stereo
    snr = 20 * np.log10(
        np.linalg.norm(stereo) / (np.linalg.norm(delta) + 1e-12)
    )
    print(
        json.dumps(
            {
                "runtime_sec": time.perf_counter() - started,
                "output": args.output,
                "deployed_score": CNNDetector().predict(
                    args.output
                )["probability"],
                "snr_db": float(snr),
                "info": info,
                "progress": progress,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
