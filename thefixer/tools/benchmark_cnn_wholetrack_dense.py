#!/usr/bin/env python3
"""Run the dense CNN prototype diagnostics on one or more audio files."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

# Allow the documented ``python tools/...`` invocation from thefixer/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cnn_differentiable_v2 import load_audio_mono
from app.cnn_wholetrack_dense_prototype import (
    benchmark_exact_certificate,
    benchmark_forward_backward,
    compare_dense_with_differentiable_standalone,
    compare_dense_with_exact,
    forward_dense_logit_grid,
    probe_alignment_offsets,
)


def run(path: str, repeats: int) -> dict:
    audio = load_audio_mono(path)
    with torch.no_grad():
        dense = forward_dense_logit_grid(
            torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        )[0].cpu().numpy()
    exact = compare_dense_with_exact(audio, dense)
    return {
        "path": str(Path(path)),
        "samples": int(len(audio)),
        "seconds": float(len(audio) / 16000),
        "dense_outputs": int(len(dense)),
        "dense_vs_exact": {
            "complete_windows": int(np.sum(exact.valid_complete)),
            "correlation": exact.correlation,
            "max_abs_logit_error": exact.max_abs_error,
            "mean_abs_logit_error": exact.mean_abs_error,
        },
        "dense_vs_local_differentiable": compare_dense_with_differentiable_standalone(
            audio, dense
        ),
        "alignment_probe": probe_alignment_offsets(audio, dense),
        "exact_certificate": benchmark_exact_certificate(audio),
        "forward_backward": benchmark_forward_backward(audio, repeats=repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="audio paths at any ffmpeg-readable format")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"files": [run(path, args.repeats) for path in args.paths]}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
