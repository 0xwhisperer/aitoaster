"""Test temporal smoothing of an existing isolated CNN correction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cnn_adaptive_dense_prototype import exact_union_scan
from app.cnn_differentiable_v2 import load_audio_mono
from app.cnn_real_scanner import ParallelRealScoreScanner


def smooth_stft_magnitude(
    correction: np.ndarray,
    sr: int,
    smooth_sec: float,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Smooth rapid magnitude modulation while retaining STFT phase."""
    output = np.zeros_like(correction)
    window = torch.hann_window(n_fft)
    frame_rate = sr / hop_length
    for channel in range(correction.shape[1]):
        signal = torch.from_numpy(correction[:, channel])
        transformed = torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        magnitude = transformed.abs().numpy()
        phase = transformed / (transformed.abs() + 1e-12)
        smoothed_magnitude = gaussian_filter1d(
            magnitude,
            sigma=max(0.01, smooth_sec * frame_rate),
            axis=1,
            mode="nearest",
        )
        reconstructed = torch.istft(
            phase * torch.from_numpy(smoothed_magnitude),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            length=len(signal),
        )
        output[:, channel] = reconstructed.numpy()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed")
    parser.add_argument("cnn_overlay")
    parser.add_argument("--seconds", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--output-prefix", default="/tmp/cnn_smooth")
    args = parser.parse_args()
    fixed, sr = sf.read(args.fixed, dtype="float32", always_2d=True)
    overlay, overlay_sr = sf.read(
        args.cnn_overlay, dtype="float32", always_2d=True
    )
    if overlay_sr != sr or overlay.shape != fixed.shape:
        raise ValueError("fixed file and CNN overlay must align")
    without_cnn = fixed - overlay
    results = []
    with ParallelRealScoreScanner() as scanner:
        for seconds in [float(v) for v in args.seconds.split(",")]:
            smoothed = smooth_stft_magnitude(overlay, sr, seconds)
            candidate = without_cnn + smoothed
            output_path = f"{args.output_prefix}_{seconds:g}s.wav"
            sf.write(output_path, candidate, sr, subtype="PCM_16")
            mono = torch.from_numpy(load_audio_mono(output_path))
            _, scan = exact_union_scan(
                mono, torch.zeros_like(mono), scanner
            )
            results.append(
                {
                    "seconds": seconds,
                    "path": output_path,
                    "passed": scan.passed,
                    "worst": scan.worst_score,
                    "failures": len(scan.failing_positions),
                    "overlay_rms_db": float(
                        20
                        * np.log10(
                            np.sqrt(np.mean(smoothed**2)) + 1e-30
                        )
                    ),
                    "change_snr_db": float(
                        20
                        * np.log10(
                            np.linalg.norm(fixed)
                            / (
                                np.linalg.norm(candidate - fixed)
                                + 1e-30
                            )
                        )
                    ),
                }
            )
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
