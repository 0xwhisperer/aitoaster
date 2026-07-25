"""Cached, mathematically equivalent CNN audio-quality penalties."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .cnn_gradient_optimizer_v2 import (
    STFT_WIN,
    _PERCEPTUAL_WEIGHT,
    _worst_chunk_mean,
)


def _silence_gate(
    original: torch.Tensor,
    *,
    win_sec: float = 0.02,
    sr: int = 16000,
) -> torch.Tensor:
    """Return the fixed sample-domain gate used by the existing guard."""
    win = max(1, int(win_sec * sr))
    n = original.shape[-1]
    n_blocks = (n + win - 1) // win
    pad = n_blocks * win - n
    original_pad = F.pad(original, (0, pad)) if pad else original
    blocks = original_pad.view(n_blocks, win)
    block_rms = torch.sqrt((blocks.detach() ** 2).mean(dim=1) + 1e-12)
    block_db = 20 * torch.log10(block_rms + 1e-8)
    gate = torch.clamp((block_db + 70.0) / 35.0, 0.0, 1.0) ** 2
    return (
        gate.unsqueeze(1)
        .expand(n_blocks, win)
        .reshape(-1)[:n]
        .contiguous()
    )


class CachedCNNQualityPenalty:
    """Reuse every quality term that depends only on the original audio.

    The former functions rebuilt Hann windows, the silence envelope, and the
    original-audio STFT on every optimizer step.  Those values are constant
    for the complete solve.  This class preserves the same loss and gradient
    while computing the constant side once.
    """

    def __init__(
        self,
        original: torch.Tensor,
        *,
        sr: int = 16000,
        lo_hz: float = 400,
        hi_hz: float = 8000,
        perceptual_n_fft: int = STFT_WIN,
        band_n_fft: int = 2048,
    ):
        if original.ndim != 1:
            raise ValueError("CNN quality reference must be one-dimensional")
        self.original = original.detach()
        self.sr = int(sr)
        self.perceptual_n_fft = int(perceptual_n_fft)
        self.band_n_fft = int(band_n_fft)
        self.gate = _silence_gate(self.original, sr=self.sr)
        self.perceptual_window = torch.hann_window(
            self.perceptual_n_fft,
            dtype=self.original.dtype,
            device=self.original.device,
        )
        self.band_window = torch.hann_window(
            self.band_n_fft,
            dtype=self.original.dtype,
            device=self.original.device,
        )
        with torch.no_grad():
            original_stft = torch.stft(
                self.original,
                n_fft=self.perceptual_n_fft,
                hop_length=self.perceptual_n_fft // 4,
                window=self.perceptual_window,
                return_complex=True,
            )
            original_db = 20 * torch.log10(original_stft.abs() + 1e-6)
            masking = torch.clamp(
                (original_db + 50.0) / 40.0, 0.0, 1.0
            )
            self.masking_mult = 0.05 + 0.65 * masking
        frequencies = torch.fft.rfftfreq(
            self.band_n_fft,
            1 / self.sr,
            device=self.original.device,
        )
        self.out_of_band = (frequencies < lo_hz) | (
            frequencies > hi_hz
        )
        self.perceptual_weight = _PERCEPTUAL_WEIGHT.to(
            device=self.original.device, dtype=self.original.dtype
        ).unsqueeze(1)

    def perceptual(self, delta: torch.Tensor) -> torch.Tensor:
        transformed = torch.stft(
            delta * self.gate,
            n_fft=self.perceptual_n_fft,
            hop_length=self.perceptual_n_fft // 4,
            window=self.perceptual_window,
            return_complex=True,
        )
        power = (
            transformed.abs().square()
            * self.perceptual_weight
            * self.masking_mult
        )
        return _worst_chunk_mean(power)

    def band(self, delta: torch.Tensor) -> torch.Tensor:
        transformed = torch.stft(
            delta,
            n_fft=self.band_n_fft,
            hop_length=self.band_n_fft // 4,
            window=self.band_window,
            return_complex=True,
        )
        out_of_band_power = transformed.abs().square()[self.out_of_band]
        if out_of_band_power.shape[0] == 0:
            return delta.new_tensor(0.0)
        return _worst_chunk_mean(out_of_band_power)

    def tonality(
        self,
        delta: torch.Tensor,
        *,
        chunk_frames: int = 44,
    ) -> torch.Tensor:
        transformed = torch.stft(
            delta,
            n_fft=self.perceptual_n_fft,
            hop_length=self.perceptual_n_fft // 4,
            window=self.perceptual_window,
            return_complex=True,
        )
        power = transformed.abs().square()
        n_freq, n_time = power.shape
        if n_time <= chunk_frames:
            frequency_energy = power.mean(dim=1)
            total = frequency_energy.sum() + 1e-12
            return ((frequency_energy / total) ** 2).sum()
        n_chunks = n_time // chunk_frames
        chunked = power[:, : n_chunks * chunk_frames].reshape(
            n_freq, n_chunks, chunk_frames
        ).mean(dim=2)
        normalized = chunked / (chunked.sum(dim=0) + 1e-12).unsqueeze(0)
        return normalized.square().sum(dim=0).max()

    def loss(
        self,
        delta: torch.Tensor,
        *,
        lambda_perceptual: float,
        lambda_band: float,
        lambda_tonality: float,
    ) -> torch.Tensor:
        return (
            lambda_perceptual * self.perceptual(delta)
            + lambda_band * self.band(delta)
            + lambda_tonality * self.tonality(delta)
        )
