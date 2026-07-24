import numpy as np
import torch
import torch.nn.functional as F
from .cnn_differentiable_v2 import forward_logit_differentiable, forward_score_differentiable, SR

STFT_WIN = 1024


def a_weighting_approx(f):
    f = np.maximum(f, 1.0)
    ra = (12194.0 ** 2 * f ** 4) / (
        (f ** 2 + 20.6 ** 2)
        * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2))
        * (f ** 2 + 12194.0 ** 2)
    )
    weight_db = 20 * np.log10(ra) + 2.0
    return 10 ** (weight_db / 20)


def _build_perceptual_weight(sr, n_fft):
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    w = a_weighting_approx(freqs)
    return torch.tensor(w / w.max(), dtype=torch.float32)


_PERCEPTUAL_WEIGHT = _build_perceptual_weight(SR, STFT_WIN)


def _silence_guard(delta_1d, original_1d, win_sec=0.02, sr=None):
    """Direct time-domain absolute-loudness gate: compute the original's own
    RMS envelope in small (~20ms) blocks and force the delta toward zero,
    block by block, wherever that envelope is near true silence - independent
    of any per-STFT-bin masking math. The STFT masking penalty alone left a
    small floor allowance active everywhere, which was still audibly louder
    than a genuinely near-silent (-60 to -70dBFS) track opening."""
    sr = sr or SR
    win = max(1, int(win_sec * sr))
    n = original_1d.shape[-1]
    n_blocks = (n + win - 1) // win
    pad = n_blocks * win - n
    if pad:
        original_pad = F.pad(original_1d, (0, pad))
        delta_pad = F.pad(delta_1d, (0, pad))
    else:
        original_pad = original_1d
        delta_pad = delta_1d

    o_blocks = original_pad.view(n_blocks, win)
    d_blocks = delta_pad.view(n_blocks, win)
    block_rms = torch.sqrt((o_blocks.detach() ** 2).mean(dim=1) + 1e-12)
    block_db = 20 * torch.log10(block_rms + 1e-8)

    # true silence (<= -70dBFS) -> essentially zero allowance; -35dBFS -> full allowance
    floor_db, ceiling_db = -70.0, -35.0
    gate = torch.clamp((block_db - floor_db) / (ceiling_db - floor_db), 0.0, 1.0)
    gate = gate ** 2  # steeper falloff near the floor

    gated_delta = d_blocks * gate.unsqueeze(1)
    return gated_delta.reshape(-1)[:n]


def perceptual_penalty(delta_1d, original_1d):
    """Returns a scalar penalty for how audible delta_1d is likely to be.

    CRITICAL: this must NOT be a flat .mean() over the whole track's STFT
    frames. Confirmed as the actual root cause of CNN's fix failing on every
    full-length track this session while succeeding on a short 47s clip: a
    flat mean lets a LOCALLY loud/aggressive correction (which is exactly
    what the optimizer needs on windows the real model still flags) hide
    behind a long quiet average elsewhere in the track - the penalty barely
    rises even though a human (or the real detector, checking short 10s
    windows independently) would notice that one loud region just fine.
    Using the WORST local window's mean penalty (grouped into ~1s chunks,
    close to the real detector's own analysis granularity) instead of a
    single track-wide average means a locally loud correction is penalized
    at full strength regardless of how long or quiet the rest of the track
    is - matching how both a human ear and the real per-window detector
    actually judge audibility (by the worst moment, not the average)."""
    gated_delta = _silence_guard(delta_1d, original_1d)

    window = torch.hann_window(STFT_WIN)
    D = torch.stft(gated_delta, n_fft=STFT_WIN, hop_length=STFT_WIN // 4, window=window, return_complex=True)
    O = torch.stft(original_1d, n_fft=STFT_WIN, hop_length=STFT_WIN // 4, window=window, return_complex=True)
    d_mag = D.abs()
    o_mag = O.abs().detach()

    o_db_abs = 20 * torch.log10(o_mag + 1e-6)
    o_db_floor, o_db_ceiling = -50.0, -10.0
    masking_mult = torch.clamp((o_db_abs - o_db_floor) / (o_db_ceiling - o_db_floor), 0.0, 1.0)
    masking_mult = 0.05 + 0.65 * masking_mult

    pw = _PERCEPTUAL_WEIGHT.unsqueeze(1)
    weighted_power = (d_mag ** 2) * pw * masking_mult  # [freq, time]
    return _worst_chunk_mean(weighted_power)


def _worst_chunk_mean(power_ft, chunk_frames=44):
    """Mean penalty per ~1-second chunk of STFT frames (chunk_frames=44 at
    STFT_WIN=1024, hop=256, SR=16000: 256/16000*44 ~= 0.70s - close enough
    to the real detector's per-window granularity to catch local problems),
    then take the WORST chunk rather than averaging over all chunks - see
    perceptual_penalty's docstring for why this matters."""
    n_freq, n_time = power_ft.shape
    if n_time <= chunk_frames:
        return power_ft.mean()
    n_chunks = n_time // chunk_frames
    trimmed = power_ft[:, : n_chunks * chunk_frames]
    per_chunk_mean = trimmed.reshape(n_freq, n_chunks, chunk_frames).mean(dim=(0, 2))
    return per_chunk_mean.max()


def apply_silence_guard_to_delta(delta_1d, original_1d, win_sec=0.02, sr=None):
    """Post-processing pass: apply the same absolute silence gate directly to
    the FINAL delta before saving, so the guard is not just a soft training
    penalty but a hard guarantee in the shipped audio."""
    with torch.no_grad():
        return _silence_guard(delta_1d, original_1d, win_sec=win_sec, sr=sr)


def band_limit_penalty(delta_1d, lo_hz, hi_hz, sr, n_fft=2048):
    """Penalize energy outside the model's actual analysis band (500-8000Hz
    per config.yaml's cqt.fmin=500). Guard band added to avoid penalizing
    the exact edges needed to influence boundary bins. Uses worst-chunk
    (not whole-track-mean) for the same reason as perceptual_penalty - a
    long track's average otherwise hides a local out-of-band spike."""
    window = torch.hann_window(n_fft)
    D = torch.stft(delta_1d, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
    freqs = torch.fft.rfftfreq(n_fft, 1 / sr)
    out_of_band = (freqs < lo_hz) | (freqs > hi_hz)
    d_power = D.abs() ** 2
    d_power_oob = d_power[out_of_band]
    if d_power_oob.shape[0] == 0:
        return torch.tensor(0.0)
    return _worst_chunk_mean(d_power_oob)


def tonality_penalty(delta_1d, n_fft=1024, chunk_frames=44):
    """Discourage energy concentrating into a few narrow bins (audible
    whine/ring) via an inverse-participation-ratio style concentration
    measure. Computed per ~1s time-chunk and the WORST (most concentrated,
    most tonal-sounding) chunk is used, not one measure over the whole
    track's time-averaged spectrum - a brief tonal whine partway through a
    long track would otherwise average out against the rest of the track's
    broadband noise and barely register, the same track-length-dilution
    problem fixed in perceptual_penalty/band_limit_penalty above."""
    window = torch.hann_window(n_fft)
    D = torch.stft(delta_1d, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
    d_power = D.abs() ** 2  # [freq, time]
    n_freq, n_time = d_power.shape
    if n_time <= chunk_frames:
        freq_energy = d_power.mean(dim=1)
        total = freq_energy.sum() + 1e-12
        return ((freq_energy / total) ** 2).sum()
    n_chunks = n_time // chunk_frames
    trimmed = d_power[:, : n_chunks * chunk_frames]
    chunked = trimmed.reshape(n_freq, n_chunks, chunk_frames).mean(dim=2)  # [freq, n_chunks]
    totals = chunked.sum(dim=0) + 1e-12  # [n_chunks]
    normalized = chunked / totals.unsqueeze(0)
    concentration_per_chunk = (normalized ** 2).sum(dim=0)  # [n_chunks]
    return concentration_per_chunk.max()


def optimize_segment(
    audio_seg,
    target=0.05,
    lambda_perceptual=2000.0,
    lambda_band=5000.0,
    lambda_tonality=50.0,
    lr=0.00002,
    max_steps=400,
    min_steps=150,
    verbose=True,
):
    """Gradient-based optimization of ONE 10-second segment (matching the real
    evaluator's per-segment analysis), using the same discipline as the fixed
    linear-model optimizer: optimize the logit (avoids sigmoid saturation),
    slow learning rate + minimum step count (lets perceptual/anti-tonality
    penalties actually shape the solution instead of one huge overshoot step)."""
    original = audio_seg.clone()
    delta = torch.zeros_like(original, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    logit_target = torch.logit(torch.tensor(target), eps=1e-6)
    best_delta = None
    best_delta_norm = float("inf")

    for step in range(max_steps):
        optimizer.zero_grad()
        perturbed = original + delta
        logit = forward_logit_differentiable(perturbed.unsqueeze(0))
        percep = perceptual_penalty(delta, original)
        band_pen = band_limit_penalty(delta, lo_hz=400, hi_hz=8000, sr=SR)
        tonal_pen = tonality_penalty(delta)
        logit_loss = torch.relu(logit - logit_target + 1.0)
        loss = logit_loss + lambda_perceptual * percep + lambda_band * band_pen + lambda_tonality * tonal_pen
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cur_score = forward_score_differentiable(original + delta).item()
            cur_norm = delta.norm().item()
            if cur_score < target and cur_norm < best_delta_norm:
                best_delta_norm = cur_norm
                best_delta = delta.detach().clone()

        if verbose and step % 50 == 0:
            snr = 20 * torch.log10(original.norm() / (delta.norm() + 1e-8)).item()
            print(f"    step {step:3d}: score={cur_score:.5f}  percep={percep.item():.6f}  "
                  f"band={band_pen.item():.6f}  tonal={tonal_pen.item():.4f}  SNR={snr:.1f}dB")

        if cur_score < target * 0.5 and step >= min_steps:
            break

    if best_delta is None:
        best_delta = delta.detach().clone()
    return best_delta
