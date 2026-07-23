import numpy as np
import torch
from .linear_differentiable import (
    forward_score_differentiable, forward_logit_differentiable, load_audio_torch, get_real_score,
    SAMPLE_RATE, MAX_DURATION,
)

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


def build_perceptual_weight(sr, n_fft):
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    w = a_weighting_approx(freqs)
    return torch.tensor(w / w.max(), dtype=torch.float32)


PERCEPTUAL_WEIGHT = build_perceptual_weight(SAMPLE_RATE, STFT_WIN)


def perceptual_penalty(delta_1d, original_1d):
    window = torch.hann_window(STFT_WIN)
    D = torch.stft(delta_1d, n_fft=STFT_WIN, hop_length=STFT_WIN // 4, window=window, return_complex=True)
    O = torch.stft(original_1d, n_fft=STFT_WIN, hop_length=STFT_WIN // 4, window=window, return_complex=True)
    d_mag = D.abs()
    o_mag = O.abs().detach()
    o_db = 20 * torch.log10(o_mag + 1e-6)
    o_db_norm = torch.clamp((o_db - o_db.min()) / (o_db.max() - o_db.min() + 1e-6), 0, 1)
    masking_mult = 1.0 - 0.7 * o_db_norm  # loud original -> more masking -> lower penalty
    pw = PERCEPTUAL_WEIGHT.unsqueeze(1)
    weighted_power = (d_mag ** 2) * pw * masking_mult
    energy_penalty = weighted_power.mean()
    return energy_penalty


def tonality_penalty(delta_1d, n_fft=1024):
    """Discourage energy concentrating into a few narrow bins (an audible
    whine/ring) vs. spreading broadband. Penalizes peak-bin-energy relative to
    mean-bin-energy directly, on its own natural scale (not folded into the
    ear-sensitivity-weighted perceptual penalty, whose tiny values were
    swamping this term almost to zero)."""
    window = torch.hann_window(n_fft)
    D = torch.stft(delta_1d, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
    d_power = D.abs() ** 2
    freq_energy = d_power.mean(dim=1)  # mean over time, per frequency bin
    total = freq_energy.sum() + 1e-12
    # penalize concentration via a normalized "effective number of bins used"
    # measure (inverse participation ratio): low when energy is spread across
    # many bins, high when concentrated in one - directly on a 0-1ish scale
    normalized = freq_energy / total
    concentration = (normalized ** 2).sum()  # in [1/n_bins, 1]; 1 = all energy in one bin
    return concentration


def band_limit_penalty(delta_1d, lo_hz=800, hi_hz=8200, sr=44100, n_fft=4096):
    """Penalize any energy the optimizer puts OUTSIDE the model's actual analysis
    band (1-8kHz) - that energy does nothing to fool the detector, so it should
    never be worth adding. Small guard band (800-8200) avoids penalizing the
    exact edges needed to influence boundary bins."""
    window = torch.hann_window(n_fft)
    D = torch.stft(delta_1d, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
    freqs = torch.fft.rfftfreq(n_fft, 1 / sr)
    out_of_band = (freqs < lo_hz) | (freqs > hi_hz)
    d_mag = D.abs()
    return (d_mag[out_of_band] ** 2).mean()


def _real_score_for_delta(delta, audio_orig):
    """Ground-truth check: write perturbed audio to a real WAV and run it
    through the actual ONNX+librosa pipeline, bypassing the differentiable
    surrogate entirely. The surrogate has been observed to diverge sharply
    from the real model specifically AFTER optimization (baseline match can
    be near-perfect while the optimized result is far off), so trusting the
    surrogate's own score alone is not sufficient - this must be checked
    directly and periodically, not assumed."""
    import tempfile, os, soundfile as sf
    perturbed = (audio_orig + delta).detach().numpy()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        path = tf.name
    try:
        sf.write(path, perturbed, SAMPLE_RATE, subtype="PCM_16")
        return get_real_score(path)
    finally:
        os.unlink(path)


def optimize(audio_orig, lambda_perceptual=2000.0, lambda_band=5000.0, lambda_tonality=50.0,
             target=0.01, real_target=0.05, max_steps=400, lr=0.00002,
             real_check_interval=50, verbose=True):
    """Full per-sample gradient optimization: delta is a free-form waveform
    perturbation (not constrained to any fixed set of frequencies), optimized
    to minimize [detector score + perceptual penalty + out-of-band penalty]
    jointly. Periodically re-verifies against the REAL (non-differentiable)
    model rather than trusting the surrogate's own score, and only accepts a
    candidate as "best" once the real model confirms it - the surrogate can
    look perfect (score ~0) while the real model still flags the result."""
    delta = torch.zeros_like(audio_orig, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    best_delta = None
    best_real_score = 1.0
    best_delta_norm = float("inf")
    extra_steps = 0

    # target the logit directly (has real gradients everywhere) rather than
    # the sigmoid score, which saturates to exactly 1.0 (zero gradient) when
    # the model is very confident - as it is on a raw, unmodified AI track
    logit_target = torch.logit(torch.tensor(target), eps=1e-6)

    step = 0
    while step < max_steps:
        optimizer.zero_grad()
        perturbed = audio_orig + delta
        logit = forward_logit_differentiable(perturbed)
        percep = perceptual_penalty(delta, audio_orig)
        band_penalty = band_limit_penalty(delta, sr=16000, lo_hz=800, hi_hz=8000)
        tonal_penalty = tonality_penalty(delta)
        # push logit down toward (and past) the target threshold's logit value
        logit_loss = torch.relu(logit - logit_target + 1.0)  # margin of 1.0 past target
        loss = logit_loss + lambda_perceptual * percep + lambda_band * band_penalty + lambda_tonality * tonal_penalty
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cur_score = forward_score_differentiable(audio_orig + delta).item()

        if verbose and step % 50 == 0:
            snr = 20 * torch.log10(audio_orig.norm() / (delta.norm() + 1e-8)).item()
            print(f"  step {step:3d}: surrogate_score={cur_score:.5f}  percep_penalty={percep.item():.6f}  band_penalty={band_penalty.item():.6f}  SNR={snr:.1f}dB")

        # periodically verify against the REAL model - this is the only
        # source of truth for whether we can actually stop or call a delta "best"
        if cur_score < target * 0.5 and step > 0 and step % real_check_interval == 0:
            real_score = _real_score_for_delta(delta, audio_orig)
            cur_norm = delta.norm().item()
            if verbose:
                print(f"    [real check @ step {step}] real_score={real_score:.5f} (surrogate said {cur_score:.5f})")
            if real_score < real_target and cur_norm < best_delta_norm:
                best_delta_norm = cur_norm
                best_delta = delta.detach().clone()
                best_real_score = real_score
            elif real_score < best_real_score:
                best_real_score = real_score
                best_delta = delta.detach().clone()

            if real_score < real_target and step >= 150:
                if verbose:
                    print(f"  converged (real-verified) at step {step}")
                break
            if real_score >= real_target:
                # surrogate thinks we're done but the real model disagrees -
                # keep optimizing past max_steps rather than silently
                # returning a delta that doesn't actually work
                extra_steps += real_check_interval
                if step + 1 >= max_steps and extra_steps <= max_steps * 3:
                    max_steps = step + 1 + real_check_interval

        step += 1

    if best_delta is None:
        # never got a real-model check below real_target - do one final
        # direct check on whatever we ended with so we never silently ship
        # something that was never actually verified against the real model
        real_score = _real_score_for_delta(delta, audio_orig)
        if verbose:
            print(f"  WARNING: no candidate cleared real_target during search; final real_score={real_score:.5f}")
        if real_score < 0.5:
            best_delta = delta.detach().clone()
            best_real_score = real_score

    return best_delta, best_real_score
