"""
The Fixer - deterministic mastering-chain DSP tools.

Every function takes/returns (stereo_float_array[N,2], sr) and a small info
dict describing what changed, so the web UI can show a plain-English log of
what was actually done to the audio. Nothing here touches AI-detector scoring
directly (see detector.py / linear_fix.py / cnn_fix.py for that) - this module
is the "ordinary mastering engineer" toolbox: silence trim, DC offset, high
pass, RMS/LUFS normalization, multiband compression, stereo correlation,
true-peak limiting.
"""
import numpy as np
from scipy import signal
import pyloudnorm as pyln


def trim_silence(audio, sr, threshold=0.0005, pad_ms=5):
    """Trim leading/trailing near-silence. threshold is linear amplitude;
    default catches true digital silence plus low-level noise floor dither."""
    mono_abs = np.abs(audio).max(axis=1)
    n = len(audio)
    above = mono_abs > threshold
    if not above.any():
        return audio, {"applied": False, "reason": "entire file below threshold"}

    lead_idx = int(np.argmax(above))
    trail_from_end = int(np.argmax(above[::-1]))
    trail_idx = n - trail_from_end

    pad = int(pad_ms / 1000 * sr)
    lead_idx = max(0, lead_idx - pad)
    trail_idx = min(n, trail_idx + pad)

    if lead_idx == 0 and trail_idx == n:
        return audio, {"applied": False, "lead_ms": 0, "trail_ms": 0}

    trimmed = audio[lead_idx:trail_idx]
    return trimmed, {
        "applied": True,
        "lead_ms": round(lead_idx / sr * 1000, 1),
        "trail_ms": round(trail_from_end / sr * 1000, 1),
        "lead_samples": lead_idx,
        "samples_removed": n - len(trimmed),
    }


def fix_dc_offset(audio, sr):
    dc = audio.mean(axis=0)
    if np.abs(dc).max() < 1e-7:
        return audio, {"applied": False, "dc_l": float(dc[0]), "dc_r": float(dc[1])}
    out = audio - dc
    return out, {
        "applied": True,
        "dc_l_before": float(dc[0]),
        "dc_r_before": float(dc[1]),
        "dc_l_after": float(out[:, 0].mean()),
        "dc_r_after": float(out[:, 1].mean()),
    }


def high_pass_filter(audio, sr, cutoff_hz=30, order=2):
    """Gentle high-pass to remove sub-sonic rumble/DC residue that a plain
    mean-subtraction can't catch (e.g. slow drift). 30Hz is below all
    audible fundamentals so it is inaudible on music but keeps the mix
    clean for LUFS metering and limiting."""
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    out = np.stack([signal.sosfiltfilt(sos, audio[:, ch]) for ch in range(audio.shape[1])], axis=1)
    return out, {"applied": True, "cutoff_hz": cutoff_hz, "order": order}


def detect_transients(audio, sr, jump_threshold=0.35, envelope_ratio_threshold=8.0, min_gap_sec=0.5):
    """Find genuine click/pop/glitch artifacts - NOT ordinary musical
    transients like kick/snare hits, which have fast-but-natural attacks and
    must never trigger this. A real digital pop is characterized by an
    unnaturally large SAMPLE-TO-SAMPLE jump (near-instant discontinuity that
    no acoustic instrument or synth envelope produces) that also spikes far
    above a longer (200ms) local RMS envelope. Both conditions must hold, and
    the threshold is deliberately strict so ordinary dynamic music is left
    untouched - this is for a handful of one-off glitches per track, not a
    general-purpose limiter."""
    mono = audio.mean(axis=1)
    n = len(mono)

    # sample-to-sample jump: true discontinuities (pops/clicks) show up as a
    # single large derivative spike; smooth attacks (even fast percussive
    # ones) rise over many samples and never produce this
    jump = np.abs(np.diff(mono, prepend=mono[0]))

    # longer 200ms RMS envelope so normal loud passages (which have already-high
    # local energy) don't get flagged as "far above the envelope"
    win = max(1, int(0.2 * sr))
    win = win | 1
    envelope = np.sqrt(signal.medfilt(mono ** 2, kernel_size=win))
    envelope = np.maximum(envelope, 1e-6)
    ratio = np.abs(mono) / envelope

    candidates = []
    i = 0
    min_gap = int(min_gap_sec * sr)
    while i < n:
        if jump[i] > jump_threshold and ratio[i] > envelope_ratio_threshold:
            lo = max(0, i - min_gap // 2)
            hi = min(n, i + min_gap // 2)
            local_peak_idx = lo + int(np.argmax(np.abs(mono[lo:hi])))
            candidates.append(local_peak_idx)
            i = local_peak_idx + min_gap
        else:
            i += 1
    return [{"time_sec": round(c / sr, 3), "peak": float(np.abs(mono[c]))} for c in candidates]


def fix_transient(audio, sr, time_sec, target_peak=None, attack_ms=3, release_ms=60, context_sec=0.3):
    """Raised-cosine gain-envelope pop suppression at one specific point.
    target_peak, if not given explicitly, is derived from THIS song's own
    recent musical context (the RMS envelope just before the glitch, scaled
    up to a reasonable peak headroom) rather than a fixed absolute value -
    a pop in a quiet passage and a pop in a loud passage need very
    different amounts of reduction."""
    n = len(audio)
    center = int(time_sec * sr)
    search = int(0.005 * sr)
    lo, hi = max(0, center - search), min(n, center + search)
    local_peak = np.abs(audio[lo:hi]).max()

    if target_peak is None:
        ctx_samples = int(context_sec * sr)
        ctx_lo = max(0, center - ctx_samples - search)
        ctx_hi = max(0, center - search)
        if ctx_hi > ctx_lo:
            context_rms = np.sqrt(np.mean(audio[ctx_lo:ctx_hi] ** 2))
            target_peak = min(local_peak * 0.9, max(context_rms * 4, 0.05))
        else:
            target_peak = local_peak * 0.7

    if local_peak <= target_peak * 1.05:
        return audio, {"applied": False, "reason": "already under target peak"}

    reduction_ratio = target_peak / local_peak
    attack = int(attack_ms / 1000 * sr)
    release = int(release_ms / 1000 * sr)
    gain = np.ones(n)
    for i in range(attack):
        idx = center - attack + i
        if 0 <= idx < n:
            frac = i / attack
            eased = 0.5 - 0.5 * np.cos(np.pi * frac)
            gain[idx] = 1.0 - eased * (1.0 - reduction_ratio)
    for i in range(release):
        idx = center + i
        if 0 <= idx < n:
            frac = i / release
            eased = 0.5 - 0.5 * np.cos(np.pi * frac)
            gain[idx] = reduction_ratio + eased * (1.0 - reduction_ratio)

    out = audio * gain[:, None]
    return out, {
        "applied": True,
        "time_sec": time_sec,
        "peak_before": float(local_peak),
        "peak_after": float(np.abs(out[lo:hi]).max()),
        "reduction_db": float(20 * np.log10(reduction_ratio)),
    }


def measure_lufs(audio, sr):
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(audio)


def normalize_lufs(audio, sr, target_lufs=-14.0):
    """Standard streaming loudness target (-14 LUFS covers most platforms;
    Apple Music targets -16, but -14 is the safer general default and won't
    get turned down further on playback)."""
    current = measure_lufs(audio, sr)
    if not np.isfinite(current):
        return audio, {"applied": False, "reason": "silent/invalid audio"}
    gain_db = target_lufs - current
    gain_linear = 10 ** (gain_db / 20)
    out = audio * gain_linear
    peak = np.abs(out).max()
    if peak > 0.999:
        out = out * (0.999 / peak)
        gain_db_actual = 20 * np.log10(0.999 / peak) + gain_db
    else:
        gain_db_actual = gain_db
    return out, {
        "applied": True,
        "lufs_before": float(current),
        "lufs_target": target_lufs,
        "gain_db": float(gain_db_actual),
        "lufs_after": float(measure_lufs(out, sr)),
    }


def stereo_correlation(audio):
    """Phase/mono-compatibility check: +1 = perfectly in phase (mono-safe),
    0 = uncorrelated (wide), -1 = out of phase (mono-collapse risk)."""
    l, r = audio[:, 0], audio[:, 1]
    if np.std(l) < 1e-9 or np.std(r) < 1e-9:
        return 1.0
    return float(np.corrcoef(l, r)[0, 1])


def fix_phase_issues(audio, sr, min_correlation=0.0):
    """If stereo correlation is negative enough to risk mono cancellation,
    blend a small amount of mid-channel back in to restore mono safety
    without collapsing the stereo image entirely."""
    corr = stereo_correlation(audio)
    if corr >= min_correlation:
        return audio, {"applied": False, "correlation": corr}

    mid = audio.mean(axis=1)
    side = (audio[:, 0] - audio[:, 1]) / 2
    blend = min(0.5, (min_correlation - corr))
    l_fixed = mid + side * (1 - blend)
    r_fixed = mid - side * (1 - blend)
    out = np.stack([l_fixed, r_fixed], axis=1)
    return out, {
        "applied": True,
        "correlation_before": corr,
        "correlation_after": stereo_correlation(out),
        "side_blend_reduction": blend,
    }


def multiband_compress(audio, sr, bands=((0, 200), (200, 2000), (2000, 20000)),
                        ratio=1.3, threshold_db=-12.0):
    """Gentle 3-band downward compression for tonal-balance smoothing -
    reduces peaky dynamic imbalance between low/mid/high without touching
    overall spectral tilt aggressively. Conservative defaults (low ratio,
    higher threshold) by design: this should only be shaping the loudest
    peaks in each band, not continuously riding gain on the whole track -
    least change necessary to smooth genuine imbalance."""
    nyq = sr / 2
    out = np.zeros_like(audio)
    info_bands = []
    prev_hi = 0
    for lo, hi in bands:
        hi = min(hi, nyq - 1)
        if lo == 0:
            sos = signal.butter(4, hi, btype="lowpass", fs=sr, output="sos")
        elif hi >= nyq - 1:
            sos = signal.butter(4, lo, btype="highpass", fs=sr, output="sos")
        else:
            sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")

        band_audio = np.stack([signal.sosfiltfilt(sos, audio[:, ch]) for ch in range(audio.shape[1])], axis=1)

        env = np.abs(band_audio).max(axis=1)
        win = max(1, int(0.02 * sr)) | 1
        env_smooth = signal.medfilt(env, kernel_size=win)
        env_db = 20 * np.log10(np.maximum(env_smooth, 1e-8))
        over = np.maximum(env_db - threshold_db, 0)
        gain_db = -over * (1 - 1 / ratio)
        gain = 10 ** (gain_db / 20)
        out += band_audio * gain[:, None]
        info_bands.append({"range_hz": [lo, round(hi)], "max_reduction_db": float(gain_db.min())})

    return out, {"applied": True, "ratio": ratio, "threshold_db": threshold_db, "bands": info_bands}


def true_peak_limit(audio, sr, ceiling_db=-1.0, oversample=4):
    """True-peak (inter-sample peak) limiter: oversample to approximate the
    reconstructed analog waveform's real peak (which can exceed sample-peak
    after D/A), then apply a lookahead-free gain-reduction so the true peak
    never exceeds ceiling_db. Simple/safe brick-wall style limiter suitable
    as a final safety stage, not a loudness-maximizing one."""
    ceiling = 10 ** (ceiling_db / 20)
    up = signal.resample_poly(audio, oversample, 1, axis=0)
    true_peak = np.abs(up).max()
    sample_peak = np.abs(audio).max()

    if true_peak <= ceiling:
        return audio, {
            "applied": False,
            "true_peak_db": float(20 * np.log10(true_peak + 1e-12)),
            "ceiling_db": ceiling_db,
        }

    gain = ceiling / true_peak
    out = audio * gain
    return out, {
        "applied": True,
        "true_peak_db_before": float(20 * np.log10(true_peak + 1e-12)),
        "sample_peak_db_before": float(20 * np.log10(sample_peak + 1e-12)),
        "ceiling_db": ceiling_db,
        "gain_reduction_db": float(20 * np.log10(gain)),
    }


def spectral_tilt_report(audio, sr):
    """Simple tonal-balance measurement: energy ratio across low/mid/high
    thirds, useful for the UI's 'before vs after' EQ curve display."""
    mono = audio.mean(axis=1)
    freqs, psd = signal.welch(mono, sr, nperseg=8192)
    bands = {"low (20-250Hz)": (20, 250), "mid (250-4000Hz)": (250, 4000), "high (4000-20000Hz)": (4000, min(20000, sr / 2 - 1))}
    report = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        report[name] = float(10 * np.log10(psd[mask].mean() + 1e-15))
    return report, freqs.tolist(), (10 * np.log10(psd + 1e-15)).tolist()
