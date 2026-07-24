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
import subprocess
import json as _json

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
    """Returns the ITU-R BS.1770 integrated loudness in LUFS, or NaN if the
    clip is too short for pyloudnorm's measurement block size (it raises a
    ValueError rather than returning a degraded estimate) - callers should
    treat NaN the same way they already treat -inf (silent audio): as "no
    meaningful LUFS value available", not as an error."""
    meter = pyln.Meter(sr)
    try:
        return meter.integrated_loudness(audio)
    except ValueError:
        return float("nan")


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
    after D/A), then reduce gain ONLY where the true peak actually exceeds
    ceiling_db (an attack/release-smoothed per-sample gain envelope), rather
    than scaling the ENTIRE track down by one flat gain derived from its
    single loudest moment.

    Why this changed from flat gain-scaling: a flat scale-down affects the
    file's overall LUFS by the exact same amount as its single loudest
    transient's overshoot, even if that transient is a brief outlier and the
    rest of the track has plenty of headroom - confirmed directly on a real
    production run where a post-chain LUFS drift-correction pass raised gain
    toward -14 LUFS, then this limiter (flat-scaling at the time) undid most
    of that correction, delivering -17.9 LUFS instead of -14. Real
    dynamics-style limiting only pulls down where it must, preserving far
    more of the track's actual loudness and the LUFS target it was set to."""
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

    # per-oversampled-sample gain needed to keep that instant under ceiling
    # (1.0 = no reduction needed there)
    up_abs = np.abs(up)
    instant_gain = np.minimum(1.0, ceiling / np.maximum(up_abs, 1e-12))

    # attack/release smoothing so gain reduction ramps in/out rather than
    # switching instantaneously (which would itself add audible distortion) -
    # fast attack (catch the peak before it happens isn't possible without
    # lookahead, so this favors fast reduction) and a slower release so gain
    # recovers smoothly after the loud moment passes, matching standard
    # limiter design practice.
    release_ms = 50.0
    sr_up = sr * oversample
    release_coeff = np.exp(-1.0 / (release_ms * 0.001 * sr_up))

    n_ch = up.shape[1] if up.ndim > 1 else 1
    # limiting must apply the SAME gain to both channels at each instant to
    # avoid shifting stereo balance - use the more-reducing (smaller) gain
    # of the two channels at every instant.
    if n_ch > 1:
        instant_gain = np.min(instant_gain, axis=1, keepdims=True)[:, 0]
    col = instant_gain

    # vectorized one-pole attack/release smoothing (equivalent to the
    # sample-by-sample recurrence g = coeff*g + (1-coeff)*target, but a
    # naive python loop over ~49M oversampled samples on a full track is
    # too slow to run multiple times per pipeline - confirmed directly:
    # that's on the order of tens of millions of iterations for a ~5min
    # track at 4x oversample). Splitting into falling (attack) and rising
    # (release) segments and applying scipy's C-implemented IIR filter
    # per-segment reproduces the same asymmetric-coefficient recurrence at
    # native speed. Because attack/release only differs in decay RATE, not
    # direction logic, run the fast (attack) filter first pass over the
    # whole signal to get a supersharp response, then blend using proper
    # asymmetric one-pole logic via a single explicit small-batch pass:
    # process the largely-flat 1.0 gain track in chunks-by-changepoint is
    # overengineering here - instead use lfilter with the RELEASE (slower)
    # coefficient as a safe smoothing pass, then clip to never exceed the
    # instantaneous requirement (this stays inside the safe/no-clip
    # envelope while remaining smooth, and is a standard simplification for
    # a lookahead-free limiter where attack must be near-instant anyway).
    b_release = [1 - release_coeff]
    a_release = [1, -release_coeff]
    smoothed = signal.lfilter(b_release, a_release, col)
    # smoothing can only RAISE gain relative to the instant requirement
    # (never lower it enough) since it's a low-pass toward the target - clip
    # back down to the strict per-instant ceiling wherever that happens, which
    # gives the near-instant attack this limiter needs without a python loop.
    smoothed = np.minimum(smoothed, col)
    gain_env = np.repeat(smoothed[:, None], n_ch, axis=1) if n_ch > 1 else smoothed

    up_limited = up * gain_env
    out = signal.resample_poly(up_limited, 1, oversample, axis=0)
    n = min(len(out), len(audio))
    out = out[:n]

    final_true_peak = np.abs(signal.resample_poly(out, oversample, 1, axis=0)).max()
    if final_true_peak > ceiling:
        # smoothing/downsampling can leave a residual overshoot on rare
        # sharp transients - a final flat trim (not a full re-limit) closes
        # the gap without discarding the dynamics-preserving work above.
        out = out * (ceiling / final_true_peak)

    return out, {
        "applied": True,
        "true_peak_db_before": float(20 * np.log10(true_peak + 1e-12)),
        "sample_peak_db_before": float(20 * np.log10(sample_peak + 1e-12)),
        "ceiling_db": ceiling_db,
        "gain_reduction_db": float(20 * np.log10(np.min(gain_env) + 1e-12)),
        "dynamics_limited": True,
    }


def detect_spectral_rolloff(audio, sr, cutoff_hz=17000.0):
    """Check whether a track has a hard high-frequency rolloff (a common
    artifact of lossy encoding, low-quality AI generation, or resampling
    from a lower source rate) starting around cutoff_hz - measured as a
    steep drop in average spectral energy right at that frequency compared
    to what the track's own lower-frequency rolloff slope would predict.
    Returns (has_rolloff: bool, detected_cutoff_hz: float or None,
    deficit_db: float) so callers can decide whether spectral_revive is
    worth running, rather than always applying it regardless of whether
    the file actually needs it."""
    mono = audio.mean(axis=1)
    win = 32768
    if len(mono) < win * 2:
        return False, None, 0.0
    window = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    mags = []
    for start in range(0, len(mono) - win, win // 4):
        seg = mono[start:start + win] * window
        mags.append(np.abs(np.fft.rfft(seg)))
    avg_spec = np.array(mags).mean(axis=0)
    avg_db = 20 * np.log10(avg_spec + 1e-12)
    norm_db = avg_db - avg_db.max()

    nyquist = sr / 2
    if cutoff_hz >= nyquist - 500:
        return False, None, 0.0

    fit_lo, fit_hi = 3000.0, cutoff_hz - 500.0
    fit_mask = (freqs >= fit_lo) & (freqs <= fit_hi)
    if fit_mask.sum() < 10:
        return False, None, 0.0
    slope, intercept = np.polyfit(np.log2(freqs[fit_mask]), norm_db[fit_mask], 1)

    predicted_db = slope * np.log2(cutoff_hz + 1000) + intercept
    actual_mask = (freqs >= cutoff_hz + 500) & (freqs <= cutoff_hz + 1500)
    if actual_mask.sum() < 2:
        return False, None, 0.0
    actual_db = norm_db[actual_mask].mean()

    deficit_db = predicted_db - actual_db
    # a real, natural rolloff still loses SOME energy at the top - only flag
    # this as an artificial cutoff worth reviving if the actual level is
    # substantially below what the track's own established slope predicts
    has_rolloff = bool(deficit_db > 6.0)
    return has_rolloff, (cutoff_hz if has_rolloff else None), float(max(0, deficit_db))


def spectral_revive(audio, sr, cutoff_hz=None, seed=42):
    """Fill in high-frequency content above a hard rolloff (commonly left by
    lossy encoding, low-quality AI generation, or resampling from a lower
    source rate) using ONLY this track's own spectral characteristics - no
    external reference file or fixed target curve is used anywhere:

    1. Fits this track's OWN rolloff slope (dB/octave) in the region just
       below the cutoff, then extrapolates that same line past it. Natural
       spectral decay is close to linear in log-frequency space, so this
       gives a physically plausible target level derived entirely from the
       file's own measured characteristics.
    2. Projects harmonics from each frame's own detected spectral peaks
       upward past the cutoff (a harmonic at 2x/3x/4x... the fundamental
       decays by a fixed dB/octave rate) rather than adding disconnected
       synthetic content.
    3. Adds broadband texture at the self-derived target level, modulated
       frame-by-frame by this track's OWN dynamics (how bright this exact
       moment is relative to the track's own average brightness) so the
       fill breathes with the music instead of being a static hiss.

    Returns (audio, info) with the fitted rolloff slope and the frequency
    range extended, for the "what was done" report."""
    if cutoff_hz is None:
        has_rolloff, detected_cutoff, deficit_db = detect_spectral_rolloff(audio, sr)
        if not has_rolloff:
            return audio, {"applied": False, "reason": "no artificial high-frequency rolloff detected"}
        cutoff_hz = detected_cutoff
    else:
        deficit_db = None

    n_total = len(audio)
    WIN = 4096
    HOP = WIN // 4
    window = np.hanning(WIN)
    freqs = np.fft.rfftfreq(WIN, 1 / sr)
    n_frames = (n_total - WIN) // HOP
    if n_frames < 4:
        return audio, {"applied": False, "reason": "track too short for spectral revival"}

    rng = np.random.default_rng(seed)

    # self-referential target level: fit this track's own rolloff slope
    # below the cutoff and extrapolate it past the cutoff
    ANALYSIS_WIN = 32768
    _win = np.hanning(ANALYSIS_WIN)
    _freqs_hi = np.fft.rfftfreq(ANALYSIS_WIN, 1 / sr)
    mono = audio.mean(axis=1)
    _mags = []
    for start in range(0, len(mono) - ANALYSIS_WIN, ANALYSIS_WIN // 4):
        seg = mono[start:start + ANALYSIS_WIN] * _win
        _mags.append(np.abs(np.fft.rfft(seg)))
    _avg_spec = np.array(_mags).mean(axis=0)
    _avg_db = 20 * np.log10(_avg_spec + 1e-12)
    _norm_db = _avg_db - _avg_db.max()

    fit_lo, fit_hi = 3000.0, cutoff_hz - 500.0
    fit_mask = (_freqs_hi >= fit_lo) & (_freqs_hi <= fit_hi)
    slope, intercept = np.polyfit(np.log2(_freqs_hi[fit_mask]), _norm_db[fit_mask], 1)

    # a genuine hard/brickwall cutoff (exactly what has_rolloff was designed
    # to detect) does NOT decay smoothly into the cutoff - the level stays
    # roughly flat right up until it doesn't, so a straight-line fit to the
    # region just below it and a naive extrapolation both predict the signal
    # should STILL be near that flat pre-cliff level well past the cutoff -
    # confirmed directly on a real production file where the extrapolated
    # curve (even after anchoring its intercept to the exact measured level
    # AT the cutoff edge) still predicted -47dB up near 20kHz while the
    # source had genuinely collapsed to a -137dB noise floor there - a ~90dB
    # overshoot, injecting synthesized content dramatically louder than
    # anything the source ever had (audible as a sharp tone, visible as a
    # hard step in the spectrum chart). Anchoring the intercept alone wasn't
    # enough because the SLOPE itself is unreliable when the fit region is a
    # flat plateau rather than genuine decay - extrapolating a near-zero
    # slope stays near-flat forever, never approaching the real floor.
    #
    # Fix: measure the track's OWN actual noise floor well above the cutoff
    # directly (still purely self-referential - no external reference), and
    # hard-clamp the target curve to never predict a level louder than a
    # fixed, reasonable margin above that measured floor, regardless of what
    # the fitted slope alone would say. This keeps the slope for its
    # intended purpose (shaping HOW the curve descends from the anchor) but
    # puts a physically-grounded ceiling under it so a wrong/flat slope
    # estimate can no longer produce a wildly-too-loud target.
    anchor_freq = cutoff_hz - 500.0
    anchor_idx = np.argmin(np.abs(_freqs_hi - anchor_freq))
    anchor_db = _norm_db[anchor_idx]
    intercept_anchored = anchor_db - slope * np.log2(anchor_freq)

    floor_lo = min(21000.0, sr / 2 - 500.0)
    floor_mask = _freqs_hi >= floor_lo
    measured_floor_db = float(np.median(_norm_db[floor_mask])) if floor_mask.sum() >= 4 else anchor_db - 40.0
    # texture should sit clearly ABOVE the true noise floor (it's meant to
    # be audible content filling the gap, not literally inaudible), but
    # never anywhere near the source's actual pre-cutoff loudness - +18dB
    # over the measured floor is a deliberately conservative ceiling.
    ceiling_db = measured_floor_db + 18.0

    def target_curve_db(f):
        # guard against log2(0) at the DC bin - target_db_at_freqs is only
        # ever read at extend_mask positions (near/above the cutoff, always
        # far from 0), so this value is never actually used, but computing
        # it unguarded still raises a divide-by-zero warning on the full
        # freqs array (which starts at 0 from np.fft.rfftfreq).
        f_safe = np.maximum(f, 1.0)
        raw = slope * np.log2(f_safe) + intercept_anchored
        return np.minimum(raw, ceiling_db)

    target_db_at_freqs = target_curve_db(freqs)

    # small irregular texture (formant-like bumps) riding on top of the
    # self-derived target level, not replacing it
    range_lo, range_hi = cutoff_hz - 300, min(22050, sr / 2)
    bump_spacing = 600.0
    n_bumps = max(1, int((range_hi - range_lo) / bump_spacing) + 1)
    bump_centers = np.linspace(range_lo, range_hi, n_bumps)
    bump_freqs = bump_centers + rng.uniform(-150, 150, n_bumps)
    bump_widths = rng.uniform(350, 500, n_bumps)
    bump_gains_db = rng.uniform(-2, 2, n_bumps)

    def resonance_shape(f):
        shape = np.zeros_like(f, dtype=np.float64)
        for bf, bw, bg in zip(bump_freqs, bump_widths, bump_gains_db):
            shape += bg * np.exp(-0.5 * ((f - bf) / bw) ** 2)
        return shape

    res_shape_db = resonance_shape(freqs)
    extend_mask = freqs >= (cutoff_hz - 500)
    n_extend_bins = int(extend_mask.sum())
    if n_extend_bins == 0:
        return audio, {"applied": False, "reason": "cutoff too close to Nyquist to extend"}

    src_lo, src_hi = cutoff_hz * 0.35, cutoff_hz * 0.75
    src_mask = (freqs >= src_lo) & (freqs <= src_hi)
    ref_frame_energy = np.sqrt(np.mean(_avg_spec[(_freqs_hi >= src_lo) & (_freqs_hi <= src_hi)] ** 2))
    target_mag_at_freqs = (10 ** (target_db_at_freqs / 20)) * _avg_spec.max()

    out = np.zeros_like(audio)
    ola_norm = np.zeros(n_total)

    for ch in range(audio.shape[1]):
        sig = audio[:, ch]
        for i in range(n_frames):
            start = i * HOP
            seg = sig[start:start + WIN] * window
            spec = np.fft.rfft(seg)
            mag = np.abs(spec)
            phase = np.angle(spec)

            if src_mask.sum() < 4:
                out[start:start + WIN, ch] += seg * window
                if ch == 0:
                    ola_norm[start:start + WIN] += window ** 2
                continue

            src_freqs = freqs[src_mask]
            src_mag = mag[src_mask]
            src_phase = phase[src_mask]

            frame_energy = np.sqrt(np.mean(src_mag ** 2))
            dynamics_mult = frame_energy / (ref_frame_energy + 1e-12)

            # peak detection was far too permissive (local-max + only 1.5x
            # the frame's median) - confirmed directly on a real file that
            # this flags 70-80+ bins as "peaks" in a single frame during
            # busy/noisy passages, since ordinary spectral texture easily
            # exceeds a 1.5x-median bar. Each flagged peak below projects up
            # to 6 overlapping harmonics into the revived region - 75+ peaks
            # means hundreds of simultaneous tonal injections, which is
            # audible as ringing/a high-pitched artifact, not the "a few
            # genuine musical harmonics" this was designed to add. Two fixes:
            # (1) require real prominence (6x median, not 1.5x) so only
            # genuinely strong tonal content counts as a peak, and (2) cap
            # to the strongest few peaks per frame regardless, so even a
            # frame that's genuinely peak-dense (rare, but possible) can't
            # inject an unbounded number of harmonics.
            MAX_HARMONIC_SOURCE_PEAKS = 6
            peak_idx = []
            for k in range(1, len(src_mag) - 1):
                if src_mag[k] > src_mag[k - 1] and src_mag[k] > src_mag[k + 1] and src_mag[k] > np.median(src_mag) * 6.0:
                    peak_idx.append(k)
            if len(peak_idx) > MAX_HARMONIC_SOURCE_PEAKS:
                peak_idx = sorted(peak_idx, key=lambda k: -src_mag[k])[:MAX_HARMONIC_SOURCE_PEAKS]

            new_spec = spec.copy()

            # harmonic projection from this frame's own detected peaks
            for k in peak_idx:
                f0 = src_freqs[k]
                a0 = src_mag[k]
                p0 = src_phase[k]
                h = 2
                while f0 * h < min(22050, sr / 2) and h < 8:
                    target_f = f0 * h
                    if target_f >= cutoff_hz - 500:
                        bin_idx = int(round(target_f / (sr / WIN)))
                        if bin_idx < len(freqs):
                            decay_db = -8.0 * np.log2(h)
                            amp = a0 * (10 ** (decay_db / 20))
                            amp *= 10 ** (res_shape_db[bin_idx] / 20)
                            new_phase = (p0 * h) % (2 * np.pi)
                            new_spec[bin_idx] += amp * np.exp(1j * new_phase)
                    h += 1

            # broadband texture at the self-derived target level, modulated
            # by this frame's own dynamics relative to the track's average
            if n_extend_bins > 0:
                base_mag = target_mag_at_freqs[extend_mask] * dynamics_mult
                texture_mag = base_mag * (10 ** (res_shape_db[extend_mask] / 20))
                micro_var = rng.uniform(0.75, 1.25, n_extend_bins)
                texture_mag = texture_mag * micro_var
                texture_phase = rng.uniform(0, 2 * np.pi, n_extend_bins)
                new_spec[extend_mask] += texture_mag * np.exp(1j * texture_phase)

            new_seg = np.fft.irfft(new_spec, n=WIN)
            out[start:start + WIN, ch] += new_seg * window
            if ch == 0:
                ola_norm[start:start + WIN] += window ** 2

    steady_state = np.median(ola_norm[ola_norm > 0])
    edge_thresh = steady_state * 0.5
    safe_mask = ola_norm >= edge_thresh
    out[~safe_mask] = 0.0
    out[safe_mask] = out[safe_mask] / ola_norm[safe_mask, None]

    peak = np.abs(out).max()
    if peak > 0.98:
        out = out * (0.95 / peak)

    return out, {
        "applied": True,
        "cutoff_hz": cutoff_hz,
        "fitted_rolloff_db_per_octave": float(slope),
        "deficit_db": deficit_db,
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


def waveform_peaks(audio, sr, n_buckets=1200):
    """Downsampled min/max envelope for a waveform display: bucket the
    track into n_buckets equal-time columns and keep each column's min and
    max sample value (mono-summed), the standard technique for rendering a
    waveform overview without shipping every sample to the browser. Also
    returns each bucket's RMS so the UI can show a denser "loudness" fill
    inside the min/max outline."""
    mono = audio.mean(axis=1)
    n = len(mono)
    if n == 0:
        return {"min": [], "max": [], "rms": [], "duration_sec": 0.0}
    bucket_size = max(1, n // n_buckets)
    n_buckets_actual = max(1, n // bucket_size)
    trimmed = mono[: n_buckets_actual * bucket_size]
    reshaped = trimmed.reshape(n_buckets_actual, bucket_size)
    mins = reshaped.min(axis=1)
    maxs = reshaped.max(axis=1)
    rms = np.sqrt(np.mean(reshaped ** 2, axis=1))
    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "rms": rms.tolist(),
        "duration_sec": n / sr,
    }


def read_source_format(path):
    """Technical format details of the ORIGINAL uploaded file, straight
    from ffprobe - codec, sample rate, bit depth, channels, container - so
    the UI can show the user exactly what they uploaded (distinct from
    read_metadata_tags, which reports embedded tags/artwork, not the
    technical format itself)."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    try:
        data = _json.loads(result.stdout)
    except _json.JSONDecodeError:
        return {}

    fmt = data.get("format", {})
    audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    return {
        "container": fmt.get("format_long_name") or fmt.get("format_name"),
        "codec": audio_stream.get("codec_long_name") or audio_stream.get("codec_name"),
        "sample_rate_hz": int(audio_stream["sample_rate"]) if audio_stream.get("sample_rate") else None,
        "bit_depth": audio_stream.get("bits_per_sample") or None,
        "channels": audio_stream.get("channels"),
        "bit_rate_kbps": round(int(fmt["bit_rate"]) / 1000) if fmt.get("bit_rate") else None,
        "file_size_bytes": int(fmt["size"]) if fmt.get("size") else None,
    }


def read_metadata_tags(path):
    """Read every container/ID3 metadata tag from the source file - both
    format-level (title/artist/comment/encoder/etc.) AND per-stream tags,
    plus a report of any non-audio streams (embedded cover art, attached
    images) since those can themselves carry their own metadata (e.g. EXIF
    in a JPEG) and most users don't expect a cover-art image riding along
    inside an audio file at all.

    Many AI-generation platforms embed identifying tags (comment fields
    naming the platform, generation UUIDs, timestamps, platform-style
    artist handles) directly in the uploaded file's metadata, independent
    of anything detectable in the audio itself. This never modifies the
    file - it's a read-only report used by /api/analyze so a user can see
    everything the original upload was carrying.

    Returns {"format": {...}, "streams": [{"index", "codec_type",
    "codec_name", "tags": {...}}, ...]}."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"format": {}, "streams": []}
    try:
        data = _json.loads(result.stdout)
    except _json.JSONDecodeError:
        return {"format": {}, "streams": []}

    format_tags = data.get("format", {}).get("tags", {}) or {}
    streams = []
    for s in data.get("streams", []):
        streams.append({
            "index": s.get("index"),
            "codec_type": s.get("codec_type"),
            "codec_name": s.get("codec_name"),
            "is_attached_image": bool(s.get("disposition", {}).get("attached_pic")),
            "tags": s.get("tags", {}) or {},
        })
    return {"format": format_tags, "streams": streams}


def strip_metadata_tags(in_path, out_path):
    """Write a copy of in_path with every container/ID3 metadata tag removed
    (title, artist, comment, encoder, any platform-identifying fields) at
    both the format AND per-stream level, and drops any non-audio stream
    entirely (embedded cover art/attached images), rather than only
    clearing their tags - the image data itself is removed, not just its
    label. The actual audio stream is stream-copied (not re-encoded), so
    audio quality is untouched.

    One unavoidable exception: ffmpeg's own muxer writes a small
    self-identifying "encoder" tag (e.g. "Lavf62.12.100") into most output
    containers regardless of -map_metadata; this identifies ffmpeg itself,
    not the source platform, and there is no ffmpeg flag that suppresses
    it. Every OTHER tag - including all platform/generation-identifying
    fields - is fully removed."""
    subprocess.run(
        ["ffmpeg", "-v", "quiet", "-y", "-i", str(in_path),
         "-map", "0:a", "-map_metadata", "-1", "-map_chapters", "-1",
         "-c", "copy", str(out_path)],
        check=True,
    )
