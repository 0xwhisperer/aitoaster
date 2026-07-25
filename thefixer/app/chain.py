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


def high_pass_filter(audio, sr, cutoff_hz=30, order=4):
    # order raised from 2 to 4: verified directly that a 2nd-order filter
    # only asymptotically approaches a clean sub-cutoff floor (-24.3dB ->
    # -31.6dB across 4 repeated passes on a real file, never actually
    # converging in one pass) - a steeper 4th-order rolloff clears
    # meaningful sub-30Hz content in a SINGLE pass instead, which is what
    # makes the "does this still need running" check below meaningful.
    # Still fully inaudible either way - 30Hz sits below all audible
    # musical fundamentals regardless of filter steepness.
    """Gentle high-pass to remove sub-sonic rumble/DC residue that a plain
    mean-subtraction can't catch (e.g. slow drift). 30Hz is below all
    audible fundamentals so it is inaudible on music but keeps the mix
    clean for LUFS metering and limiting.

    BUG FIX: this used to always report applied=True even when there was
    negligible sub-cutoff content to begin with - meaning a file already
    run through this app would be unconditionally recommended for this
    tool again on re-analysis. An earlier attempt at this fix compared how
    much energy THIS PASS removed against a threshold - verified directly
    that this is the wrong measurement: a gentle 2nd-order Butterworth
    doesn't have an infinitely steep cutoff, so a SECOND pass on already-
    filtered audio still finds a bit more to remove (measured directly:
    -24.3dB, then -26.9dB, then -28.8dB across three successive passes on
    the same file, converging toward zero but never actually reaching it) -
    "did this pass remove something" is true on essentially every pass,
    forever. The right question is whether the track's own REMAINING
    sub-cutoff content, measured directly (not by how much a filter pass
    happens to strip), is large enough to plausibly matter for LUFS/
    limiting headroom in the first place."""
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    out = np.stack([signal.sosfiltfilt(sos, audio[:, ch]) for ch in range(audio.shape[1])], axis=1)

    # BUG FIX: measuring the whole 0-cutoff_hz band conflates genuine deep
    # rumble with ordinary bass content right at the boundary - verified
    # directly on a real bass-heavy track: 20-30Hz content alone measured
    # -35.7dB (legitimate music, not rumble), while true sub-sonic content
    # (0-10Hz, where rumble/rumbling-truck/HVAC/handling-noise artifacts
    # actually live) measured a full 14dB lower at -42.8dB. A single
    # cutoff_hz-wide lowpass measurement can never tell these apart, since
    # any filter's transition band always retains some energy from
    # content legitimately just above the passband edge - no amount of
    # raising the filter order fixes that, because the content itself is
    # real and close to the boundary, not filter residue. Measure only
    # the bottom THIRD of the sub-cutoff range, where genuine rumble
    # actually concentrates, not the whole band.
    deep_rumble_hz = cutoff_hz / 3.0
    lowpass_sos = signal.butter(order, deep_rumble_hz, btype="lowpass", fs=sr, output="sos")
    sub_cutoff_content = signal.sosfiltfilt(lowpass_sos, audio, axis=0)
    # BUG FIX (third adversarial audit round, verified directly): sosfiltfilt
    # on a signal with no natural silence/fade at its own boundaries leaves
    # real ringing/edge-transient artifacts concentrated in roughly the
    # first/last fraction of a second - confirmed directly: a pure 440Hz
    # tone (zero genuine sub-10Hz content anywhere) measured -30.3dB of
    # "sub-cutoff content" using the whole-signal RMS, comfortably above
    # the -40dB bar, purely from edge artifacts (measured directly:
    # start-of-signal RMS was 0.022, dead-center RMS was exactly 0.0).
    # Trimming a small edge margin before measuring removes the artifact
    # almost entirely (verified: 0.2s margin drops the false-positive
    # reading by two orders of magnitude) while still covering the vast
    # majority of any real track's duration - genuine sub-sonic rumble is
    # rarely confined to only the first/last 0.2s of a file, so this
    # doesn't meaningfully weaken real-rumble detection.
    edge_margin = min(len(sub_cutoff_content) // 4, int(0.2 * sr))
    if len(sub_cutoff_content) > edge_margin * 2:
        measured_region = sub_cutoff_content[edge_margin:-edge_margin] if edge_margin > 0 else sub_cutoff_content
    else:
        measured_region = sub_cutoff_content
    sub_cutoff_rms = float(np.sqrt(np.mean(measured_region ** 2)))
    overall_rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-12
    # -40dB relative to the track's own overall level, now measured only
    # in the genuine-rumble range - real deep rumble in a file that needs
    # this filter should sit well above this bar; ordinary bass content
    # near the cutoff edge (excluded by the narrower measurement above)
    # never triggers it in the first place.
    meaningful = (sub_cutoff_rms / overall_rms) > 10 ** (-40 / 20)

    return out, {"applied": bool(meaningful), "cutoff_hz": cutoff_hz, "order": order}


def detect_transients(audio, sr, jump_threshold=0.35, envelope_ratio_threshold=8.0, min_gap_sec=0.01,
                      burst_window_sec=0.03, max_burst_crossings=8):
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
    # BUG FIX HISTORY (two adversarial audit rounds, both confirmed
    # directly against real test cases):
    #   Round 1: the OLD design reported local_peak_idx - the loudest
    #   sample within +-min_gap_sec/2 of the trigger - as the click's
    #   location, then skipped the FULL min_gap_sec (0.5s default) before
    #   scanning again. Two real problems: (a) the reported location was
    #   often just a nearby loud musical passage, not the actual
    #   discontinuity, so fix_transient corrected the wrong sample
    #   entirely; (b) skipping a full 0.5s after every hit meant two
    #   genuinely separate clicks 0.3s apart could never both be found.
    #   Round 1's fix replaced the peak search with a tiny fixed +-1ms
    #   window (jump_search, below) - correct, and replaced the skip
    #   distance with a small fixed value too - which fixed (a) and (b)
    #   but broke min_gap_sec entirely: the parameter stopped being used
    #   ANYWHERE in the function, so passing a different min_gap_sec had
    #   no effect at all, and a genuine BIPHASIC click (a single real
    #   glitch event whose own waveform has two discontinuities a few ms
    #   apart - e.g. a digital dropout's drop AND recovery edge) got
    #   reported as two separate detections instead of one, risking two
    #   overlapping corrective passes on what should be one bridge.
    #
    #   Round 2 fix: min_gap_sec is restored to real, active use as the
    #   actual dedup distance (its own name's stated purpose), but its
    #   DEFAULT is lowered from 0.5s to 0.01s (10ms) - the old 0.5s value
    #   was sized for the REMOVED wide-peak-search radius, not for
    #   deduping a single glitch's own few-millisecond ringing, and was
    #   never validated as the right value for that different purpose.
    #   10ms comfortably covers a biphasic click's own multi-part ringing
    #   while staying far below the 0.3s separation Round 1's own test
    #   case required to keep working - callers needing a wider dedup
    #   window for a specific use case can still pass a larger
    #   min_gap_sec explicitly.
    min_gap = max(1, int(min_gap_sec * sr))
    jump_search = max(1, int(0.001 * sr))  # +-1ms peak-location search,
    # independent of min_gap - see Round 1 history above for why the
    # PEAK search radius and the dedup SKIP distance must be separate
    # values, not the same one.
    # BUG FIX (direct user report, "the transient tool is blowing out the t's
    # in the vocal", verified on the reported track): a vocal PLOSIVE or
    # FRICATIVE ("t", "s", "k") satisfies both tests above and was being
    # deleted as if it were a digital click.
    #
    # Two things conspire. A consonant is a broadband burst whose waveform
    # oscillates fast enough to clear jump_threshold over and over, and
    # singing often sits in a quiet gap where the 200ms envelope has
    # collapsed - measured on the reported track at 35.9s, the local envelope
    # fell to 0.0417 from 0.1410 half a second earlier, so the burst also
    # cleared the 8.0 envelope ratio easily. 25 separate samples triggered
    # inside 90ms there, and 19 of the track's 25 total detections were
    # consonants rather than clicks.
    #
    # The discriminator is DURATION, which is what physically separates the
    # two events. A digital click is a near-instantaneous discontinuity: one
    # or two samples cross the jump threshold and the waveform is continuous
    # either side. A consonant is a SUSTAINED burst lasting tens of
    # milliseconds, so it crosses the threshold hundreds of times in a row
    # (measured: 115-182 crossings per 60ms window at the reported spots,
    # versus 2-5 for the genuine one-off clicks elsewhere in the same track).
    #
    # This matters more than a normal false positive because fix_transient
    # repairs a click by DELETING it - replacing the region with linear
    # interpolation between clean neighbours - so a false positive here does
    # not merely duck the consonant, it erases it (measured: 42% of the
    # consonant's 4-12kHz energy gone, -4.8dB).
    burst_window = max(1, int(burst_window_sec * sr))
    over_jump = (jump > jump_threshold).astype(np.int32)
    # Count threshold crossings in the window centred on each sample, via a
    # cumulative sum so this stays O(n) on a full-length track.
    cumulative = np.concatenate([[0], np.cumsum(over_jump)])
    half = burst_window // 2

    while i < n:
        if jump[i] > jump_threshold and ratio[i] > envelope_ratio_threshold:
            lo = max(0, i - half)
            hi = min(n, i + half)
            crossings = int(cumulative[hi] - cumulative[lo])
            if crossings > max_burst_crossings:
                # Sustained broadband burst - a consonant or other musical
                # texture, not a discontinuity. Skip past the whole burst
                # rather than one sample, so its remaining crossings do not
                # each get re-tested.
                i = hi
                continue
            peak_lo = max(0, i - jump_search)
            peak_hi = min(n, i + jump_search)
            click_peak_idx = peak_lo + int(np.argmax(np.abs(mono[peak_lo:peak_hi])))
            candidates.append(click_peak_idx)
            i = i + min_gap
        else:
            i += 1
    return [{"time_sec": round(c / sr, 3), "peak": float(np.abs(mono[c]))} for c in candidates]


def fix_transient(audio, sr, time_sec, target_peak=None, attack_ms=3, release_ms=60, context_sec=0.3):
    """De-click a genuine sample-to-sample discontinuity by bridging across it
    with an interpolated ramp from the clean samples on either side, THEN
    applying a raised-cosine gain envelope over that bridged region for any
    remaining loudness excess.

    BUG FIX (direct user report, real production job): the previous version
    only ever applied a gain-envelope attenuation AROUND the click - it never
    touched the click's own two discontinuous samples. A genuine pop is a
    single-sample (or few-sample) near-instantaneous jump; multiplying it by
    a gain <1 makes it QUIETER but the discontinuity itself - the actual
    thing detect_transients' jump-vs-envelope test measures - survives
    almost completely intact (confirmed directly: a raw jump of ~1.7 became
    a "fixed" jump of 1.485, still 4x detect_transients' own jump_threshold
    of 0.35). This is why the results table and the post-chain corrective
    pass kept reporting "still flagged" after a fix had supposedly already
    run - it wasn't a detection bug, it was that this function's approach
    could never clear detect_transients' own criteria for a genuinely sharp
    click, no matter how many corrective rounds ran (verified: looping the
    old fix 5x on the same click still left it detected every time, slowly
    drifting to adjacent samples as each pass's own envelope edges
    introduced new smaller jumps).

    The fix: DELETE the discontinuity itself. A short window straddling the
    click's exact peak sample is replaced by linear interpolation between
    the clean sample just before the window and the clean sample just after
    it - this is the standard declick technique (the same principle as
    audio restoration tools' interpolative declickers), not a novel
    approach. target_peak/context logic is kept for backward compatibility
    of the info dict (peak_before/peak_after/reduction_db) and to size the
    surrounding gain-smoothing taper, but the interpolation bridge is what
    actually removes the jump detect_transients measures."""
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

    # bridge window: wide enough to span the actual discontinuity (a real
    # click's sample-to-sample jump is essentially instantaneous, so a few
    # samples of true bridge is enough) plus a short cosine-eased taper on
    # each side so the interpolated region doesn't itself start/end with a
    # new discontinuity relative to the untouched audio around it.
    #
    # BUG FIX (Grok #9 / Fable N3, verified directly): audited concern was
    # that this fixed ~30-sample width is single-file-tuned and would only
    # partially bridge a wider dropout/glitch (not a single-sample pop).
    # Verified directly this is out of scope by construction, not an
    # unbounded risk: detect_transients (this function's only caller inside
    # the pipeline) requires a sample-to-sample derivative jump above
    # jump_threshold=0.35 - at real sample rates (44.1kHz+), a genuinely
    # WIDE dropout's edges (even a full drop into silence within loud
    # content) never produce a single-sample delta anywhere near that bar,
    # since the signal itself is band-limited and changes gradually sample
    # to sample. Confirmed directly: a 10ms dropout inside a 200Hz tone at
    # 44.1kHz produces a max single-sample jump of ~0.023, nowhere near
    # 0.35, while a genuine single-sample click (spiking instantly to 0.9
    # against a quiet background) produces a jump of ~0.91 and IS detected.
    # A 1kHz sample-rate repro of the same dropout DOES falsely cross the
    # threshold (coarser sampling makes the same physical edge look like a
    # much bigger single-sample jump) - that artifact does not occur at any
    # sample rate this app actually processes audio at.
    bridge_half = max(2, int(0.0007 * sr))  # ~30 samples at 44.1kHz
    taper = max(1, int(0.002 * sr))  # ~2ms cosine taper into/out of the bridge
    b_lo = max(0, center - bridge_half)
    b_hi = min(n, center + bridge_half)
    edge_lo = max(0, b_lo - taper)
    edge_hi = min(n, b_hi + taper)

    if edge_hi - edge_lo < 4 or b_hi <= b_lo:
        return audio, {"applied": False, "reason": "too close to track boundary to bridge safely"}

    out = audio.copy()
    anchor_lo = audio[edge_lo]
    anchor_hi = audio[edge_hi - 1] if edge_hi - 1 < n else audio[-1]
    span = edge_hi - edge_lo
    t = np.linspace(0.0, 1.0, span)[:, None]
    bridged = anchor_lo[None, :] * (1 - t) + anchor_hi[None, :] * t

    # cosine crossfade between the original signal and the bridged/interpolated
    # signal across the taper regions, so only the very center (the actual
    # discontinuity) is pure interpolation - most of the taper blends smoothly
    # back to the real, untouched waveform on either side.
    fade = np.ones(span)
    left_taper_n = b_lo - edge_lo
    right_taper_n = edge_hi - b_hi
    for i in range(left_taper_n):
        frac = i / max(1, left_taper_n)
        fade[i] = 0.5 - 0.5 * np.cos(np.pi * frac)
    for i in range(right_taper_n):
        frac = i / max(1, right_taper_n)
        fade[span - right_taper_n + i] = 0.5 + 0.5 * np.cos(np.pi * frac)
    fade = fade[:, None]

    out[edge_lo:edge_hi] = audio[edge_lo:edge_hi] * (1 - fade) + bridged * fade

    # any remaining loudness excess right around the (now de-clicked) region
    # still gets the original gentle gain taper, sized from target_peak, so a
    # click in a otherwise-loud passage doesn't leave a locally over-hot
    # bridge relative to the surrounding music.
    post_bridge_peak = np.abs(out[lo:hi]).max()
    if post_bridge_peak > target_peak * 1.05:
        reduction_ratio = target_peak / post_bridge_peak
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
        out = out * gain[:, None]

    return out, {
        "applied": True,
        "time_sec": time_sec,
        "peak_before": float(local_peak),
        "peak_after": float(np.abs(out[lo:hi]).max()),
        "reduction_db": float(20 * np.log10(max(1e-9, np.abs(out[lo:hi]).max()) / max(1e-9, local_peak))),
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
    without collapsing the stereo image entirely. Falls back to an
    all-pass phase shift ONLY in the narrow case where mid/side blending
    mathematically cannot reach the target (see below).

    BUG FIX (third adversarial audit round, verified directly): the
    mid/side blend approach works correctly for realistic broadband
    stereo content (verified: a real decorrelated-noise fixture reaches
    its target correlation with the ORIGINAL formula, confirmed directly
    by re-running it in isolation) - but is mathematically broken for a
    PERFECTLY out-of-phase signal (L = -R exactly): mid = (L+R)/2 is
    IDENTICALLY ZERO in that case, so "blending in more mid" blends in
    nothing. Confirmed directly: L=-R with min_correlation=0.1 reported
    applied=True with correlation_after=-1.0, completely unchanged. For
    any single-frequency tone under ANY linear amplitude blend,
    correlation is mathematically bimodal (lands at exactly -1, exactly
    +1, or undefined - no continuous path through 0.1).

    FIRST FIX ATTEMPT (this same audit pass): replaced mid/side blending
    entirely with an all-pass phase shift, which fixed the degenerate
    pure-tone case but REGRESSED the existing broadband-noise test - all-
    pass shifts a single channel's phase without reference to the OTHER
    channel's actual content, and verified directly that for broadband
    noise the correlation-vs-coefficient relationship is much weaker and
    can move in either direction depending on the specific signal,
    sometimes never reaching the target at all even at the strongest
    shift tried. Mid/side blending is provably the right primary tool
    for realistic content; all-pass is only needed for the pure-tone
    degeneracy. Final fix: try mid/side first (as before), and ONLY fall
    back to all-pass if that genuinely fails to reach the target - this
    preserves correctness on realistic audio while still fixing the
    narrow degenerate case."""
    corr = stereo_correlation(audio)
    if corr >= min_correlation:
        return audio, {"applied": False, "correlation": corr}

    mid = audio.mean(axis=1)
    side = (audio[:, 0] - audio[:, 1]) / 2
    blend = min(0.5, (min_correlation - corr))
    l_fixed = mid + side * (1 - blend)
    r_fixed = mid - side * (1 - blend)
    out = np.stack([l_fixed, r_fixed], axis=1)
    corr_after = stereo_correlation(out)

    if corr_after >= min_correlation:
        return out, {
            "applied": True,
            "correlation_before": corr,
            "correlation_after": corr_after,
            "method": "mid_side_blend",
            "side_blend_reduction": blend,
        }

    # mid/side blend did not reach the target (the near-zero-mid
    # degenerate case) - fall back to an all-pass phase shift, which
    # breaks the anti-phase relationship a linear amplitude blend cannot
    # touch. Binary search the coefficient for the smallest shift that
    # reaches min_correlation, checking the REAL resulting correlation on
    # each trial rather than assuming a formula.
    def _allpass(x, coeff):
        b = [-coeff, 1.0]
        a = [1.0, -coeff]
        return signal.lfilter(b, a, x)

    left = audio[:, 0]
    right = audio[:, 1]
    lo, hi = 0.0, 0.999
    best_coeff = hi
    best_corr = None
    best_right = right
    for _ in range(24):
        mid_coeff = (lo + hi) / 2
        trial_right = _allpass(right, mid_coeff)
        trial_corr = stereo_correlation(np.stack([left, trial_right], axis=1))
        if trial_corr >= min_correlation:
            best_coeff = mid_coeff
            best_corr = trial_corr
            best_right = trial_right
            hi = mid_coeff
        else:
            lo = mid_coeff

    if best_corr is None:
        best_right = _allpass(right, hi)
        best_corr = stereo_correlation(np.stack([left, best_right], axis=1))
        best_coeff = hi

    out2 = np.stack([left, best_right], axis=1)
    return out2, {
        "applied": True,
        "correlation_before": corr,
        "correlation_after": best_corr,
        "method": "allpass_fallback",
        "allpass_coeff": float(best_coeff),
        "target_met": bool(best_corr >= min_correlation),
    }


def _band_envelope_db(audio, sr, lo, hi, nyq):
    """Shared band-split + envelope extraction used by both
    detect_band_peakiness (a real, independent measurement of THIS FILE's
    own condition) and multiband_compress (the correction itself) - kept
    as one function so the two can never silently drift apart the way
    detect_spectral_rolloff and spectral_revive once did this session."""
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
    return band_audio, env_db


def detect_band_peakiness(audio, sr, bands=None,
                           threshold_db=-12.0):
    """Measures whether THIS FILE genuinely has peaky/imbalanced dynamics in
    each band - a real, independent property of the audio itself, not "did
    multiband_compress's own gain math find something to shave off."

    BUG FIX (direct user report): the previous approach ran the compressor
    itself and checked whether its OWN reduction exceeded a small dB bar -
    but a diminishing-returns tool asked "would you still find something"
    will keep saying "a little" almost forever on a real peaky file, since
    a gentle ratio (1.3 by design) only closes part of the gap each pass.
    Verified directly: running multiband_compress 6 times in a row on the
    same peaky signal, its own max_reduction_db barely dropped (-0.53dB ->
    -0.48dB by pass 6, still comfortably above any reasonable "done" bar),
    while the file's OWN measured over-threshold condition in that band
    (peak_over_db) barely moved either (2.30dB -> 2.11dB) - because the
    compressor's gentle ratio was never designed to fully close a real gap
    in one pass, by design. The file WAS still genuinely peaky the whole
    time; that's not a bug in the compressor, it's the wrong question being
    asked of it for recommendation purposes.

    The right signal is a property of the FILE, not the compressor's own
    trace: how much of the track's duration is spent meaningfully over
    threshold. A single loud transient can briefly spike a band's peak
    level even on a well-mastered file (confirmed directly: a real,
    already-processed file measured 4.75dB of peak_over_db in its low band
    - HIGHER than a deliberately-built peaky test signal's 1.08dB) without
    that file actually being peaky/imbalanced - but that file's time spent
    meaningfully over threshold was only 1.3% of its duration, versus
    16-25% for the genuinely peaky signal. frac_time_over is the real,
    stable, file-only signal; a brief transient doesn't move it much, a
    genuinely imbalanced track does."""
    nyq = sr / 2
    # BUG FIX (adversarial audit, verified directly): the default top band
    # used to be hardcoded to end at 20000Hz regardless of sample rate -
    # but _band_envelope_db's bandpass filter is a real 4th-order
    # Butterworth, not a brick wall, so it still has genuine attenuating
    # effect well past its nominal edge. At 44.1kHz (nyquist=22050Hz),
    # that left a real ~2050Hz gap (20000-22050Hz) where the filter was
    # still measurably reducing content, but neither this detector nor
    # multiband_compress's own recommendation logic ever scanned that
    # range - confirmed directly: a pure 21kHz tone measured genuine
    # ~48dB of real attenuation from multiband_compress, while this
    # function (and therefore the /api/analyze recommendation) reported
    # zero peakiness there and multiband_compress's "applied" flag came
    # back False, because the tool's own declared band boundary and its
    # actual filter reach didn't match. Default bands now derive the top
    # edge from the ACTUAL sample rate's Nyquist frequency, so the
    # declared coverage always matches what the filter genuinely reaches,
    # at any sample rate - not just the 44.1kHz case this bug happened to
    # be found on.
    if bands is None:
        bands = ((0, 200), (200, 2000), (2000, nyq))
    # BUG FIX (adversarial audit, verified directly): this guard band
    # (0.5dB) and multiband_compress's own "applied" bar (0.3dB of actual
    # gain reduction) were two independently-chosen numbers with no
    # guaranteed relationship - confirmed directly a real 0.38-amplitude
    # 1kHz tone crossed THIS threshold (recommend=True) while its actual
    # correction stayed under multiband_compress's own applied bar
    # (applied=False), producing exactly the "recommended, but reports no
    # change needed" contradiction. multiband_compress's gain math is
    # max_reduction_db = -over_db * (1 - 1/ratio) - solving for the over_db
    # that guarantees crossing that function's own 0.3dB applied bar (at
    # the default ratio=1.3) gives 1.3dB. Using that SAME derived value
    # here (not a second independently-chosen number) guarantees anything
    # this function recommends will also cross multiband_compress's own
    # applied threshold - the two can never disagree again because they're
    # now tied to the same underlying relationship, not two separate taste
    # calls. This constant does NOT need updating by hand if ratio ever
    # changes elsewhere, since it's a fixed property of the DEFAULT ratio
    # this function's own default threshold_db/ratio pairing represents -
    # if a caller passes a different ratio, the two functions can still
    # disagree, but the SHARED default configuration (what /api/analyze and
    # the pipeline actually use) is now guaranteed consistent.
    MEANINGFULLY_OVER_DB = 1.3
    results = []
    for lo, hi in bands:
        _, env_db = _band_envelope_db(audio, sr, lo, hi, nyq)
        over = np.maximum(env_db - threshold_db, 0)
        results.append({
            "range_hz": [lo, round(min(hi, nyq - 1))],
            "peak_over_db": float(over.max()) if len(over) else 0.0,
            "frac_time_over": float((over > MEANINGFULLY_OVER_DB).mean()) if len(over) else 0.0,
        })
    return results


def multiband_compress(audio, sr, bands=None,
                        ratio=1.3, threshold_db=-12.0):
    """Gentle 3-band downward compression for tonal-balance smoothing -
    reduces peaky dynamic imbalance between low/mid/high without touching
    overall spectral tilt aggressively. Conservative defaults (low ratio,
    higher threshold) by design: this should only be shaping the loudest
    peaks in each band, not continuously riding gain on the whole track -
    least change necessary to smooth genuine imbalance."""
    nyq = sr / 2
    # default bands derive their top edge from the actual sample rate's
    # Nyquist frequency, matching detect_band_peakiness's own default (see
    # that function's comment for the real bug this fixes - a hardcoded
    # 20000Hz edge left a real gap versus this tool's actual filter reach).
    if bands is None:
        bands = ((0, 200), (200, 2000), (2000, nyq))
    out = np.zeros_like(audio)
    info_bands = []
    for lo, hi in bands:
        band_audio, env_db = _band_envelope_db(audio, sr, lo, hi, nyq)
        over = np.maximum(env_db - threshold_db, 0)
        gain_db = -over * (1 - 1 / ratio)
        gain = 10 ** (gain_db / 20)
        out += band_audio * gain[:, None]
        info_bands.append({"range_hz": [lo, round(min(hi, nyq - 1))], "max_reduction_db": float(gain_db.min())})

    # BUG FIX (adversarial audit, verified directly): splitting into bands,
    # applying independent per-band gain reduction, and summing back
    # together can produce a RECOMBINED peak higher than the original
    # signal's own peak, even though every individual band's gain is <=1 -
    # the band-split filters shift each band's phase differently, so the
    # three signals can align more constructively after filtering than
    # they did in the original unfiltered signal. Confirmed directly: a
    # real -1.085dBTP input (correctly NOT recommended for the limiter)
    # came out of multiband_compress at 1.0488 sample peak / +0.67dBTP -
    # genuine digital clipping introduced by a tool the limiter precheck
    # had already cleared. Emergency anti-clip safety net, same pattern
    # already used in linear_fix.py/cnn_fix.py for the identical
    # recombination-can-exceed-input-peak risk in their own delta-transfer
    # steps - this is not the app's real loudness ceiling (true_peak_limit
    # is), just a hard floor under any accidental clipping this tool's own
    # band recombination could introduce.
    peak = np.abs(out).max()
    if peak > 0.97:
        out *= 0.97 / peak

    # "applied" reports whether THIS pass did anything real (any nonzero
    # reduction, filtering out pure floating-point noise) - this is
    # correct and unchanged; recommending the tool again on re-analysis is
    # now a SEPARATE decision made by detect_band_peakiness against the
    # file's own condition, not by asking this function to grade its own
    # diminishing-returns homework (see that function's docstring for why
    # the old approach here was wrong).
    any_real_reduction = any(b["max_reduction_db"] < -0.3 for b in info_bands)
    # BUG FIX (adversarial audit, verified directly): even with zero real
    # gain reduction in every band, the band-split -> sosfiltfilt ->
    # recombine process still introduces small, real differences from the
    # original samples - not from compression (every band's gain is
    # exactly 1.0 in this case), but from filtfilt's inherent edge-
    # transient behavior (zero-phase forward-backward filtering settles in
    # over its first/last several samples). Confirmed directly: a quiet
    # 21kHz tone with real_reduction=False still differed from the
    # original by up to 0.0051 at sample 0, decaying to exactly 0.0 by the
    # steady-state middle of the signal - a real DSP artifact, not a
    # correctness bug in the gain math, but still a genuine violation of
    # "no change needed" when info["applied"] is False. When nothing was
    # actually reduced, return the untouched original directly rather than
    # the filtered-and-recombined-at-unity-gain version, so "applied:
    # False" is an honest, exact guarantee, not just "no gain reduction,
    # but still some filter-processing residue."
    if not any_real_reduction:
        return audio, {"applied": False, "ratio": ratio, "threshold_db": threshold_db, "bands": info_bands}
    return out, {"applied": any_real_reduction, "ratio": ratio, "threshold_db": threshold_db, "bands": info_bands}


def sample_peak_safety_clamp(audio, ceiling_db=-1.0):
    """Minimal, surgical safety net: flat gain scale-down ONLY if the raw
    sample peak actually exceeds ceiling, no oversampling/resampling at
    all. Not a substitute for true_peak_limit's real inter-sample-peak
    awareness or its dynamics-preserving attack/release shaping - use this
    ONLY for a re-verification safety pass where the goal is "don't ship
    something that clips," not full-quality mastering-grade limiting.

    BUG FIX (direct user report, real production job): the pipeline's
    post-CNN-reverification safety limiter re-run used the SAME
    true_peak_limit function as the main pipeline - but that function's
    oversample/downsample round-trip (resample_poly up then down, only
    triggered when actual limiting is needed) introduces real, broadband
    reconstruction noise across the ENTIRE signal, not just at the
    limited peak. Measured directly: a pure 4x-then-1x resample round-trip
    alone (with zero gain change) still produced up to -46dB of real
    sample-to-sample difference across a whole test signal - exactly the
    magnitude of noise this session has repeatedly found is enough to
    destabilize a fragile CNN-optimized adversarial correction. Confirmed
    on a real production job: a CNN result re-verified down to a genuinely
    passing 14.17% regressed to 77.147% with this limiter re-run as the
    only step in between, and the pipeline's own ping-pong guard then
    refused to re-run CNN again, shipping the broken 77% file. A
    re-verification safety pass doesn't need full inter-sample-peak
    awareness or dynamics-preserving shaping - it only needs to guarantee
    the file doesn't ship with raw digital clipping introduced by
    whatever ran just before it. A flat clamp with no resampling at all
    achieves that without reintroducing broadband noise into an
    already-fragile signal."""
    ceiling = 10 ** (ceiling_db / 20)
    peak = np.abs(audio).max()
    if peak <= ceiling:
        return audio, {"applied": False, "sample_peak_db": float(20 * np.log10(peak + 1e-12)), "ceiling_db": ceiling_db}
    scaled = audio * (ceiling / peak)
    return scaled, {
        "applied": True,
        "sample_peak_db_before": float(20 * np.log10(peak + 1e-12)),
        "ceiling_db": ceiling_db,
        "gain_reduction_db": float(20 * np.log10(ceiling / peak)),
    }


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
    # BUG FIX (third adversarial audit round, verified directly): lfilter
    # with no explicit initial state (zi) implicitly assumes the signal
    # was SILENT before sample 0 - for a track that needs gain reduction
    # right from the start (e.g. a sustained loud passage beginning at
    # t=0), the filter has to "ramp up" from that false zero-state,
    # producing a real, severe fade-in completely disconnected from the
    # actual overshoot. Confirmed directly: a steady 1kHz tone needing
    # only ~0.55dB of reduction (true_peak_db_before=-0.44 vs ceiling=-1.0)
    # measured a reported gain_reduction_db of -78.9dB and an RMS ratio of
    # 0.405 in the first 50ms - a real, audible fade-in the limiter had no
    # legitimate reason to produce. Seed lfilter's initial state to the
    # gain target AT sample 0 (via scipy's lfilter_zi scaled by that
    # target) instead of the implicit zero-signal assumption, so the
    # filter starts already at the correct steady-state gain when the
    # track opens already loud, and only genuinely ramps for an ACTUAL
    # transition into a loud passage partway through the track.
    zi = signal.lfilter_zi(b_release, a_release) * col[0]
    smoothed, _ = signal.lfilter(b_release, a_release, col, zi=zi)
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
    the file actually needs it.

    BUG FIX: this used to compare the region past cutoff_hz against what
    the PRE-cutoff slope predicts, with no upper bound - but
    spectral_revive's own fill is deliberately capped at
    measured_floor_db + 18dB (a hard ceiling added to fix a real overshoot
    bug), so its fill NEVER reaches what the natural slope would predict,
    by design. That meant a file this app already revived would still
    measure a "deficit" against the slope forever, and get recommended for
    the same fix again on every re-analysis - a real, confirmed bug: the
    tool's own target and this detector's own bar were fundamentally
    incompatible. Fixed by additionally checking whether the region just
    past cutoff_hz already looks like spectral_revive's own signature fill
    (a level near a plateau roughly measured_floor+18dB above the file's own
    established noise floor, rather than a genuinely absent/near-silent
    cliff) - if so, this file has already been revived as much as this
    tool is ever going to revive it, and there is nothing left to fix."""
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

    # Already-revived check - REMOVED (second adversarial audit round,
    # verified directly, three separate attempts each falsified by a real
    # counterexample):
    #   1. A single scalar deficit-magnitude band: falsified by an
    #      untouched hard -18dB step landing inside the "revived" band.
    #   2. A near-vs-far shape/decay-drop check (does the curve keep
    #      decaying past the cutoff): falsified because a genuinely gentle,
    #      real, never-touched gradual rolloff (as mild as 50dB/octave)
    #      ALSO keeps decaying past the cutoff, showing a 5.7dB+ drop -
    #      well inside the same range genuine revival shows (9.3dB+). A
    #      hard step's drop stays under ~0.15dB across a wide sweep, so
    #      "is this NOT a hard step" is reliable, but "is this
    #      specifically spectral_revive's OWN glide, not just any real
    #      gradual rolloff" is not answerable from spectral shape alone -
    #      confirmed directly, a genuinely gentle real rolloff and
    #      spectral_revive's own curve produce nearly IDENTICAL decay
    #      rates (~100-105dB/octave both) just past cutoff.
    #   3. Watermark-payload state tracking (embedding a real "this file
    #      was revived by this app" bit - the only fully reliable option
    #      found): scoped as too costly for this feature given the
    #      payload is already at capacity and extending it would require
    #      re-running the full survival-testing matrix - a deliberate
    #      product decision, not a technical dead end.
    #
    # Net result: there is no known reliable way to distinguish "this
    # app's own recent revival" from "a real, untouched, moderately steep
    # natural rolloff" using spectral shape alone without a real false-
    # positive OR false-negative risk in one direction or the other. Given
    # a false POSITIVE here (wrongly saying "already revived" on a file
    # that still genuinely needs it) is worse than a false NEGATIVE
    # (recommending spectral_revive again on a file this app already
    # fixed - a minor, harmless redundant recommendation, not a quality
    # regression), this check is deliberately left OUT rather than shipped
    # with an exploitable blind spot. spectral_revive itself is
    # unaffected - re-running it on its own already-revived output is not
    # destructive, just unnecessary.

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

    # BUG FIX (found via direct testing, real file): a hard min(raw,
    # ceiling_db) clamp produces a sharp corner exactly where raw first
    # crosses the ceiling, then holds PERFECTLY FLAT for the rest of the
    # extend range - confirmed directly on a real file where the natural
    # slope predicted a level louder than the ceiling at EVERY frequency
    # past the cutoff, so the "curve" was actually a dead-flat line the
    # whole way to Nyquist. Visually and audibly, a flat plateau reads as
    # obviously synthetic - real spectral content never holds one exact
    # level across multiple octaves. Fixed with a smooth exponential glide:
    # right at the cutoff, the curve still carries the natural slope's
    # character; over roughly one octave, it bends smoothly toward the
    # ceiling instead of hitting a hard corner; past that, it continues
    # with a small residual slope (a fraction of the original fitted
    # slope) rather than going perfectly flat, so there's always some real
    # decay character even far above the cutoff.
    GLIDE_OCTAVES = 1.0  # how many octaves past cutoff the blend takes to
    # mostly complete - shorter = snaps to the ceiling faster (more like
    # the old hard clamp), longer = more gradual, more natural-looking.
    RESIDUAL_SLOPE_FRACTION = 0.15  # a small fraction of the original
    # slope keeps applying even once the glide has mostly finished, so
    # the curve never goes perfectly flat far above the cutoff.

    def target_curve_db(f):
        # guard against log2(0) at the DC bin - target_db_at_freqs is only
        # ever read at extend_mask positions (near/above the cutoff, always
        # far from 0), so this value is never actually used, but computing
        # it unguarded still raises a divide-by-zero warning on the full
        # freqs array (which starts at 0 from np.fft.rfftfreq).
        f_safe = np.maximum(f, 1.0)
        raw = slope * np.log2(f_safe) + intercept_anchored

        octaves_past_cutoff = np.maximum(0.0, np.log2(f_safe / cutoff_hz))
        # decay_weight: 1.0 right at the cutoff (full natural-slope
        # character), smoothly -> 0 as octaves_past_cutoff grows past
        # GLIDE_OCTAVES - this is what makes the transition a bend
        # instead of a corner.
        decay_weight = np.exp(-octaves_past_cutoff / GLIDE_OCTAVES)
        blended = ceiling_db + (raw - ceiling_db) * decay_weight

        # residual slope: even once decay_weight has mostly decayed to 0,
        # keep a small fraction of the ORIGINAL fitted slope applying past
        # the cutoff, anchored at the ceiling - this is what keeps the far
        # end of the curve from ever going perfectly dead flat.
        residual = ceiling_db + slope * RESIDUAL_SLOPE_FRACTION * octaves_past_cutoff

        # never predict louder than the raw natural-slope extrapolation
        # (the blend/residual logic above only pulls the curve DOWN toward
        # or below the ceiling, it should never push it up past what the
        # track's own fitted slope would say)
        return np.minimum(np.maximum(blended, residual), raw)

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


def detect_synthesis_artifact(audio, sr, band_lo_hz=6000, band_hi_hz=18000,
                               notch_lo_hz=7500, notch_hi_hz=9500,
                               win=4096, hop=1024):
    """Find short, narrowband high-frequency bursts riding on transients
    (consonants/percussive hits) - the signature of a neural
    vocoder/AI-generation reconstruction artifact, not genuine sibilance or
    cymbal content, both of which are broadband (spectrally flat) in this
    range. Distinguishes the two via spectral flatness (geometric mean /
    arithmetic mean of magnitude): a real fricative or cymbal fills
    band_lo_hz-band_hi_hz roughly evenly (high flatness, close to 1); a
    narrowband ring concentrates energy in a few bins (low flatness).

    Empirically found and localized on a real file this session via direct
    spectral analysis (not guessed): scanning the whole track for short,
    unusually tonal high-frequency bursts found 52 instances, 40% of them
    landing in the same 500Hz sub-band (8.5-9kHz), each 46-280ms long and
    each coincident with a consonant/transient - consistent with a model
    artifact tied to a specific internal frequency resolution boundary,
    not random noise or normal vocal sibilance (which is broadband and
    would NOT cluster this tightly on peak frequency across dozens of
    independent instances).

    notch_lo_hz/notch_hi_hz default to the empirically observed cluster
    range (the full histogram spread found, not just its 8.5-9kHz peak, so
    genuine instances slightly off the median center aren't missed).

    Returns a list of {time_sec, duration_sec, peak_hz, flatness} dicts,
    one per detected burst."""
    mono = audio.mean(axis=1)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    band_mask = (freqs >= band_lo_hz) & (freqs <= band_hi_hz)
    band_freqs = freqs[band_mask]

    n_frames = max(0, (len(mono) - win) // hop)
    if n_frames < 4:
        return []

    window = np.hanning(win)
    mags = np.empty((band_mask.sum(), n_frames), dtype=np.float64)
    for i in range(n_frames):
        seg = mono[i * hop:i * hop + win] * window
        spec = np.abs(np.fft.rfft(seg))
        mags[:, i] = spec[band_mask]

    gmean = np.exp(np.mean(np.log(mags + 1e-12), axis=0))
    amean = np.mean(mags, axis=0)
    flatness = gmean / (amean + 1e-12)
    energy_db = 20 * np.log10(amean + 1e-9)

    finite_energy = energy_db[np.isfinite(energy_db)]
    if finite_energy.size < 4:
        return []
    active = energy_db > np.percentile(finite_energy, 60)
    if not active.any():
        return []
    active_flatness = flatness[active]
    tonal_threshold = np.percentile(active_flatness, 10)
    candidates = active & (flatness < tonal_threshold)

    idx = np.where(candidates)[0]
    if len(idx) == 0:
        return []

    # group adjacent candidate frames into bursts (allow a small gap so one
    # burst isn't split by a single non-qualifying frame in its middle)
    runs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev > 3:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))

    hop_sec = hop / sr
    results = []
    for s, e in runs:
        seg = mags[:, s:e + 1]
        peak_idx = int(np.argmax(seg.mean(axis=1)))
        peak_hz = float(band_freqs[peak_idx])
        if not (notch_lo_hz <= peak_hz <= notch_hi_hz):
            continue
        results.append({
            "time_sec": round(s * hop_sec, 3),
            "duration_sec": round(max(1, e - s + 1) * hop_sec, 3),
            "peak_hz": round(peak_hz, 1),
            "flatness": round(float(flatness[s:e + 1].mean()), 4),
        })
    return results


def fix_synthesis_artifact(audio, sr, artifacts=None, shelf_hz=6000.0,
                            cliff_hz=9800.0, reduction_db=-9.0):
    """Surgical, transient-triggered gentle high-shelf: NOT a static EQ cut
    (would dull real cymbals/sibilance everywhere in the track) and NOT a
    single narrow notch (tried first this session and found wrong - direct
    spectral inspection of multiple detected instances showed this isn't a
    single resonant tone; it's a broader, unnaturally FLAT-TOPPED plateau
    across roughly 6-9.6kHz with a real, consistent 8-12dB drop into a hard
    cliff around 10kHz, confirmed across 8 independently spot-checked
    instances - consistent with a bandwidth-limited AI reconstruction
    ceiling, not a resonance). A notch targeting one frequency inside a
    broad flat plateau does nothing meaningful to the plateau's own
    unnatural flatness/level - confirmed directly: deepening a notch's
    reduction_db from -14 to -40dB made post-fix detection WORSE (53->46
    got worse to 53->53), because a steep notch carves its OWN new sharp
    edge into an already-abnormal region rather than smoothing it.

    The actual problem is the plateau sitting unnaturally LOUD/flat
    relative to the track's own natural high-frequency decay right at
    the cliff - the fix is a gentle, wide shelf that tilts shelf_hz-cliff_hz
    down toward what a natural rolloff into the cliff would look like,
    only during each detected burst's own time window (crossfaded in the
    same way fix_transient/fix_synthesis_artifact's earlier notch version
    used, for the same "no new discontinuity at the correction's own
    edges" reason).

    artifacts: the list returned by detect_synthesis_artifact - if None,
    runs detection internally with default parameters."""
    if artifacts is None:
        artifacts = detect_synthesis_artifact(audio, sr)

    if not artifacts:
        return audio, {"applied": False, "reason": "no synthesis artifacts detected", "instances_found": 0}

    n = len(audio)
    out = audio.copy()
    fixed_instances = []

    taper_sec = 0.015

    # gentle wide shelf: a single 2nd-order Butterworth low-pass at
    # shelf_hz, blended back with the original by `blend` - this tilts
    # energy above shelf_hz down smoothly (not a hard cut), landing the
    # gentlest reduction right at shelf_hz and the full reduction_db by
    # the region approaching cliff_hz, roughly matching how the track's
    # OWN natural rolloff already behaves below the artifact band (the
    # same self-referential principle spectral_revive uses elsewhere in
    # this file, just in reverse - taming an unnatural excess instead of
    # filling a genuine deficit).
    shelf_b, shelf_a = signal.butter(2, shelf_hz / (sr / 2), btype="low")
    blend = 10 ** (reduction_db / 20)

    for a in artifacts:
        start_sample = max(0, int((a["time_sec"] - taper_sec) * sr))
        end_sample = min(n, int((a["time_sec"] + a["duration_sec"] + taper_sec) * sr))
        if end_sample - start_sample < int(0.01 * sr):
            continue

        segment = out[start_sample:end_sample]
        shelved_low = signal.filtfilt(shelf_b, shelf_a, segment, axis=0)
        # shelved_low is everything BELOW shelf_hz (smoothly rolled off
        # above it) - the tilted-down high content is segment - shelved_low,
        # scaled by blend, added back to shelved_low so the correction
        # only ever REDUCES the above-shelf_hz region, never boosts or
        # phase-shifts the below-shelf_hz content it shares with shelved_low.
        above_shelf = segment - shelved_low
        corrected = shelved_low + above_shelf * blend

        taper_n = max(1, int(taper_sec * sr))
        fade = np.ones(len(segment))
        for i in range(min(taper_n, len(fade))):
            frac = i / taper_n
            fade[i] = 0.5 - 0.5 * np.cos(np.pi * frac)
        for i in range(min(taper_n, len(fade))):
            frac = i / taper_n
            fade[len(fade) - 1 - i] = 0.5 - 0.5 * np.cos(np.pi * frac)
        fade = fade[:, None]

        out[start_sample:end_sample] = segment * (1 - fade) + corrected * fade
        fixed_instances.append({
            "time_sec": a["time_sec"],
            "shelf_hz": shelf_hz,
            "reduction_db": reduction_db,
        })

    if not fixed_instances:
        return audio, {"applied": False, "reason": "detected instances too short to correct safely", "instances_found": len(artifacts)}

    return out, {
        "applied": True,
        "instances_found": len(artifacts),
        "instances_fixed": len(fixed_instances),
        "details": fixed_instances,
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
