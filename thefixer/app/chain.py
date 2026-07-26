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
from scipy import signal, ndimage
import pyloudnorm as pyln

# Limiter attack and lookahead. The attack is how gradually gain reduction is
# allowed to arrive; the lookahead must be at least as long, or the envelope
# cannot physically be in place by the time the peak lands.
#
# Measured on a pure 1kHz tone with one swell over the ceiling (added harmonic
# distortion, lower is better):
#
#     no lookahead at all                  69.9 dB
#     1.5ms lookahead, no envelope smooth  51.6 dB
#     lookahead + smoothed envelope        46.9 dB   <- current
#
# That is a 23dB reduction in limiter-induced distortion. The remainder is
# inherent: any gain envelope multiplied against a tone produces sidebands,
# and sweeping the attack from 1.5ms to 12ms moves the result only between
# 44.4 and 47.7dB, so the envelope's SHAPE - not its speed - sets the floor.
# The 4x resample round-trip contributes nothing measurable (+0.0dB).
#
# 1.5ms sits in the 1-5ms range professional limiters use: gradual enough to
# cut the sideband distortion, short enough not to audibly duck the material
# before each peak (a long attack on a dense mix reads as pumping).
LIMITER_ATTACK_MS = 1.5
LIMITER_LOOKAHEAD_MS = 1.5


FADE_MIN_MS = 10
FADE_MAX_MS = 10000


def apply_fade(audio, sr, fade_in_ms=10, fade_out_ms=3000):
    """Apply a fade-in and/or fade-out to the track's head and tail.

    Durations are in milliseconds and clamped to [10, 10000] - the range the
    UI sliders expose. Passing 0 (or a negative) for either disables that
    side, so a fade-out-only or fade-in-only run is possible.

    Uses a raised-cosine (sine-squared) S-curve.

    NAMING NOTE: this was originally labelled "equal-power", which is wrong.
    Equal-power is a CROSSFADE term - it describes a sin/cos pair whose
    squares sum to 1 so the summed power stays constant through a transition
    between two sources. A single fade to silence has no such constraint,
    and there is no single industry-standard curve for one; DAWs offer a
    choice (linear, exponential, logarithmic, S-curve).

    Measured at the fade midpoint:
        linear       -> -6.0 dB
        sine         -> -3.0 dB   (the actual "equal-power" curve)
        sine-squared -> -6.0 dB   (this one)

    So this matches linear at the halfway point but, unlike linear, has zero
    slope at BOTH ends - no corner at the start or the end of the fade. That
    is the usual reason to prefer an S-curve for a fade-out.

    Handles the degenerate cases the sliders make reachable: a fade longer
    than the track itself, and a fade-in plus fade-out that would otherwise
    overlap. Both are bounded so the two curves meet at most once and never
    multiply into a doubled gain or read past the buffer.
    """
    n = len(audio)
    if n == 0:
        return audio, {"applied": False, "reason": "empty audio"}

    requested_in = int(fade_in_ms or 0)
    requested_out = int(fade_out_ms or 0)
    in_ms = 0 if requested_in <= 0 else int(np.clip(requested_in, FADE_MIN_MS, FADE_MAX_MS))
    out_ms = 0 if requested_out <= 0 else int(np.clip(requested_out, FADE_MIN_MS, FADE_MAX_MS))

    if in_ms <= 0 and out_ms <= 0:
        return audio, {"applied": False, "reason": "both fade durations are zero"}

    in_samples = min(int(in_ms / 1000.0 * sr), n)
    out_samples = min(int(out_ms / 1000.0 * sr), n)

    # A 10s fade on a shorter track, or a long fade at both ends, would make
    # the two curves overlap and multiply - audibly a hole in the middle.
    # Scale both down proportionally so they meet at most at a single point.
    total = in_samples + out_samples
    if total > n and total > 0:
        scale = n / total
        in_samples = int(in_samples * scale)
        out_samples = int(out_samples * scale)

    envelope = np.ones(n, dtype=np.float64)
    if in_samples > 1:
        # sin^2(0..pi/2): 0 -> 1, S-curve (zero slope at both ends)
        envelope[:in_samples] = np.sin(
            np.linspace(0.0, np.pi / 2, in_samples)
        ) ** 2
    elif in_samples == 1:
        envelope[0] = 0.0
    if out_samples > 1:
        envelope[n - out_samples:] = np.cos(
            np.linspace(0.0, np.pi / 2, out_samples)
        ) ** 2
    elif out_samples == 1:
        envelope[-1] = 0.0

    if audio.ndim > 1:
        out = (audio * envelope[:, None]).astype(audio.dtype)
    else:
        out = (audio * envelope).astype(audio.dtype)

    return out, {
        "applied": True,
        "fade_in_ms": in_ms,
        "fade_out_ms": out_ms,
        "fade_in_samples": int(in_samples),
        "fade_out_samples": int(out_samples),
        "curve": "raised-cosine S-curve (sine squared)",
    }


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

    # BUG FIX (external chain audit, reproduced): this returned the FILTERED
    # audio unconditionally while reporting applied=False when there was no
    # meaningful rumble. So on a clean track it changed 100% of samples - at
    # 31.0dB SNR, attenuating 30Hz by 6.02dB and 40Hz by 0.83dB - and told
    # the user it had done nothing. Both the log line and the results table
    # were wrong, and "doesn't touch anything audible" was false.
    #
    # applied=False must mean the audio is untouched. Return the input.
    if not meaningful:
        return audio, {"applied": False, "cutoff_hz": cutoff_hz,
                       "order": order,
                       "reason": "no meaningful sub-cutoff content"}
    return out, {"applied": True, "cutoff_hz": cutoff_hz, "order": order}


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


# Below this, stereo information is collapsed to mono. 120Hz is the standard
# pop/club choice: the ear localises very little down here (wavelengths are
# long relative to head spacing), the low end carries most of a mix's energy,
# and out-of-phase bass is what actually cancels on mono playback and causes
# cutting problems on vinyl and club systems.
BASS_MONO_HZ = 120.0
# Phase correction is confined below this. Mono compatibility is a low-end
# problem in practice; correcting the whole spectrum needlessly narrows an
# image that was fine.
PHASE_BAND_HZ = 300.0


def _band_split(audio, sr, cutoff_hz, order=4):
    """Split into (below, above) complementary parts that sum back exactly.

    The high band is derived by SUBTRACTION rather than a second filter, which
    is what makes reconstruction exact by construction: low + high == input,
    always, for any lowpass. (Not Linkwitz-Riley - see
    split_bands_complementary for why that label was wrong here.)
    """
    nyq = sr / 2.0
    cutoff = min(max(cutoff_hz, 1.0), nyq - 1.0)
    sos = signal.butter(order, cutoff / nyq, btype="lowpass", output="sos")
    low = np.stack(
        [signal.sosfiltfilt(sos, audio[:, ch]) for ch in range(audio.shape[1])],
        axis=1,
    )
    return low, audio - low


def _correlation_of(audio):
    left, right = audio[:, 0], audio[:, 1]
    if left.std() < 1e-9 or right.std() < 1e-9:
        return 1.0
    value = float(np.corrcoef(left, right)[0, 1])
    return 1.0 if not np.isfinite(value) else value


def stereo_field_correct(audio, sr, bass_mono_hz=BASS_MONO_HZ,
                          phase_band_hz=PHASE_BAND_HZ):
    """Bass-mono and low-band phase repair - both automatic.

    Replaces fix_phase_issues' whole-track approach, which measured ONE
    correlation figure across the entire file and applied ONE mid/side blend
    to all of it. Real phase problems are frequency-dependent and usually
    confined to the low end, so a whole-track average simultaneously
    under-corrects a genuine bass issue and narrows an image that was fine.

    Two stages, in this order:

    1. BASS-MONO below bass_mono_hz. Sums the low band to mono. The ear
       localises very little down there, that band carries most of the
       energy, and out-of-phase bass is precisely what cancels when a track
       is summed for mono playback - or what makes a cut unplayable on
       vinyl. This alone removes the failure mode the old tool existed for.

    2. PHASE correction between bass_mono_hz and phase_band_hz, if that band
       is still negatively correlated after step 1. Mid/side blending, the
       same mechanism as before but confined to where it belongs.

    This function deliberately does NOT widen. An earlier draft had a third
    "gentle width" stage and this docstring still described it long after the
    stage was removed. Widening is a separate concern with its own mono-
    compatibility risk, and it must not be entangled with a function whose
    job is to make the low end MORE mono.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio, {"applied": False, "reason": "not stereo"}

    audio = np.asarray(audio, dtype=np.float32)
    correlation_before = _correlation_of(audio)

    # A genuinely mono source must stay mono - never fabricate width.
    if np.allclose(audio[:, 0], audio[:, 1], atol=1e-7):
        return audio, {
            "applied": False,
            "reason": "source is mono",
            "bass_mono_hz": bass_mono_hz,
            "correlation_before": correlation_before,
            "correlation_after": correlation_before,
        }

    low, high = _band_split(audio, sr, bass_mono_hz)

    # 1. bass to mono.
    #
    # NOT a plain (L+R)/2. When the low band is anti-phase - which is exactly
    # the case this stage exists for - the average is ZERO, so summing would
    # "fix" the cancellation by deleting the bass outright. Verified: on a
    # perfectly out-of-phase 60Hz test signal the mono sum retained 20% of the
    # original bass energy rather than the full amount.
    #
    # Instead, when the two sides oppose each other, flip one before summing
    # so their energy ADDS. Below 120Hz the ear cannot localise the result, so
    # inverting one side is inaudible - while the energy it preserves is the
    # whole point of the stage.
    # The inversion is CROSSFADED and gated, never a hard branch on the sign.
    # Branching on `low_corr < 0` fires identically at -0.001 and at -1.0, and
    # near zero the two branches produce essentially uncorrelated outputs
    # (measured r = 0.02) - so bass polarity would be decided by estimator
    # noise, and two renders of near-identical material could disagree. There
    # is also nothing to rescue at -0.1: (L-R)/2 there discards the correlated
    # bass note and keeps only the decorrelated room, which is the opposite of
    # preserving energy.
    #
    # So: do nothing until the bass is DECISIVELY anti-phase, then ramp the
    # inversion in smoothly between -0.2 and -0.5.
    low_corr = _correlation_of(low)
    invert_weight = float(np.clip((-low_corr - 0.2) / 0.3, 0.0, 1.0))
    if invert_weight > 0.0:
        right_effective = low[:, 1] * (1.0 - 2.0 * invert_weight)
        bass_mono = (low[:, 0] + right_effective) / 2.0
    else:
        bass_mono = low.mean(axis=1)
    low_out = np.stack([bass_mono, bass_mono], axis=1)

    # 2. phase repair in the band just above it
    phase_corrected = False
    if phase_band_hz > bass_mono_hz:
        # A steeper split here than the bass crossover uses. With a 4th-order
        # split the 300Hz skirt is shallow enough that anti-phase content from
        # ABOVE the band bleeds down into it: measured, zeroing the side
        # entirely still left the 130-290Hz band at -0.917 correlation,
        # because most of what remained was leakage rather than band content.
        # At 8th order the same test lands at +0.666. The bass crossover keeps
        # its gentler slope - it is summing to mono, not isolating a band.
        band, rest = _band_split(high, sr, phase_band_hz, order=8)
        band_corr = _correlation_of(band)
        if band_corr < 0.0:
            # blend mid back in until the band is no longer anti-correlated;
            # mid/side is the right tool for broadband content (see
            # fix_phase_issues' own history for why all-pass is not).
            mid = band.mean(axis=1)
            side = (band[:, 0] - band[:, 1]) / 2.0
            # scale side down enough to bring correlation non-negative
            keep = float(np.clip(1.0 + band_corr, 0.0, 1.0))
            band = np.stack([mid + side * keep, mid - side * keep], axis=1)
            phase_corrected = True
        high = band + rest

    # NO WIDTH STAGE HERE, deliberately.
    #
    # An earlier version widened the image above the bass crossover toward a
    # target correlation. It was removed: this tool is a mono-SAFETY
    # correction, and widening is the one operation in it that made the
    # measured track less mono-compatible (0.905 -> 0.832 correlation on a
    # real master, from a 1.457x side boost that the copy called "gentle").
    # It was also backwards - a near-mono mix got the largest boost and barely
    # moved, while an already-healthy mix at 0.6 was pushed to 0.473.
    #
    # Width is an image/taste decision, not a safety correction, and it does
    # not belong entangled with bass-mono. If it returns it should be its own
    # stage, positioned after multiband (which changes the L/R balance any
    # width decision would be based on), solving for the boost that actually
    # reaches a target rather than applying an open-loop gain, and guarding
    # the OUTCOME rather than just capping the gain.
    out = (low_out + high).astype(np.float32)

    # never let the recombination clip - the band operations can sum to a
    # larger peak than the input even when each part is smaller
    peak = float(np.abs(out).max())
    peak_rescale = 1.0
    if peak > 0.999:
        peak_rescale = 0.999 / peak
        out = (out * peak_rescale).astype(np.float32)

    return out, {
        "applied": True,
        "bass_mono_hz": bass_mono_hz,
        "phase_band_hz": phase_band_hz,
        "phase_corrected": phase_corrected,
        "bass_polarity_inverted": round(invert_weight, 3),
        "peak_rescaled": round(peak_rescale, 5),
        "correlation_before": correlation_before,
        "correlation_after": _correlation_of(out),
    }


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
        # SAME bands the compressor uses - these two must never disagree, or
        # the detector reports a band the compressor cannot act on.
        bands = default_bands(sr)
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


def multiband_compress(audio, sr, bands=None, ratio=1.3, threshold_db=-12.0,
                        max_passes=1, target_over_db=0.5):
    """Gentle 4-band tonal-balance smoothing. ONE pass by default.

    CRITICAL FIX (adversarial mastering audit): this briefly defaulted to
    iterating up to 12 passes, to stop the UI telling users to re-run the
    file by hand. That was the wrong answer and it silently turned a gentle
    tool into a crusher.

    The gain law is `gain_db = -over * (1 - 1/ratio)` applied to the ALREADY
    COMPRESSED output each pass, so the over-threshold excess decays as
    (1/ratio)^passes and the EFFECTIVE ratio compounds:

        1 pass   -> 1.30:1        6 passes  ->  4.83:1
        3 passes -> 2.20:1        9 passes  -> 10.60:1   <- typical run
                                 12 passes  -> 23.30:1

    Nine passes at a -12dB threshold is hard multiband limiting on the BODY
    of a pop master, not peak control - while the log still called it "up to
    1.3dB gentle reduction". Worse, the exit condition was an absolute
    measure of the FILE, so a dense, commercially-loud mix (which legitimately
    sits well over -12dB in the 200-2000Hz band) always drove it to the
    ceiling: the more normal the source, the harder it was crushed.

    _multiband_compress_pass now has real per-band attack and release (see
    _envelope_follower), so it is a compressor rather than spectral gain-
    riding. That makes the gentleness still matter: one pass, a couple of dB.

    max_passes stays a parameter (a caller can still ask for more) but the
    DEFAULT is 1. Target <=2dB total gain reduction for pop.
    """
    # `bands` stays None here on purpose: both the per-pass compressor and
    # detect_band_peakiness derive the same Nyquist-aware default from sr,
    # so passing None keeps those two in agreement instead of freezing a
    # copy of the default at this level.
    current = audio
    passes = 0
    last_info = None
    pass_infos = []
    previous_worst = None

    for _ in range(int(max_passes)):
        peakiness = detect_band_peakiness(current, sr, bands=bands,
                                           threshold_db=threshold_db)
        worst = max((b["peak_over_db"] for b in peakiness), default=0.0)
        if worst <= target_over_db:
            break
        # stop if the previous pass bought essentially nothing - a real file
        # can sit slightly over target in a band the compressor cannot reach
        # (e.g. content right at a band edge), and spinning to max_passes on
        # it would just add filter round-trips for no measurable gain.
        if previous_worst is not None and previous_worst - worst < 0.05:
            break
        previous_worst = worst

        stepped, info = _multiband_compress_pass(
            current, sr, bands=bands, ratio=ratio, threshold_db=threshold_db
        )
        if not info["applied"]:
            break
        current = stepped
        last_info = info
        pass_infos.append(info)
        passes += 1

    if passes == 0:
        return audio, {
            "applied": False, "ratio": ratio, "threshold_db": threshold_db,
            "passes": 0,
            "bands": (last_info or _multiband_compress_pass(
                audio, sr, bands=bands, ratio=ratio, threshold_db=threshold_db
            )[1])["bands"],
        }

    final_peakiness = detect_band_peakiness(current, sr, bands=bands,
                                             threshold_db=threshold_db)
    return current, {
        "applied": True,
        "ratio": ratio,
        "threshold_db": threshold_db,
        "passes": passes,
        # Report the CUMULATIVE reduction, not the last pass's. Reporting
        # last_info alone understated a 9-pass run as "up to 0.4dB" when the
        # worst single pass was 1.3dB and the total far more, which read as
        # the tool having barely done anything.
        "bands": [
            {**band,
             "max_reduction_db": min(
                 (info["bands"][i]["max_reduction_db"] for info in pass_infos),
                 default=band["max_reduction_db"]),
             "max_gain_db": max(
                 (info["bands"][i]["max_gain_db"] for info in pass_infos),
                 default=band["max_gain_db"])}
            for i, band in enumerate(last_info["bands"])
        ],
        "worst_over_db_after": float(
            max((b["peak_over_db"] for b in final_peakiness), default=0.0)
        ),
    }


# 4-band split points chosen for pop. The old 200/2000 split put vocal presence
# (2-5kHz, the range that makes a lead vocal intelligible), snare crack and
# guitar bite in the SAME band as cymbals and air - so a loud cymbal ducked
# the vocal along with it. 100/800/5000 gives presence its own band.
DEFAULT_CROSSOVERS_HZ = (100.0, 800.0, 5000.0)

# Per-band time constants. Low frequencies need slow attack (a 60Hz cycle is
# 16ms long - reacting faster than that tracks the WAVEFORM rather than the
# envelope, which is distortion) and slow release. Highs can be much quicker.
BAND_ATTACK_MS = (30.0, 15.0, 8.0, 3.0)
BAND_RELEASE_MS = (200.0, 150.0, 100.0, 60.0)


def default_bands(sr, crossovers=DEFAULT_CROSSOVERS_HZ):
    """Band edges as (lo, hi) pairs, top edge derived from the real Nyquist."""
    nyq = sr / 2.0
    edges = [0.0] + [float(c) for c in crossovers] + [nyq]
    return tuple((edges[i], edges[i + 1]) for i in range(len(edges) - 1))


def split_bands_complementary(audio, sr, bands=None):
    """Split into complementary bands that sum back to the input exactly.

    Each successive band is peeled off with a zero-phase Butterworth lowpass
    and the remainder carried forward by SUBTRACTION. The subtraction is what
    makes reconstruction exact - it holds for ANY lowpass, so the bands always
    sum to the original signal (measured error 3e-08, i.e. float32 epsilon).
    That replaces independent Butterworth BANDPASSES, which do not sum flat.

    NOT Linkwitz-Riley, despite an earlier version of this docstring saying
    so. Measured, this is a steep zero-phase split (measured ~-30dB/octave across
    the crossover itself, steepening further out until it meets the numerical
    floor),
    and at a nominal 100Hz crossover the two bands read -11.9dB and -2.5dB.
    LR's defining property is -6.02/-6.02 summing to unity, so the label was
    simply wrong. Being zero-phase there is also no phase response for an LR
    design to align. The flat summing here comes from the subtraction, not
    from any LR property.

    One consequence worth knowing: because the -6dB point sits well below the
    nominal frequency (measured 80.3Hz for a nominal 100Hz), the `range_hz` values
    reported downstream are nominal split points, not -6dB crossover points.
    """
    if bands is None:
        bands = default_bands(sr)
    nyq = sr / 2.0
    remaining = np.asarray(audio, dtype=np.float32)
    out = []
    for lo, hi in bands[:-1]:
        cutoff = min(max(float(hi), 1.0), nyq - 1.0)
        sos = signal.butter(2, cutoff / nyq, btype="lowpass", output="sos")
        low = remaining
        for _ in range(2):
            low = np.stack(
                [signal.sosfiltfilt(sos, low[:, ch]) for ch in range(low.shape[1])],
                axis=1,
            )
        out.append(low.astype(np.float32))
        remaining = (remaining - low).astype(np.float32)
    out.append(remaining)
    return out


def _follow_scalar(level, attack_coeff, release_coeff):
    """The two-branch recurrence, written out literally.

        rising : env = env + (1 - a_att) * (x - env)
        falling: env = max(x, env * a_rel)

    JIT-compiled when numba is importable (measured 0.025s per band on a 150s
    track), otherwise this same loop runs in plain Python (1.13s per band,
    46x slower - measured, not estimated). Slow but
    correct; correctness is the constraint here, since three successive
    attempts to vectorise this were all wrong in a different way.
    """
    n = len(level)
    out = np.empty(n, dtype=np.float64)
    prev = level[0]
    for i in range(n):
        x = level[i]
        if x > prev:
            prev = prev + (1.0 - attack_coeff) * (x - prev)
        else:
            decayed = prev * release_coeff
            prev = x if x > decayed else decayed
        out[i] = prev
    return out


try:  # optional acceleration; the pure-Python fallback is identical in output
    from numba import njit as _njit
    from numba.core import errors as _numba_errors
    _follow_fast = _njit(_follow_scalar)
    # Narrow: only numba's own compilation failures fall back. A numerical
    # bug or MemoryError inside the compiled kernel must still raise.
    _NUMBA_ERRORS = (_numba_errors.NumbaError, TypeError)
except Exception:  # pragma: no cover - numba absent
    _follow_fast = _follow_scalar
    _NUMBA_ERRORS = ()


_FOLLOW_FELL_BACK = False


def _follow(level, attack_coeff, release_coeff):
    """Run the JIT path, falling back to plain Python if it cannot compile.

    numba's njit is LAZY - it compiles on first CALL, not at decoration. So the
    try/except around the import above catches only a MISSING numba, never a
    compilation failure. Without this guard a compile error would propagate
    out of a render and 500 the request.

    The fallback is the same recurrence in plain Python: bit-identical output,
    measured 46x slower (1.13s vs 0.025s per band on a 150s track). An earlier
    version of this function called ITSELF here instead of _follow_fast, so
    numba never ran at all: ~987 frames of recursion, a RecursionError
    swallowed by a bare `except Exception`, and the slow path every time. The
    output was bit-identical, so the entire test suite passed while the
    compressor ran 46x slower than intended - which is why
    test_jit_path_is_actually_exercised watches the CALL, not the output.

    It is caught NARROWLY - only numba's own compilation
    errors - so a genuine numerical bug or MemoryError in the compiled kernel
    still raises instead of being silently absorbed into a slow path. The
    fallback also announces itself once, because a silent 40x regression in
    production with no signal is its own kind of bug.
    """
    global _FOLLOW_FELL_BACK
    if _follow_fast is _follow_scalar:
        return _follow_scalar(level, attack_coeff, release_coeff)
    try:
        return _follow_fast(level, attack_coeff, release_coeff)
    except _NUMBA_ERRORS:
        if not _FOLLOW_FELL_BACK:
            _FOLLOW_FELL_BACK = True
            print("[chain] numba could not compile the envelope follower; "
                  "using the plain-Python fallback (same output, ~40x slower)")
        return _follow_scalar(level, attack_coeff, release_coeff)


def _envelope_follower(level, sr, attack_ms, release_ms):
    """Asymmetric peak-follower: ramps up over the attack, decays over the
    release. The standard compressor envelope.

        rising : env[n] = env[n-1] + (1 - a_att) * (x[n] - env[n-1])
        falling: env[n] = max(x[n], env[n-1] * a_rel)

    This is computed by the literal recurrence above, NOT by composing filters.
    Three vectorised attempts were each blocked by audit, and every one failed
    in the same place - the moment after a transient ends:

      1. A one-pole fed with a max-dilated peak. On a short transient the pole
         only charged for the attack window, so once the peak left the window
         the envelope collapsed 22.1dB in TWO samples - a step discontinuity
         in the gain, audible as a click on every percussive hit.

      2. Adding an attack ramp on top of that dilation. Dilation raises the
         TARGET, not the STATE, so the two composed into lag: peak gain
         reduction landed +40ms AFTER a kick, delivering 9.5% of the intended
         reduction while the transient was present and the rest onto whatever
         followed.

      3. Taking min(ramp, decaying-max). The pole was still driven by the
         decayed target rather than the signal, so the envelope RISED 3.03dB
         over 21.9ms while the input was exactly zero, peaking 22ms into
         silence. It also broke the env >= |x| invariant, letting a spike pass
         25.7dB above the envelope with no gain reduction at all.

    The recurrence has none of those failure modes because the envelope only
    ever moves in response to the current sample: it cannot rise while the
    input falls, and it cannot lag behind a peak it has already seen.

    One consequence worth stating plainly: during the attack ramp the envelope
    is BELOW the input by design, so a very short spike gets little or no gain
    reduction - a single-sample full-scale spike sees essentially none, with
    the envelope sitting 33-53dB under it. That is inherent to having an
    attack time at all, and it is why the recombination clamp in
    _multiband_compress_pass and the downstream true-peak limiter both matter.
    """
    level = np.asarray(level, dtype=np.float64)
    if len(level) == 0:
        return level

    attack_samples = max(1.0, attack_ms * 0.001 * sr)
    release_samples = max(1.0, release_ms * 0.001 * sr)
    # 3 time constants inside the window -> ~95% arrival by its end, so the
    # ramp completes within the stated attack time rather than reaching only
    # 63% of the way there.
    attack_coeff = float(np.exp(-3.0 / attack_samples))
    release_coeff = float(np.exp(-1.0 / release_samples))
    return _follow(level, attack_coeff, release_coeff)


def _multiband_compress_pass(audio, sr, bands=None,
                              ratio=1.3, threshold_db=-12.0):
    """ONE pass of gentle 4-band downward compression -
    reduces peaky dynamic imbalance between low/mid/high without touching
    overall spectral tilt aggressively. Conservative defaults (low ratio,
    higher threshold) by design: this should only be shaping the loudest
    peaks in each band, not continuously riding gain on the whole track -
    least change necessary to smooth genuine imbalance."""
    nyq = sr / 2
    if bands is None:
        bands = default_bands(sr)
    split = split_bands_complementary(audio, sr, bands)
    out = np.zeros_like(audio)
    info_bands = []
    for index, ((lo, hi), band_audio) in enumerate(zip(bands, split)):
        # Envelope with REAL attack and release, per band. The previous
        # version computed gain per-sample from a 20ms median with no time
        # constants at all - that is spectral gain-riding, not compression:
        # it cannot let a transient through, so drums lose their punch.
        # Low bands get slow constants (a 60Hz cycle is 16ms long; reacting
        # faster tracks the waveform rather than the envelope), highs fast.
        attack_ms = BAND_ATTACK_MS[min(index, len(BAND_ATTACK_MS) - 1)]
        release_ms = BAND_RELEASE_MS[min(index, len(BAND_RELEASE_MS) - 1)]
        level = np.abs(band_audio).max(axis=1)
        env = _envelope_follower(level, sr, attack_ms, release_ms)
        env_db = 20 * np.log10(np.maximum(env, 1e-8))

        over = np.maximum(env_db - threshold_db, 0)
        gain_db = -over * (1 - 1 / ratio)
        gain = 10 ** (gain_db / 20)
        out += band_audio * gain[:, None]
        info_bands.append({
            "range_hz": [round(lo), round(min(hi, nyq - 1))],
            "max_reduction_db": float(gain_db.min()),
            # A compressor may only ever REDUCE. This is the most positive
            # gain the band saw, and it must never exceed 0dB. Exposed
            # because two generations of tests tried to catch an expander
            # bug by measuring a band's RMS and both were blind to it: the
            # boost multiplies a near-silent band, so its absolute level
            # barely moves while the gain itself reaches +22dB.
            "max_gain_db": float(gain_db.max()),
            "attack_ms": attack_ms,
            "release_ms": release_ms,
        })

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


# ------------------------------------------------------------------ saturation
# tanh soft saturation with level-independent drive, 4x oversampling and a DC
# guard. Every number here was measured; see saturate() for the derivations.
SATURATION_DRIVES = {"light": 0.9, "medium": 1.6, "strong": 3.0}
SATURATION_OPERATING_RMS = 0.125      # -18 dBFS, the classic 0VU reference
SATURATION_OVERSAMPLE = 4
SATURATION_MIN_DURATION_SEC = 2.0
SATURATION_MIN_RMS = 1e-3             # -60 dBFS program RMS
SATURATION_MAX_MAKEUP_DB = 3.0


def _program_rms(audio, sr, percentile=95.0, block_sec=0.4):
    """95th-percentile block RMS - the level the drive is normalised against.

    A plain whole-file RMS is dragged down by intros, outros and quiet
    verses, so the loud sections would saturate harder than the quiet ones on
    the same setting. The 95th percentile of short blocks tracks what the
    track actually sits at when it is playing.
    """
    # Per-channel MAX, not the mono sum. A perfectly out-of-phase track
    # (L = -R) sums to exactly zero, so a mono-sum estimator reported 0.0 and
    # the stage skipped it claiming "program level below -60dBFS" - which was
    # false, each channel measured 0.20 RMS. Anti-phase material is rare but
    # a wrong reason in the log is worse than a right refusal.
    if audio.ndim > 1:
        channels = [audio[:, c] for c in range(audio.shape[1])]
    else:
        channels = [audio]

    block = max(1, int(block_sec * sr))
    best = 0.0
    for chan in channels:
        n = (len(chan) // block) * block
        if n < block:
            best = max(best, float(np.sqrt((chan.astype(np.float64) ** 2).mean())))
            continue
        blocks = chan[:n].astype(np.float64).reshape(-1, block)
        rms = np.sqrt((blocks ** 2).mean(axis=1))
        rms = rms[rms > 0]
        if len(rms):
            best = max(best, float(np.percentile(rms, percentile)))
    return best


def saturate(audio, sr, amount="medium", oversample=SATURATION_OVERSAMPLE):
    """Gentle tanh saturation: odd harmonics, level-independent, LUFS-neutral.

    Chosen over the alternatives on measurement. At a matched 1.00% THD on a
    1kHz tone: tanh gives a fast-decaying odd series (H3 -40.0, H5 -80.0,
    H7 -117.8dB) which is the console/transformer character pop bus glue
    wants. Asymmetric "tube" saturation measured lower IMD (3.48% vs 6.09%)
    but puts H2 an octave above the fundamental, where pop vocal and snare
    content already lives, and carries -43.3 dBFS of DC. Wavefolding is
    disqualified outright: 482% IMD and a harmonic series that does not decay
    (H3 -44.7 through H9 -47.3) - a synthesis effect, not a mastering one.

    THREE THINGS THIS GETS RIGHT that a naive implementation does not:

    1. LEVEL-INDEPENDENT DRIVE. The signal arrives mid-chain at whatever
       level it happens to be, and a fixed drive therefore distorts a loud
       track far harder than a quiet one. Measured on real material, a naive
       fixed drive varied the distortion residual by 240x across an 18dB
       input swing (0.00048 -> 0.11517 RMS). This normalises to a fixed
       internal operating point first, saturates, then scales back - measured
       constant THD to three decimal places across the same 18dB range.

    2. 4x OVERSAMPLING, not optional. A nonlinearity generates harmonics
       above Nyquist which fold back as inharmonic content, and it lands
       BELOW 8kHz - inside both detectors' analysis band, where nothing
       downstream can remove it. Measured two-tone alias products below 8kHz
       -30 to -50dB at 1x depending on drive, and -78 to -97dB at 4x, with
       no further improvement at 8x or 16x. 4x is the knee. (At the default
       drive specifically: -40.3dB at 1x, -87.6dB at 4x.) It costs about 1.5s
       on a 150s track, under 2% of pipeline runtime.

    3. A DC GUARD, FOR THE SYMMETRIC CURVE TOO. tanh is an odd function, so
       the intuition is that it cannot rectify a DC term - but real program
       material is itself asymmetric, so it does. Measured on a real track,
       unguarded tanh at drive 1.6 produced 1.12e-03 of DC: 19x over this
       app's own lossy re-check floor (6e-5) and 112x over the lossless one
       (1e-5). Since dc_offset runs at step 3 and this runs at step 8,
       nothing downstream would catch it and the delivered file would
       re-recommend dc_offset on re-upload. Per-channel mean subtraction
       lands at 3.6e-07.

    Output level is RMS-matched to input. That makes the stage nearly
    loudness-neutral, but NOT exactly: RMS matching is not LUFS matching,
    because LUFS is K-weighted and gated. Measured integrated loudness change
    is at most +0.07 LU at light, +0.14 LU at medium and +0.35 LU at strong
    (worst case across two real tracks). Small enough that normalize_lufs downstream reclaims almost
    none of it - unlike a broad EQ move, of which only 28-65% survives - but
    an earlier version of this docstring claimed "within 0.002 LU", which was
    wrong by 50-180x.

    What this does NOT do: it is not an EQ. Broadband tonal balance moves by
    at most 0.10dB at light, 0.21dB at medium and 0.56dB at strong (largest
    octave-band change, worst case across two real tracks). The audible effect is
    peak-density reduction - short-term crest falls 0.46-0.77dB at drive 1.6 -
    not tonal colour. It is not "warmth" in the tonal sense.
    """
    from scipy import signal as _sig

    drive = SATURATION_DRIVES.get(amount) if isinstance(amount, str) else amount
    if drive is None:
        return audio, {"applied": False, "reason": f"unknown amount {amount!r}"}

    if not np.isfinite(audio).all():
        # A single NaN sample became 176,400 NaN samples: resample_poly
        # spreads it across its filter length and the DC guard's mean then
        # spreads it across the whole channel. Refuse rather than silently
        # destroy the track and report "pass".
        return audio, {"applied": False, "amount": amount,
                       "reason": "input contains non-finite samples"}

    duration = len(audio) / float(sr)
    if duration < SATURATION_MIN_DURATION_SEC:
        # Below this the percentile RMS estimator is unreliable: measured on
        # a 0.05s file the normalisation factor blew up to 112,948, and a
        # 0.2s file overshot the THD target by 2.6x.
        return audio, {"applied": False, "amount": amount,
                       "reason": f"track is {duration:.2f}s; needs "
                                 f"{SATURATION_MIN_DURATION_SEC:.0f}s"}

    level = _program_rms(audio, sr)
    if level < SATURATION_MIN_RMS:
        # Near-silence measured 12.3% THD - the estimator divides by a level
        # that is essentially noise.
        return audio, {"applied": False, "amount": amount,
                       "reason": "program level below -60dBFS"}

    audio = np.asarray(audio, dtype=np.float32)
    norm = SATURATION_OPERATING_RMS / level

    up = _sig.resample_poly(audio * norm, oversample, 1, axis=0)
    # Divide by `drive`, NOT by tanh(drive). tanh(x*d)/d has unity slope at
    # the origin, so quiet passages pass through at their own level and only
    # the loud parts compress - which is what a saturator is. Normalising by
    # tanh(d) instead makes the curve hit +/-1 at full scale, which BOOSTS
    # overall level (measured +1.88dB at drive 0.9, +4.47dB at 1.6, +8.59dB
    # at 3.0) and left the makeup-gain clamp pinned at its -3dB limit in
    # every case - a clamp that always binds is a bug, not a safety net.
    sat = np.tanh(up * drive) / drive
    out = _sig.resample_poly(sat, 1, oversample, axis=0).astype(np.float64)
    # Defensive only, and knowingly untestable: for every integer oversample
    # factor (2,3,4,5,7,8,16) and every length checked, resample_poly up then
    # down returns EXACTLY the input length, so this branch is unreachable and
    # no test can observe its removal. Kept because relying on that as a
    # documented guarantee of scipy's would be unwise, not because it fires.
    if out.shape[0] != audio.shape[0]:
        if out.shape[0] > audio.shape[0]:
            out = out[:audio.shape[0]]
        else:
            pad = [(0, audio.shape[0] - out.shape[0])] + \
                  [(0, 0)] * (out.ndim - 1)
            out = np.pad(out, pad)

    # DC guard - see point 3 above. Before auto-gain, so the makeup gain is
    # measured on the signal that will actually be delivered.
    dc_before = float(np.abs(np.atleast_1d(out.mean(axis=0))).max())
    out = out - out.mean(axis=0, keepdims=True)

    out /= norm
    # RMS auto-gain, clamped: a runaway estimator must not silently become a
    # volume change.
    out_level = _program_rms(out.astype(np.float32), sr)
    makeup_db = 0.0
    if out_level > 0:
        makeup = level / out_level
        makeup_db = float(np.clip(20 * np.log10(makeup),
                                  -SATURATION_MAX_MAKEUP_DB,
                                  SATURATION_MAX_MAKEUP_DB))
        out *= 10 ** (makeup_db / 20)

    out = out.astype(np.float32)
    return out, {
        "applied": True,
        "amount": amount,
        "drive": drive,
        "oversample": oversample,
        "makeup_db": round(makeup_db, 3),
        "dc_removed": dc_before,
        "peak_before": float(np.abs(audio).max()),
        "peak_after": float(np.abs(out).max()),
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
    # LOOKAHEAD (adversarial mastering audit). Without it the gain ramp began
    # AT the peak, so the transient's leading edge passed through unreduced
    # and the envelope had to jump underneath it - and that jump is itself
    # distortion. Measured on a pure 1kHz tone with one smooth swell over the
    # ceiling (a signal with no harmonics of its own, so anything harmonic in
    # the output is the limiter's): the limiter added 69.9dB of harmonic
    # distortion, taking a -121dB-clean signal to -51dB.
    #
    # Every professional limiter looks ahead 1-5ms for exactly this reason.
    # Implemented as a running MINIMUM over the lookahead window: each instant
    # adopts the smallest gain required at any point within the next
    # LOOKAHEAD_MS, so the envelope is already at the correct value when the
    # peak arrives and never has to step. minimum_filter1d is O(n) and runs on
    # the oversampled grid without a Python loop.
    #
    # The window is centred (origin shifted) so the reduction leads the peak
    # rather than lagging it, which also means NO net delay is introduced -
    # the output stays sample-aligned with the input and needs no compensating
    # trim. That alignment is load-bearing: this limiter runs after the
    # detector fixes, which are position-sensitive.
    lookahead_samples = max(1, int(LIMITER_LOOKAHEAD_MS * 0.001 * sr_up))
    if lookahead_samples > 1:
        # shift the window so it looks FORWARD from each sample
        col = ndimage.minimum_filter1d(
            col, size=lookahead_samples,
            origin=-(lookahead_samples // 2), mode="nearest",
        )
    zi = signal.lfilter_zi(b_release, a_release) * col[0]
    smoothed, _ = signal.lfilter(b_release, a_release, col, zi=zi)
    # smoothing can only RAISE gain relative to the instant requirement
    # (never lower it enough) since it's a low-pass toward the target - clip
    # back down to the strict per-instant ceiling wherever that happens. With
    # lookahead in place `col` is already the forward-looking minimum, so this
    # clip now rarely bites; it remains as the hard guarantee that the ceiling
    # is never exceeded.
    smoothed = np.minimum(smoothed, col)
    # The clip above leaves CORNERS in the gain envelope wherever it bites,
    # and a corner in a multiplied envelope is a discontinuity in the first
    # derivative - which is exactly what shows up as harmonic distortion.
    # Lookahead alone took the added distortion from 69.9dB to 51.6dB;
    # rounding those corners with a short Hann smoothing of the envelope
    # removes the rest. The window is tied to the lookahead so the envelope
    # can still track a genuine transient, and the result is clamped back
    # under `col` so smoothing can never raise gain above what the ceiling
    # allows - the hard guarantee survives.
    # Smooth over the ATTACK time, not just the lookahead window. Measured on
    # the pure-tone test: lookahead alone 51.6dB, smoothing over the 1.5ms
    # lookahead 46.9dB, while an ideal envelope that reduces gently across the
    # whole transient adds only -0.3dB. The gap was the envelope still moving
    # far faster than it needs to. Smoothing across the attack time lets the
    # reduction arrive gradually - which is what the lookahead bought us the
    # room to do - while the clamp below keeps the ceiling guaranteed.
    attack_samples = max(3, int(LIMITER_ATTACK_MS * 0.001 * sr_up))
    if attack_samples > 2:
        kernel = np.hanning(attack_samples)
        kernel /= kernel.sum()
        smoothed = np.convolve(smoothed, kernel, mode="same")
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


# --------------------------------------------------------------- tonal cleanup
# (centre_hz, label, trigger_db). The trigger is PER BAND because the noise
# floor of the p10 statistic differs by band: the analysis window holds far
# more cycles at 3150Hz than at 250Hz, so the frame-to-frame spread is much
# smaller up there. Measured across 5 noise seeds with NO resonance present:
#     250Hz   p10 floor -3.43 to -3.83     a real +7dB resonance reads -0.88
#    3150Hz   p10 floor -0.61 to -0.77     a real +7dB resonance reads +2.03
# Each trigger sits ~1.3dB above its own floor, which is roughly halfway to
# the resonance case. A single global trigger cut pure noise by 1.33dB in the
# harshness band while missing real resonances in the boxiness band.
TONAL_REGIONS = (
    (250.0, "boxiness", -2.0),
    (3150.0, "harshness", 0.6),
)
TONAL_Q = 1.4
# Triggers are per-band, in TONAL_REGIONS above. They are NEGATIVE at 250Hz
# because p10 is the quietest decile, not the average - even a genuine
# resonance dips below its own local trend in some frames.
TONAL_PERSISTENCE_PCT = 10.0  # percentile of per-frame excess: the "even at its
                              # quietest" floor. A resonance rings in every frame;
                              # a note only while it plays.
TONAL_MAX_CUT_DB = 1.5
TONAL_MIN_DURATION_SEC = 30.0
# How lopsided the shoulders on either side of a region may be before it is
# treated as a filter skirt rather than a resonance. Measured on the 3150Hz
# band: pure noise and a real resonance both sit well under 6dB of shoulder
# tilt, while a 4th-order lowpass at 2kHz produces far more.
TONAL_MAX_SHOULDER_TILT_DB = 12.0
# How many frames must SURVIVE the skirt veto before the p10 statistic is
# trustworthy. Measured: a real resonance keeps 100% of its frames, a fully
# vetoed skirt keeps 0% - but the awkward cases in between keep 19% (a
# 2nd-order lowpass) and 65% (an 8th-order highpass), and on that biased
# remnant the p10 rises above the trigger and draws a cut on material with
# no resonance at all. Below this fraction there is no reliable sample left
# to judge, so the region is skipped rather than guessed at.
TONAL_MIN_USABLE_FRAMES = 0.85
TONAL_FRAME_SEC = 0.20


def _peaking_sos(freq_hz, q, gain_db, sr):
    """RBJ peaking-EQ biquad. Minimum-phase, causal, no pre-ringing.

    Verified by an earlier audit as textbook-correct: measured gain matches
    the request to under 0.001dB at every frequency/Q/gain, measured Q is
    1.401 against 1.4 requested, and all poles and zeros sit strictly inside
    the unit circle from 8k to 192kHz.
    """
    a_ = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b = np.array([1 + alpha * a_, -2 * cos_w0, 1 - alpha * a_])
    a = np.array([1 + alpha / a_, -2 * cos_w0, 1 - alpha / a_])
    return (b / a[0]).tolist() + (a / a[0]).tolist()


def _frame_excess_db(audio, sr, freq_hz, q, frame_sec=TONAL_FRAME_SEC):
    """Per-frame excess of a narrow region over its own local spectral trend.

    THIS IS THE WHOLE DESIGN. A previous version measured the excess ONCE,
    over the whole track, and it did not work - it fired on sustained musical
    notes and ignored the resonances it was built to find. Measured on that
    version: a bass note 26dB BELOW the noise bed drew the full cut (+3.08dB
    excess) while a genuine +7dB Q=1.4 resonance drew nothing (+2.80dB), and
    a plain 4th-order lowpass with no resonance at all drew the full cut
    (+7.18dB).

    The reason is structural: any excess-over-neighbours statistic on a
    full-track average is really a narrowband-energy detector, and in music
    the dominant narrowband energy IS NOTES. No threshold change fixes that,
    because the two are indistinguishable in a time-averaged spectrum.

    The discriminator has to be TEMPORAL. A resonance is excited by whatever
    content passes through it, so it is present in nearly every frame that
    has energy at all. A note is present only while it is played - typically
    a few percent to a third of a track, and intermittently. So: measure the
    excess frame by frame, and later require it in a high FRACTION of active
    frames (see tonal_cleanup). A tonic note occupies too few frames to
    qualify no matter how loud it is.

    Returns (excess_per_frame, frame_is_active).
    """
    from scipy import signal as _sig

    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    nper = int(frame_sec * sr)
    if nper < 256 or len(mono) < nper * 4:
        return None, None
    hop = nper // 2
    n_frames = 1 + (len(mono) - nper) // hop
    if n_frames < 8:
        return None, None

    half_bw = freq_hz / (2.0 * q)
    lo_edge, hi_edge = freq_hz - half_bw, freq_hz + half_bw
    # probe bands a quarter-octave clear of the region, both sides
    probes = []
    for mult in (2 ** 0.25, 2 ** 0.5, 2 ** 0.75):
        for centre in (lo_edge / mult, hi_edge * mult):
            if 0 < centre < sr / 2:
                probes.append(centre)
    if len(probes) < 3:
        return None, None

    window = np.hanning(nper)
    freqs = np.fft.rfftfreq(nper, 1 / sr)
    region_sel = (freqs >= lo_edge) & (freqs < min(hi_edge, sr / 2))
    probe_sels = [((freqs >= c / 1.06) & (freqs < min(c * 1.06, sr / 2)), np.log2(c))
                  for c in probes]
    probe_sels = [(s, lf) for s, lf in probe_sels if s.any()]
    if not region_sel.any() or len(probe_sels) < 3:
        return None, None

    idx = np.arange(n_frames) * hop
    frames = np.lib.stride_tricks.sliding_window_view(mono, nper)[idx] * window
    psd = np.abs(np.fft.rfft(frames, axis=1)) ** 2

    total = psd.mean(axis=1)
    active = total > (np.percentile(total, 95) * 1e-3)

    here = 10 * np.log10(psd[:, region_sel].mean(axis=1) + 1e-20)
    x = np.array([lf for _, lf in probe_sels])
    y = np.stack([10 * np.log10(psd[:, s].mean(axis=1) + 1e-20)
                  for s, _ in probe_sels], axis=1)
    # quadratic baseline per frame: stiff enough not to bend around a Q=1.4
    # bell, flexible enough to follow real spectral tilt and its curvature
    coeffs = np.polyfit(x, y.T, 2)
    baseline = np.polyval(coeffs, np.log2(freq_hz))

    # MONOTONICITY VETO. A quadratic cannot follow a filter knee: on an
    # 8th-order lowpass at 2kHz the fit undershoots so badly that the 3150Hz
    # region reads p10 = 16.73dB - HIGHER than a genuine +7dB resonance
    # (2.03dB). No threshold on the excess alone can separate them, because
    # the skirt scores higher than the thing being looked for.
    #
    # But a skirt is MONOTONIC: the spectrum falls steadily across the whole
    # probe span. A resonance is a local peak - the probes below it rise
    # toward it and the probes above it fall away. So compare the mean of the
    # lower probes against the mean of the upper probes: if the region sits on
    # a steep one-way slope rather than on a bump, veto it.
    n_side = len(probe_sels) // 2
    lo_probes = np.array([v for _, v in probe_sels[:n_side]])
    hi_probes = np.array([v for _, v in probe_sels[n_side:]])
    if len(lo_probes) and len(hi_probes):
        lo_mean = y[:, :n_side].mean(axis=1)
        hi_mean = y[:, n_side:].mean(axis=1)
        # a genuine resonance has roughly balanced shoulders; a skirt has one
        # shoulder far above the other
        tilt = np.abs(lo_mean - hi_mean)
        steep = tilt > TONAL_MAX_SHOULDER_TILT_DB
        # EXCLUDE the skirt-like frames from the statistic rather than
        # scoring them hugely negative. An earlier version set them to -99
        # and then took the p10 across ALL frames, so on real music - where
        # 15.5% of frames legitimately look skirt-like - those -99s dragged
        # the percentile to -99 and vetoed the whole band. The right
        # treatment is "this frame carries no information about a
        # resonance", i.e. drop it, and judge on the frames that remain.
        return here - baseline, active & ~steep
    return here - baseline, active


def tonal_cleanup(audio, sr, regions=TONAL_REGIONS, q=TONAL_Q,
                  max_cut_db=TONAL_MAX_CUT_DB,
                  min_duration_sec=TONAL_MIN_DURATION_SEC):
    """Cut-only correction of two problem regions, gated on PERSISTENCE.

    Corrects boxiness (250Hz) and harshness (3.15kHz) only where the excess
    over the local spectral trend is present in nearly EVERY frame - measured
    as the 10th percentile of the per-frame excess. That is the property that
    separates a resonance from music, and it took four failed detectors to
    find it:

      1. Whole-track average excess. Fired on sustained notes and missed real
         resonances - on a full-track average a narrowband detector cannot
         tell a room mode from a bass line, because both are narrowband
         energy.
      2. Per-frame OCCUPANCY above a threshold. Better, but the frame-to-
         frame measurement noise on plain pink noise has a standard deviation
         of 3.39dB - as large as the resonance being looked for - so a real
         +7dB resonance and a sustained note both scored ~0.57 occupancy and
         were indistinguishable.
      3/4. Sideband mean and log-linear interpolation for the baseline: both
         invented resonances on steep filter skirts (a 2kHz lowpass with no
         resonance at all read +7.18dB and drew a full cut).

    The 10th percentile works because of what each thing IS in time:

        signal                    median    p10     IQR
        pink noise, no resonance    0.51   -3.83   4.75
        REAL +7dB resonance         3.42   -0.88   4.73
        intermittent bass note      1.68   -3.34   6.95
        4th-order lowpass skirt     0.49   -3.83   4.72

    A resonance rings whenever ANY content excites it, so even its quietest
    frames sit above the local trend - p10 is high. A note is loud while
    played and gone otherwise, so its p90 is high but its p10 collapses and
    its spread is wide. A filter skirt is not a local peak at all, so the
    quadratic baseline follows it and everything reads near zero.

    Cut-only, and only the excess beyond the trigger, capped. There are no
    boosts: every headroom and homogenisation problem measured during this
    design came from boosting.
    """
    from scipy import signal as _sig

    duration = len(audio) / float(sr)
    if duration < min_duration_sec:
        return audio, {"applied": False, "bands": [],
                       "reason": f"track is {duration:.0f}s; needs "
                                 f"{min_duration_sec:.0f}s"}

    bands, sos_list = [], []
    for freq_hz, label, trigger_db in regions:
        if freq_hz >= sr / 2:
            continue
        excess, active = _frame_excess_db(audio, sr, freq_hz, q)
        if excess is None or active is None or not active.any():
            continue
        usable = float(active.mean())
        if usable < TONAL_MIN_USABLE_FRAMES:
            # Too much of this region looks like a filter skirt for the
            # remaining frames to be a fair sample of it.
            bands.append({
                "freq_hz": freq_hz, "label": label, "trigger_db": trigger_db,
                "usable_frames": round(usable, 3),
                "persistent_db": None, "spread_db": None, "median_db": None,
                "cut_db": 0.0,
                "skipped": "region reads as a filter slope, not a resonance",
            })
            continue
        ea = excess[active]
        # the floor of the distribution: what this region is over the trend
        # even in its quietest frames
        persistent = float(np.percentile(ea, TONAL_PERSISTENCE_PCT))
        spread = float(np.percentile(ea, 75) - np.percentile(ea, 25))
        cut_db = 0.0
        if persistent > trigger_db:
            cut_db = -min(persistent - trigger_db, max_cut_db)
            # a computed cut of 0.00dB is not a correction - do not build a
            # filter for it, and do not report the stage as applied
            if cut_db <= -0.01:
                sos_list.append(_peaking_sos(freq_hz, q, cut_db, sr))
            else:
                cut_db = 0.0
        bands.append({
            "freq_hz": freq_hz, "label": label,
            "trigger_db": trigger_db,
            "usable_frames": round(usable, 3),
            "persistent_db": round(persistent, 2),
            "spread_db": round(spread, 2),
            "median_db": round(float(np.median(ea)), 2),
            "cut_db": round(cut_db, 2),
        })

    if not sos_list:
        return audio, {"applied": False, "bands": bands,
                       "reason": "no region rings persistently enough"}
    out = _sig.sosfilt(np.asarray(sos_list, dtype=np.float64),
                       audio, axis=0).astype(np.float32)
    return out, {"applied": True, "bands": bands, "q": q,
                 "max_cut_db": max_cut_db}


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


def _read_id3_frame_tags(path):
    """Enumerate every raw ID3v2 (and legacy ID3v1) frame via mutagen.

    ffprobe's -show_format/-show_streams JSON only maps a handful of
    well-known frames (TIT2->title, TPE1->artist, COMM->comment, etc) into
    its tags dict - it silently drops frame types it doesn't have a mapping
    for, including WXXX/WOAS/WOAR (URL frames), TXXX (arbitrary
    user-defined text - where AI platforms often stuff generation IDs,
    prompts, model names), PRIV (private binary frames), UFID (unique file
    identifiers), and USLT (lyrics). Confirmed on a real Suno export: a
    WOAS frame carrying a direct link to the source song page
    (https://suno.com/song/<uuid>) was completely invisible to ffprobe's
    JSON output while mutagen read it straight off the frame table.

    Returns a flat {frame_id_or_desc: str(value)} dict, or {} if the file
    has no ID3 tag or isn't ID3-taggable (e.g. FLAC/Vorbis comments, which
    ffprobe already maps completely). Frames ffprobe already surfaces under
    a friendlier key (TIT2/TPE1/COMM -> title/artist/comment) are skipped
    here to avoid reporting the same value twice under two different keys."""
    try:
        from mutagen.id3 import ID3
    except ImportError:
        return {}
    try:
        tags = ID3(str(path))
    except Exception:
        return {}
    skip_prefixes = ("APIC", "TIT2", "TPE1", "COMM")
    frames = {}
    for key, frame in tags.items():
        if key.startswith(skip_prefixes):
            continue
        frames[key] = str(frame)
    return frames


def read_metadata_tags(path):
    """Read every container/ID3 metadata tag from the source file - both
    format-level (title/artist/comment/encoder/etc.) AND per-stream tags,
    plus a report of any non-audio streams (embedded cover art, attached
    images) since those can themselves carry their own metadata (e.g. EXIF
    in a JPEG) and most users don't expect a cover-art image riding along
    inside an audio file at all. Also merges in raw ID3v2 frames ffprobe's
    JSON output doesn't map (see _read_id3_frame_tags) so e.g. a WOAS
    source-URL frame is surfaced even though ffprobe drops it silently.

    Many AI-generation platforms embed identifying tags (comment fields
    naming the platform, generation UUIDs, timestamps, platform-style
    artist handles, source-page URLs) directly in the uploaded file's
    metadata, independent of anything detectable in the audio itself. This
    never modifies the file - it's a read-only report used by /api/analyze
    so a user can see everything the original upload was carrying.

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

    format_tags = dict(data.get("format", {}).get("tags", {}) or {})
    format_tags.update(_read_id3_frame_tags(path))
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
         # +bitexact also suppresses ffmpeg's own "Lavf<version>" encoder
         # tag, which -map_metadata -1 does not remove (see _STRIP_ARGS in
         # server.py for the per-format verification).
         "-fflags", "+bitexact", "-flags:a", "+bitexact",
         "-c", "copy", str(out_path)],
        check=True,
    )
