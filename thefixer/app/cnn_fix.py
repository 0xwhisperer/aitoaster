"""
CNN-model AI-detector fix: whole-track joint gradient optimization with
real-model verification loop, targeting the CQT-cepstrum CNN model.
"""
import numpy as np
import subprocess
import tempfile
import os
import soundfile as sf

from .cnn_differentiable_v2 import SR as CNN_SR, get_real_score_segment
from .cnn_wholetrack_optimizer_v2 import (
    optimize_whole_track_verified,
    optimize_eot_verified,
    _worst_shift_score,
    scan_real_scores,
)


def _transfer_delta_to_stereo(stereo_audio, sr, delta_16k):
    """Transfer a model-rate mono correction onto both native channels."""
    if len(delta_16k) == 0:
        return stereo_audio.copy()
    delta_peak = float(np.abs(delta_16k).max(initial=0))
    if delta_peak < 1e-9:
        return stereo_audio.copy()
    # Normalizing before ffmpeg resampling prevents a very quiet correction
    # from being rounded away.  Undo the normalization after transfer.
    scale = 0.9 / delta_peak
    normalized = delta_16k * scale
    native_normalized = (
        _resample_mono(normalized, CNN_SR, sr)
        if sr != CNN_SR
        else normalized
    )
    native_delta = native_normalized / scale
    output = stereo_audio.copy()
    n_delta = min(len(output), len(native_delta))

    # BUG FIX (audible flutter at track start, measured directly): the
    # optimizers gate the delta against near-silence at the model's 16kHz
    # rate, but that guarantee does not survive the resample above.
    # Polyphase resampling rings across the gate boundary, so a delta that
    # is a HARD ZERO across a silent intro at 16kHz comes back with real
    # energy there at 44.1kHz - measured at ~1.45x the active-region RMS
    # from a literally all-zero input region.
    #
    # On the real "North Star" track (a -68dBFS fade-in opening) this put
    # correction energy at ~1.0x the source level across the first 40ms,
    # against ~-24dB everywhere else: a ~20dB outlier sitting on top of a
    # near-silent intro, which is what made the flutter audible at the
    # beginning specifically. The silence guard's whole purpose is to
    # prevent exactly this, so re-apply it at the NATIVE rate against the
    # NATIVE source, after the resample that breaks it.
    native_mono = stereo_audio[:n_delta].mean(axis=1)
    native_delta = _apply_native_silence_guard(
        native_delta[:n_delta],
        native_mono,
        sr,
    )

    # BUG FIX (the actually-audible flutter, measured on "North Star"): the
    # guard above is BROADBAND, and broadband loudness is the wrong question
    # for this model.  The CNN is a CQT-cepstrum detector over 500Hz-8kHz,
    # and a cepstrum is scale-invariant - it sees spectral SHAPE, not level.
    # On a bass-only intro the track reads as loud (-25dBFS broadband, guard
    # gate 1.000) while 500Hz-8kHz sits at -45 to -74dBFS, so a tiny absolute
    # change swings the cepstral shape enormously.  Measured: the model's own
    # gradient there is 1.58e+03 versus ~2-10 in the body of the track, a
    # grad/source ratio of 2.6e+05 versus ~20-100.
    #
    # Nothing downstream restrained that.  perceptual_penalty's masking
    # multiplier bottoms out at 0.05 for quiet bins, which makes injecting
    # into an EMPTY band ~10x CHEAPER than injecting into a loud one -
    # backwards, since unmasked energy is precisely what the ear picks out.
    # The delivered correction reached -0.4dB relative to the in-band source
    # at t=0.60s, and 23 of the 30 worst blocks across 277s landed in the
    # first 5 seconds: 4.57dB of envelope modulation in 0-5s against
    # 0.11-0.22dB through the rest of the track.
    #
    # Cap the correction against the source's LOCAL IN-BAND level so it stays
    # masked by the music it actually sits under.
    native_delta = _apply_inband_audibility_ceiling(
        native_delta, native_mono, sr
    )
    output[:n_delta, 0] += native_delta
    output[:n_delta, 1] += native_delta
    return output


# The model's own analysis band (models/config.yaml: cqt.fmin=500, 48 bins
# at 12 per octave -> 4 octaves -> 8kHz).  Correction energy is concentrated
# here because this is the only place it can influence the detector.
_BAND_LO_HZ = 500
_BAND_HI_HZ = 8000
# Headroom below the in-band source level.  Chosen by sweeping this value
# against the real "North Star" correction and measuring both flutter (dB
# std of the 1-8kHz envelope ratio over 0-5s) and how much of the detector
# correction survives:
#
#     headroom   flutter 0-5s   correction retained
#      -12dB        2.707dB           99.79%
#      -18dB        1.256dB           99.71%
#      -24dB        0.467dB           96.77%   <- knee
#      -30dB        0.191dB           59.21%
#      -36dB        0.092dB           29.68%
#
# -24dB is the knee: flutter lands near the 0.11-0.22dB baseline measured
# across the untouched body of the track while still keeping ~97% of the
# correction.  Going further buys little audible improvement and starts
# destroying the fix outright.
_INBAND_HEADROOM_DB = -24.0


def _band_block_rms(signal, sr, win, lo_hz=_BAND_LO_HZ, hi_hz=_BAND_HI_HZ):
    """Per-block RMS of `signal` restricted to the model's analysis band."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    spectrum[(freqs < lo_hz) | (freqs >= hi_hz)] = 0
    filtered = np.fft.irfft(spectrum, len(signal))
    n_blocks = (len(filtered) + win - 1) // win
    pad = n_blocks * win - len(filtered)
    if pad:
        filtered = np.pad(filtered, (0, pad))
    blocks = filtered.reshape(n_blocks, win)
    return np.sqrt((blocks ** 2).mean(axis=1) + 1e-20), n_blocks


def _apply_inband_audibility_ceiling(delta, mono_source, sr, win_sec=0.02):
    """Hold the correction a fixed margin below the local in-band source.

    Measures both the source and the correction inside 500Hz-8kHz only, then
    attenuates any block where the correction rises above
    `_INBAND_HEADROOM_DB` relative to the source in that band.  Blocks that
    are already masked are returned untouched, so this costs the detector fix
    nothing wherever the music actually covers it.

    The per-block gain is smoothed into a per-sample envelope (and clamped so
    smoothing can only ever attenuate) because a hard 20ms gain step is
    itself a 50Hz amplitude modulation - the same artifact being removed.
    """
    n = len(delta)
    if n == 0:
        return delta
    win = max(1, int(win_sec * sr))
    source_band, n_blocks = _band_block_rms(mono_source, sr, win)
    delta_band, _ = _band_block_rms(delta, sr, win)

    ceiling = source_band * (10.0 ** (_INBAND_HEADROOM_DB / 20.0))
    gain = np.ones(n_blocks, dtype=np.float64)
    over = delta_band > ceiling
    gain[over] = ceiling[over] / (delta_band[over] + 1e-20)

    # Take a running minimum over each block and its neighbours BEFORE
    # interpolating.  Clamping after interpolation (with np.minimum against
    # each block's own gain) leaves a one-sample cliff at a quiet->loud
    # boundary: the envelope is held down through the quiet block and then
    # released instantly, measured as a 0.031 -> 0.494 jump at exactly the
    # boundary sample.  Shrinking the neighbourhood first means the ramp
    # starts from an already-safe value and rises smoothly, so the envelope
    # is both conservative and continuous.
    padded_gain = np.concatenate(([gain[0]], gain, [gain[-1]]))
    safe_gain = np.minimum.reduce(
        [padded_gain[:-2], padded_gain[1:-1], padded_gain[2:]]
    )
    centers = np.arange(n_blocks) * win + win / 2.0
    envelope = np.interp(np.arange(n), centers, safe_gain)
    return (delta * envelope.astype(delta.dtype)).astype(delta.dtype)


def _apply_native_silence_guard(delta, mono_source, sr, win_sec=0.02):
    """Gate a native-rate delta wherever the native-rate source is silent.

    Mirrors _silence_guard in cnn_gradient_optimizer_v2 (same 20ms blocks,
    same -70dBFS floor / -35dBFS ceiling, same squared falloff) so the
    delivered file honors the identical rule the optimizer certified
    against, rather than a weaker one.  Implemented on numpy here because
    this runs on the native-rate stereo transfer path, outside the torch
    optimization graph.

    The block gain is linearly interpolated across each block instead of
    applied as a hard per-block step: a 20ms staircase on the correction is
    itself a 50Hz amplitude modulation, which is the same class of artifact
    this is meant to remove.
    """
    n = len(delta)
    if n == 0:
        return delta
    win = max(1, int(win_sec * sr))
    n_blocks = (n + win - 1) // win
    pad = n_blocks * win - n

    padded_source = np.pad(mono_source, (0, pad)) if pad else mono_source
    blocks = padded_source.reshape(n_blocks, win)
    block_rms = np.sqrt((blocks ** 2).mean(axis=1) + 1e-12)
    block_db = 20 * np.log10(block_rms + 1e-8)

    floor_db, ceiling_db = -70.0, -35.0
    gate = np.clip((block_db - floor_db) / (ceiling_db - floor_db), 0.0, 1.0)
    gate = gate ** 2

    # Interpolate the per-block gain onto a per-sample envelope, anchored at
    # each block's center, so the gate ramps smoothly instead of stepping.
    #
    # Interpolating alone is not safe at a silence->music boundary: the ramp
    # between a silent block and the loud one after it would re-open the gate
    # over the tail of the silent block, which is exactly where the resampler
    # deposits its ringing. Take the per-sample minimum of the interpolated
    # envelope and each sample's OWN block gain, so smoothing can only ever
    # lower the gain, never raise it above what that block's own loudness
    # permits.
    centers = np.arange(n_blocks) * win + win / 2.0
    envelope = np.interp(np.arange(n), centers, gate)
    own_block_gate = np.repeat(gate, win)[:n]
    envelope = np.minimum(envelope, own_block_gate)
    return (delta * envelope.astype(delta.dtype)).astype(delta.dtype)


def _resample_mono(audio, sr_in, sr_out):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_in:
        in_path = tf_in.name
    out_path = in_path + "_rs.wav"
    try:
        sf.write(in_path, audio, sr_in, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-i", in_path, "-ar", str(sr_out), out_path], check=True)
        data, _ = sf.read(out_path, dtype="float32")
        return data
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def fix_cnn(stereo_audio, sr, max_steps=300, min_steps=100, hop_sec=0.5,
            real_check_interval=10, progress_cb=None, step_progress_cb=None,
            mode="eot", parallel_lr=0.00005):
    """Whole-track CNN fix. stereo_audio: [N,2] float32 at native sr.
    Returns (fixed_stereo, info). progress_cb receives log-line strings;
    step_progress_cb(step, max_steps, max_surrogate_score, real_check_extra)
    receives per-step optimizer detail for UI progress bars, since this
    stage's internal optimization loop can run for many minutes - the last
    two args mirror what already prints to the server log (the live
    surrogate estimate every step, plus the periodic real-model re-check's
    max score and how many windows are still failing) so the browser isn't
    left with only a bare step count while all the detail stays server-side.

    Three modes:
    - "simple": optimizes ONLY the real detector's own 5 fixed evaluation
      positions, at their exact spot, no shift-robustness - what an
      unmodified, off-the-shelf deployment of this detector would check.
      Cheapest, but verified this session to be fragile: gradient descent
      finds a non-robust fix that can fail the instant the delivered file's
      exact byte-for-byte alignment differs even slightly from what was
      optimized.
    - "eot" (recommended): optimizes the same 5 positions, but trains
      against random +-0.5s shifts each step (Expectation-over-
      Transformation) so the fix is robust to exactly the kind of
      positional drift that broke "simple" - see optimize_eot_verified's
      docstring for the full reasoning (this came out of an independent
      architectural review after "thorough" proved unreliable despite
      multiple rounds of tuning). Costs ~5-6x more than "simple" but far
      less than "thorough", while directly targeting the actual failure
      mode instead of brute-forcing coverage around it.
    - "thorough": parallel exact-window optimization over every 0.5-second
      production start plus the deployed detector's fractional starts and
      timing neighborhoods. Safe windows leave the repeated gradient set but
      remain covered by sentinels, mandatory rescans, and the complete native
      delivered-file certificate."""
    mono = stereo_audio.mean(axis=1)
    mono_16k = _resample_mono(mono, sr, CNN_SR) if sr != CNN_SR else mono.copy()

    if progress_cb:
        mode_desc = {
            "simple": "the real detector's own 5 fixed positions only, no shift-robustness",
            "eot": "the real detector's 5 fixed positions, trained for +-0.5s shift-robustness",
            "thorough": "parallel exact windows across the whole track with native-output verification",
        }.get(mode, mode)
        progress_cb(f"cnn: optimizing {len(mono_16k)/CNN_SR:.1f}s of audio ({mode_desc}) - this can take a while")

    try:
        if mode == "thorough":
            from .cnn_adaptive_dense_prototype import (
                required_exact_positions,
            )
            from .cnn_parallel_optimizer import (
                optimize_parallel_active,
            )

            # Native transfer is part of the still-live optimizer session.
            # If this certificate fails at step 50, the same correction,
            # Adam moments, weights, and worker pools continue to step 60+
            # with the failed starts reactivated.  No full-track restart.
            def _delivery_transform(candidate_delta):
                candidate = _transfer_delta_to_stereo(
                    stereo_audio, sr, candidate_delta
                )
                candidate_mono = candidate.mean(axis=1)
                return (
                    _resample_mono(
                        candidate_mono, sr, CNN_SR
                    )
                    if sr != CNN_SR
                    else candidate_mono
                )

            parallel_max_steps = min(max(1, int(max_steps)), 80)
            total_positions = len(
                required_exact_positions(mono_16k)[0]
            )

            def _parallel_progress(
                phase, step, total, estimate, active_count
            ):
                if step_progress_cb is None:
                    return
                real_extra = None
                if phase == "delivery":
                    real_extra = {
                        "real_max_score": estimate,
                        "n_windows_bad": active_count,
                        "n_windows": (
                            total_positions
                        ),
                    }
                step_progress_cb(
                    step, total, estimate, real_extra
                )

            delta_16k, parallel_scan, parallel_timing = (
                optimize_parallel_active(
                    mono_16k,
                    max_steps=parallel_max_steps,
                    min_steps=min(40, parallel_max_steps),
                    real_check_interval=10,
                    full_check_interval=30,
                    lr=parallel_lr,
                    max_repair_rounds=0,
                    delivery_transform=_delivery_transform,
                    delivery_check_steps=(50, 60, 70, 80),
                    progress_cb=_parallel_progress,
                )
            )
            positions = parallel_scan.positions
            total_positions = len(positions)
            seg_len = 10 * CNN_SR
            pre_transfer_worst = parallel_scan.worst_score
        elif mode == "eot":
            delta_16k, positions, seg_len, pre_transfer_worst = optimize_eot_verified(
                mono_16k, max_steps=max_steps, min_steps=min_steps,
                real_check_interval=real_check_interval, verbose=True,
                progress_cb=step_progress_cb,
            )
        else:
            delta_16k, positions, seg_len, pre_transfer_worst = optimize_whole_track_verified(
                mono_16k, max_steps=max_steps, min_steps=min_steps, hop_sec=hop_sec,
                real_check_interval=real_check_interval, verbose=True,
                progress_cb=step_progress_cb, mode=mode,
            )
    except ValueError as e:
        return stereo_audio, {"applied": False, "reason": str(e)}

    if progress_cb:
        if mode == "thorough" and parallel_timing.get(
            "accepted_delivery"
        ):
            progress_cb(
                "cnn: in-session native-output certificate reached "
                f"{pre_transfer_worst * 100:.2f}% worst-window AI"
            )
        else:
            progress_cb(f"cnn: pre-transfer certified worst score (post-silence-guard): {pre_transfer_worst * 100:.2f}% AI")

    # tracks shorter than the real detector's segment length get zero-padded
    # internally before optimization (see optimize_whole_track_verified) so
    # the optimizer analyzes the same padded representation the real
    # detector scores - truncate the delta back down to the real audio's
    # length here, since the padded tail doesn't exist in the delivered file.
    if len(delta_16k) > len(mono_16k):
        delta_16k = delta_16k[:len(mono_16k)]

    delta_peak = np.abs(delta_16k).max()
    if delta_peak < 1e-9:
        return stereo_audio, {"applied": False, "reason": "optimizer found zero-magnitude delta"}

    out = _transfer_delta_to_stereo(stereo_audio, sr, delta_16k)
    n = len(out)

    # emergency anti-clipping safety net ONLY (not the app's real loudness
    # ceiling) - see linear_fix.py's identical clamp for why this alone does
    # not guarantee -1dBTP compliance without the true-peak limiter also
    # running as the final chain stage.
    peak = np.abs(out).max()
    if peak > 0.97:
        out *= 0.97 / peak

    orig_rms = np.sqrt(np.mean(stereo_audio[:n, 0] ** 2))
    delta_rms = np.sqrt(np.mean((out[:, 0] - stereo_audio[:n, 0]) ** 2))
    snr_db = 20 * np.log10(orig_rms / (delta_rms + 1e-12))

    # final verification AFTER the resample-to-native-rate transfer: the
    # transfer is a second place (distinct from optimization itself) where a
    # verified-good delta at 16kHz can still degrade once resampled back up
    # and mixed into the real stereo file, so re-check every window for real
    # on what the detector would actually see if it analyzed the delivered
    # file (mono-downmixed back to the model's own 16kHz), not on the pre-
    # transfer 16kHz signal alone.
    #
    # BUG FIX (adversarial review, verified directly): this used to check
    # only the EXACT stored positions, even for EOT mode - but EOT's whole
    # guarantee is about a +-eot_jitter_sec NEIGHBORHOOD around each
    # position, not the exact point. The resample/mix transfer is exactly
    # the kind of small positional perturbation EOT was built to be robust
    # to, so checking only the exact position after transfer silently
    # dropped back to the same false-convergence pattern (pass at one exact
    # point, fail nearby) that motivated building EOT in the first place.
    # For EOT mode, re-run the same worst-shift scan on the transferred
    # output instead of a single-point check.
    if progress_cb:
        progress_cb("cnn: verifying final transferred stereo output against the real model")
    out_mono_native = out.mean(axis=1)
    out_mono_16k = _resample_mono(out_mono_native, sr, CNN_SR) if sr != CNN_SR else out_mono_native
    n_16k = len(out_mono_16k)

    if mode == "thorough":
        from .cnn_adaptive_dense_prototype import required_exact_positions
        from .cnn_real_scanner import ParallelRealScoreScanner

        post_positions, post_seg_len = required_exact_positions(
            out_mono_16k
        )
        with ParallelRealScoreScanner() as scanner:
            post_transfer_scores = scanner.scan(
                out_mono_16k, post_positions, post_seg_len
            )
        positions = post_positions
        seg_len = post_seg_len
    elif mode == "eot":
        # BUG FIX (adversarial audit, verified directly): _worst_shift_score
        # now returns None (not 0.0) when a position has no valid in-bounds
        # window at all - filter those out explicitly here, the same way
        # the "else" (thorough/simple) branch below already filters
        # too-short segments, rather than letting a None silently coerce
        # into "0.0 = perfectly clean" via max().
        post_transfer_scores = [
            s for s in (
                _worst_shift_score(out_mono_16k, pos, seg_len, n_16k,
                                    shift_range_sec=0.5, shift_step_sec=0.1, sr=CNN_SR)
                for pos in positions
            )
            if s is not None
        ]
    else:
        valid_positions = [
            pos for pos in positions
            if len(out_mono_16k[pos:pos + seg_len]) >= seg_len
        ]
        post_transfer_scores = scan_real_scores(
            out_mono_16k, valid_positions, seg_len
        )
    worst_after_transfer = max(post_transfer_scores) if post_transfer_scores else None

    # this was being computed and silently dropped into the returned info
    # dict with no corresponding progress_cb call - meaning the live log
    # would show "Double-checking the CNN fix..." immediately followed by
    # "Done (Xs)" with NO indication of what that check actually found.
    # Report it explicitly, mirroring the equivalent post-check log lines
    # already emitted for the linear model (see linear_fix.py's "attempt N
    # result checked against the REAL detector" line).
    if progress_cb:
        if worst_after_transfer is not None:
            # BUG FIX: .2f formatting on a genuinely very small score (e.g.
            # 0.00003) rounds down to a confusing "0.00%" that reads as
            # broken/impossible next to a non-pass badge, rather than
            # showing the real, meaningful precision that explains it.
            # Use more decimal places specifically when the value is small
            # enough that 2 decimals would hide it.
            _pct = worst_after_transfer * 100
            _precision = 2 if _pct >= 0.01 else 5
            progress_cb(f"cnn: worst window after transfer scored {_pct:.{_precision}f}% AI")
        else:
            progress_cb("cnn: no windows survived the post-transfer check (all fell outside the delivered audio's length)")

    # 0.08 matches the optimizer's own real_target - a genuine safety
    # margin below the model's raw 0.5 pass/fail boundary, not just
    # barely crossing it (see cnn_wholetrack_optimizer_v2.py's real_target
    # comment for why: a thin margin gets erased by any later stage).
    verified = bool(worst_after_transfer is not None and worst_after_transfer < 0.08)

    # BUG FIX (adversarial review, verified directly): "applied" used to be
    # hardcoded True regardless of whether verified_after_transfer was True
    # or False - meaning a delta that failed its own final certification
    # was still reported as a successful fix. applied now means "a
    # correction was computed and is being returned," which is always true
    # at this point in the function (the zero-magnitude-delta case already
    # returned early above); verified_after_transfer is the separate,
    # authoritative signal for whether that correction actually passed.
    # Callers (server.py) already branch on verified_after_transfer
    # specifically for the retry/re-run decision - this only fixes what the
    # generic "applied" flag claims, it does not change existing retry logic.
    return out, {
        "applied": True,
        "mode": mode,
        "snr_db": float(snr_db),
        "n_windows": len(positions),
        "window_positions_sec": [p / CNN_SR for p in positions],
        "pre_transfer_worst_score": float(pre_transfer_worst),
        "worst_score_after_transfer": float(worst_after_transfer) if worst_after_transfer is not None else None,
        "verified_after_transfer": verified,
        **(
            {"parallel_timing": parallel_timing}
            if mode == "thorough"
            else {}
        ),
    }
