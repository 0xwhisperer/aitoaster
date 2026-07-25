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
            mode="eot"):
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
    - "thorough": the original dense-whole-track approach (hundreds of
      overlapping windows). Expensive (can take 10+ minutes) and, per the
      same review, treats a symptom rather than the cause - kept available
      for comparison/fallback, not the recommended default going forward."""
    mono = stereo_audio.mean(axis=1)
    mono_16k = _resample_mono(mono, sr, CNN_SR) if sr != CNN_SR else mono.copy()

    if progress_cb:
        mode_desc = {
            "simple": "the real detector's own 5 fixed positions only, no shift-robustness",
            "eot": "the real detector's 5 fixed positions, trained for +-0.5s shift-robustness",
            "thorough": "dense overlapping windows across the whole track",
        }.get(mode, mode)
        progress_cb(f"cnn: optimizing {len(mono_16k)/CNN_SR:.1f}s of audio ({mode_desc}) - this can take a while")

    try:
        if mode == "eot":
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

    scale = 0.9 / delta_peak
    delta_norm_16k = delta_16k * scale
    delta_native_norm = _resample_mono(delta_norm_16k, CNN_SR, sr) if sr != CNN_SR else delta_norm_16k
    delta_native = delta_native_norm / scale

    # apply the correction only where a delta exists (resampling can leave
    # delta_native a few samples shorter/longer than stereo_audio) but carry
    # the FULL original track through unchanged elsewhere - never truncate
    # the delivered audio down to however much the delta happened to cover.
    n_delta = min(len(stereo_audio), len(delta_native))
    out = stereo_audio.copy()
    out[:n_delta, 0] += delta_native[:n_delta]
    out[:n_delta, 1] += delta_native[:n_delta]
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

    if mode == "eot":
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
    }
