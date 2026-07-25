"""
Linear-model AI-detector fix: gradient-based adversarial correction targeting
the fakeprint logistic-regression model specifically. Wraps the verified
linear_gradient_optimizer for use as one stage of the full mastering chain.
"""
import numpy as np
import torch
import subprocess
import tempfile
import os
import soundfile as sf

from .linear_differentiable import SAMPLE_RATE as LIN_SR, MAX_DURATION, get_real_score
from .linear_gradient_optimizer import optimize as _optimize_linear


def _resample_mono(audio_44k_mono, sr_in, sr_out):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_in:
        in_path = tf_in.name
    out_path = in_path + "_rs.wav"
    try:
        sf.write(in_path, audio_44k_mono, sr_in, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-i", in_path, "-ar", str(sr_out), out_path], check=True)
        data, _ = sf.read(out_path, dtype="float32")
        return data
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def _score_stereo_array(stereo_audio, sr):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        path = tf.name
    try:
        sf.write(path, np.clip(stereo_audio, -1, 1), sr, subtype="PCM_16")
        return get_real_score(path)
    finally:
        os.unlink(path)


ACCEPT_THRESHOLD = 0.01  # the user's actual bar is <1% AI, not just "under 50%"
REAL_TARGET_FLOOR = 0.00001
SURROGATE_TARGET_FLOOR = 0.0001


def _tighten_retry_targets(real_target, surrogate_target):
    """Tighten both retry targets monotonically, using role-specific floors.

    The real-model gate can use a smaller floor because it is only an
    acceptance measurement. The surrogate target shapes every optimization
    step; forcing its logit toward still smaller values after 1e-4 adds
    optimization pressure without demonstrated transfer benefit.
    """
    return (
        min(real_target, max(REAL_TARGET_FLOOR, real_target * 0.3)),
        min(surrogate_target, max(SURROGATE_TARGET_FLOOR, surrogate_target * 0.3)),
    )


def fix_linear(stereo_audio, sr, target=0.005, real_target=0.008, max_steps=225,
                max_retries=3, progress_cb=None, step_progress_cb=None,
                prefer_feature_solver=True, feature_min_snr_db=35.0,
                feature_max_gain_db=0.75):
    """Apply the gradient-based linear-model fix. stereo_audio: [N,2] float32
    at native sr (typically 44100). Returns (fixed_stereo, info).

    max_steps was 400; lowered to 225 (independently reviewed and measured:
    ~794ms/step on a 300s track, so 400 steps was ~318s/attempt even when
    convergence happened much earlier). Safe to lower because
    linear_gradient_optimizer.optimize() has its own adaptive safety net
    independent of this starting value: it breaks out early via real-model
    verification once genuinely converged (as early as step 150), forces a
    real check at whatever step the budget actually ends on (not just at
    real_check_interval multiples - an earlier version could miss this,
    verified: with max_steps=225 and real_check_interval=50 the last check
    landed at step 200, and the never-satisfied `step+1 >= max_steps`
    condition meant the loop silently ran out with no further verification
    - fixed by forcing a check at the true final step and driving extension
    off a separate fixed absolute_max_steps instead of a mutated max_steps),
    and extends itself past the cap (up to 4x the original) if the real
    model still disagrees when the budget ends - so lowering the starting
    cap only cuts wasted steps on runs that would have converged early
    anyway; harder files still get as many steps as they actually need via
    the extension path.

    Verifies the REAL detector score on the final transferred (resampled,
    stereo) output - not just the surrogate's score during optimization -
    since the resample-to-native-rate transfer is itself a second place
    where a surrogate-verified delta can still fail to hold up against the
    real model. If the final check fails, retries with a stricter
    real_target rather than silently shipping an unverified result. The
    acceptance bar is <1% (ACCEPT_THRESHOLD), not merely "under the model's
    50% decision boundary" - a result like 3% would flip the label to "Real"
    but is nowhere near the near-zero scores achieved on other tracks."""
    mono = stereo_audio.mean(axis=1)
    mono_model_sr = _resample_mono(mono, sr, LIN_SR) if sr != LIN_SR else mono.copy()

    max_samples = MAX_DURATION * LIN_SR
    truncated = len(mono_model_sr) > max_samples
    analysis = mono_model_sr[:max_samples] if truncated else mono_model_sr

    if progress_cb and truncated:
        progress_cb(f"linear: model only ever scores the first {MAX_DURATION}s of a track "
                    f"(matches the detector's own limit) - only that portion will be corrected")

    # Fast first-stage solve. The detector reduces the complete analysis prefix
    # to one mean log spectrum before applying its 3,585-feature classifier, so
    # optimize that sufficient statistic directly and reconstruct only once.
    # The existing waveform optimizer below remains the correctness fallback:
    # the feature candidate must survive the exact same native-rate stereo
    # transfer check, plus conservative SNR and spectral-gain guards, before it
    # can return.
    if prefer_feature_solver:
        try:
            from .linear_feature_optimizer import optimize_feature_eq

            if progress_cb:
                progress_cb(
                    "linear: trying fast feature-domain solve before the "
                    "waveform optimizer"
                )
            feature_result = optimize_feature_eq(analysis, target_score=0.00005)
            feature_delta = feature_result.audio - analysis
            feature_delta_peak = float(np.abs(feature_delta).max())

            if feature_delta_peak >= 1e-9:
                scale = 0.9 / feature_delta_peak
                normalized = feature_delta * scale
                native_normalized = (
                    _resample_mono(normalized, LIN_SR, sr)
                    if sr != LIN_SR else normalized
                )
                native_delta = native_normalized / scale
                n_delta = min(len(stereo_audio), len(native_delta))
                feature_out = stereo_audio.copy()
                feature_out[:n_delta, 0] += native_delta[:n_delta]
                feature_out[:n_delta, 1] += native_delta[:n_delta]

                peak = np.abs(feature_out).max()
                if peak > 0.97:
                    feature_out *= 0.97 / peak

                transferred_score = _score_stereo_array(feature_out, sr)
                # Measure the actual delivered stereo result. A channel-only
                # check can misreport quality on asymmetric material (for
                # example a silent or deliberately sparse left channel).
                orig_rms = np.sqrt(np.mean(stereo_audio ** 2))
                delta_rms = np.sqrt(np.mean((feature_out - stereo_audio) ** 2))
                transferred_snr = float(
                    20 * np.log10(orig_rms / (delta_rms + 1e-12))
                )
                quality_ok = (
                    transferred_snr >= feature_min_snr_db
                    and feature_result.gain_peak_db <= feature_max_gain_db
                )

                if progress_cb:
                    progress_cb(
                        "linear: feature-domain result checked on transferred "
                        f"stereo output: {transferred_score * 100:.5f}% AI, "
                        f"SNR {transferred_snr:.1f}dB, peak spectral adjustment "
                        f"{feature_result.gain_peak_db:.2f}dB"
                    )

                if transferred_score < ACCEPT_THRESHOLD and quality_ok:
                    return feature_out, {
                        "applied": True,
                        "method": "feature_domain",
                        "snr_db": transferred_snr,
                        "target": target,
                        "final_real_score": transferred_score,
                        "attempts": 1,
                        "feature_solver_sec": feature_result.elapsed_sec,
                        "feature_gain_rms_db": feature_result.gain_rms_db,
                        "feature_gain_peak_db": feature_result.gain_peak_db,
                    }

                if progress_cb:
                    reason = (
                        f"score remained {transferred_score * 100:.3f}%"
                        if transferred_score >= ACCEPT_THRESHOLD
                        else "perceptual quality guard was not met"
                    )
                    progress_cb(
                        f"linear: fast feature-domain candidate rejected ({reason}); "
                        "falling back to the full waveform optimizer"
                    )
            elif progress_cb:
                progress_cb(
                    "linear: feature-domain solve found no correction; falling "
                    "back to the full waveform optimizer"
                )
        except Exception as exc:
            if progress_cb:
                progress_cb(
                    "linear: feature-domain solve could not complete "
                    f"({exc}); falling back to the full waveform optimizer"
                )

    # track the best-scoring attempt across all retries, not just the last one -
    # a later retry can converge WORSE than an earlier one (each retry tightens
    # cur_real_target/target, which can make the optimizer's job harder within
    # the same max_steps budget), so always ship whichever attempt actually
    # scored lowest rather than whatever happened to run last.
    best_out = None
    best_final_score = 1.0
    best_n = None
    best_attempt = None

    # the resample-to-native-rate transfer (16kHz mono analysis -> native
    # stereo delivered file) reliably costs real accuracy on its own, even
    # when the optimizer's internal 16kHz real-check already looks near-
    # perfect. Confirmed directly on two independent production optimization
    # runs on the same file: internal 16kHz score -> final post-transfer
    # score was 0.00012 -> 0.03106 (258x) and 0.00005 -> 0.01406 (281x) - a
    # consistent PROPORTIONAL multiplier (~260-280x), not a fixed additive
    # loss. This is why most files need 2+ full attempts: attempt 1 aims at
    # the nominal target, discovers only AFTER the expensive transfer that
    # the real jump was ~270x worse, then attempt 2 has to redo the whole
    # optimization with a tighter target. Dividing by the observed
    # multiplier (rather than subtracting a flat margin, which undershot
    # last time: 1.4% still missed the <1% bar) budgets attempt 1's
    # internal target proportionally so it has a real shot at surviving
    # the transfer. Capped at a floor so this doesn't demand an
    # unreachably tiny target the optimizer can never actually hit within
    # max_steps.
    TRANSFER_LOSS_MULTIPLIER = 270.0  # observed ~258-281x across two real runs
    cur_real_target = max(0.00005, min(real_target, ACCEPT_THRESHOLD / TRANSFER_LOSS_MULTIPLIER))
    for attempt in range(max_retries + 1):
        if progress_cb:
            progress_cb(f"linear: attempt {attempt + 1} of {max_retries + 1} - optimizing (the live "
                        f"percentage shown during this step is a fast internal estimate, not the "
                        f"final verified score - it will be re-checked against the real detector "
                        f"once this attempt finishes)")

        audio_t = torch.tensor(analysis, dtype=torch.float32)
        cur_attempt = attempt

        def _on_step(step, mx, cur_score, _attempt=cur_attempt):
            if step_progress_cb is not None:
                step_progress_cb(step, mx, cur_score, _attempt + 1, max_retries + 1)

        delta_t, best_real_score = _optimize_linear(audio_t, target=target, real_target=cur_real_target,
                                                      max_steps=max_steps, verbose=False, progress_cb=_on_step,
                                                      retry_index=attempt)
        if delta_t is None:
            if attempt < max_retries:
                cur_real_target = min(0.5, cur_real_target * 3)
                continue
            return stereo_audio, {"applied": False, "reason": f"optimizer never reached a verified result (best real score {best_real_score:.4f})"}
        delta = delta_t.numpy()

        delta_peak = np.abs(delta).max()
        if delta_peak < 1e-9:
            return stereo_audio, {"applied": False, "reason": "optimizer found zero-magnitude delta"}

        # precision-preserving transfer: normalize near full-scale before the
        # WAV round-trip / resample so a tiny delta doesn't collapse into a
        # handful of int16 quantization levels, then undo the scale after.
        scale = 0.9 / delta_peak
        delta_norm = delta * scale
        delta_44k_norm = _resample_mono(delta_norm, LIN_SR, sr) if sr != LIN_SR else delta_norm
        delta_native = delta_44k_norm / scale

        # apply the correction only to the analyzed prefix - the model itself
        # only ever scores the first MAX_DURATION seconds of a track (see the
        # "truncated" check above), so that's the only region a delta exists
        # for. Anything past that must be carried through UNCHANGED rather
        # than truncated away: out must always cover the FULL original track
        # length, not just however much was analyzed.
        n_delta = min(len(stereo_audio), len(delta_native))
        out = stereo_audio.copy()
        out[:n_delta, 0] += delta_native[:n_delta]
        out[:n_delta, 1] += delta_native[:n_delta]
        n = len(out)

        # emergency anti-clipping safety net ONLY (not the app's real
        # loudness ceiling) - this just prevents raw digital clipping if the
        # correction happens to push a peak past 1.0. It allows peaks up to
        # ~-0.26dBFS, well above the -1dBTP the true-peak limiter targets;
        # if that limiter isn't also selected/run as the final chain stage,
        # this alone does not guarantee -1dBTP compliance.
        peak = np.abs(out).max()
        if peak > 0.97:
            out *= 0.97 / peak

        final_score = _score_stereo_array(out, sr)
        if progress_cb:
            progress_cb(f"linear: attempt {attempt + 1} result checked against the REAL detector "
                        f"(not the fast estimate) on the actual delivered audio: {final_score * 100:.3f}%")

        improved = final_score < best_final_score
        if improved:
            best_final_score = final_score
            best_out = out
            best_n = n
            best_attempt = attempt + 1

        if final_score < ACCEPT_THRESHOLD:
            orig_rms = np.sqrt(np.mean(stereo_audio[:n, 0] ** 2))
            delta_rms = np.sqrt(np.mean((out[:, 0] - stereo_audio[:n, 0]) ** 2))
            snr_db = 20 * np.log10(orig_rms / (delta_rms + 1e-12))
            return out, {
                "applied": True,
                "snr_db": float(snr_db),
                "target": target,
                "final_real_score": final_score,
                "attempts": attempt + 1,
            }

        # a retry only has a shot at doing better if it's actually being asked
        # to hit a tighter target than the previous attempt - once cur_real_target
        # has bottomed out at its floor, a further retry starts from the same
        # conditions and reliably reproduces the same result (confirmed on a
        # real production run: attempts 3 and 4 both landed on 1.178% exactly),
        # so stop rather than burn ~2-7 minutes on a repeat that cannot improve.
        # Floor value must track the min()-based retry step above (0.00001) -
        # this used to reference a stale 0.002 floor left over from before
        # TRANSFER_LOSS_MULTIPLIER dropped the initial target to 0.00005,
        # which meant this check could never actually trigger against the
        # real floor the retry step was using.
        at_target_floor = cur_real_target <= REAL_TARGET_FLOOR + 1e-9
        stalled = attempt > 0 and not improved and at_target_floor
        if progress_cb:
            if attempt < max_retries and not stalled:
                progress_cb(f"linear: real score {final_score * 100:.2f}% is above the <{ACCEPT_THRESHOLD*100:.0f}% target - retrying with a stricter internal target")
            elif stalled:
                progress_cb(f"linear: real score {final_score * 100:.2f}% is above the <{ACCEPT_THRESHOLD*100:.0f}% target and did not "
                            f"improve on the previous attempt at the same internal target floor - further retries would repeat this "
                            f"result, so stopping early after {attempt + 1} of {max_retries + 1} attempts and shipping the best-scoring "
                            f"attempt found")
            else:
                progress_cb(f"linear: real score {final_score * 100:.2f}% is above the <{ACCEPT_THRESHOLD*100:.0f}% target "
                            f"after all {max_retries + 1} attempts - shipping the best-scoring attempt found and continuing to the next tool")
        if stalled:
            break
        # BUG FIX (adversarial review, verified directly): the 0.002 floor
        # here predates TRANSFER_LOSS_MULTIPLIER dropping the INITIAL target
        # down to 0.00005 - once that happened, max(0.002, 0.00005*0.3)
        # evaluates to 0.002, a target 40x LOOSER than attempt 1 even though
        # the log line right above claims "retrying with a stricter internal
        # target." A retry is only useful if it's actually harder than the
        # attempt that just failed. Floor removed in favor of an explicit
        # min() against the current value, so this can now only ever move
        # the same direction the log message promises.
        cur_real_target, target = _tighten_retry_targets(cur_real_target, target)

    # exhausted retries: ship whichever attempt scored BEST across the whole
    # loop, not whatever the last attempt happened to produce - a later retry
    # can converge worse than an earlier one, so always use best_out here.
    if best_out is not None and best_final_score < 0.5:
        orig_rms = np.sqrt(np.mean(stereo_audio[:best_n, 0] ** 2))
        delta_rms = np.sqrt(np.mean((best_out[:, 0] - stereo_audio[:best_n, 0]) ** 2))
        snr_db = 20 * np.log10(orig_rms / (delta_rms + 1e-12))
        return best_out, {
            "applied": True,
            "snr_db": float(snr_db),
            "target": target,
            "final_real_score": best_final_score,
            "attempts": max_retries + 1,
            "best_attempt": best_attempt,
            "warning": f"did not reach the <{ACCEPT_THRESHOLD*100:.0f}% target after {max_retries + 1} attempts; "
                       f"best achieved was {best_final_score*100:.2f}% (on attempt {best_attempt})",
        }

    return stereo_audio, {"applied": False, "reason": "could not verify a working fix after transfer within retry budget"}
