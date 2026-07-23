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


def fix_linear(stereo_audio, sr, target=0.005, real_target=0.008, max_steps=400,
                max_retries=3, progress_cb=None, step_progress_cb=None):
    """Apply the gradient-based linear-model fix. stereo_audio: [N,2] float32
    at native sr (typically 44100). Returns (fixed_stereo, info).

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

    # track the best-scoring attempt across all retries, not just the last one -
    # a later retry can converge WORSE than an earlier one (each retry tightens
    # cur_real_target/target, which can make the optimizer's job harder within
    # the same max_steps budget), so always ship whichever attempt actually
    # scored lowest rather than whatever happened to run last.
    best_out = None
    best_final_score = 1.0
    best_n = None
    best_attempt = None

    cur_real_target = real_target
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
                                                      max_steps=max_steps, verbose=False, progress_cb=_on_step)
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

        if final_score < best_final_score:
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

        if progress_cb:
            if attempt < max_retries:
                progress_cb(f"linear: real score {final_score * 100:.2f}% is above the <{ACCEPT_THRESHOLD*100:.0f}% target - retrying with a stricter internal target")
            else:
                progress_cb(f"linear: real score {final_score * 100:.2f}% is above the <{ACCEPT_THRESHOLD*100:.0f}% target "
                            f"after all {max_retries + 1} attempts - shipping the best-scoring attempt found and continuing to the next tool")
        cur_real_target = max(0.002, cur_real_target * 0.3)
        target = max(0.001, target * 0.3)

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
