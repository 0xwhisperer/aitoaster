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
from .cnn_wholetrack_optimizer_v2 import optimize_whole_track_verified


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


def fix_cnn(stereo_audio, sr, max_steps=300, min_steps=100, hop_sec=2.5,
            real_check_interval=25, progress_cb=None, step_progress_cb=None):
    """Whole-track CNN fix. stereo_audio: [N,2] float32 at native sr.
    Returns (fixed_stereo, info). progress_cb receives log-line strings;
    step_progress_cb(step, max_steps) receives raw optimizer step counts for
    UI progress bars, since this stage's internal optimization loop can run
    for many minutes with no other visible progress signal."""
    mono = stereo_audio.mean(axis=1)
    mono_16k = _resample_mono(mono, sr, CNN_SR) if sr != CNN_SR else mono.copy()

    if progress_cb:
        progress_cb(f"cnn: optimizing {len(mono_16k)/CNN_SR:.1f}s of audio (this can take a while)")

    try:
        delta_16k, positions, seg_len = optimize_whole_track_verified(
            mono_16k, max_steps=max_steps, min_steps=min_steps, hop_sec=hop_sec,
            real_check_interval=real_check_interval, verbose=True,
            progress_cb=step_progress_cb,
        )
    except ValueError as e:
        return stereo_audio, {"applied": False, "reason": str(e)}

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
    if progress_cb:
        progress_cb("cnn: verifying final transferred stereo output against the real model")
    out_mono_native = out.mean(axis=1)
    out_mono_16k = _resample_mono(out_mono_native, sr, CNN_SR) if sr != CNN_SR else out_mono_native
    post_transfer_scores = []
    for pos in positions:
        seg = out_mono_16k[pos:pos + seg_len]
        if len(seg) < seg_len:
            continue
        post_transfer_scores.append(get_real_score_segment(seg))
    worst_after_transfer = max(post_transfer_scores) if post_transfer_scores else None

    return out, {
        "applied": True,
        "snr_db": float(snr_db),
        "n_windows": len(positions),
        "window_positions_sec": [p / CNN_SR for p in positions],
        "worst_score_after_transfer": float(worst_after_transfer) if worst_after_transfer is not None else None,
        "verified_after_transfer": bool(worst_after_transfer is not None and worst_after_transfer < 0.5),
    }
