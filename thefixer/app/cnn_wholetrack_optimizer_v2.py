import numpy as np
import torch
from .cnn_differentiable_v2 import (
    forward_logit_differentiable, forward_score_differentiable,
    get_real_score_segment, SR, SEGMENT_SAMPLES,
)
from .cnn_gradient_optimizer_v2 import perceptual_penalty, band_limit_penalty, tonality_penalty, apply_silence_guard_to_delta


# The differentiable optimization path (nnAudio's CQT feeding the CNN's
# pooling layers) genuinely crashes below this many samples - verified
# directly: 3200 samples raises "Output size is too small" inside a pooling
# layer, 500 samples raises a padding-size error in the CQT itself, while
# 4000 samples (250ms @ 16kHz) is the smallest input that runs cleanly. The
# real (non-differentiable, librosa-based) scoring path tolerates much
# shorter input without crashing, so this floor applies ONLY to what the
# optimizer can operate on, not to scoring.
MIN_VIABLE_SEGMENT_SAMPLES = 4000


def build_sliding_windows(n_samples, hop_sec=2.5, segment_sec=10.0, sr=SR, edge_guard_sec=0.5):
    hop = int(hop_sec * sr)
    seg_len = int(segment_sec * sr)
    edge_guard = int(edge_guard_sec * sr)

    if n_samples < seg_len + 2 * edge_guard:
        # track too short for even one full-length window plus edge guards on
        # both sides - shrink the window to fit the track instead, still
        # respecting the edge guard on both sides where possible.
        seg_len = n_samples - 2 * edge_guard
        if seg_len < MIN_VIABLE_SEGMENT_SAMPLES:
            # still too short even after dropping the edge guard entirely -
            # a previous version returned [0], n_samples here unconditionally,
            # which silently fed a too-short segment into the differentiable
            # CQT/pooling pipeline and crashed with a low-level tensor-shape
            # error. Try the whole track with no edge guard as a last resort,
            # and only proceed if that itself clears the real minimum -
            # otherwise the caller must not attempt optimization at all.
            if n_samples >= MIN_VIABLE_SEGMENT_SAMPLES:
                return [0], n_samples
            raise ValueError(
                f"track too short ({n_samples} samples = {n_samples / sr:.2f}s at {sr}Hz) "
                f"for the CNN optimizer, which needs at least "
                f"{MIN_VIABLE_SEGMENT_SAMPLES} samples ({MIN_VIABLE_SEGMENT_SAMPLES / sr:.2f}s)"
            )
        return [edge_guard], seg_len

    # a window must not touch the literal first OR last sample of the signal -
    # CQT boundary handling produces a NaN-gradient singularity at either edge
    # (confirmed: start-of-track and end-of-track both trigger this)
    last_valid_start = n_samples - seg_len - edge_guard
    positions = list(range(edge_guard, max(edge_guard + 1, last_valid_start), hop))
    if not positions or positions[-1] < last_valid_start:
        positions.append(max(edge_guard, last_valid_start))
    return positions, seg_len


def optimize_whole_track_verified(
    audio_np,
    target=0.05,
    real_target=0.35,  # be conservative on the REAL score target, since surrogate is optimistic
    lambda_perceptual=2000.0,
    lambda_band=5000.0,
    lambda_tonality=50.0,
    lr=0.00002,
    max_steps=600,
    min_steps=150,
    hop_sec=2.5,
    real_check_interval=25,
    verbose=True,
    progress_cb=None,
):
    """Same joint multi-window optimization as before, but periodically checks
    the REAL (librosa-based) score at each window and, for any window where
    the real score still exceeds real_target despite the surrogate claiming
    success, boosts that window's weight in the loss so the optimizer keeps
    working on it specifically rather than trusting the surrogate blindly.

    For tracks shorter than the real detector's own analysis segment length
    (10s), the real detector zero-pads the clip out to a full segment before
    scoring it (see detector.py's extract_segments) - if the optimizer only
    ever analyzed the real (shorter, unpadded) audio, it would be optimizing
    against a DIFFERENT signal than what the detector actually scores.
    Zero-pad here too so both sides analyze the same representation; the
    returned delta is only as long as the padding needed, and the caller
    (cnn_fix.fix_cnn) truncates it back down to the real track length before
    applying it, since the padded tail doesn't exist in the delivered file."""
    n_real = len(audio_np)
    if n_real < SEGMENT_SAMPLES:
        padded = np.zeros(SEGMENT_SAMPLES, dtype=audio_np.dtype)
        padded[:n_real] = audio_np
        audio_np = padded
    n = len(audio_np)
    positions, seg_len = build_sliding_windows(n, hop_sec=hop_sec)
    print(f"track length: {n/SR:.1f}s, {len(positions)} overlapping windows "
          f"(hop={hop_sec}s, window={seg_len/SR:.1f}s)")

    audio = torch.tensor(audio_np, dtype=torch.float32)
    delta = torch.zeros_like(audio, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    logit_target = torch.logit(torch.tensor(target), eps=1e-6)
    # per-window extra weight, boosted when the real model disagrees with the surrogate
    window_weight = {pos: 1.0 for pos in positions}

    best_delta = None
    best_real_max = 1.0

    for step in range(max_steps):
        optimizer.zero_grad()
        perturbed = audio + delta

        total_logit_loss = 0.0
        max_surrogate_score = 0.0
        for pos in positions:
            seg = perturbed[pos:pos + seg_len]
            logit = forward_logit_differentiable(seg.unsqueeze(0))
            w = window_weight[pos]
            total_logit_loss = total_logit_loss + w * torch.relu(logit - logit_target + 1.0)
            with torch.no_grad():
                s = torch.sigmoid(logit).item()
                max_surrogate_score = max(max_surrogate_score, s)

        percep = perceptual_penalty(delta, audio)
        band_pen = band_limit_penalty(delta, lo_hz=400, hi_hz=8000, sr=SR)
        tonal_pen = tonality_penalty(delta)

        loss = total_logit_loss + lambda_perceptual * percep + lambda_band * band_pen + lambda_tonality * tonal_pen
        loss.backward()

        # a very short track zero-padded up to the detector's 10s segment
        # length can put most of a window over pure digital silence, which
        # has produced non-finite gradients (verified: a 2s clip padded to
        # 10s NaNs on the very first optimizer step) - a different failure
        # mode than the documented CQT edge-of-track singularity. Detect and
        # fail cleanly here rather than let a corrupted delta silently
        # propagate into a later real-model check (which crashes deep inside
        # librosa with a much less diagnosable "buffer is not finite" error).
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise ValueError(
                "gradient became non-finite during CNN optimization - this can happen on "
                "very short tracks padded with a large proportion of silence; the CNN fix "
                "is not currently reliable on tracks this short"
            )

        optimizer.step()

        if progress_cb is not None:
            progress_cb(step, max_steps)

        if verbose and step % 5 == 0:
            snr = 20 * torch.log10(audio.norm() / (delta.norm() + 1e-8)).item()
            print(f"  step {step:3d}: max_surrogate_score={max_surrogate_score:.4f}  "
                  f"total_logit_loss={total_logit_loss.item():.2f}  SNR={snr:.1f}dB", flush=True)

        # periodically verify against the REAL model and re-weight problem windows
        if step > 0 and step % real_check_interval == 0:
            with torch.no_grad():
                perturbed_np = (audio + delta).numpy()
            real_scores = {}
            for pos in positions:
                seg_np = perturbed_np[pos:pos + seg_len]
                real_scores[pos] = get_real_score_segment(seg_np)
            real_max = max(real_scores.values())
            n_bad = sum(1 for v in real_scores.values() if v > real_target)
            if verbose:
                print(f"    [real check @ step {step}] max_real_score={real_max:.4f}, "
                      f"{n_bad}/{len(positions)} windows still above real_target={real_target}", flush=True)

            # boost weight for windows where real score is still too high
            for pos in positions:
                if real_scores[pos] > real_target:
                    window_weight[pos] = min(window_weight[pos] * 1.5, 20.0)
                else:
                    window_weight[pos] = max(window_weight[pos] * 0.9, 1.0)

            cur_norm = delta.norm().item()
            if real_max < best_real_max:
                best_real_max = real_max
                best_delta = delta.detach().clone()

            if real_max < real_target and step >= min_steps:
                print(f"  converged (real-verified) at step {step}")
                break

    if best_delta is None:
        best_delta = delta.detach().clone()
        print("  WARNING: real verification never confirmed full convergence, using best available")
    else:
        print(f"  best real_max achieved during search: {best_real_max:.4f}")

    # hard guarantee (not just a training-time penalty): zero out the delta
    # in genuinely near-silent passages of the ORIGINAL track, regardless of
    # what the optimizer converged to.
    best_delta = apply_silence_guard_to_delta(best_delta, audio)

    return best_delta.numpy(), positions, seg_len
