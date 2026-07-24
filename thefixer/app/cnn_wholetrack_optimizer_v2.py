import numpy as np
import torch
import torch.nn.functional as F
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


def build_sliding_windows(n_samples, hop_sec=0.5, segment_sec=10.0, sr=SR, edge_guard_sec=0.5):
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
    # (confirmed: start-of-track and end-of-track both trigger this). The
    # fallback below used to unconditionally append a window starting at
    # EXACTLY last_valid_start - which places its END exactly at
    # n_samples - edge_guard, i.e. right back at the boundary the guard
    # exists to avoid. Verified directly: on a real 277s track this produced
    # a window ending 0.46s from the literal last sample and its gradient
    # came back non-finite on step 0 (logit 14.7, near-total confidence,
    # right where the singularity bites hardest). Clamp the fallback to
    # last_valid_start - hop instead so the final window keeps the full
    # edge_guard margin like every other window, even if that means a
    # slightly larger gap before it than `hop`.
    last_valid_start = n_samples - seg_len - edge_guard
    positions = list(range(edge_guard, max(edge_guard + 1, last_valid_start), hop))
    if not positions:
        positions.append(edge_guard)
    elif positions[-1] < last_valid_start:
        safe_last = last_valid_start - hop
        if safe_last > positions[-1]:
            positions.append(safe_last)
    return positions, seg_len


def optimize_whole_track_verified(
    audio_np,
    target=0.05,
    real_target=0.08,  # a real production run showed accepting anything under
    # the model's raw 0.5 decision boundary (its old default of 0.35, or even
    # accepting up to 0.5) leaves almost no safety margin: a file that reached
    # 0.48 internally regressed to 0.999 the moment ANY later stage (here: the
    # linear re-verification pass) nudged the audio at all. Linear's own
    # target is 50x stricter than its 0.5 pass boundary (0.01 vs 0.5) for
    # exactly this reason - a small margin gets erased by any later change.
    # 0.08 mirrors that same margin-over-boundary ratio for cnn.
    lambda_perceptual=2000.0,
    lambda_band=5000.0,
    lambda_tonality=50.0,
    lr=0.00002,
    max_steps=600,
    min_steps=150,
    hop_sec=0.5,  # was 2.5 - confirmed directly on a real production file that
    # the real (non-differentiable) detector's score oscillates wildly
    # (0% to 99.99%+) with a period of roughly 0.8-0.9s on CNN-corrected
    # audio - a 2.5s hop can skip 2-3 full oscillation cycles between
    # sampled windows entirely, so the optimizer "converges" against its own
    # sparse sampling while leaving most of the actual timeline unfixed
    # (verified: optimizer reported 0/67 windows failing while the real
    # detector's fixed 5-segment scan still landed on two ~100%-flagged
    # positions the optimizer never checked). This same instability does
    # NOT exist on unmodified source audio (verified: rock-stable ~99.9%
    # regardless of sub-second offset) - it's specifically introduced by
    # the correction overfitting to its own sparse sampling grid. 0.5s
    # keeps at least one sample inside every observed "good" oscillation
    # window instead of being able to skip over sub-second corrected
    # regions entirely.
    real_check_interval=10,  # was 25 - the real-model check is the only trustworthy
    # progress signal (the surrogate/live-estimate is known to diverge sharply from
    # the real model - see the transfer-loss investigation elsewhere in this file's
    # history), so checking every 10 steps instead of 25 means the UI's meaningful
    # progress detail (windows still failing, real max score) updates ~2.5x more
    # often during a run that can otherwise take 9+ minutes with long silent gaps.

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

    # seed each window's weight from the REAL model's score on the untouched
    # original audio, not just 1.0 - confirmed on a real production file that
    # the differentiable surrogate can be badly wrong on specific windows
    # (one window: surrogate said 17% AI, real model said 92% AI) even before
    # any optimization happens. Waiting for the first periodic real-check
    # (step 25) to notice this via the *= 1.5 boost means ~25 steps run with
    # near-zero pressure on exactly the windows that need it most, since the
    # main gradient loss trusts a surrogate that already thinks they're fine.
    # Seeding from real scores up front means blind-spot windows get strong
    # pressure from step 0 - the periodic re-check loop below still runs and
    # keeps adjusting weight_window as delta evolves, this only fixes the
    # starting point.
    print("checking initial real-model score per window (before any optimization)...", flush=True)
    window_weight = {}
    audio_np_orig = audio_np
    for pos in positions:
        seg_np = audio_np_orig[pos:pos + seg_len]
        real0 = get_real_score_segment(seg_np)
        # same 1.0-20.0 scale and threshold the periodic re-check already uses,
        # just applied before step 0 instead of first appearing at step 25
        window_weight[pos] = min(20.0, 1.0 + 19.0 * max(0.0, real0 - real_target) / max(1e-6, 1.0 - real_target)) \
            if real0 > real_target else 1.0
    n_seeded_hot = sum(1 for w in window_weight.values() if w > 1.0)

    # a real production file had 107/108 windows already scoring above
    # real_target pre-optimization - applying the FULL 1.0-20.0 boost range
    # simultaneously across nearly the entire track made the combined loss
    # (and its gradient through the single shared delta tensor) blow up to
    # non-finite on step 0 (verified directly: this exact crash reproduced
    # with the un-normalized boost). Most windows needing SOME extra push is
    # normal and fine; ALL of them needing the same near-max push at once is
    # not something the existing lr/loss scale was tuned to handle. Rescale
    # so the total pressure across all windows starts near what the un-seeded
    # code would apply (average weight ~1.0), preserving each window's
    # RELATIVE priority from the real-score gap without inflating the sum.
    total_weight = sum(window_weight.values())
    if total_weight > 0:
        norm = len(positions) / total_weight
        window_weight = {pos: w * norm for pos, w in window_weight.items()}

    print(f"  {n_seeded_hot}/{len(positions)} windows started with boosted weight "
          f"(real score already above real_target={real_target} pre-optimization), "
          f"rescaled so total pressure matches the un-seeded baseline", flush=True)

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
            margin_term = logit - logit_target + 1.0
            # CRITICAL FIX: plain relu(margin_term) is EXACTLY zero, with
            # EXACTLY zero gradient, the instant a window's surrogate score
            # drops low enough to clear the margin. Confirmed as the actual
            # mechanism stalling every full-track CNN run this session: once
            # a window "looks solved" to the differentiable surrogate, this
            # term goes flat and boosting window_weight on it (even up to
            # the 20x cap) multiplies zero by up to 20 and gets zero back -
            # no amount of per-window weight can revive a term with no slope
            # to amplify. That's why stuck windows stayed frozen at whatever
            # score they'd already reached instead of continuing to improve,
            # even while the REAL detector still flagged them.
            # Fix: leaky_relu instead of relu - full gradient strength above
            # the margin (identical behavior to before there), but a small
            # residual negative slope below it, so a window the real model
            # still disagrees with keeps getting pushed even after clearing
            # its own surrogate's satisfied point. The residual slope is
            # weighted by w same as before, so window_weight's per-window
            # boosting is meaningful again everywhere, not just before a
            # window first clears its margin.
            total_logit_loss = total_logit_loss + w * F.leaky_relu(margin_term, negative_slope=0.02)
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
            progress_cb(step, max_steps, max_surrogate_score, None)

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
            if progress_cb is not None:
                progress_cb(step, max_steps, max_surrogate_score, {
                    "real_max_score": real_max, "n_windows_bad": n_bad, "n_windows": len(positions),
                })

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
