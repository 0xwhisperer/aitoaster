import numpy as np
import torch
import torch.nn.functional as F
from .cnn_differentiable_v2 import (
    forward_logit_differentiable, forward_score_differentiable,
    get_real_score_segment, get_real_evaluator_segments, SR, SEGMENT_SAMPLES,
)
from .cnn_real_scanner import get_default_real_score_scanner
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

# BUG FIX (direct, repeated user report): max_steps/min_steps for Thorough
# mode used to be flat constants (300/150) regardless of how many 10-second
# windows a track actually has - a 30s file (38 windows) and a 600s file
# (~1200 windows) got the identical budget, which makes no sense: all
# windows share ONE joint delta tensor, so more windows plausibly need more
# steps to jointly converge. A prior attempt at scaling this was reverted
# for being unvalidated against real data (see build_sliding_windows'
# nearby history) - this time it's grounded in an actual multi-length
# benchmark, not a guess.
#
# Measured directly (real optimizer runs, not simulated): a track's own
# convergence step count grows roughly with log2(window count), not
# linearly - consistent with heavy window overlap (10s windows at 0.5s hop
# share ~95% of their samples with neighbors), so a correction that fixes
# one window generalizes partially to its neighbors; more windows mostly
# means more redundant content to VERIFY, not proportionally more distinct
# problems to solve.
#   38 windows (30s)  -> converged at step 40
#  158 windows (90s)  -> converged at step 70
#  338 windows (180s) -> converged at step 80
# step/log2(windows) settles to ~9.5 for the two larger cases. This is only
# 3 data points, not a proven law - the formula below applies real safety
# margin (1.5x for min_steps, 4x that again for the max_steps ceiling)
# rather than fitting tightly to these exact numbers, and never reduces
# below the previous flat defaults for typical-length files.
def scaled_step_budget(n_windows):
    """Returns (min_steps, max_steps) scaled to how many overlapping windows
    a track actually has, instead of a flat constant regardless of length."""
    if n_windows <= 1:
        return 60, 300
    raw = 9.5 * np.log2(n_windows)
    min_steps = max(60, int(raw * 1.5))
    max_steps = max(300, min_steps * 4)
    return min_steps, max_steps


def scan_real_scores(audio_np, positions, seg_len):
    """Scan exact real-model scores in position order using the reusable pool."""
    return get_default_real_score_scanner().scan(audio_np, positions, seg_len)


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
    mode="thorough",  # "thorough" (default): dense overlapping windows every
    # hop_sec across the WHOLE track, covering far more than any real
    # detector deployment would actually check - built to be robust against
    # a checker sampling differently than expected. "simple": optimize ONLY
    # the exact 5 fixed positions the real detector itself checks
    # (CNNDetector.predict's own n_segments=5 logic, reproduced exactly via
    # get_real_evaluator_segments) - i.e. what a standard, uncustomized
    # deployment of this detector would actually test against, nothing
    # more. Added specifically to let "simple" be tested in isolation
    # against "thorough" to see whether the extra coverage is worth its
    # much higher cost, rather than always paying for the expensive version.
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
    if mode == "simple":
        seg_len = SEGMENT_SAMPLES
        positions = get_real_evaluator_segments(audio_np, n_segments=5)
        print(f"track length: {n/SR:.1f}s, SIMPLE mode: optimizing only the "
              f"real detector's own 5 fixed evaluation positions "
              f"(window={seg_len/SR:.1f}s)")
    else:
        positions, seg_len = build_sliding_windows(n, hop_sec=hop_sec)
        print(f"track length: {n/SR:.1f}s, {len(positions)} overlapping windows "
              f"(hop={hop_sec}s, window={seg_len/SR:.1f}s)")
        # BUG FIX (direct, repeated user report): scale the step budget to
        # this track's ACTUAL window count via scaled_step_budget (see its
        # definition above build_sliding_windows for the real multi-length
        # benchmark this is grounded in), rather than leaving every track
        # stuck with the same flat default regardless of length. Only ever
        # RAISES min_steps/max_steps above whatever the caller passed in -
        # never lowers them - so an explicit override always still wins;
        # this only fills in more headroom for tracks with enough windows
        # that the flat default might not be (or might barely be) enough.
        scaled_min, scaled_max = scaled_step_budget(len(positions))
        if scaled_min > min_steps:
            print(f"  scaling min_steps {min_steps} -> {scaled_min} for {len(positions)} windows")
            min_steps = scaled_min
        if scaled_max > max_steps:
            print(f"  scaling max_steps {max_steps} -> {scaled_max} for {len(positions)} windows")
            max_steps = scaled_max

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
    real0_by_pos = {}
    audio_np_orig = audio_np
    initial_real_scores = scan_real_scores(audio_np_orig, positions, seg_len)
    for pos, real0 in zip(positions, initial_real_scores):
        real0_by_pos[pos] = real0
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

    # EARLY EXIT: the seeding scan above already ran the real (non-
    # differentiable) model on every window - if every single one already
    # scores under real_target with zero correction applied, there is
    # nothing for the gradient loop to do. Without this, an already-clean
    # file still paid for min_steps (150 by default) of Adam optimization
    # against a delta that starts at exactly zero and has no pressure
    # pushing it anywhere, purely because the real-check break condition at
    # the bottom of the loop is gated on step >= min_steps and can't fire
    # any earlier - confirmed directly this was true "wasted work" per user
    # report ("3 of 4 passes do nothing but take time"). real0 (not the
    # per-step real_scores dict, which doesn't exist yet) is what the
    # seeding loop already measured per window with zero delta applied, so
    # reusing it here costs nothing extra.
    if n_seeded_hot == 0:
        zero_delta = torch.zeros_like(audio)
        already_worst = max(real0_by_pos.values()) if real0_by_pos else 0.0
        print(f"  0/{len(positions)} windows above real_target={real_target} "
              f"before any optimization - skipping the gradient loop entirely, "
              f"nothing to fix", flush=True)
        return zero_delta.numpy(), positions, seg_len, already_worst

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
            real_scores = dict(zip(
                positions, scan_real_scores(perturbed_np, positions, seg_len)
            ))
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

            # BUG FIX (direct user report, watching a live job): min_steps
            # used to gate EVERY real-verified pass, even a comfortably-clear
            # one (e.g. max_real_score=0.0018 against real_target=0.08 - 44x
            # under the line) at an early step. min_steps exists to give a
            # BORDERLINE result more training before accepting it, not to
            # force an already-clearly-converged result to keep grinding
            # with nothing left to gain. Only apply the min_steps floor when
            # the result is still close to the line (within a real margin of
            # real_target); a comfortably-clear pass breaks immediately
            # regardless of step count, exactly like the analogous fix
            # already applied to the linear optimizer's step>=150 floor.
            comfortably_clear = real_max < real_target * 0.5
            if real_max < real_target and (step >= min_steps or comfortably_clear):
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
    #
    # BUG FIX (adversarial review, verified directly): this used to mutate
    # best_delta with no re-check after it had already been selected and
    # validated above - the same "certified, then silently changed" gap
    # found in optimize_eot_verified. Re-check every position on the
    # POST-guard delta before returning, so the reported score matches what
    # actually ships.
    guarded_delta = apply_silence_guard_to_delta(best_delta, audio)
    with torch.no_grad():
        guarded_perturbed_np = (audio + guarded_delta).numpy()
    post_guard_scores = scan_real_scores(guarded_perturbed_np, positions, seg_len)
    post_guard_worst = max(post_guard_scores) if post_guard_scores else 1.0
    print(f"  post-silence-guard worst real score: {post_guard_worst:.4f} "
          f"(pre-guard best_real_max was {best_real_max:.4f})")
    if post_guard_worst > best_real_max + 1e-6:
        print("  WARNING: silence guard degraded the certified score - "
              "this is exactly why this re-check exists")

    return guarded_delta.numpy(), positions, seg_len, post_guard_worst


def _worst_shift_score(perturbed_np, center_pos, seg_len, n, shift_range_sec=1.0,
                        shift_step_sec=0.05, sr=SR):
    """Scan the REAL (non-differentiable) score across a range of small time
    shifts around center_pos and return the worst (highest) one found - the
    metric Fable's review specifically recommended adopting instead of a
    single-point check: passing at exactly zero-shift is exactly the lie
    that's been shipping all session (a delta that scores 0% at its exact
    optimized position but 90%+ just 0.25s away is NOT actually fixed, even
    though a single-point check would have called it a pass)."""
    # BUG FIX (adversarial audit, verified directly): worst used to
    # initialize to 0.0 and simply never update if EVERY candidate offset
    # fell outside [0, n) - meaning "genuinely scanned and confirmed
    # perfectly clean" and "never scanned a single valid window" both
    # returned the identical 0.0, with no way for a caller to tell them
    # apart. Confirmed directly: a post-transfer track shrunk by
    # resampling to just under one segment length (a real, reachable shape
    # after a 44.1kHz->16kHz resample, e.g. 440,998 samples -> 159,999,
    # one sample short of a full 160,000-sample segment) produced a
    # position with NO valid in-bounds shift at all, returned 0.0, and got
    # certified as verified_after_transfer=True for a segment that was
    # never actually scored by the real model. Return None when nothing
    # was scanned, so every caller can distinguish "verified clean" from
    # "could not verify" and fail closed on the latter instead of treating
    # an absence of data as a passing score.
    shift_samples = int(shift_range_sec * sr)
    step_samples = max(1, int(shift_step_sec * sr))
    offsets = []
    for offset in range(-shift_samples, shift_samples + 1, step_samples):
        pos = center_pos + offset
        if pos < 0 or pos + seg_len > n:
            continue
        offsets.append(pos)
    if not offsets:
        return None
    return max(scan_real_scores(perturbed_np, offsets, seg_len))


def optimize_eot_verified(
    audio_np,
    target=0.05,
    real_target=0.08,
    lambda_perceptual=2000.0,
    lambda_band=5000.0,
    lambda_tonality=50.0,
    lr=0.00002,
    max_steps=300,
    min_steps=60,
    eot_samples=6,       # random shift samples averaged per step per region
    eot_jitter_sec=0.5,  # +-0.5s jitter range, matching the actual measured
                          # instability period found earlier this session
                          # (score oscillation period ~0.8-0.9s on corrected
                          # audio - +-0.5s comfortably covers a full cycle)
    real_check_interval=10,
    n_segments=5,         # optimize the REAL detector's own fixed positions,
                          # not a dense whole-track grid - see module-level
                          # docstring below for why
    verbose=True,
    progress_cb=None,
):
    """Expectation-over-Transformation (EOT) optimizer: fixes an architectural
    problem in optimize_whole_track_verified, not just a tuning issue,
    identified via an independent design review (see project history) after
    that approach proved unreliable even after multiple rounds of tuning.

    THE ROOT CAUSE optimize_whole_track_verified doesn't address: gradient
    descent against a fixed audio position finds a MINIMAL-NORM, NON-ROBUST
    perturbation - a classic adversarial-example failure mode, not a bug.
    Verified directly and repeatedly this session: the real detector's score
    on CNN-corrected audio swings wildly (0% to 99.99%+) within a QUARTER
    SECOND of the position the delta was optimized at, while the SAME scan
    on unmodified source audio is rock-stable regardless of offset. The old
    approach's response - densely tile the whole track with overlapping
    windows so no checked position is missed - treats the SYMPTOM (gaps
    between brittle points) with brute force, at 5-10x the cost, and still
    doesn't reliably transfer once the file goes through resample/encode
    (which shifts exactly where the real detector's fixed positions land).

    THE FIX: don't discover a brittle point and then try to cover the gaps
    around it - make the point itself robust to being off by up to
    eot_jitter_sec, by training against RANDOM shifted samples every step
    (Expectation-over-Transformation, a standard adversarial-robustness
    technique). This also lets us go back to optimizing only the ~5 real
    positions the deployed detector actually checks (score = median of 5,
    so only 3 of 5 need to pass) instead of hundreds of windows across the
    whole track - a delta that's robust across +-0.5s at 5 positions is a
    much easier, cheaper target than "every possible 10s window everywhere,
    with zero robustness margin at any of them."

    Cost: n_segments x eot_samples forward/backward passes per step
    (5 x 6 = 30 by default) vs. hundreds of windows in the old approach -
    roughly an order of magnitude cheaper per step, on top of directly
    targeting the actual failure mode instead of working around it.

    Convergence uses _worst_shift_score (max real score across a +-1s scan
    at each position), not a single-point check - passing only at the exact
    optimized position is precisely the false convergence this session kept
    hitting."""
    n_real = len(audio_np)
    if n_real < SEGMENT_SAMPLES:
        padded = np.zeros(SEGMENT_SAMPLES, dtype=audio_np.dtype)
        padded[:n_real] = audio_np
        audio_np = padded
    n = len(audio_np)
    seg_len = SEGMENT_SAMPLES
    positions = get_real_evaluator_segments(audio_np, n_segments=n_segments)
    print(f"track length: {n/SR:.1f}s, EOT mode: optimizing the real detector's "
          f"{len(positions)} fixed positions with +-{eot_jitter_sec}s shift "
          f"robustness ({eot_samples} shift samples/step)", flush=True)

    audio = torch.tensor(audio_np, dtype=torch.float32)
    delta = torch.zeros_like(audio, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    logit_target = torch.logit(torch.tensor(target), eps=1e-6)
    jitter_samples = int(eot_jitter_sec * SR)

    window_weight = {pos: 1.0 for pos in positions}

    # EARLY EXIT: only 5 positions in EOT mode, so a worst-shift pre-scan
    # here is cheap (unlike Thorough's per-window seeding scan, this isn't
    # reusing work the rest of the function needs anyway - it's a small
    # deliberate up-front check). If every position is already robust
    # across the full jitter range with zero delta applied, there is
    # nothing to train against - skip straight to returning a zero delta
    # instead of paying for min_steps of Adam optimization with no pressure
    # pushing it anywhere.
    with torch.no_grad():
        audio_np_orig = audio.numpy()
    # BUG FIX (adversarial audit, verified directly): _worst_shift_score
    # returns None (not a numeric score) when a position has no valid
    # in-bounds shift window at all - filter those out before max() rather
    # than let a None crash the comparison or (worse, if a default were
    # used to paper over it) get silently treated as a passing score. If
    # EVERY position failed to scan, treat this as "cannot confirm clean"
    # (default=1.0, the worst possible score) so the early-exit below
    # correctly does NOT fire on unverifiable input.
    pre_scan_results = [
        s for s in (
            _worst_shift_score(audio_np_orig, pos, seg_len, n,
                                shift_range_sec=eot_jitter_sec, shift_step_sec=0.1, sr=SR)
            for pos in positions
        )
        if s is not None
    ]
    pre_scan_worst = max(pre_scan_results, default=1.0) if len(pre_scan_results) == len(positions) else 1.0
    if pre_scan_worst < real_target:
        zero_delta = torch.zeros_like(audio)
        print(f"  worst-shift score across all {len(positions)} positions is already "
              f"{pre_scan_worst:.4f} (under real_target={real_target}) before any "
              f"optimization - skipping the gradient loop entirely, nothing to fix", flush=True)
        return zero_delta.numpy(), positions, seg_len, pre_scan_worst

    best_delta = None
    best_worst_shift_max = 1.0
    rng = np.random.default_rng(1234)

    for step in range(max_steps):
        optimizer.zero_grad()
        perturbed = audio + delta

        total_logit_loss = 0.0
        max_surrogate_score = 0.0
        for pos in positions:
            w = window_weight[pos]
            for _ in range(eot_samples):
                jitter = int(rng.integers(-jitter_samples, jitter_samples + 1))
                shifted_pos = max(0, min(n - seg_len, pos + jitter))
                seg = perturbed[shifted_pos:shifted_pos + seg_len]
                logit = forward_logit_differentiable(seg.unsqueeze(0))
                margin_term = logit - logit_target + 1.0
                # same leaky-relu fix as optimize_whole_track_verified - a
                # plain relu here would have the identical zero-gradient
                # stalling problem that fix addressed.
                total_logit_loss = total_logit_loss + (w / eot_samples) * F.leaky_relu(margin_term, negative_slope=0.02)
                with torch.no_grad():
                    s = torch.sigmoid(logit).item()
                    max_surrogate_score = max(max_surrogate_score, s)

        percep = perceptual_penalty(delta, audio)
        band_pen = band_limit_penalty(delta, lo_hz=400, hi_hz=8000, sr=SR)
        tonal_pen = tonality_penalty(delta)

        loss = total_logit_loss + lambda_perceptual * percep + lambda_band * band_pen + lambda_tonality * tonal_pen
        loss.backward()

        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise ValueError(
                "gradient became non-finite during CNN EOT optimization - this can happen on "
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

        if step > 0 and step % real_check_interval == 0:
            with torch.no_grad():
                perturbed_np = (audio + delta).numpy()
            # worst-shift score per position (Fable-recommended metric) - not
            # just the exact-position score, which is precisely the check
            # that let non-robust deltas look converged all session.
            # BUG FIX (adversarial audit, verified directly): a None from
            # _worst_shift_score (no valid in-bounds window at all for this
            # position) must never be compared against real_target or fed
            # into max() as if it were a real score - treat it as "cannot
            # confirm clean," i.e. as bad as scoring 1.0 (the worst
            # possible), so it counts toward n_bad and can never make
            # worst_shift_max look artificially low.
            worst_shift_scores = {}
            for pos in positions:
                raw = _worst_shift_score(
                    perturbed_np, pos, seg_len, n,
                    shift_range_sec=eot_jitter_sec, shift_step_sec=0.1, sr=SR)
                worst_shift_scores[pos] = raw if raw is not None else 1.0
            worst_shift_max = max(worst_shift_scores.values())
            n_bad = sum(1 for v in worst_shift_scores.values() if v > real_target)
            if verbose:
                print(f"    [real check @ step {step}] worst_shift_max={worst_shift_max:.4f}, "
                      f"{n_bad}/{len(positions)} positions still above real_target={real_target} "
                      f"(checked across +-{eot_jitter_sec}s shifts, not just exact position)", flush=True)
            if progress_cb is not None:
                progress_cb(step, max_steps, max_surrogate_score, {
                    "real_max_score": worst_shift_max, "n_windows_bad": n_bad, "n_windows": len(positions),
                })

            for pos in positions:
                if worst_shift_scores[pos] > real_target:
                    window_weight[pos] = min(window_weight[pos] * 1.5, 20.0)
                else:
                    window_weight[pos] = max(window_weight[pos] * 0.9, 1.0)

            if worst_shift_max < best_worst_shift_max:
                best_worst_shift_max = worst_shift_max
                best_delta = delta.detach().clone()

            # same fix as optimize_whole_track_verified above: only hold a
            # comfortably-clear result to the min_steps floor when it's
            # actually still close to the line, not just because a step
            # counter hasn't hit an arbitrary number yet.
            comfortably_clear = worst_shift_max < real_target * 0.5
            if worst_shift_max < real_target and (step >= min_steps or comfortably_clear):
                print(f"  converged (shift-robust, real-verified) at step {step}")
                break

    if best_delta is None:
        best_delta = delta.detach().clone()
        print("  WARNING: shift-robust real verification never confirmed full convergence, using best available")
    else:
        print(f"  best worst-shift real score achieved during search: {best_worst_shift_max:.4f}")

    # BUG FIX (adversarial review, verified directly): best_delta was
    # selected and worst-shift-VALIDATED above, then MUTATED by the silence
    # guard below with no re-check - so a delta that passed certification
    # could ship with samples zeroed out afterward in a region that had been
    # load-bearing for that passing score, and nothing would ever know. This
    # is the exact "looked converged, wasn't" failure class EOT itself was
    # built to fix, reintroduced one step later. Re-run the same worst-shift
    # scan on the POST-guard delta before returning, so the number this
    # function reports (and that cnn_fix.py uses to decide whether to accept
    # the result) reflects what's actually being shipped, not a pre-mutation
    # snapshot.
    guarded_delta = apply_silence_guard_to_delta(best_delta, audio)
    with torch.no_grad():
        guarded_perturbed_np = (audio + guarded_delta).numpy()
    # BUG FIX (adversarial audit, verified directly): this is the FINAL
    # certification value returned to cnn_fix.py as pre_transfer_worst -
    # the last line of defense before a delta gets shipped. A None from
    # _worst_shift_score (no valid in-bounds window) must fail closed here
    # above anywhere else in this file: treat it as the worst possible
    # score (1.0), never silently drop it from the max() via a filter that
    # could let an all-None scan report a false "clean" default.
    post_guard_results = [
        s if s is not None else 1.0
        for s in (
            _worst_shift_score(guarded_perturbed_np, pos, seg_len, n,
                                shift_range_sec=eot_jitter_sec, shift_step_sec=0.1, sr=SR)
            for pos in positions
        )
    ]
    post_guard_worst = max(post_guard_results, default=1.0)
    print(f"  post-silence-guard worst-shift real score: {post_guard_worst:.4f} "
          f"(pre-guard was {best_worst_shift_max:.4f})")
    if post_guard_worst > best_worst_shift_max + 1e-6:
        print("  WARNING: silence guard degraded the certified score - "
              "this is exactly why this re-check exists")

    return guarded_delta.numpy(), positions, seg_len, post_guard_worst
