# Adversarial PM Audit — The Fixer (Codex 5.6)

## Your role

You are an adversarial staff engineer / PM doing a hostile design review, not a
cheerleader. Assume the current implementation is wrong until proven otherwise
by reading the actual code and doing the actual math. Do not accept any claim
in this prompt (including timing numbers and thresholds) as ground truth —
verify them against the code, and call out anywhere this prompt itself is
stale or wrong.

The team has been iterating on this feature for a long session, fixing bug
after bug reactively. That process produces working-but-encrusted code:
patches on patches, magic constants justified by one anecdote, defaults tuned
against a single test file. You are here specifically to find what a from-
scratch redesign would do differently, not to nitpick the patches. Do not
just validate the existing approach — your job is to find a genuinely faster
way to solve the same problem, or to explain with numbers why one doesn't
exist.

**Think out of the box.** The obvious moves (reduce steps, cache STFTs,
batch forward passes) have mostly already been tried — see "Already
investigated" below. We need genuinely different angles: different
optimization formulations, different loss landscapes, precomputation
strategies, warm-starts, model-specific shortcuts, or a case for why
"thorough" mode's cost is actually irreducible given the guarantee it's
providing.

## The app, in one paragraph

"The Fixer" (`/Users/daniel/Desktop/audio/thefixer/`) takes an audio file
that a pretrained AI-music detector flags as AI-generated, and computes a
minimal, perceptually-hidden waveform perturbation that flips the detector's
verdict to "human," while also doing legitimate mastering/cleanup (silence
trim, DC offset, transient repair, spectral fill-in, loudness/true-peak
limiting). There are two independent pretrained detector models being
targeted — a fast linear model and a slower CNN — each fixed via a separate
gradient-based adversarial-optimization module. This audit is about whether
those two optimization loops can be made meaningfully faster without losing
the correctness guarantees they were built (through painful trial and error)
to have.

## Files to read, in this order

1. `app/detector.py` — the REAL (non-differentiable) scoring pipeline for
   both models. This is ground truth for what "passing" means. Note
   `LinearDetector.compute_fakeprint` (time-averaged spectral residual over
   the WHOLE track, 1-8kHz) vs `CNNDetector.extract_segments`/`predict`
   (exactly 5 fixed 10-second windows, evenly spaced, skipping first/last 5s,
   aggregated by MEDIAN — meaning only 3 of 5 windows need to pass).
2. `models/config.yaml` — confirms the model's native rate is 16kHz
   (`sample_rate: 16000  # matches SONICS dataset`) — both fix pipelines
   downsample to 16kHz for scoring/optimizing regardless of the source
   file's rate, and never touch the delivered file's own sample rate.
3. `app/linear_differentiable.py` and `app/linear_gradient_optimizer.py` —
   the linear model's differentiable surrogate and its Adam-based waveform
   optimizer (`optimize()`).
4. `app/linear_fix.py` — orchestrates `optimize()`: retry loop, transfer-loss
   compensation, real-model re-verification.
5. `app/cnn_differentiable_v2.py` and `app/cnn_gradient_optimizer_v2.py` —
   the CNN model's differentiable surrogate (CQT + cepstrum reimplemented in
   torch) and its shared loss-penalty helpers.
6. `app/cnn_wholetrack_optimizer_v2.py` — the CNN optimizer, THE most
   important file for this audit. Contains three code paths:
   - `optimize_whole_track_verified(..., mode="thorough")`: dense sliding
     windows every `hop_sec=0.5s` across the WHOLE track (can be 60-100+
     overlapping 10s windows on a 3-5 min track).
   - `optimize_whole_track_verified(..., mode="simple")`: optimizes ONLY the
     real detector's exact 5 evaluation positions, no shift-robustness.
     Fast but was found to be brittle (see "Already investigated" below).
   - `optimize_eot_verified(...)`: optimizes the same 5 positions, but
     trains against `eot_samples=6` random ±`eot_jitter_sec=0.5s` shifts per
     position per step (Expectation-over-Transformation / EOT), to make the
     fix robust rather than a brittle point solution. This is the current
     default (`cnn_fix.py`'s `mode="eot"`).
7. `app/cnn_fix.py` — orchestrates the CNN optimizer call, resample transfer,
   final re-verification.
8. `app/server.py` — search for `TOOL_ORDER`, `run_pipeline`, `cnn_mode`,
   and the CNN/linear re-verification blocks to see how these fixes compose
   into the full processing chain (order matters: CNN runs before linear,
   and there's a mandatory unconditional CNN re-check after linear + any
   later stage, because later stages were found to silently un-fix CNN).

## The core question

**"Thorough" mode is still offered because it's believed to be the safest,
but it can take 10+ minutes on a single track.** EOT mode is faster and
already targets the actual robustness problem (shift-invariance) rather than
brute-force coverage — but is it actually as safe as thorough, or just
"safe enough on the one file we tested it on"? And regardless of the answer
to that: is there a fundamentally different, faster way to get either mode's
guarantee?

Concretely, answer:

1. **Is dense whole-track tiling ("thorough") solving a real problem, or a
   symptom?** The code's own comments claim EOT supersedes it by directly
   targeting shift-robustness instead of brute-force covering the track. If
   that's true, is "thorough" mode now dead weight that should be removed
   entirely (one less code path, one less UI option, one less thing to
   maintain) rather than kept "for comparison"? If it's NOT true — if there's
   a failure mode thorough catches that EOT provably doesn't — name it
   specifically with a mechanism, not a vague "more coverage is safer."

2. **Can EOT itself be made faster without weakening its guarantee?**
   Specifically look at:
   - `eot_samples=6` — is 6 shift-samples per position per step
     empirically necessary, or was it picked without a sensitivity sweep?
     Could fewer shift samples per step (e.g. 2-3) with more total steps, or
     a schedule (start narrow, widen jitter range as loss drops), reach the
     same worst-shift-score guarantee for less total compute? The real cost
     driver is `n_segments x eot_samples` forward/backward passes per step
     (30 today) — read the loop in `optimize_eot_verified` and quantify.
   - `real_check_interval=10` and the `_worst_shift_score` scan
     (`shift_step_sec=0.1` across `±eot_jitter_sec`) — this is a real
     (non-differentiable) call to the actual ONNX/librosa detector for EVERY
     position at EVERY check, at 0.1s resolution across a 1s range (~10-11
     calls per position, x5 positions = 50-55 real-model calls every 10
     steps). Is this scan resolution/frequency defensible, or could it be
     made adaptive (coarse scan first, only fine-scan positions near the
     threshold) for a real speedup with no loss of guarantee?
   - The per-step loss also computes `perceptual_penalty`, `band_limit_penalty`,
     and `tonality_penalty` on the FULL delta every step (see
     `cnn_gradient_optimizer_v2.py`) even though only a handful of 10s windows
     around 5 positions are being optimized — is full-track penalty
     computation on every step wasteful given the delta is heavily
     concentrated near just 5 regions? Could these penalties be computed only
     over the active regions (± padding) instead of the whole track?

3. **Warm-starting / transfer between the two models.** Linear runs first (or
   does it — check current `TOOL_ORDER`) and both models are being fixed on
   the SAME underlying audio. Is there any reusable structure — shared
   spectral-domain analysis, a shared "what does this file's fakeprint/CQT
   signature look like" precompute — that could inform or seed either
   optimizer's starting point, rather than each running as a fully
   independent cold-start Adam optimization from a zero-init delta every
   time? Be skeptical here: the two models operate on different feature
   representations (fakeprint spectral residual vs CQT-cepstrum), so a naive
   "reuse the delta" transfer may not be principled — evaluate whether ANY
   transfer is defensible or whether this is a dead end, and say so plainly
   if it is.

4. **Is Adam + per-step full backward pass the right optimizer at all for
   this problem shape?** This is a small number of free parameters relative
   to typical deep learning (a single time-domain delta vector), targeting a
   small number of fixed evaluation points through a frozen pretrained
   network. Would something like L-BFGS (better for this size of smooth,
   near-convex problem, typically far fewer iterations to converge) be worth
   testing here instead of first-order Adam? What about precomputing the
   frozen CQT/CNN's local Jacobian/linearization around the original audio
   once, and solving a much cheaper local linear (or few-Newton-step)
   problem instead of hundreds of full nonlinear forward/backward passes —
   would that hold up given the model's actual nonlinearity, or break down
   too fast to be useful? Give a real judgment, not just "worth trying."

5. **Precompute once per file, not per mode/per run.** If a user runs the
   tool once, gets a result, and re-runs with different tool selections
   (common in this app's actual usage pattern — the user iterates a lot),
   is anything being unnecessarily recomputed from scratch across runs on
   the SAME source file (e.g. the CQT-cepstrum of the untouched original
   audio, or its fakeprint) that could be cached keyed by file hash?

6. **Question the guarantees themselves, not just the code that provides
   them.** `real_target=0.08` for CNN and `real_target≈0.008` for linear
   (`ACCEPT_THRESHOLD=0.01` in `linear_fix.py`) were picked based on specific
   observed regression incidents this session (documented in the codebase's
   comments — search for "TRANSFER_LOSS_MULTIPLIER" and "0.08 mirrors that
   same margin-over-boundary ratio"), not from first-principles analysis. Is
   the margin-over-boundary logic sound, or is it cargo-culted from one or
   two anecdotes? Would a smaller, principled margin (backed by measuring
   the actual variance of the transfer-loss multiplier across many files,
   not the two data points currently cited) let the optimizer stop earlier
   on most files while still being safe?

## Already investigated — do not re-recommend without new evidence

A previous review (internal, not this one) already checked these and found
they do NOT help; if you want to re-raise one, you must show why the earlier
finding was wrong, not just re-propose it:

- **Batching CNN forward passes** (stacking multiple windows/positions into
  one batched call): measured only ~4% overall gain; CNN-forward-only was
  actually SLOWER when batched (102ms→161ms) because the bottleneck is the
  CNN's fully-connected layers on CPU, not the CQT transform (which does
  batch ~1.6x but is only ~15-20% of per-step cost).
- **Apple Silicon MPS GPU backend**: hung for 90+ seconds on first real
  operation in a direct test on this machine — not viable without much
  deeper work (likely specific ops in the CQT/cepstrum path lacking MPS
  kernels, falling back to slow CPU paths or worse).
- **"Simple" mode (5 fixed positions, no shift-robustness) as the default**:
  already implemented and available, but confirmed unreliable — the real
  detector's score on simple-mode-corrected audio was measured to swing from
  0% to 99.99%+ within a quarter-second of positional shift, because the
  optimizer finds a minimal-norm adversarial perturbation that only works at
  the exact position it was optimized against (a textbook non-robust
  adversarial example). This is WHY EOT mode exists. Don't recommend
  "just use simple mode, it's faster" without addressing this directly —
  if you think the shift-robustness problem can be solved a different,
  cheaper way than EOT, that's exactly the kind of idea this audit wants.
- **Position drift from resample/encode rounding**: directly tested
  (`_resample_mono` output vs a full WAV-save-then-ffmpeg-redecode
  round-trip) — zero drift found on the tested file/path. Not the dominant
  cause of anything currently. Don't spend much time here unless you find a
  path (different codec, different sample rate ratio) where it actually
  matters.
- **Lowering `max_steps` alone / caching the linear model's masking-STFT**:
  already done for the linear optimizer (`max_steps` 400→225,
  `compute_masking_mult` hoisted out of the per-step loop) — verified
  bit-exact and cut a real test run from ~300-400s to 53.3s. The CNN
  optimizer has NOT received equivalent treatment — this is exactly the kind
  of gap this audit should be looking for MORE of, applied to the CNN side.

## What "good" looks like in your report

Do not write generic advice ("consider caching", "profile the code"). For
every recommendation:

1. **Name the exact function/line** the change applies to.
2. **State the mechanism** — why this is actually faster, in terms of what
   computation is avoided or reordered, not just "should help."
3. **State the risk** — what correctness guarantee could this weaken, and
   how would you verify it wasn't (what test, what metric, on what file).
4. **Estimate the win** — rough order of magnitude (2x? 10%? only on long
   tracks?) and your confidence in that estimate.
5. **Flag your confidence explicitly** — separate "I verified this by reading
   the actual tensor shapes / doing the arithmetic" from "this is a
   plausible hypothesis I did not verify." Do not present a guess as a
   finding.

Rank recommendations by (estimated win) x (confidence) ÷ (implementation
risk), highest first. If your honest conclusion is that "thorough" mode's
cost is actually justified and nothing meaningfully faster exists without
giving up real safety margin, say that clearly and explain why — a report
that pushes back on the premise is more useful than one that invents weak
wins to look productive.
