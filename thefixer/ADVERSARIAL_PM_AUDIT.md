# Adversarial PM Audit — The Fixer

## Executive verdict

The current design is not one slow optimizer with a clean guarantee. It is
three different contracts conflated in comments and UI:

1. the deployed detector contract: median of five deterministic 10-second
   windows;
2. EOT's contract: every one of five windows must pass every independently
   shifted start in a ±0.5-second neighborhood;
3. thorough mode's contract: sampled coverage of nearly every 10-second window
   in the track, but only on a 0.5-second grid and without between-grid
   robustness.

EOT is therefore not a cheaper implementation of thorough's guarantee. It
solves a different, much narrower location-robustness problem. Thorough catches
one specific thing EOT does not: a detector choosing a window start outside the
five ±0.5-second neighborhoods (for example, a different `n_segments`, a later
leading trim, or a different deployment policy). None of those happens in the
current ground-truth `CNNDetector.predict` path. Thorough should be removed
from the product/UI contract and retained, if desired, only as an offline
stress-test. It is dead weight for the known deployed detector, but not because
EOT mathematically supersedes whole-track coverage.

The bigger finding is that EOT over-solves the deployed contract. A real
alignment drift is one common shift applied to all five positions, and the
detector takes their median. The robust condition is:

`max_shift median(position_scores_at_that_same_shift) < threshold`

The code instead approximates:

`max_position max_independent_shift position_score < threshold`

That requires all five positions to pass all independent shifts, when the real
verdict needs only three positions to pass under one common shift. This is a
genuine redesign opportunity, not a constant tweak.

The advertised EOT guarantee is also not preserved to the delivered audio.
`optimize_eot_verified` validates a candidate, then changes it with the silence
guard; `fix_cnn` then resamples/transfers it and checks only exact positions,
not the worst-shift scan. A candidate may also be returned when it never passed
`real_target`. The first priority is to define and verify one end-to-end
contract after every mutation.

## What I verified

### Ground truth and stale claims

- `app/detector.py:98-100` truncates the linear model to the first 300 seconds.
  The prompt's “whole track” description is false for tracks over five minutes.
- `models/config.yaml:14` sets both models' native rate to 16 kHz.
- `app/detector.py:170-202` chooses the CNN positions, and
  `app/detector.py:215` aggregates them by median.
- `app/server.py:370-375` currently runs CNN before linear. The prompt correctly
  asks the reader to verify this; any “linear runs first” premise is stale.
- At the current 0.5-second hop, `build_sliding_windows` produced 532 windows
  on the supplied 276.93-second fixture, not “60-100+.” EOT uses 30 surrogate
  evaluations per step. Thorough therefore has 17.7 times as many window
  evaluations per step on that file. On the 30-second fixture, thorough has
  only 38 windows versus EOT's 30, so the “order-of-magnitude cheaper” claim is
  only true for long tracks.
- The EOT docstring at `app/cnn_wholetrack_optimizer_v2.py:405-408` says
  convergence scans ±1 second, but the call at `:488-490` passes ±0.5 second.
- `eot_samples=6` at `:359` has no sensitivity result or derivation in the
  repository. It is a tuning guess.
- `fix_cnn` passes `min_steps=100` by default (`app/cnn_fix.py:30`), overriding
  `optimize_eot_verified`'s apparent 60-step default.

### Measurements on this machine

These are targeted microbenchmarks, not end-to-end production timing:

| Measurement | 30.00 s fixture | 276.93 s fixture |
|---|---:|---:|
| One current 30-sample EOT step | 0.963 s | 2.179 s |
| 30 surrogate forward graph builds | 0.323 s | 0.316 s |
| Three penalty forward computations | 0.056 s | 0.430 s |
| Combined backward | 0.582 s | 1.423 s |
| Active EOT union | 21 s (70.0%) | 55 s (19.9%) |

A warm real librosa+ONNX segment score took a median 43.1 ms. The 11 shifts ×
five positions scan therefore costs about 2.37 seconds per check. With
`real_check_interval=10` and a 100-step minimum, the run pays about 23.7
seconds for ten dense scans before it is even allowed to stop.

A differentiable 10-second forward+backward took a median 20.4 ms in isolation.
The full model has a 24×313 cepstral input and about 533 million convolutional
MACs per window. Its two fully connected layers have only 8,256 MACs. The prior
claim that fully connected layers are the CNN compute bottleneck is inconsistent
with the actual graph by roughly five orders of magnitude. This does not
invalidate the measured result that batching was slower on this CPU; it does
invalidate the explanation attached to that result. I do not recommend
reintroducing batching without a profiler showing why the measured batched
path regressed.

On the supplied “fixed” 30-second fixture, a shared shift scan still moved the
real median from approximately 0.00001 to 0.99988 over ±0.5 seconds. That
confirms the positional brittleness is real; it does not establish that six
random shifts per position per step is the minimum-cost remedy.

## Correctness blockers before speed tuning

### 1. EOT certification is invalidated after certification

Locations:

- `app/cnn_wholetrack_optimizer_v2.py:508-522`
- `app/cnn_fix.py:130-160`

The optimizer stores `best_delta` using a pre-guard worst-shift scan, then
applies `apply_silence_guard_to_delta`. `fix_cnn` resamples and mixes the
changed delta, then verifies only the exact stored positions. It never reruns
the ±0.5-second scan on the transferred output. It also returns
`"applied": True` even when `verified_after_transfer` is false.

Required correction: define a single certification function over the actual
post-guard, post-resample, post-stereo-transfer PCM. Only a candidate that
passes that function may be returned as verified. For the deployed contract,
that function should use the median under common shifts, not max over every
position. The final saved/encoded file should receive the same check if lossy
output is in scope.

Risk: more candidates will be reported as failures until the optimizer is
aligned to the real contract. That is exposure of existing failure, not a new
regression.

### 2. The linear “adaptive extension” does not work with `max_steps=225`

Location: `app/linear_gradient_optimizer.py:178-203`.

Real checks occur only at multiples of 50. With `max_steps=225`, the last
periodic check is step 200, where `step + 1 >= max_steps` is false. The loop
then ends at 225 without reaching another periodic check, so the claimed
extension never triggers. The safety argument in `app/linear_fix.py:50-60` is
false for the current numbers. The mutable `max_steps` is also used in its own
extension bound, so the intended maximum is not expressed cleanly.

Required correction: keep an immutable `initial_max_steps`; always do a real
check at the cap; extend from that check when necessary; bound against an
explicit `absolute_max_steps`.

### 3. The linear retry target moves in the wrong direction

Locations: `app/linear_fix.py:109-110` and `:193-215`.

The first target is `max(0.00005, 0.01/270) = 0.00005`. After a failed first
attempt, line 214 computes `max(0.002, 0.00005*0.3) = 0.002`: a 40-times
looser target, despite logging that the retry is stricter. The 0.002 “floor”
is stale relative to the new 0.00005 initial target.

The two cited transfer examples are more naturally described as logit shifts:
about +5.59 and +5.65. Applying the worse observed shift to a 1% acceptance
boundary requires an internal probability near 0.0000355. The hard floor
0.00005 is already too loose by the code's own two examples.

### 4. Short tracks duplicate the same CNN position five times

Locations: `app/cnn_differentiable_v2.py:88-104` and
`app/cnn_wholetrack_optimizer_v2.py:409-446`.

After a sub-10-second track is padded to exactly 10 seconds, the five computed
positions are all zero. EOT then evaluates the identical segment 30 times per
step. Deduplicating positions gives an immediate five-times window-evaluation
win on these files and matches the real detector, which emits one padded
segment in `app/detector.py:181-187`.

## Ranked recommendations

Ranking is estimated win × confidence ÷ implementation/correctness risk.

### 1. Hoist all CNN original-only regularizer state

Locations:

- `_silence_guard`, `app/cnn_gradient_optimizer_v2.py:29-59`
- `perceptual_penalty`, `:62-94`
- call site, `app/cnn_wholetrack_optimizer_v2.py:456-458`

Mechanism: the silence gate, original-audio STFT, masking multiplier, Hann
windows, frequency bins, and out-of-band masks do not change during a run.
The code recomputes them every step. Mirror the already-correct linear
`compute_masking_mult` hoist.

Estimated win: 8-12% of a long-track EOT step on the measured fixture, or
roughly 20 seconds over 100 steps. Smaller on short tracks.

Risk: low. Verify scalar penalties and delta gradients before/after are equal
within float tolerance on silence, loud music, and mixed-level fixtures.

Confidence: high; I read the dependency graph and separately measured about
0.2 seconds for the original 276.93-second STFT/mask work.

### 2. Parameterize only the EOT-active union and regularize only that union

Locations:

- delta construction at `app/cnn_wholetrack_optimizer_v2.py:421-423`
- penalty calls at `:456-458`

Mechanism: on a long track the detector loss touches only five 11-second
regions (10-second window plus ±0.5-second start range). The measured fixture's
active union is 55 seconds, 19.9% of the track. Detector gradients outside it
are exactly zero, and a zero-initialized CNN delta remains zero there. Store
only active-region parameters, scatter them into the five model segments, and
compute the worst-chunk penalties over active regions with enough STFT padding.
Align penalty chunk boundaries to the original global grid so this does not
silently change the regularizer definition.

Estimated win: 35-45% end-to-end per step on 3-5 minute tracks; little benefit
on short tracks whose regions overlap. My measured long-track step rose from
0.963 seconds at 30 seconds to 2.179 seconds at 276.93 seconds almost entirely
because the regularizers scale with full duration.

Risk: medium. A careless crop can miss STFT boundary energy or change which
0.7-second chunk is the maximum. Test penalty values/gradients against the
full-track implementation for random deltas supported in the union, then run
post-transfer real scans and objective audio metrics.

Confidence: high on avoided work and approximate win; medium on exact
implementation parity.

### 3. Replace random-average EOT with common-shift, median constraint generation

Locations: the nested loop and convergence logic at
`app/cnn_wholetrack_optimizer_v2.py:434-514`.

Mechanism:

1. Use common shifts across all five positions, because encoding/resampling
   creates a common alignment transform.
2. Optimize the third order statistic (the median condition), rather than
   forcing all five positions below target.
3. Maintain a small active set of failing shifts. At each real validation,
   add the current worst shared shift; optimize only the three positions that
   control the median at those active shifts. Remove constraints only after a
   full scan certifies them.

This is a cutting-plane/minimax formulation. It spends gradients on the
currently violated real constraints instead of averaging 30 randomly selected
constraints, many of which may already be irrelevant to the median.

Estimated win: 2-5 times in CNN gradient work if the active set stays at one to
three shared shifts and three controlling positions (3-9 forwards versus 30).
Combined with active-region penalties, a 2-times overall EOT speedup is
plausible. This is not yet verified by a convergence sweep.

Risk: medium-high. The active median positions can switch, and the surrogate
can disagree with the real model. Keep real-model constraint generation,
piecewise-differentiable `kthvalue`/top-k handling, and mandatory final dense
certification. Compare success rate, worst shared-shift median, SNR, and
wall-clock time on a held-out corpus.

Confidence: high that the current objective over-solves the real contract;
medium that constraint generation will converge in fewer total steps.

### 4. Make real verification fail-fast and adaptive

Locations:

- `_worst_shift_score`, `app/cnn_wholetrack_optimizer_v2.py:326-346`
- check loop, `:480-514`

Mechanism: early in training, a position usually fails on the first or second
shift. Window weights only use pass/fail, so scanning the remaining shifts to
compute an exact maximum adds no control information. Stop a position scan as
soon as it exceeds `real_target`. Run cheap exact-position or coarse checks for
guidance, and run the complete 0.1-second grid only when a candidate appears
to pass, at `min_steps`, and after every output mutation. Only a complete scan
may mark `best_delta` certified.

Estimated win: worst-case unchanged near convergence; early verifier cost can
fall from 55 calls to about five, from 2.37 seconds to roughly 0.22 seconds on
this machine. Over the 100-step minimum, likely 5-10% total runtime.

Risk: low if incomplete scans are never used for certification or exact
“best” ranking. Coarse scans alone cannot guarantee safety because the measured
score spikes are narrow.

Confidence: high.

### 5. Replace six random samples with a deterministic stratified schedule

Location: `app/cnn_wholetrack_optimizer_v2.py:432-451`.

Mechanism: six has no empirical support. With 100 minimum steps it draws 600
continuous offsets per position. A rotating schedule of two or three offsets
per step can cover the 11 certification-grid offsets repeatedly, always
including or frequently revisiting the endpoints. This reduces gradient
variance relative to unstratified random samples and makes experiments
reproducible.

Estimated win: reducing six to three cuts the surrogate portion approximately
in half. On the current measured long-track step that is about 20-25% overall
before active-region optimization, potentially 30-40% after regularizers are
trimmed.

Risk: medium. More steps could erase the per-step gain. Run a factorial sweep
over 2/3/4/6 samples and schedules on at least 30 files, measuring time to
post-transfer certification, not steps to surrogate loss.

Confidence: high that six is unjustified; medium-low that three is the final
optimum without the sweep.

### 6. Add content-addressed stage caching, not just feature caching

Locations:

- random upload identity, `app/server.py:216-242`
- repeated analysis, `:262-280`
- pipeline entry, `:394-408`

Mechanism: `file_id` is random and there is no derived-artifact cache. Key each
stage output by canonical input-PCM hash, ordered upstream tools and options,
model/config hashes, and code/cache schema version. An identical rerun can
reuse the verified stage output and certification record. Cache initial
fakeprint/five cepstra for repeated analysis calls, but do not expect untouched
original features to accelerate gradient steps: every step changes the audio.

Estimated win: near-total elimination of detector-fix time on an exact rerun;
small benefit when upstream choices change the PCM. This is the largest
conditional user-visible win.

Risk: medium. A stale key would reuse an invalid adversarial correction. Include
every upstream PCM-affecting option and model/code version, and always
re-certify the final transferred/saved artifact.

Confidence: high for exact reruns.

### 7. Prevent cross-model regression; do not reuse one model's delta

Locations:

- order, `app/server.py:370-375`
- linear/CNN rework loop, `:520-658`

The CNN runs first, so the linear optimizer already starts from CNN-corrected
audio. Reapplying or seeding with the CNN delta would double a perturbation
whose direction has no principled relation to the linear fakeprint objective.
The representations, spatial support, aggregation, and loss geometry differ.
There is no defensible shared fakeprint/CQT cache after CNN has modified the
input.

The defensible transfer is a preservation constraint: during the later linear
optimization, periodically include the CNN's deployed median loss (or project
the linear update away from a small set of active CNN gradients). This may cost
a few CNN forwards but can avoid a complete multi-minute CNN rerun.

Estimated win: zero on runs where the models do not interfere; potentially
minutes when it prevents the re-run at `app/server.py:610-618`.

Risk: high. Gradient conflict may make the linear target harder. Test frequency
of re-runs, both final real scores, and SNR against the current sequential
baseline.

Confidence: high that raw delta reuse is a dead end; medium-low on the net win
from a joint preservation loss until measured.

## Optimizer judgment

### L-BFGS: reject

The prompt's “small number of free parameters” premise is wrong. The current
delta has one parameter per 16 kHz sample: 4,430,880 parameters on the measured
276.93-second track. Adam is memory-heavy but predictable. L-BFGS adds large
history vectors and invokes the full closure multiple times per outer step.
With non-smooth ReLUs, max-pooling, worst-chunk maxima, and tonality maxima, the
objective is neither smooth nor near-convex. L-BFGS is a poor fit and is more
likely to increase memory and wall time than reduce them.

### One cached local Jacobian: reject as the main solver

The CQT is followed by magnitude, log, ReLU, three max-pools, global pooling,
and another ReLU. Small adversarial changes can switch max-pool winners and
move low-magnitude bins sharply through `log(abs(.))`. The repository already
documents that a surrogate matching the baseline diverges after optimization;
that is direct evidence against trusting one local linearization over the full
move. An explicit 30×N Jacobian is also enormous, and current backprop already
computes the vector-Jacobian products needed for an update.

A trust-region linearized first step could be tested as a warm-start, accepted
only if the real constraint scan improves, but I estimate at most a modest
iteration reduction with low confidence. It does not outrank removing the
wrong constraints and full-track penalty work.

### Precomputing the untouched CQT does not remove the changing transform

CQT is linear before magnitude, so in principle
`CQT(original + delta) = CQT(original) + CQT(delta)`. But the current path
computes one CQT of the sum. Splitting it still requires a CQT of every changing
delta segment; it does not eliminate the transform. Sharing a whole-track CQT
across windows changes segment-local padding and frame/pooling alignment, the
very phase behavior implicated in the failure. This is not a safe high-priority
optimization.

## Thresholds and evidence required

The CNN comment at `app/cnn_wholetrack_optimizer_v2.py:75-83` is arithmetically
wrong: 0.01 is 50 times below 0.5, while 0.08 is only 6.25 times below 0.5.
More importantly, the linear 1% value is the product acceptance requirement,
not an empirically calibrated transfer margin. Copying its ratio is not a
principle.

Use additive logit degradation, not raw probability multipliers:

`transfer_shift = logit(post_transfer_score) - logit(pre_transfer_score)`

Across a representative corpus and each actual output path (WAV, MP3, FLAC,
M4A; relevant sample-rate ratios), estimate a high quantile of this shift.
Set the internal target from:

`logit(internal_target) = logit(delivery_threshold) - q99(transfer_shift) - reserve`

For CNN, measure the shift of the actual certification statistic (shared-shift
median), not a single segment or max over all segments. Use a tuning set and a
held-out validation set; at least 30 diverse files is a minimum sensitivity
study, not a claim of production-level confidence.

The two linear observations from one file are surprisingly consistent in
logit space, but they estimate no cross-file variance. The one CNN regression
incident supplies no distribution at all. Both current margins are anecdotal.
A smaller margin may let many files stop earlier, but the repository contains
insufficient data to estimate how much or to claim it is safe.

## Recommended decision

1. Remove thorough from the normal UI and document it as an offline
   alternative-deployment stress test, not “safer EOT.”
2. Fix certification so it runs after silence guarding, transfer, and final
   encoding, using one declared statistic.
3. Fix the linear cap/retry bugs before trusting any timing or threshold
   experiment.
4. Implement the exact low-risk CNN hoists and active-region regularizers.
5. Prototype common-shift median constraint generation against the existing
   EOT baseline.
6. Run the missing corpus sweep before changing `eot_samples` or either safety
   margin.

My honest conclusion is not that EOT's present settings are “safe enough.”
There is no evidence for six samples or 0.08, and the validated delta is changed
after validation. Nor is thorough's cost irreducible: on the current detector
it pays to solve hundreds of constraints the verdict never evaluates. The best
path is to align the optimization and certification to the detector's actual
order statistic, then spend compute only on constraints that can control that
statistic.
