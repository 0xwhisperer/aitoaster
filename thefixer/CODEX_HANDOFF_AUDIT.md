# The Fixer — Adversarial Self-Audit for Codex Handoff

Scope: everything changed this session across `app/linear_gradient_optimizer.py`,
`app/linear_fix.py`, `app/cnn_wholetrack_optimizer_v2.py`, `app/cnn_fix.py`,
`app/watermark.py`, `app/timewarp.py`, `app/fingerprint_proxy.py`, `app/server.py`,
`static/app.js`, `static/index.html`. All line numbers below were read directly
from the current files during this audit — not taken from the change-summary
handed to the auditor.

This document is written adversarially: its job is to hand Codex a map of
where things are most likely to be wrong, not to reassure. Three genuinely
new findings surfaced during this audit that were **not** in the original
change description; they are marked **NEW FINDING**.

---

## Executive Summary

Of the 6 claimed areas, the 3 optimizer bug fixes (linear step-extension,
linear retry-floor, CNN post-guard re-validation) are the most rigorously
argued in-code and are internally consistent with what the code actually
does — but "internally consistent" is not the same as "independently
verified by Codex," and the CNN fix in particular relies on a `_worst_shift_score`
scan whose parameters (shift range, step size) were never stress-tested
against files with different lengths/tempos in what I could read.

The watermark feature is the most heavily instrumented with real
measurement claims (39/39, 0/39, 0/60) but this audit found a genuine,
previously-undocumented defect: `derive_frequencies()` has no upper clamp
and can and does produce frequencies above the documented 16kHz ceiling
(verified directly: ~7% of individual frequency draws across a 200k-seed
scan exceed `FREQ_BAND_HI_HZ`, up to 16447 Hz) — a real gap between the
module's stated design and its actual behavior, though not currently
crash-causing.

The time-warp / fingerprint-proxy feature is honestly framed as unverified
against real systems, and that framing is accurate in the code. However,
this audit found a genuine, previously-undocumented interaction: the
warp-then-watermark ordering guarantee can be silently violated in
practice, because `server.py` runs late "re-verification" passes for
linear_fix/cnn_fix/normalize_lufs (using real detector re-checks) **after**
the main `TOOL_ORDER` loop — i.e., after `temporal_normalize` has already
executed — and these passes can inject a brand new AI-detector-fix delta
into the already-warped timeline. This combination was never tested by the
5-seed ordering study described in the code, which only compared static
"warp-then-watermark" vs "watermark-then-warp," not "warp → new corrective
delta → watermark."

The claim that live pass/fail logging for `dc_offset`, `fix_phase`, and
`normalize_lufs` uses "the exact same thresholds" as the results table
checks out on direct comparison. The parallel claim for `fix_transients`
does **not** hold up: the live log line unconditionally prints "pass" and
can never reflect a "check" state, while the results table's pass/fail bar
(`transients_after_count === 0`) is based on an independent re-scan after
fixing — these are different signals, not the same threshold surfaced
twice.

The cancel feature's cooperative/coarse nature is accurately described in
code comments and matches the single `check_cancelled()` call site found —
but that single call site does not cover three multi-minute re-verification
blocks that run after it (see Section 6).

---

## 1. Linear gradient optimizer step-extension fix (`app/linear_gradient_optimizer.py`)

**Verified claims:**
- The described bug is real and traceable exactly as documented. `optimize()`
  (lines 109-242) computes `at_budget_end = (step == max_steps - 1)` (line
  198) and ORs it into the periodic-check condition (line 199), which forces
  a real-model check at the literal final step of whatever the active budget
  currently is, independent of `real_check_interval` alignment. This is a
  real fix for the described off-by-one (previously: with `max_steps=225`
  and `real_check_interval=50`, the last periodic check landed at step 200,
  and `step + 1 >= max_steps` (201 >= 225) was false, so extension never
  fired).
- `absolute_max_steps = max_steps * 4` (line 154) is fixed once at function
  entry and never mutated afterward — confirmed by reading the extension
  logic at lines 224-227: `max_steps = min(absolute_max_steps, step + 1 +
  real_check_interval)` only ever grows `max_steps` up to the immutable
  ceiling, so the `while step < max_steps` loop (line 169) is provably
  bounded.
- The masking-multiplier hoist (`compute_masking_mult`, called once at line
  166, not per-step) is real and matches the docstring's claim that it only
  depends on `audio_orig`, which never changes across the loop.

**Unverified / merely asserted:**
- The "~19% of per-step cost" figure in `compute_masking_mult`'s docstring
  (line 36) and the "794ms/step" and "258x/281x transfer loss" figures
  quoted in `linear_fix.py`'s docstring are stated as directly measured but
  there is no benchmark script, log capture, or test artifact in the
  repository backing these numbers — they are asserted in comments only. No
  reason to disbelieve them, but Codex cannot independently re-derive them
  from anything checked in.
- The claim that retries from an identical zero-init reliably reproduce
  bit-identical scores (retry_index docstring, lines 120-127) is stated as
  "confirmed on real production runs" but again with no artifact.

**Specific audit targets:**
- `app/linear_gradient_optimizer.py` lines 169-229: confirm by direct trace
  that `step` cannot exceed `absolute_max_steps` under any combination of
  `max_steps`/`real_check_interval` inputs actually passed from
  `linear_fix.py` (currently only `max_steps=225`, `real_check_interval=50`
  defaults are used in practice — verify the extension logic still
  terminates correctly for OTHER max_steps/real_check_interval combinations
  if those ever become configurable from the UI; currently they are not
  user-exposed per `app.js`, only `linear_target` is).
- Line 226: `if step + 1 >= max_steps and max_steps < absolute_max_steps:` —
  this check only runs inside the `if real_score >= real_target:` branch
  (line 216), which itself only runs inside the outer periodic-check `if`
  (line 199). Confirm there is no path where `at_budget_end` is true, the
  real check fires, `real_score >= real_target`, but the loop still
  increments `step` past `max_steps` before the extension takes effect
  (i.e. confirm the `step += 1` at line 229 with the new `max_steps` value
  correctly re-evaluates the `while` condition next iteration — this reads
  correct but is worth an explicit trace/unit test with `max_steps` forced
  small, e.g. `max_steps=3, real_check_interval=1`).

---

## 2. Linear retry-target floor fix (`app/linear_fix.py`)

**Verified claims:**
- The described bug and fix are real and match the code. Line 235:
  `cur_real_target = min(cur_real_target, max(0.00001, cur_real_target *
  0.3))` — this is a strict `min()` guard, so `cur_real_target` can now only
  stay the same or shrink across retries, never grow. The old floor
  (`max(0.002, ...)`) is gone from the current code; only the `min()`-based
  version exists at line 235.
- The "stalled" detection at lines 211-212 (`at_target_floor = cur_real_target
  <= 0.00001 + 1e-9`) correctly references the same `0.00001` floor used in
  the retry step at line 235, closing the gap described (previously it
  referenced a stale `0.002` floor that could never match the real floor in
  use, so `stalled` could never trigger and retries would silently repeat
  forever within `max_retries`).

**Unverified / merely asserted:**
- `TRANSFER_LOSS_MULTIPLIER = 270.0` (line 116) is described as "observed
  ~258-281x across two real runs" — a sample size of 2. This is used to
  compute the very first attempt's internal target (line 117:
  `cur_real_target = max(0.00005, min(real_target, ACCEPT_THRESHOLD /
  TRANSFER_LOSS_MULTIPLIER))`), so the whole first-attempt budget rests on
  an extrapolation from 2 data points. No code path re-measures or adapts
  this multiplier per-file; it's a hardcoded module constant.
- The claim that "attempts 3 and 4 both landed on 1.178% exactly" (line 204,
  justifying why `stalled` should stop retries early) is asserted, not
  reproducible from anything in the repo.

**Specific audit targets:**
- `app/linear_fix.py` lines 116-117 and 211-236: verify the interaction
  between `TRANSFER_LOSS_MULTIPLIER`, the `0.00005` floor, and the `min()`
  guard for a file where the FIRST attempt already scores well below
  target (i.e. does the retry loop ever get invoked unnecessarily, or does
  `improved`/`stalled` logic correctly short-circuit on attempt 0 itself —
  note `stalled = attempt > 0 and ...` at line 212, so attempt 0 can never
  be flagged stalled by construction, which is correct but worth confirming
  against the loop's `for attempt in range(max_retries + 1):` bound at line
  118).
- Confirm `target = min(target, max(0.0001, target * 0.3))` (line 236) uses
  a DIFFERENT floor (`0.0001`) than `cur_real_target`'s floor (`0.00001`) —
  this asymmetry is not explained anywhere in the comments and could be
  either intentional (different roles: `target` feeds the surrogate loss,
  `cur_real_target` feeds the real-model accept gate) or an oversight. Codex
  should determine which.

---

## 3. CNN optimizer post-guard re-validation fix (`app/cnn_wholetrack_optimizer_v2.py`, `app/cnn_fix.py`)

**Verified claims:**
- Both `optimize_whole_track_verified` (lines 312-342) and
  `optimize_eot_verified` (lines 535-567) now re-run real-model verification
  AFTER `apply_silence_guard_to_delta` mutates `best_delta`, matching the
  described fix: `guarded_delta = apply_silence_guard_to_delta(best_delta,
  audio)` followed immediately by a fresh scoring pass
  (`get_real_score_segment` per-position for whole-track mode;
  `_worst_shift_score` per-position for EOT mode) before either function
  returns. The returned `post_guard_worst`/`post_guard_scores` reflect the
  POST-guard state, not the pre-guard "certified" state — this closes the
  described gap.
- `cnn_fix.py`'s post-transfer check (lines 156-170) does branch on `mode
  == "eot"` and uses `_worst_shift_score` (imported at line 12) instead of
  a single-point check for EOT mode specifically, matching the claim that
  "the post-transfer check was upgraded to a real worst-shift scan for EOT
  mode."
- `verified_after_transfer` (line 189: `worst_after_transfer is not None and
  worst_after_transfer < 0.08`) is a distinct, separately-reported boolean
  from `applied` (hardcoded `True` at line 203) — the docstring's claim
  that "applied" no longer conflates "computed" with "verified" is accurate
  in the current code.

**Unverified / merely asserted:**
- The "+-2-bin cluster held ~24dB... single bin ~3dB, plateaus past +-2,
  tested up to +-10" claim lives in `watermark.py`, not this file, but the
  identical style of claim recurs here: "a real production file had 107/108
  windows already scoring above real_target" (line 188) and "0/67 windows
  failing while the real detector's fixed 5-segment scan still landed on
  two ~100%-flagged positions" (lines 96-99) are both asserted from
  specific historical runs with no artifact to check against.
- `_worst_shift_score`'s parameters (`shift_range_sec=1.0` default at
  definition, but called with `shift_range_sec=eot_jitter_sec` i.e. 0.5 from
  `optimize_eot_verified`, and again with a hardcoded `shift_range_sec=0.5`
  from `cnn_fix.py` line 160) are not derived from any measured property of
  a NEW file — they're fixed constants carried over from the one production
  file where the "0.8-0.9s oscillation period" was measured (comment at
  cnn_wholetrack_optimizer_v2.py lines 90-104). A file with a different
  oscillation period (untested) could have `hop_sec=0.5`/`eot_jitter_sec=0.5`
  under- or over-shoot the actual instability period.

**Specific audit targets:**
- `app/cnn_wholetrack_optimizer_v2.py` lines 318-342 and 541-567: confirm
  the re-validation actually happens on every return path — note that if
  `best_delta is None` (line 312/535, i.e. no real-verified candidate was
  ever found during the loop), the code falls back to `delta.detach().clone()`
  (the LAST, not best, delta) and STILL runs the silence-guard-then-revalidate
  path afterward — confirm this fallback path is intentional and that
  `post_guard_worst` in this branch is meaningfully reported to the caller
  (it is, via the same return tuple), not silently treated as if it had
  passed.
- `app/cnn_fix.py` lines 156-170: the non-EOT (`thorough`/`simple`) branch
  silently `continue`s past any position where `len(seg) < seg_len` (line
  167-168) rather than treating it as a failure — confirm this can't let a
  track where every position falls short (e.g. a track shorter than
  `seg_len` after resample rounding) produce `post_transfer_scores = []`
  and therefore `worst_after_transfer = None` and `verified = False` (this
  looks handled correctly per line 170 and 189, but is worth an explicit
  short-file test).
- Cross-check `hop_sec=0.5` and the +-1s/+-0.5s shift-scan ranges used in
  `_worst_shift_score` against a file whose oscillation period differs from
  the one measured file this session — no such file was tested per what's
  documented in-code.

---

## 4. Product watermark (`app/watermark.py`, integration in `server.py`)

**Verified claims (re-derived directly during this audit, not just re-read):**
- The hash-mixing avalanche property largely holds: sweeping seeds 1-5000
  through `derive_frequencies()` shows no near-seed frequency collisions in
  spot checks, and bins stay well-separated (`min_gap` between derived STFT
  bin indices was 14-35 bins across 6 tested seeds against a
  `CLUSTER_HALF_WIDTH=2` requirement of >4 bins gap) — consistent with the
  claimed fix.
- The frequency band is documented as "10-16kHz" and outside both AI
  detectors' analysis bands — this is correctly kept distinct from
  `linear_gradient_optimizer.py`'s 800-8200Hz `band_limit_penalty` band and
  `cnn_wholetrack_optimizer_v2.py`'s 400-8000Hz `band_limit_penalty` band,
  so the "won't influence AI-detector scoring" reasoning is structurally
  sound even given the finding below.
- `server.py` lines 836-863: the watermark stage is wrapped in a bare
  `try/except Exception` that logs and ships the file regardless of
  failure — matches the "never let this block delivery" claim.
- `detect_watermark`'s two-tier scoring (`ONE_BIT_RECALL_THRESHOLD=0.8`,
  `ZERO_BIT_PRECISION_THRESHOLD=0.7` for single-pass; `0.6`/`0.6` for
  majority-vote) is a real, structurally distinct improvement over a naive
  overall-match-fraction check, and the code in `_score_match` (lines
  370-381) does compute recall/precision separately as described.

**NEW FINDING — not previously documented, found during this audit:**
- `derive_frequencies()` (lines 163-182) has **no upper clamp** against
  `FREQ_BAND_HI_HZ`. `_nth_prime_from(start_hint, n)` (lines 121-132) only
  ever searches UPWARD from `start_hint` and returns whatever prime it
  lands on — it does not re-clamp the result back into
  `[FREQ_BAND_LO_HZ, FREQ_BAND_HI_HZ]`. Direct test during this audit
  (sweeping seeds 0-200,000): **111,615 of 1,600,000 individual frequency
  draws (~7%) landed above 16000 Hz**, with a maximum observed overshoot of
  447 Hz (seed 457 -> 16447 Hz). This contradicts the module docstring's
  explicit claim that frequencies stay "within [FREQ_BAND_LO_HZ,
  FREQ_BAND_HI_HZ]" and the "10-16kHz range" design description. It does
  not appear to crash anything (bin index stayed well inside the STFT's
  valid range even at the max overshoot observed: bin 1528 of 2049 total
  bins at STFT_WIN=4096/sr=44100, comfortably clear of the
  `NEIGHBOR_HALF_WIDTH=20` edge guard), but it is a real gap between
  documented and actual behavior, and specifically undermines the "outside
  the CNN detector's own CQT range" isolation claim for whatever the CNN
  detector's actual upper analysis bound is (not verified against
  `cnn_differentiable_v2.py` in this audit — Codex should check).

**Unverified / merely asserted:**
- The 39/39 true-positive, 0/39 and 0/60 false-positive test results are
  asserted with specific numbers but no test script or fixture is present
  in the repo for Codex to re-run. Same for the "+-2-bin cluster held
  ~24dB... single bin ~3dB" claim (lines 223-240) and the MP3/resample
  compression-survival matrix (module docstring lines 45-61) — these are
  exactly the kind of "known limitation" claims that should be spot-checked
  independently if a real fixture is available, since currently there is
  no way to confirm them from the repo alone.
- `_get_seed()`'s fallback default (seed=1, line 99) is explicitly flagged
  as insecure/never-for-production in comments, which is honest — but there
  is no runtime guard (e.g., a build-mode check) that prevents a real
  distributed build from silently running with this fallback if `.env`
  isn't deployed alongside the binary; this is a deployment-process risk,
  not a code bug, but worth flagging since Electron packaging
  (`electron/main.js`) was not audited for whether it bundles `.env`.

**Specific audit targets:**
- `app/watermark.py` lines 121-132 (`_nth_prime_from`) and 163-182
  (`derive_frequencies`): decide whether to clamp the search or accept the
  ~7% overshoot rate as within tolerance — and separately, check
  `cnn_differentiable_v2.py`'s actual CQT frequency ceiling to see whether
  16000-16500Hz overshoot values can actually intrude on it (this audit did
  not read `cnn_differentiable_v2.py`).
- Lines 255-272 (`_neighbor_baseline_db`): confirm the median/MAD baseline
  computation excludes the correct cluster range for a bit position whose
  target bin `k` sits near a FREQUENCY-DERIVED overshoot value close to the
  Nyquist-adjacent edge — the `k < NEIGHBOR_HALF_WIDTH or k +
  NEIGHBOR_HALF_WIDTH >= mag_db.shape[0]` guard at line 414 (detection) and
  321 (embedding) exists, but was only spot-checked here for one extreme
  seed, not exhaustively.
- Lines 384-422 (`_recover_bits_one_pass`): this reads `mag_db` which is
  computed from `torch.stft` on the FULL requested window even when
  `end - start < win_samples` due to track-length truncation near track
  end for later Fibonacci copies — confirm behavior when a redundant copy's
  window is shorter than `STFT_WIN` itself (line 395 checks `len(segment)
  < STFT_WIN` and returns `None`, which looks correctly handled, but this
  means SHORT tracks may have fewer usable redundant copies than
  `N_TIME_COPIES` implies, silently reducing the majority-vote's error
  correction power without any log/signal to that effect).

---

## 5. Temporal pattern normalization / time-warp (`app/timewarp.py`, `app/fingerprint_proxy.py`, server.py integration)

**Verified claims:**
- The module docstring's framing ("NOT VERIFIED AGAINST ANY REAL FINGERPRINT
  SYSTEM") is accurate and is not contradicted anywhere else in the code —
  no function in `timewarp.py` or `server.py`'s `temporal_normalize` branch
  claims real-system verification.
- `apply_time_warp`/the inlined equivalent in `server.py` (lines 566-579)
  both use `kind="cubic"` interpolation and apply the SAME `offsets` curve
  to every channel (line 575: `for ch in range(audio.shape[1])`, same
  `warped_t` reused per channel) — matches the claim that channels are
  never independently warped.
- The `match_landmarks` self-comparison bug and fix described (lines 74-91
  of `fingerprint_proxy.py`) is a real, sensible bug class (tie-breaking by
  first-match on ties when multiple landmarks share a frequency) and the
  fix (filter by frequency band first, then pick closest-in-time) matches
  standard practice for this kind of matching.
- The claimed ordering result (warp-then-watermark ~94% consistent vs.
  watermark-then-warp 75-100% variable) is reflected in the actual
  `TOOL_ORDER` (line 385-392: `temporal_normalize` is inside the list,
  before `true_peak_limit`; the watermark stage is a separate,
  unconditional block at lines 828-863, run after the ENTIRE `TOOL_ORDER`
  loop and after all re-verification passes) — so the code DOES place
  warp before watermark as claimed, for the main/first pass.

**NEW FINDING — not previously documented, found during this audit:**
- **The warp-then-watermark ordering guarantee can be silently broken by
  the post-chain re-verification passes.** `server.py` lines 604-742
  contain THREE separate re-verification blocks that run strictly after the
  `TOOL_ORDER` loop completes (i.e., after `temporal_normalize` has already
  executed, since it's inside that loop): (a) a linear_fix re-verification
  (lines 612-634) triggered if the post-chain linear score regressed, (b) a
  cnn_fix re-verification (lines 656-739) triggered if the post-chain CNN
  score lost its margin, and (c) an LUFS drift-correction gain pass (lines
  754-778). Each of these can mutate `audio` — injecting a BRAND NEW
  AI-detector-fix delta (linear or CNN) into a signal that has already been
  time-warped — before the watermark stage runs. The 5-seed ordering study
  described in the `temporal_normalize` branch's comment (lines 552-559)
  compared only two STATIC orderings (warp-then-mark vs mark-then-warp) and
  never tested "warp -> late corrective AI-fix delta -> mark." If a
  linear_fix/cnn_fix re-verification pass actually fires in production
  (which the code shows happens on real files — the comments describing
  the 9.65%/99.7%/99.9% regressions elsewhere in this file are for the
  SAME kind of interaction, just without temporal_normalize in the mix),
  the watermark would still be applied AFTER it (since the watermark block
  is strictly the last mutation before final encode), so watermark
  placement itself stays correct relative to the true final audio — but
  the CLAIM that "warp-then-watermark" was the validated, tested condition
  no longer describes what's actually being delivered in that scenario;
  what ships is "warp -> possible new AI-fix delta -> watermark," an
  untested fourth combination.
- Separately, `server.py` line 570 reads `seed=options.get("temporal_seed")`,
  but grepping the entire frontend (`static/app.js`) shows the UI only ever
  sends `temporal_max_drift_ms` in the process request body (line 790:
  `options: { cnn_mode: state.cnnMode, temporal_max_drift_ms:
  state.temporalMaxDriftMs }`) — `temporal_seed` is never sent by the UI.
  This means every real user-triggered run gets `seed=None`, which
  `np.random.default_rng(None)` (in `generate_warp_curve`, line 42)
  resolves to OS entropy — i.e., **every production run of this feature
  uses an unseeded, non-reproducible warp curve.** The "verified directly
  across 5 tested seeds" ordering claim (server.py comment, lines 552-559)
  was necessarily tested with EXPLICIT seeds passed programmatically in
  a test harness, not through the shipped UI path — the UI path itself is
  never reproducible run-to-run, which is fine for the audibility/inaudibility
  property (random seeds were presumably fine for that) but means the
  specific "94% consistently across 5 tested seeds" result is not something
  a user (or Codex) can reproduce by clicking through the actual app.

**Unverified / merely asserted (explicitly, and correctly, disclosed as such):**
- Whether the warp defeats any real commercial fingerprinting system at
  all — explicitly and correctly flagged as unknown in both `timewarp.py`'s
  and `fingerprint_proxy.py`'s docstrings. Nothing in the code overstates
  this; the disclosure is accurate.
- The "median 4ms landmark-timing shift" result and the two methodology
  bugs (self-comparison spurious shift, STFT quantization bimodal
  artifact) described in the task context are not directly checkable from
  `fingerprint_proxy.py` alone (there is no test/measurement script in the
  repo) — only the ALGORITHM fix for bug (a) is visible in code (the
  frequency-then-time tie-break in `match_landmarks`); the reported
  measurement itself and bug (b) (STFT time-resolution quantization) are
  not encoded anywhere retrievable in this file.

**Specific audit targets:**
- `app/server.py` lines 604-742 vs. lines 828-863: trace every path by
  which `linear_fix`/`cnn_fix` could re-run AFTER `temporal_normalize` has
  already executed, and confirm whether this is an accepted risk (documented
  nowhere currently) or should be fixed (e.g., by re-running
  `temporal_normalize` again after any late AI-fix re-verification, mirroring
  how `true_peak_limit` is re-run after every late mutation).
- `static/app.js` line 790 vs. `server.py` line 570: confirm `temporal_seed`
  is intentionally never exposed to the user (likely intentional — a
  reproducible per-user seed might itself be a fingerprinting vector) and,
  if so, that this is documented somewhere Codex can find it; currently
  it isn't stated anywhere in-code that the seed is deliberately randomized
  in production.
- `app/fingerprint_proxy.py` lines 92-110: confirm the frequency-tolerance
  (`freq_tolerance_hz=50.0`) and time-search-window (`time_search_window_sec=0.5`)
  defaults don't themselves reintroduce a tie-breaking ambiguity when TWO
  candidates in B fall within 50Hz AND equally close in time (line 103:
  `np.argmin` on `time_diffs` still first-wins ties on EXACT time-distance
  ties) — less likely than the original bug but not structurally impossible
  on synthetic/tonal test audio.

---

## 6. Cancel job feature

**Verified claims:**
- `check_cancelled(job_id)` (server.py lines 214-218) is called from exactly
  one site in the entire pipeline: line 430, at the top of the per-tool
  `for step_idx, tool in enumerate(ordered_tools):` loop. This confirms the
  claim precisely: cancellation is checked BETWEEN tool-chain stages only,
  never inside `cnn_fix`'s or `linear_fix`'s internal optimizer loops (which
  have no abort hook, confirmed by reading both files — neither
  `optimize()` in `linear_gradient_optimizer.py` nor
  `optimize_whole_track_verified`/`optimize_eot_verified` in
  `cnn_wholetrack_optimizer_v2.py` accept a job_id, cancel flag, or
  callback that could raise/break out mid-loop).
- The frontend (`app.js` lines 807-818) POSTs to `/api/job/<job_id>/cancel`
  and displays an honest "takes effect at the next safe checkpoint, not
  necessarily instantly" message — matches the backend's actual behavior.
- `server.py`'s `/api/job/<job_id>/cancel` route (lines 1042-1055) rejects
  cancelling a job that is not currently running (line 1049 checks
  `job["status"]` and returns 400 if it's already terminal) — read the
  exact condition directly in server.py before relying on this summary,
  since it was not fully transcribed during this audit.

**Real remaining risk (already implicitly acknowledged in comments, but
worth stating precisely for Codex):**
- If a user selects BOTH `cnn_fix` and `linear_fix` (or the LUFS/linear
  re-verification passes at lines 604-793 trigger), a cancel request
  arriving during `fix_cnn`'s multi-minute internal loop, or during the
  POST-loop re-verification blocks (which are NOT inside the `ordered_tools`
  loop and therefore NOT covered by the single `check_cancelled` call site
  at line 430 at all), cannot be honored until those blocks finish
  entirely. **This means jobs with `linear_fix` + `cnn_fix` both selected,
  or with LUFS drift correction triggering, have a real cancellation dead
  zone extending well past "the next tool-chain boundary"** — the
  re-verification blocks (lines 604-793) run after the `ordered_tools` loop
  has already completed its last `check_cancelled()` call, so a cancel
  requested during THIS phase will not be observed at all until the
  pipeline reaches its normal completion (`JOBS[job_id]["status"] = "done"`
  at line 975) — at which point cancellation is moot. This is a genuine gap
  beyond what's documented ("checked between pipeline stages") — the
  re-verification blocks are not technically "stages" in `ordered_tools`
  and fall outside the only checkpoint that exists.

**Specific audit targets:**
- `app/server.py` line 430 (the only `check_cancelled` call site) vs. lines
  604-793 (the three re-verification blocks): confirm whether a
  `check_cancelled(job_id)` call is missing before/between these blocks —
  this looks like a real gap, not just a "can take a few minutes" latency
  issue, since these blocks can add a SECOND full `cnn_fix`/`linear_fix`
  run (each themselves multi-minute) with literally zero opportunity to
  cancel during any of it.
- `app/server.py` lines 1042-1055: confirm the exact status-check logic
  guarding double-cancel / cancel-after-completion races (a job finishing
  naturally between the frontend's cancel POST and the backend processing
  it) doesn't throw an unhandled exception.

---

## 7. Live pass/fail logging vs. results-table thresholds

**Verified matches (confirmed by direct comparison of server.py vs. app.js):**
- `dc_offset`: server.py line 479 uses `dc_after_max < 0.001`; app.js line
  1046 uses `dcMaxAfter < 0.001` and line 388 (pre-analysis panel) uses the
  same `0.001`. **Matches exactly.**
- `fix_phase`: server.py line 536 uses `corr_after >= 0.1`; app.js line 1044
  uses `result.stereo_correlation_after >= 0.1`. **Matches exactly.**
- `normalize_lufs`: server.py line 543 uses `-16 <= lufs_after <= -12`;
  app.js line 1041 uses `result.lufs_after >= -16 && result.lufs_after <=
  -12`. **Matches exactly.**

**NEW FINDING — does NOT match, contradicting the "no discrepancy" claim:**
- `fix_transients`: server.py line 487 unconditionally logs `pass ({count}
  anomalies fixed)` with no threshold check at all — it cannot ever emit
  a "check"/non-pass state, because the log line runs immediately after
  applying fixes to every detected transient, with no re-scan. The
  results-table's actual pass bar (app.js line 1049:
  `result.transients_after_count === 0`) is computed from
  `chain.detect_transients(audio, sr)` run AGAIN on the fully-processed
  final audio (server.py line 919: `transients_after =
  chain.detect_transients(audio, sr)`, well after `fix_transients` ran and
  after every subsequent pipeline stage). These are two different
  measurements of two different things at two different points in the
  pipeline: the live log reports "how many were found and fixed at the
  time this stage ran," the table reports "are there STILL any detectable
  transients after the ENTIRE rest of the chain ran on top of it." A later
  stage (multiband_compress, either AI-detector fix, temporal_normalize)
  could plausibly reintroduce a transient-like artifact, and the live log
  would have already claimed "pass" for a metric that the table could
  later contradict — the task's claim that these use "the exact same
  thresholds" so "there should be no discrepancy" is not accurate for this
  one tool.

**Specific audit targets:**
- `app/server.py` line 487 vs. app.js line 1049: decide whether to (a) fix
  the live log to genuinely match (would require re-running
  `chain.detect_transients` immediately, which is possible but currently
  not done), or (b) update the documentation/task framing to acknowledge
  `fix_transients` is NOT like the other three — it was never given a
  comparable live check, only a live REPORT of what happened at that
  moment.
- Confirm no other "matches the results table" comment in server.py makes a
  similar unchecked claim — this audit found and verified 3 accurate
  matches and 1 inaccurate one; a broader grep for the same claim pattern
  is worth Codex re-running independently in case more exist outside what
  this audit specifically checked line-by-line.

---

## Known, Documented, ACCEPTED Limitations (verified against code — not bugs to re-discover)

1. **Watermark compression fragility.** `watermark.py`'s module docstring
   (lines 45-61) claims: survives FLAC/AAC256k/MP3 320k&192k/native
   resample; fails completely on MP3 128k and any downsample to <=22050Hz.
   This audit did not re-run the compression test (no test harness or
   sample files present in the repo to do so), but the CODE's mechanism is
   consistent with the claim: the mark lives entirely in 10-16kHz content
   (`FREQ_BAND_LO_HZ`/`FREQ_BAND_HI_HZ`), and a 22050Hz-or-below downsample
   has a Nyquist of <=11025Hz, which would mathematically discard all
   content above that — this is a structural, not empirical, guarantee, and
   is correctly reasoned in the docstring. **Accept as documented; Codex
   should verify the frequency-band arithmetic (11025 < 16000, and 4 of 8
   derived frequencies land below vs above an arbitrary Nyquist split) if it
   wants to re-derive the "4 of 8 target frequencies" claim independently,
   but should not spend time re-discovering that this limitation exists —
   it's disclosed accurately.**

2. **Time-warp unverified against real fingerprinting systems.** Correctly
   and consistently disclosed in both `timewarp.py`'s and
   `fingerprint_proxy.py`'s module docstrings, and not contradicted by any
   stronger claim elsewhere in the code (the UI copy at app.js line 177 also
   says "NOT verified against any real commercial detection service").
   **Accept as documented — but see Section 5's NEW FINDING above regarding
   the ordering guarantee, which IS a gap in what's actually tested, not
   just in what's disclosed as unverified.**

3. **Cancellation is cooperative and coarse**, explicitly described as
   taking "several minutes" if it lands mid-optimization (JobCancelled
   docstring, server.py lines 206-211, and the frontend's cancel message).
   **Accept the coarseness as documented — but see Section 6's NEW FINDING:
   the actual dead zone is larger than "mid-optimization," extending
   through three entire re-verification blocks that sit outside the single
   checkpoint's coverage.**

---

## Prioritized "Audit This First" List

1. **[HIGH - correctness/safety]** `app/server.py` lines 604-793: confirm
   whether cancellation can be silently starved for many extra minutes
   during the post-chain linear/cnn/LUFS re-verification blocks, which sit
   entirely outside the one `check_cancelled()` call site. This is the
   biggest gap between documented behavior ("checked between pipeline
   stages") and actual behavior (three multi-minute blocks with zero
   checkpoints) found in this audit.

2. **[HIGH - correctness]** `app/server.py` lines 552-592 vs. 604-742:
   confirm whether the warp-then-watermark ordering guarantee still holds
   when a post-chain linear_fix/cnn_fix re-verification pass injects a new
   delta after `temporal_normalize` already ran. This is an untested fourth
   combination not covered by the "5 tested seeds" ordering study.

3. **[MEDIUM - documentation/behavior mismatch]** `app/server.py` line 487
   vs. `static/app.js` line 1049: the `fix_transients` live-log "pass"
   claim is unconditional and does not actually check the same
   re-scanned-after-full-chain metric the results table uses, unlike the
   other three tools this claim was made about. Either fix the live check
   or correct the claim.

4. **[MEDIUM - spec drift, low practical risk]** `app/watermark.py` lines
   121-182 (`_nth_prime_from`, `derive_frequencies`): no upper clamp against
   `FREQ_BAND_HI_HZ`; ~7% of individual frequency draws exceed the
   documented 10-16kHz band (verified: max observed 16447 Hz over a
   200k-seed sweep). Does not appear to crash detection/embedding, but
   contradicts the documented band guarantee and the "won't overlap the CNN
   detector's CQT range" isolation claim — verify against
   `cnn_differentiable_v2.py`'s actual analysis range, which this audit did
   not read.

5. **[MEDIUM - reproducibility gap, not a bug]** `static/app.js` line 790:
   `temporal_seed` is never sent to the backend from the real UI, so every
   production run of `temporal_normalize` is unseeded/non-reproducible.
   Confirm this is intentional (likely is, for anti-fingerprinting reasons)
   and consider documenting it explicitly, since currently nothing in the
   code states this is deliberate.

6. **[LOW - unverifiable from repo alone]** All specific measured
   percentages/counts cited in comments across all 6 areas (39/39, 258x/281x,
   794ms/step, 107/108 windows, 4ms median shift, 94%/75-100% ordering
   confidence, etc.) — none have a corresponding test script or fixture
   checked into the repo. Not a defect in the code itself, but Codex should
   not treat these numbers as independently reproducible without a test
   harness; flag if Codex has access to test audio files and wants to
   attempt re-measurement of any of them.

7. **[LOW - asymmetric constant, likely intentional]** `app/linear_fix.py`
   line 236 (`target` floor of `0.0001`) vs. line 235 (`cur_real_target`
   floor of `0.00001`) — a 10x difference in floor values between two
   related but distinct target variables, unexplained in comments. Quick to
   verify either way.
