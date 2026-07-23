# Adversarial code audit request: "The Fixer"

## What this is

A local Flask + PyTorch web app for AI-music-detector evasion and audio
mastering. It is NOT production/multi-tenant software - it's a single-user
local tool (`./run.sh`, binds port 8090, no auth) - but within that scope
it should still be correct, not leak resources, and not silently produce
wrong or degraded output.

## Repo location

`thefixer/` inside the repo. Key files:

- `app/server.py` - Flask routes, background-thread job orchestration, the
  `TOOL_ORDER` pipeline that chains DSP tools + two adversarial
  "AI-detector fix" tools in a specific sequence.
- `app/chain.py` - DSP tools: silence trim, DC offset correction, transient/
  pop detection+fix, high-pass filter, stereo phase correction, LUFS
  normalization, multiband compression, true-peak limiting, spectral tilt
  reporting, metadata read/strip.
- `app/detector.py` - torch-free (numpy/scipy/librosa/onnxruntime only)
  reimplementation of two pretrained AI-music-detector models (a linear
  logistic-regression model over a "fakeprint" spectral feature, and a
  CQT-cepstrum CNN), used purely for scoring/verification.
- `app/linear_fix.py`, `app/linear_gradient_optimizer.py`,
  `app/linear_differentiable.py` - gradient-based adversarial correction
  targeting the linear model. The `_differentiable` module is a PyTorch
  reimplementation of the model's feature pipeline (used ONLY for
  backprop); the real ONNX model is always used for actual scoring/
  verification.
- `app/cnn_fix.py`, `app/cnn_gradient_optimizer_v2.py`,
  `app/cnn_wholetrack_optimizer_v2.py`, `app/cnn_differentiable_v2.py` -
  same pattern for the CNN model, but optimizing jointly across ALL
  overlapping 10-second analysis windows in a full track simultaneously.
- `static/index.html`, `static/app.js` - single-page frontend.
- `models/*.onnx`, `*.npz`, `config.yaml` - pretrained model weights.

## What I want from this audit

A genuinely adversarial line-by-line review - not a summary of what the
code does, but an attempt to find real bugs, security issues, correctness
problems, and design flaws, as if you were trying to get this rejected in
code review. Read every file fully before concluding anything. Reproduce
every finding with an actual runnable test before reporting it - "confirmed
with a scaled executable test" findings are far more valuable than
theoretical concerns, and every finding in this report should say which it
is.

### Specific areas of concern

1. **Concurrency/thread-safety.** Background job threads, shared model/
   session state loaded once at import time, lazy singleton initialization
   under concurrent first-requests.
2. **Resource leaks.** Every temp-file path, on every exception path.
   Whether upload/output directories are ever cleaned up (they aren't by
   default - is that acceptable for this tool's scope, or does it need a
   TTL/cleanup job?).
3. **DSP correctness.** Off-by-one errors, incorrect dB math, incorrect
   array slicing, edge cases: empty audio, all-silence audio, L/R-identical
   audio, clips shorter than a processing window, tracks over 5 minutes
   (past any hardcoded analysis-window cap).
4. **Retry/re-verification logic correctness.** Off-by-one in
   retry/attempt counting, whether a worse result can be returned instead
   of a better one already computed, whether the final safety-net step
   (e.g. a limiter) still runs after every correction stage including any
   late re-verification pass.
5. **Security, scoped to "must not be broken even for intended single-user
   local use."** Path traversal, arbitrary file read/write, unsanitized
   input used in globs/paths/subprocess args. Don't flag things that only
   matter for a hardened multi-tenant deployment (no auth, no rate
   limiting) - but DO flag anything broken for the tool's actual intended
   use.
6. **Silent failure modes.** Broad exception catches that swallow errors,
   fallbacks that silently return stale/wrong data, any path where a job
   could get stuck in a non-terminal state forever, any value that could
   serialize as invalid JSON (NaN/Infinity) and silently break the frontend.
7. **Frontend/backend contract mismatches.** Every field the frontend reads
   from an API response - confirm the backend actually sends it on every
   code path, not just the happy path.
8. **ML/numerical correctness in the gradient optimizers.** Whether the
   differentiable surrogate genuinely matches the real (non-differentiable)
   model it's meant to approximate - test with an actual side-by-side score
   comparison on a real audio sample, not just code review. Missing/
   extraneous `.detach()` calls that would silently break or leak gradient
   flow. Precision loss at any WAV/int16 round-trip used to transfer a
   correction.
9. **Scope-appropriate feature checks.** If a feature claims a specific
   guarantee (e.g. "strips all metadata," "outputs at -1dBTP," "under Xs
   this always works"), verify the claim holds under adversarial/edge-case
   input, not just the common case.

## Output format required

Every finding must be labeled with exactly one severity as the FIRST word
of its heading: **Critical**, **High**, **Medium**, **Low**. For each
finding, give:

- File + line number (or line range) for every code reference.
- A CONCRETE failure scenario: "if X happens, Y breaks because Z" - not
  "this could theoretically be an issue." State whether it was reproduced
  with an actual runnable test, or is a code-review-only concern (be
  explicit about which - both are useful, but they are not equally
  confidence-worthy).
- What the correct/expected behavior should be instead.

At the end of the report, give ONE overall verdict, chosen from exactly
these three, as its own top-level line:

- **BLOCKED** - at least one Critical or High finding exists that would
  cause data loss, silent incorrect output, a crash on a realistic input,
  or a security issue reachable in the tool's actual intended use. Do not
  ship until these are fixed.
- **APPROVED WITH NITS** - no Critical/High findings, but one or more
  Medium/Low findings exist that are worth fixing but don't block shipping
  (e.g. a resource-usage inefficiency, a cosmetic inconsistency, a low-
  impact edge case unlikely to be hit in normal use).
- **APPROVED** - no findings at all, or only findings so trivial (typos,
  pure style) that they don't warrant even a nit-level callout.

If you are not confident something is actually a bug, say so explicitly
rather than asserting it confidently - a false positive costs real triage
time. Do not fix anything in this pass - audit only, findings first.
