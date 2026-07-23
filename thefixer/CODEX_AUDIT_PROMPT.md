# Adversarial code audit request: "The Fixer"

## What this is

A local Flask + PyTorch web app for AI-music-detector evasion and audio
mastering. It was built in one long autonomous session by Claude Code
(Sonnet 5), largely tested via curl/API calls against a running server
rather than exercised through the browser UI. It is NOT production/
multi-tenant software — it's a single-user local tool (`./run.sh`, binds
port 8090, no auth) — but within that scope it should still be correct,
not leak resources, and not silently produce wrong or degraded output.

## Repo location

`thefixer/` inside the repo (branch `worktree-stateful-wobbling-bunny` on
GitHub at `0xwhisperer/aitoaster`, currently PR #1, draft). Key files:

- `app/server.py` — Flask routes, background-thread job orchestration, the
  `TOOL_ORDER` pipeline that chains DSP tools + two adversarial
  "AI-detector fix" tools in a specific sequence.
- `app/chain.py` — DSP tools: silence trim, DC offset correction, transient/
  pop detection+fix, high-pass filter, stereo phase correction, LUFS
  normalization, multiband compression, true-peak limiting, spectral tilt
  reporting.
- `app/detector.py` — torch-free (numpy/scipy/librosa/onnxruntime only)
  reimplementation of two pretrained AI-music-detector models (a linear
  logistic-regression model over a "fakeprint" spectral feature, and a
  CQT-cepstrum CNN), used purely for scoring/verification.
- `app/linear_fix.py`, `app/linear_gradient_optimizer.py`,
  `app/linear_differentiable.py` — gradient-based adversarial correction
  targeting the linear model. The `_differentiable` module is a PyTorch
  reimplementation of the model's feature pipeline (used ONLY for
  backprop); the real ONNX model is always used for actual scoring/
  verification.
- `app/cnn_fix.py`, `app/cnn_gradient_optimizer_v2.py`,
  `app/cnn_wholetrack_optimizer_v2.py`, `app/cnn_differentiable_v2.py` —
  same pattern for the CNN model, but optimizing jointly across ALL
  overlapping 10-second analysis windows in a full track simultaneously
  (not per-segment), with periodic real-model re-verification and adaptive
  per-window loss weighting.
- `static/index.html`, `static/app.js` — single-page frontend: upload,
  live analysis, selectable tool chain, before/after report, A/B player,
  step-aware progress bar, docs modal.
- `models/*.onnx`, `*.npz`, `config.yaml` — pretrained model weights,
  copied from the open-source `lofcz/ai-music-detector` project.

## What I want from this audit

A genuinely adversarial line-by-line review — not a summary of what the
code does, but an attempt to find real bugs, security issues, correctness
problems, and design flaws, as if you were trying to get this rejected in
code review. Read every file fully before concluding anything.

### Specific areas of concern

1. **Concurrency/thread-safety.** `run_pipeline` runs in a daemon thread
   per job (`threading.Thread(..., daemon=True)`); the `JOBS` dict is
   guarded by `JOBS_LOCK` in `job_log`/`job_set_step`/`job_set_sub_progress`
   and in the route handlers, but check EVERY read/write for missing lock
   coverage. Check whether `get_scorer()`'s lazy singleton init
   (`_scorer_lock`) is actually safe under concurrent first-requests.
   Check whether the ONNX `InferenceSession` objects, or the CNN's
   `onnx2torch.convert()`-produced PyTorch model (loaded once at MODULE
   IMPORT TIME in `cnn_differentiable_v2.py`, i.e. shared across all
   threads/jobs), are safe to call from two jobs running concurrently. If
   two uploads are processed at the same time, could their gradient
   optimizations interfere with each other via shared model state?

2. **Resource leaks.** Every `tempfile.NamedTemporaryFile` use across
   `linear_fix.py`, `cnn_fix.py`, `detector.py` — confirm cleanup happens
   on every code path, including exceptions. Confirm whether
   `thefixer/uploads/` and `thefixer/outputs/` ever get cleaned up at all
   (every processing run writes a full-resolution stereo WAV plus a copy
   of the original — does disk usage grow unbounded with normal use?).

3. **DSP correctness.** Read `chain.py`'s `trim_silence`,
   `fix_dc_offset`, `detect_transients`, `fix_transient`, `normalize_lufs`,
   `multiband_compress`, `true_peak_limit`, `stereo_correlation`,
   `fix_phase_issues` line by line for off-by-one errors, incorrect dB
   math, incorrect array slicing, and edge cases: empty audio, all-silence
   audio, a file where L and R channels are byte-identical, clips shorter
   than a processing window (e.g. shorter than the multiband compressor's
   or transient detector's analysis window).

4. **The re-verification/retry logic** (`fix_linear`'s retry loop in
   `linear_fix.py`; the post-chain linear re-verification block added late
   in `server.py`'s `run_pipeline`). Check for infinite loops, off-by-one
   in retry/attempt counting, cases where a WORSE result could be returned
   than a better one that was computed and discarded, or bugs in the
   "best so far" tracking.

5. **Security, scoped to "must not be broken even for intended single-
   user local use."** `_safe_download_name`/`_find_upload_path` path
   handling, the `/api/audio/<kind>/<file_id>` route, upload handling —
   any path traversal, arbitrary file read, or injection risk that would
   matter even on localhost. Don't flag things that only matter for a
   hardened multi-tenant deployment (no auth, no rate limiting, etc. are
   known/accepted for this tool's scope) — but DO flag anything broken for
   the tool's actual intended use.

6. **Silent failure modes.** Broad exception catches that swallow errors,
   fallbacks that silently return stale/wrong data, any path where a job
   could get stuck in `"running"` forever without ever reaching `"done"`
   or `"error"`.

7. **Frontend/backend contract mismatches.** Does `app.js` expect fields
   that `server.py` doesn't always return (or vice versa) across every
   code path — especially newer fields: `output_name`, `current_step_idx`,
   `total_steps`, `current_step_name`, `sub_progress`,
   `worst_score_after_transfer`, `verified_after_transfer`.

8. **ML/numerical correctness in the gradient optimizers** — genuine bugs,
   not style nitpicks. Does `forward_logit_differentiable` actually
   produce correct gradients end to end? Any missing or extraneous
   `.detach()` calls that would silently break or leak gradient flow? Any
   place a numpy/torch dtype mismatch (float32 vs float64, especially
   around the WAV int16 round-trip used to transfer a tiny adversarial
   delta at full resolution) could silently produce wrong results?

### Known, already-acknowledged issues (don't waste time re-discovering these — verify if you want, but they're already tracked)

- The CNN whole-track optimizer does not always converge on every
  overlapping window on long tracks — this is surfaced via
  `worst_score_after_transfer`/`verified_after_transfer`, not hidden.
- Running `linear_fix` then `cnn_fix` can let the CNN correction disturb
  the linear fix's precision even in the correct pipeline order; there's a
  one-shot automatic re-verification/retry patch for this, not a complete
  fix (no joint optimization across both objectives yet).
- The multiband compressor has no attack/release envelopes, no soft knee,
  no makeup gain, no lookahead, and uses plain Butterworth (not
  Linkwitz-Riley) crossovers — a deliberately simplified implementation,
  documented as such in the README and in-app docs.
- Dithering is intentionally out of scope (explicit prior instruction).

## Output format requested

For each finding: file + approximate line number, a CONCRETE failure
scenario ("if X happens, Y breaks because Z" — not "this could
theoretically be an issue"), and a severity rating (critical/high/medium/
low). Rank most severe first. If you're not confident something is
actually a bug, say so explicitly rather than asserting it confidently —
false positives waste triage time as much as missed bugs do. Do not fix
anything in this pass — audit only, findings first.
