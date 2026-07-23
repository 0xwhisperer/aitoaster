# The Fixer

A local mastering & AI-music-detector verification console. Upload a track,
see how it scores against two independent AI-music detector models (a linear
fakeprint/logistic-regression model and a CQT-cepstrum CNN), pick which
mastering-chain tools to run, and get back a processed file plus a full
before/after report — scores, SNR, LUFS, spectral tilt — with real-time A/B
playback against the original.

## Running it

```
cd thefixer
./run.sh
```

First run creates a venv and installs `requirements.txt` automatically
(numpy/scipy/librosa/onnxruntime/soundfile/flask/pyloudnorm/torch/torchaudio/
onnx2torch/nnAudio/pyyaml). Then open **http://localhost:8090**.

This is a local Python+Flask app, not a static site — it needs the server
running to do anything (audio decoding, ONNX inference, PyTorch gradient
optimization all happen server-side). It cannot be hosted on GitHub Pages or
any other static host; see [Deployment](#deployment) below.

## The signal chain, in order

The order matters and was tuned specifically through testing, not chosen
arbitrarily — see [Why this order](#why-this-order).

1. **Trim leading/trailing silence** — true-digital-silence detection
   (threshold 0.0005 linear amplitude, 5ms safety pad).
2. **DC offset correction** — subtracts each channel's mean.
3. **Surgical transient/pop fix** — detects genuine click/glitch artifacts
   (requires both a large sample-to-sample jump AND a spike far above a
   200ms local RMS envelope, so it never fires on ordinary kick/snare hits)
   and applies a raised-cosine gain-reduction envelope (3ms attack, 60ms
   release) just at that moment. Target reduction is derived from the
   track's own recent loudness context, not a fixed value.
4. **High-pass filter** — 2nd-order Butterworth, zero-phase (`sosfiltfilt`),
   30Hz cutoff. Removes sub-audible rumble/DC residue.
5. **Stereo phase/correlation correction** — checks L/R correlation; if
   negative enough to risk mono-cancellation, blends mid/side to restore
   safety.
6. **LUFS loudness normalization** — targets -14 LUFS (general
   streaming-platform standard; Apple Music specifically targets -16, most
   others including Spotify/YouTube are closer to -14).
7. **Multiband compression** — gentle 3-band (low/mid/high) downward
   compression, conservative settings (ratio 1.3:1, threshold -12dB). See
   [Multiband compressor: how it compares to a real one](#multiband-compressor-how-it-compares-to-a-real-one)
   for what's simplified here.
8. **Linear-model AI-detector fix** — gradient-based adversarial correction
   targeting the fakeprint/logistic-regression detector.
9. **CNN-model AI-detector fix** — whole-track joint gradient optimization
   targeting the CQT-cepstrum CNN detector, across all overlapping 10-second
   analysis windows simultaneously.
10. **Post-chain linear re-verification** (automatic, not a separate
    selectable tool) — if both AI fixes were selected, re-scores the linear
    model on the final post-chain audio and, if `cnn_fix` disturbed it back
    above target, re-runs `linear_fix` once more, verified. See
    [Why this order](#why-this-order).
11. **True-peak limiter** — brick-wall ceiling at -1dBTP (industry-standard
    safe margin for lossy-codec transcoding headroom), oversampled 4x to
    catch inter-sample peaks, not just sample peaks.

## Why this order

Two real bugs were found and fixed by testing the actual output, not by
reasoning about it in the abstract:

**AI-detector fixes must run after all musical mastering steps, but before
the final limiter.** Any gain/EQ/dynamics change applied after an
adversarial correction can perturb its exact spectral signature enough to
undo it, even though the fix passed real-model verification immediately
after it ran. Confirmed directly: a linear fix verified at <1% scored 16%
once `normalize_lufs`/`multiband_compress` ran on top of it in an earlier,
wrongly-ordered version of the chain. The limiter is the one exception -
it's a safety net against clipping, not a musical/spectral-shaping step, and
it needs to see the truly final signal (including whatever small amount of
energy the AI fixes add) to actually guarantee the delivered file stays
under its ceiling.

**The two AI-detector fixes can still interact with each other even in the
correct order.** Running `linear_fix` then `cnn_fix` can let the CNN fix's
broadband correction disturb the linear fix's precise correction. Confirmed
directly on a full 276.9s test track: a verified 1.56% (right after
`linear_fix` completed) became 9.65% once `cnn_fix` had also run. This is
why step 10 above exists - a mandatory re-verification pass that catches
and corrects this specific interaction automatically, rather than silently
shipping a degraded result. It is not a complete fix for the underlying
issue (see [Roadmap](#roadmap)).

## Verification discipline

Every AI-detector fix is checked against the REAL (non-differentiable, exact
librosa+ONNX pipeline) model, not just the differentiable PyTorch surrogate
used during gradient optimization - the surrogate has been observed to
diverge sharply from the real model specifically after optimization, even
when it matches near-perfectly on unperturbed audio. Verification happens:

- **During optimization** - periodically (`real_check_interval`), the actual
  ONNX model is run on the current candidate, and any window/segment where
  the real score still disagrees with the surrogate gets extra optimization
  weight rather than being trusted blindly.
- **After the resample/stereo transfer** - a second, independent point where
  a verified-good result at the model's native sample rate (16kHz mono) can
  still degrade once resampled back up and mixed into the real deliverable
  (44.1kHz stereo). Both fixes re-check the real score on the actual
  transferred audio, and `linear_fix` will retry with a stricter internal
  target (up to `max_retries` times) if the transfer degraded the result.
- **After the full chain** - if both fixes were selected, a final check
  verifies the linear model on the truly final (post-mastering,
  post-cnn_fix) audio and triggers one more corrective pass if needed (see
  above).
- **In the final job result** - `scores_before`/`scores_after` always reflect
  the REAL model scored on the actual saved output file, never a
  self-reported claim from mid-pipeline. If either model is still failing
  after everything, this is logged explicitly as a warning, not hidden.

A fix is only ever reported as `"applied": true` after the real model has
confirmed it on the actual audio that gets delivered.

## Architecture

- `app/detector.py` - torch-free (numpy/scipy/librosa/onnxruntime only)
  reimplementation of both detector models for fast, dependency-light
  scoring. Verified to match the original `lofcz/ai-music-detector` inference
  scripts exactly (confirmed identical scores on known test files).
- `app/chain.py` - the deterministic DSP mastering tools (silence trim, DC
  offset, transient/pop fix, high-pass, phase correction, LUFS
  normalization, multiband compression, true-peak limiting, spectral tilt
  reporting).
- `app/linear_differentiable.py`, `app/cnn_differentiable_v2.py` - PyTorch
  differentiable reimplementations of each model's feature pipeline, used
  only for gradient-based optimization (never for scoring/verification,
  which always goes through the real ONNX+librosa path).
- `app/linear_gradient_optimizer.py`, `app/cnn_gradient_optimizer_v2.py`,
  `app/cnn_wholetrack_optimizer_v2.py` - the actual adversarial optimization
  loops, including perceptual/masking penalties (A-weighting, absolute
  silence-loudness gating, out-of-band energy penalty, tonal-concentration
  penalty) that keep corrections inaudible.
- `app/linear_fix.py`, `app/cnn_fix.py` - the public wrapper functions the
  pipeline actually calls: handle resampling to/from the model's native
  sample rate, the precision-preserving delta transfer (normalize to
  near-full-scale before a WAV round-trip so a tiny correction doesn't
  collapse into a handful of int16 quantization levels), and all
  real-model re-verification.
- `app/server.py` - Flask app: upload, analyze, process (background job
  with live progress log and step-aware progress tracking), serve
  before/after audio for A/B playback with user-controllable output
  filenames.
- `static/` - single-page frontend (upload, live spectrum/EQ visualization,
  selectable tool chain with recommendations, before/after comparison
  table, real-time A/B player, step-aware progress bar).

## Models

`models/linear_detector.onnx`, `models/cnn_detector.onnx`,
`models/linear_weights.npz`, `models/config.yaml` are copied from
[lofcz/ai-music-detector](https://github.com/lofcz/ai-music-detector).

## Deployment

This cannot run as a static site or on GitHub Pages. There is no client-side
audio processing at all - the frontend is a thin client that calls
`/api/upload`, `/api/analyze`, `/api/process`, `/api/job` on the Flask
backend for every operation. No `localStorage`, `indexedDB`, or
`sessionStorage` is used anywhere; nothing persists in the browser. All
state lives on the server's disk (`thefixer/uploads/`, `thefixer/outputs/`)
and in an in-memory job dict that is lost on server restart.

Making this static-hostable would require a fundamentally different
architecture: onnxruntime-web (WASM) for detector scoring, a full
JavaScript/WASM reimplementation of the DSP chain, and dropping the
PyTorch gradient-based AI-detector fixes entirely (there is no practical
way to run gradient-based adversarial optimization against a real
ONNX model in a browser at acceptable speed). That is a different project,
not a deployment configuration change.

## Multiband compressor: how it compares to a real one

The concept is standard mastering practice (split into bands, compress each
independently to smooth tonal-balance imbalance without one broadband
compressor squashing everything together) - real tools like iZotope Ozone,
Waves C6, or FabFilter Pro-MB do exactly this. The implementation here is a
simplified, conservative version, not a full-featured multiband compressor:

**What's missing or simplified vs. a real mastering-grade multiband:**
- **No attack/release time constants.** Real compressors have separate
  attack and release envelopes controlling how fast gain reduction engages
  vs. releases. This implementation uses a single 20ms median-filter
  smoothing window as a stand-in for both, which is a crude approximation.
  A proper compressor would let you set attack ~5-30ms and release
  ~50-500ms independently per band.
- **No knee.** Real compressors have a soft knee - gain reduction ramps in
  gradually around the threshold rather than switching on hard. This one
  has a hard knee (`over = max(env_db - threshold_db, 0)`), which can sound
  more abrupt.
- **No makeup gain.** After compressing, real multibands typically let you
  add back gain to compensate for the loudness you just removed. This
  doesn't do that automatically.
- **No lookahead.** Can't anticipate a transient before it hits.
- **Filter crossover isn't phase-aligned/Linkwitz-Riley.** Splitting into
  bands with plain Butterworth low/high/bandpass filters (rather than a
  proper Linkwitz-Riley crossover) can introduce small phase-cancellation
  artifacts when the bands are summed back together - professional
  multiband tools use LR crossovers specifically to avoid this.

Bottom line: reasonable and safe for light tonal-balance smoothing as one
step in an automated chain, but not a substitute for a dedicated multiband
compressor plugin if you need real control over the dynamics.

## Roadmap

### Fix-interaction correctness (highest priority)

- [ ] The `cnn_fix`-disturbs-`linear_fix` interaction has a targeted
  one-shot patch (an automatic linear re-verification/retry pass after the
  full chain), but the root cause - two independent adversarial corrections
  sharing the same audio without a joint objective - is unsolved. Next
  step: either alternate the two fixes in a loop until both stay verified
  simultaneously, or design a single joint optimization that targets both
  models' loss functions at once.
- [ ] The CNN whole-track optimizer does not always converge on every
  overlapping analysis window on long tracks - on one full-track test, 5 of
  108 windows plateaued around 93-96% AI probability despite full-budget
  optimization (300 steps) and window-weight boosting (capped at 20x). This
  is surfaced honestly via `worst_score_after_transfer` /
  `verified_after_transfer` in the result, never hidden, but isn't fixed.

### Mastering-chain gaps

- [ ] **Attack/release-aware dynamics** across the board (biggest gap - see
  above). Applies to both `fix_transients` and `multiband_compress`.
- [ ] **Stereo widening/imaging tools** beyond the basic phase-correlation
  fix - mid/side EQ, stereo width control.
- [ ] **De-essing** (sibilance control) - not applicable to most full mixes
  but standard on vocal-heavy masters.
- [ ] **Harmonic exciter/saturation** - a common "analog warmth" step in
  commercial mastering, not present here.
- [ ] **Dithering** - explicitly descoped for this version per direct
  instruction; not an oversight.
- [ ] **Reference-track matching** (matching tonal curve/loudness to a
  target reference song) - a common modern mastering workflow, not
  implemented.
- [ ] **Fuller metering/monitoring** - no LRA (loudness range) metering, no
  true-peak metering displayed before the limiter runs, beyond the basic
  correlation check.
- [ ] **Multiband compressor improvements** - attack/release, soft knee,
  makeup gain, lookahead, Linkwitz-Riley crossovers (see above).

### App/UX gaps

- [ ] No undo/versioning across multiple processing runs on the same
  upload - each run is independent, nothing chains a prior output back in
  as a new input automatically (you can re-upload the output file yourself
  to iterate, but the app doesn't do this for you).
- [ ] No persistence across server restarts - the in-memory job dict and
  everything in `uploads/`/`outputs/` survives only as long as the process
  runs; there is no database.
- [ ] Single-file, single-job workflow - no batch processing of multiple
  files in one pass.
- [ ] No authentication/multi-user support - this is a local single-user
  tool by design, not hardened for shared or public deployment.
