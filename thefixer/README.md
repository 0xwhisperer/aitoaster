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

This is a local Python+Flask app - it needs the server running to do
anything (audio decoding, ONNX inference, PyTorch gradient optimization all
happen server-side). See [Deployment](#deployment) below.

## The signal chain, in order

The processing order is deliberate — see [Why this order](#why-this-order).

1. **Strip metadata & embedded images** — reports every format- and
   stream-level tag (title, artist, comment, encoder, generation-platform
   provenance) plus any embedded cover art, and flags known AI-generation
   platform keywords explicitly. Metadata is always stripped from the
   delivered output regardless of selection (every output is freshly
   encoded from raw audio, so there's no code path where a tag could
   survive) — this step exists to make that fact visible and specific,
   not to gate whether stripping happens.
2. **Trim leading/trailing silence** — true-digital-silence detection
   (threshold 0.0005 linear amplitude, 5ms safety pad).
3. **DC offset correction** — subtracts each channel's mean.
4. **Surgical transient/pop fix** — detects genuine click/glitch artifacts
   (requires both a large sample-to-sample jump AND a spike far above a
   200ms local RMS envelope, so ordinary kick/snare hits are never flagged)
   and repairs each one by interpolating across it from the clean samples
   either side, since merely reducing a discontinuity's level leaves the
   jump itself intact. Sustained broadband bursts are rejected: a click
   crosses the detection threshold once or twice, a vocal consonant
   ("s"/"t"/"k") crosses it hundreds of times, and because the repair
   deletes rather than ducks, a false positive would erase the consonant.
   The post-chain re-check corrects only anomalies that were NOT present in
   the source — compression and limiting can smooth a consonant until it
   resembles a click, and a sharp edge already in the recording is not this
   tool's to remove.
5. **High-frequency spectral fill-in (17kHz+)** — detects an artificial
   hard cutoff (common in lossy encoding or low-quality AI generation) by
   comparing the actual energy just above a candidate cutoff against what
   the track's own established rolloff slope predicts, and only flags it
   when the deficit is large. Fills content above the cutoff using
   *only* this track's own characteristics — its self-fitted rolloff
   slope extrapolated past the cutoff, harmonic projection from each
   frame's own detected spectral peaks, and broadband texture modulated
   by the track's own frame-by-frame dynamics. No external reference file
   or fixed target curve is used anywhere in the calculation.
6. **High-pass filter** — 2nd-order Butterworth, zero-phase (`sosfiltfilt`),
   30Hz cutoff. Removes sub-audible rumble/DC residue.
7. **Stereo phase/correlation correction** — checks L/R correlation; if
   negative enough to risk mono-cancellation, blends mid/side to restore
   safety.
8. **LUFS loudness normalization** — default -14 LUFS, adjustable -16 to -9 (general
   streaming-platform standard; Apple Music specifically targets -16, most
   others including Spotify/YouTube are closer to -14).
9. **Multiband compression** — gentle 4-band downward compression with
   complementary crossovers at 100Hz / 800Hz / 5kHz, ratio 1.3:1, threshold
   -12dB, ONE pass, and per-band attack/release (30ms/200ms on the low band
   down to 3ms/60ms on the top). The crossovers put vocal presence in its own
   band rather than sharing one gain control with cymbals, and the bands
   sum back to exactly the input. Runs before loudness is set. See
   [Multiband compressor: how it compares to a real one](#multiband-compressor-how-it-compares-to-a-real-one)
   for what's simplified here.
10. **CNN-model AI-detector fix** — shift-robust gradient optimization
    targeting the CQT-cepstrum CNN detector. The recommended mode trains
    across small timing shifts around the detector's five real evaluation
    positions; a slower dense whole-track mode is also available.
11. **Linear-model AI-detector fix** — gradient-based adversarial
    correction targeting the fakeprint/logistic-regression detector.
    Reports live progress during optimization: current step, which retry
    attempt, and the live surrogate score as it converges.
12. **Temporal pattern normalization** (optional) — applies a small smooth
    non-uniform timing warp before the final watermark, displacing the
    low-frequency spectral peaks fingerprint matching anchors on (94% sit
    below 500Hz, none above 4kHz). Landmark displacement saturates at the
    4ms default, since landmark timing is quantized by the ~11.6ms analysis
    hop; higher values only smear sibilants without adding disruption (a
    measured sibilant retained 98.2% of its energy at 4ms vs 91.3% at 15ms).
    Measured against a local landmark proxy, not a commercial fingerprint
    system. Production runs deliberately use a fresh random curve.
13. **True-peak limiter** — brick-wall ceiling at -1dBTP (industry-standard
    safe margin for lossy-codec transcoding headroom), oversampled 4x to
    catch inter-sample peaks, not just sample peaks.
14. **Post-chain linear/CNN re-verification** (automatic, not a separate
    selectable tool) — if both AI fixes were selected, re-scores the linear
    and CNN models on the post-chain audio and can re-run either fix once
    if later processing erased its safety margin. See [Why this order](#why-this-order).
15. **Post-chain LUFS drift correction** (automatic, not a separate
    selectable tool) — if `normalize_lufs` was selected, the truly final
    LUFS is measured after every later stage and corrected if it has
    drifted more than 0.1dB from target. This is now a safety net rather
    than the main mechanism: loudness is set second-to-last, so almost
    nothing remains after it to cause drift. The correction iterates
    (bounded), because an upward correction forces a re-limit and that
    re-limiting pulls loudness back down again.

## Output format

Choose the delivered file's format independently of the source: same as
source, WAV, MP3, or FLAC. MP3 offers a further choice between VBR-0
(libmp3lame's highest VBR quality tier, ~245kbps average — considered
transparent/near-lossless and doesn't waste bits on simple passages) and
CBR 320kbps (a flat bitrate on every frame, what most people mean literally
by "highest bitrate"). The delivered filename's extension always matches
what was actually encoded, regardless of what was typed into the filename
field.

Every delivered file is also stamped with a disclosed, fixed product
watermark for aggregate footprint measurement. It is not per-user tracking,
DRM, or a delivery gate. The current v2 mark uses seed-derived frequencies
bounded to 10-16kHz; detection remains backward-compatible with v1 files.
It is known not to survive 128kbps MP3 or downsampling to 22050Hz and below,
and a watermark failure is logged but never blocks delivery.

## Why this order

**AI-detector fixes run after all musical mastering steps, but before the
final limiter.** Any gain/EQ/dynamics change applied after an adversarial
correction can perturb its exact spectral signature enough to undo it, even
if the fix passed real-model verification immediately after it ran. For
example, a linear fix verified at under 1% can score much higher again if
LUFS normalization or multiband compression runs on top of it afterward.
The limiter is the one exception — it's a safety net against clipping, not
a musical/spectral-shaping step, and it needs to see the truly final signal
(including whatever small amount of energy the AI fixes add) to actually
guarantee the delivered file stays under its ceiling.

**The two AI-detector fixes can still interact with each other.** The CNN
fix runs first and the cheaper linear fix runs second, but either correction
or the later limiter can erase the other model's safety margin. Historical
testing also observed a linear score rise from 1.56% to 9.65% when CNN was
applied afterward. The bounded post-chain re-verification sequence therefore
checks both models and may retry each once; it does not ping-pong indefinitely.

**LUFS normalization is checked again at the very end, for the same
reason.** It runs mid-chain (step 8), but every stage after it — multiband
compression, both AI-detector fixes, the limiter — can shift overall
loudness without anything verifying the FINAL delivered file still matches
the requested target. Traced directly on a real track end-to-end: LUFS
held within a fraction of a dB across cleanup, mastering, the linear fix,
and the CNN fix in one full run, but nothing previously guaranteed that on
every file, so step 13 exists as the same kind of final safety net the
linear-fix re-verification already provided.

## Verification discipline

Every AI-detector fix is checked against the real (non-differentiable,
exact librosa+ONNX pipeline) model, not just the differentiable PyTorch
surrogate used during gradient optimization — the surrogate can diverge
from the real model specifically after optimization, even when it matches
near-perfectly on unperturbed audio. Verification happens:

- **During optimization** — periodically (`real_check_interval`), the actual
  ONNX model is run on the current candidate, and any window/segment where
  the real score still disagrees with the surrogate gets extra optimization
  weight rather than being trusted blindly.
- **After the resample/stereo transfer** — a second, independent point where
  a verified-good result at the model's native sample rate (16kHz mono) can
  still degrade once resampled back up and mixed into the real deliverable
  (44.1kHz stereo). Both fixes re-check the real score on the actual
  transferred audio, and the linear fix retries with a stricter internal
  target (up to a set number of times) if the transfer degraded the result.
- **After the full chain** — if both fixes were selected, final linear and
  CNN checks can trigger one bounded corrective pass each (see above).
- **In the final job result** — reported scores always reflect the real
  model scored on the actual saved output file, never a self-reported claim
  from mid-pipeline. If either model is still failing after everything,
  this is shown as a warning rather than hidden.

The step result distinguishes a correction being applied from it being
verified after transfer. Failed verification is surfaced rather than
silently relabeled as success.

## Architecture

- `app/detector.py` — a lightweight (numpy/scipy/librosa/onnxruntime only,
  no PyTorch) reimplementation of both detector models for fast scoring.
  Matches the original `lofcz/ai-music-detector` inference scripts exactly.
- `app/chain.py` — the deterministic DSP mastering tools (silence trim, DC
  offset, transient/pop fix, high-pass, phase correction, LUFS
  normalization, multiband compression, true-peak limiting, spectral tilt
  reporting).
- `app/linear_differentiable.py`, `app/cnn_differentiable_v2.py` — PyTorch
  differentiable reimplementations of each model's feature pipeline, used
  only for gradient-based optimization (never for scoring/verification,
  which always goes through the real ONNX+librosa path).
- `app/linear_gradient_optimizer.py`, `app/cnn_gradient_optimizer_v2.py`,
  `app/cnn_wholetrack_optimizer_v2.py` — the adversarial optimization
  loops, including perceptual/masking penalties (A-weighting, absolute
  silence-loudness gating, out-of-band energy penalty, tonal-concentration
  penalty) that keep corrections inaudible.
- `app/linear_fix.py`, `app/cnn_fix.py` — the public functions the pipeline
  calls: handle resampling to/from the model's native sample rate, the
  precision-preserving delta transfer (normalize to near-full-scale before
  a WAV round-trip so a tiny correction doesn't collapse into a handful of
  int16 quantization levels), and all real-model re-verification.
- `app/server.py` — Flask app: upload, analyze, process (background job
  with live progress log and step-aware progress tracking), serve
  before/after audio for A/B playback with user-controllable output
  filenames.
- `static/` — single-page frontend (upload, live spectrum/EQ visualization,
  selectable tool chain with recommendations, before/after comparison
  table, real-time A/B player, step-aware progress bar).

## Models

`models/linear_detector.onnx`, `models/cnn_detector.onnx`,
`models/linear_weights.npz`, `models/config.yaml` are copied from
[lofcz/ai-music-detector](https://github.com/lofcz/ai-music-detector).

## Deployment

There is no client-side audio processing at all — the frontend is a thin
client that calls `/api/upload`, `/api/analyze`, `/api/process`, `/api/job`
on the Flask backend for every operation. No `localStorage`, `indexedDB`,
or `sessionStorage` is used anywhere; nothing persists in the browser. All
state lives on the server's disk (`thefixer/uploads/`, `thefixer/outputs/`)
and in an in-memory job dict that is lost on server restart. Running it
requires the Flask server (`./run.sh`) to be up.

## Multiband compressor: how it compares to a real one

The concept is standard mastering practice (split into bands, compress each
independently to smooth tonal-balance imbalance without one broadband
compressor squashing everything together) — real tools like iZotope Ozone,
Waves C6, or FabFilter Pro-MB do exactly this. The implementation here is a
simplified, conservative version, not a full-featured multiband compressor:

**What's missing or simplified compared to a real mastering-grade multiband:**
- **Fixed attack/release, not program-dependent.** Each band has its own
  attack and release (30ms/200ms low through 3ms/60ms top), but they are
  constants; real compressors vary release with program material.

- **No knee.** Real compressors have a soft knee — gain reduction ramps in
  gradually around the threshold rather than switching on hard. This
  implementation has a hard knee, which can sound more abrupt.
- **No makeup gain.** After compressing, real multibands typically add
  gain back to compensate for the loudness that was removed. This doesn't
  happen automatically here.
- **No lookahead in the compressor.** Its envelope is strictly causal.
  (The true-peak limiter separately looks ahead 1.5ms.)
- **Crossovers are zero-phase complementary, not Linkwitz-Riley.** Bands are
  peeled off with zero-phase Butterworth lowpasses and the remainder carried
  by subtraction, so they sum to exactly the input. Measured -48dB/octave;
  the split points are nominal, not -6dB crossover points.


Bottom line: reasonable and safe for light tonal-balance smoothing as one
step in an automated chain, but not a substitute for a dedicated multiband
compressor plugin where finer dynamics control is needed.

## Roadmap

### Fix-interaction correctness (highest priority)

- [ ] The interaction where the CNN fix can disturb the linear fix has a
  targeted one-shot patch (an automatic linear re-verification/retry pass
  after the full chain), but the root cause — two independent adversarial
  corrections sharing the same audio without a joint objective — is
  unsolved. A cleaner solution would either alternate the two fixes in a
  loop until both stay verified simultaneously, or use a single joint
  optimization that targets both models' loss functions at once.
- [ ] The CNN whole-track optimizer does not always converge on every
  overlapping analysis window on long tracks — on one full-track test, 5 of
  108 windows plateaued around 93-96% AI probability despite full-budget
  optimization (300 steps) and window-weight boosting (capped at 20x). This
  is surfaced in the result (via `worst_score_after_transfer` /
  `verified_after_transfer`) rather than hidden, but isn't fixed.

### Mastering-chain gaps

- [ ] **Attack/release-aware dynamics** across the board (the biggest gap —
  see above). Applies to both the transient fix and the multiband
  compressor.
- [ ] **Stereo widening/imaging tools** beyond the basic phase-correlation
  fix — mid/side EQ, stereo width control.
- [ ] **De-essing** (sibilance control) — not applicable to most full mixes
  but standard on vocal-heavy masters.
- [ ] **Harmonic exciter/saturation** — a common "analog warmth" step in
  commercial mastering, not present here.
- [ ] **Dithering** — out of scope for this version.
- [ ] **Reference-track matching** (matching tonal curve/loudness to a
  target reference song) — a common modern mastering workflow, not
  implemented.
- [ ] **Fuller metering/monitoring** — no LRA (loudness range) metering, no
  true-peak metering displayed before the limiter runs, beyond the basic
  correlation check.
- [ ] **Multiband compressor improvements** — attack/release, soft knee,
  makeup gain, program-dependent release (see above).

### App/UX gaps

- [ ] Metadata stripping is not currently a real on/off toggle - every
  output is always freshly encoded from raw audio (WAV via soundfile
  carries no tags at all; MP3/FLAC are re-encoded from a tagless temp file
  with an explicit strip pass), so there is no existing code path where a
  tag could survive even if the tool were unchecked. Making it a genuine
  toggle would require a new "preserve original tags" feature to copy
  them forward when unchecked - not built, since doing so would also
  preserve any AI-platform provenance tags the tool exists to remove.
- [ ] The spectral revival tool's per-frame harmonic-projection loop is a
  plain Python loop over every analysis frame - roughly 5x realtime on a
  full track (about 52s for a 277s track in testing). Fine for the current
  scale, but would benefit from vectorization if used on much longer
  material or in a batch context.
- [ ] No undo/versioning across multiple processing runs on the same
  upload — each run is independent; nothing automatically chains a prior
  output back in as a new input (re-uploading the output file works as a
  manual workaround).
- [ ] No persistence across server restarts — the in-memory job dict and
  everything in `uploads/`/`outputs/` survives only as long as the process
  runs; there is no database.
- [ ] Single-file, single-job workflow — no batch processing of multiple
  files in one pass.
- [ ] No authentication/multi-user support — this is a local single-user
  tool by design, not hardened for shared or public deployment.
