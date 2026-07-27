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

**Restarting after a code change.** The server does not hot-reload, and the
browser caches `app.js`, so a change needs both:

```
pkill -f "app.server"
./run.sh                     # or: venv/bin/python -m app.server
```

then a hard reload in the browser (Cmd+Shift+R). Skipping either one means
running old code while reading new source - a stale server process and a
stale tab have both caused real confusion during development.

**Watermark seed.** Create a `.env` in this directory with:

```
FIXER_WATERMARK_SEED=<your seed>
```

It is gitignored and loaded automatically at startup. Without it the app
falls back to a built-in default seed and says so in the log; fine for local
work, not for anything distributed.

## The signal chain, in order

This is the order the app actually runs, and it is enforced by
`tests/test_certification_is_last.py` - the UI cards, the log and this list
must all agree with `TOOL_ORDER` in `app/server.py`.

**The one rule that governs the order:** a CNN certification is bound to the
exact timeline it was made on. `CNNDetector.extract_segments` derives every
analysis-window position from `len(audio)`, so anything that changes the
sample count or displaces content in time must run BEFORE the detector
fixes. Measured, a 311ms trim after certification took a file from 0.003% to
78.8%. Everything after `cnn_fix` is amplitude-domain only and has been
measured inert.

1. **Strip metadata & embedded images** - reports every format- and
   stream-level tag found (title, artist, comment, encoder, generation
   platform) and any embedded cover art. The delivered file never carries
   these regardless, since every output is freshly encoded from raw audio;
   this step surfaces exactly what was there.

2. **DC offset correction** - subtracts each channel's mean. Runs before the
   trim because a DC offset lifts otherwise-silent samples above the trim
   threshold: measured, a file with a 0.02 offset trimmed 0ms before
   correction and 995ms after.

3. **High-pass filter** - filters at 30Hz, but only engages when the track
   has genuine sub-sonic content, measured below about 15Hz. A clean track
   is left bit-for-bit untouched. Also before the trim, since deep rumble
   holds the "silence" up the same way DC does.

4. **Trim leading/trailing silence** - the only stage that changes the
   sample count, so the timeline is fixed from here on. A decaying tail is
   recognised as a fade and left alone: without that guard a 3-second
   fade-out lost 61ms of its own tail.

5. **Temporal pattern denormalization** (off by default) - a small, smooth,
   non-uniform timing drift, default 4ms. Length-preserving but it displaces
   content in time, which makes it a timeline stage: applied to an already
   certified signal it measured +97 percentage points. It therefore runs
   here, with the other timeline work, even though its card is grouped with
   the AI-detector fixes because that is what it targets.

6. **Surgical transient/pop fix** - finds genuine clicks and pops and
   bridges across just that moment, with a short taper either side. Skips
   sustained bursts like vocal consonants, and its post-chain re-check only
   corrects anomalies this chain introduced.

7. **Stereo field: bass mono & phase** - sums bass below 120Hz to mono and
   repairs phase in the 120-300Hz band. Does not widen.

8. **High-frequency spectral fill-in (17kHz+)** - detects an artificial
   cutoff and fills above it using only this track's own rolloff slope,
   harmonics and dynamics.

9. **Tonal cleanup** (off by default) - checks 250Hz boxiness and 3.15kHz
   harshness, cutting only where a region rings persistently rather than
   being a note or a filter slope. Cut only, 1.5dB maximum. Most finished
   masters get nothing.

10. **Multiband compression** - gentle 4-band downward compression,
    crossovers at 100/800/5000Hz, ratio 1.3, one pass.

11. **Saturation** (off by default) - tanh soft saturation, 4x oversampled,
    level-matched. Changes peak density rather than tone.

12. **Linear-model AI-detector fix** - gradient-based adversarial correction
    against the fakeprint logistic-regression detector.

13. **CNN-model AI-detector fix** - gradient optimization against the
    CQT-cepstrum CNN, verified against the real model. **This is the
    certification point; the timeline is locked here.**

14. **LUFS loudness normalization** - default -14 LUFS, adjustable -16 to
    -9. Second-to-last so nothing after it moves the delivered level off
    target. Measured inert against a certification (-0.003pp).

15. **True-peak limiter** - brick-wall ceiling at -1dBTP with 4x
    oversampling and 1.5ms lookahead. Measured inert (0.000pp).

16. **Fade in / out** - raised-cosine, default 10ms in and 3000ms out.
    Amplitude-only and measured bit-exactly inert.

Then, automatically and without their own cards:

- **Post-chain linear/CNN re-verification** - re-scores after the stages
  above and re-runs a fix if it lost its margin.
- **Post-chain LUFS drift correction** - iterates to hold the target.
- **Product watermark** - unconditional. Measured inert: CNN delta exactly
  0.00000000pp, linear -0.0000060pp, sample count unchanged.
- **16-bit TPDF dither** - every output is written as 16-bit PCM, so the
  bit-depth reduction is unconditional. Undithered truncation leaves
  harmonics at -24.6dB against the tone; dithered, -43.7dB.
- **Dense delivered-file certification** - the encoded bytes are scanned
  across every analysis window, not the five fixed positions
  `scorer.score()` samples. Measured on a real file, the five-position
  median read 0.00112% while the worst window was 0.12442% - a 111x
  under-report, because the worst window fell between two sampled positions.


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
  by subtraction, so they sum to exactly the input. Measured ~-30dB/octave across the crossover;
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
