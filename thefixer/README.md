# The Fixer

A local mastering & AI-music-detector verification console. Upload a track,
see how it scores against two independent AI-music detector models (a linear
fakeprint/logistic-regression model and a CQT-cepstrum CNN), pick which
mastering-chain tools to run, and get back a processed file plus a full
before/after report — scores, SNR, LUFS, spectral tilt — with real-time A/B
playback against the original.

## Running it

```
./run.sh
```

First run creates a venv and installs `requirements.txt` automatically
(numpy/scipy/librosa/onnxruntime/soundfile/flask/pyloudnorm/torch/torchaudio/
onnx2torch/nnAudio/pyyaml). Then open **http://localhost:8090**.

## What's in the signal chain

**Cleanup**
- Trim leading/trailing silence
- DC offset correction
- Surgical transient/pop fix — auto-detects genuine click/glitch artifacts
  (strict: a real sample-to-sample discontinuity far above the local RMS
  envelope) and gently limits just that moment. Deliberately conservative so
  it never mistakes an ordinary kick/snare hit for a defect.
- High-pass filter (removes sub-30Hz rumble/DC residue)

**AI-detector fixes** — gradient-based adversarial correction, verified
against the REAL (non-differentiable) detector models, not just the
differentiable surrogate used during optimization:
- Linear model fix (`app/linear_fix.py`, `app/linear_gradient_optimizer.py`)
- CNN model fix (`app/cnn_fix.py`, `app/cnn_wholetrack_optimizer_v2.py`) —
  whole-track joint optimization over overlapping windows, with an absolute
  (not just per-window-relative) silence-loudness gate so genuinely quiet
  passages never pick up an audible correction louder than the music itself.

**Mastering**
- Stereo phase/correlation correction
- LUFS loudness normalization (default -14 LUFS)
- Multiband compression (gentle, tonal-balance smoothing only)
- True-peak limiter (-1dBTP ceiling, inter-sample-peak aware)

## Architecture

- `app/detector.py` — torch-free (numpy/scipy/librosa/onnxruntime only)
  reimplementation of both detector models for fast, dependency-light
  scoring. Verified to match the original `lofcz/ai-music-detector` inference
  scripts exactly.
- `app/chain.py` — the deterministic DSP mastering tools.
- `app/linear_differentiable.py`, `app/cnn_differentiable_v2.py` — PyTorch
  differentiable reimplementations of each model's feature pipeline, used
  only for gradient-based optimization (not for scoring/verification, which
  always goes through the real ONNX+librosa path).
- `app/server.py` — Flask app: upload, analyze, process (background job with
  live progress log), serve before/after audio for A/B playback.
- `static/` — single-page frontend.

## Verification discipline

Every AI-detector fix is checked against the REAL model after every
significant step — during optimization (periodic real-score checks, not just
trusting the differentiable surrogate) and again after the final resample/
stereo-transfer (a second point where a verified-good result can still
degrade). A fix is only reported as "applied" once the real model confirms
it on the actual delivered audio.

## Models

`models/linear_detector.onnx`, `models/cnn_detector.onnx`,
`models/linear_weights.npz`, `models/config.yaml` are copied from
[lofcz/ai-music-detector](https://github.com/lofcz/ai-music-detector).

## Known limitation: fix ordering & interaction

AI-detector fixes run LAST in the chain, after all mastering steps — any
gain/EQ/dynamics change applied after an adversarial correction can perturb
its exact spectral signature enough to undo it, even though the fix passed
real-model verification immediately after it ran. This was confirmed
directly (a linear fix verified at <1% scored 16% once later mastering
steps ran on top of it) and fixed by reordering `TOOL_ORDER` in
`app/server.py`.

A related, NOT YET fully solved interaction: running `linear_fix` followed
by `cnn_fix` in the same pipeline can still let the CNN fix's broadband
correction disturb the linear fix's precise correction, even with fixes
correctly ordered last. On a full 276.9s test track, this took the linear
score from a verified 1.56% (right after `linear_fix` completed) back up to
9.65% once `cnn_fix` had also run. Both fixes are independently verified
correct in isolation (linear: <1% on its own; CNN: 0.0000–0.0006 on a
30-second sample clip with all windows converged) — the interaction between
the two adversarial corrections sharing the same audio is the open problem.
Likely next step: run `linear_fix` again, verified, AFTER `cnn_fix` rather
than only before it, or jointly optimize both objectives in one pass.

Separately, the CNN whole-track optimizer does not always converge on every
overlapping analysis window — on this same full-track run, 5 of 108 windows
plateaued around 93-96% AI probability despite full-budget optimization
(300 steps) and window-weight boosting (capped at 20x). This is tracked via
the `worst_score_after_transfer` / `verified_after_transfer` fields on the
`cnn_fix` step's info dict, so it's always visible in the result rather than
silently reported as a success.
