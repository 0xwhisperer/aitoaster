# Dense whole-track CNN prototype

This is a prototype measurement only. The production optimizer, thresholds,
EOT mode, certification windows, and exact librosa/ONNX verification path
were not changed. The exact 0.5-second scan remains authoritative.

## Implementation

`app/cnn_differentiable_v2.py` now exposes the existing converted ONNX graph as
a convolutional trunk plus the original 128→64→1 MLP head. The original
10-second differentiable scorer is numerically identical to the unsplit graph.

`app/cnn_wholetrack_dense_prototype.py` computes the nnAudio CQT/cepstrum once
for the whole track, runs the trunk once, and applies a sliding average pool
of 39 trunk time cells. The three 2×2 time pools make one trunk cell 8×512 =
4,096 samples (0.256 s). `tools/benchmark_cnn_wholetrack_dense.py` reproduces
the comparisons and timings.

## Measurements

Measurements were run on the local CPU runtime using the two requested files.
Forward/backward timings are warm-process medians for one full dense pass and
one existing differentiable 10-second window. “Estimated old grid” multiplies
the latter by the number of complete 4,096-sample windows. Peak RSS was
measured in fresh processes; it includes the Python/runtime/model footprint.

| file | complete dense windows | dense fwd+bwd | old per-window fwd+bwd | estimated old grid | surrogate speedup | dense peak RSS | one-window RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4d363ababbc6.m4a` (205.2 s) | 763 | 0.273 s | 0.0124 s | 9.48 s | 34.8× | ~1.1 GB | ~0.4–0.5 GB |
| `4bf180897cfb.wav` (277.2 s) | 1,044 | 0.391 s | 0.0135 s | 14.10 s | 36.1× | ~1.3 GB | ~0.5 GB |

The dense graph emits one additional partial right-edge cell for the second
file (1,045 outputs); it is excluded from the complete-window comparison.

The exact certificate scan remained separate and unchanged:

| file | exact 0.5 s windows | exact scan time |
| --- | ---: | ---: |
| `4d363ababbc6.m4a` | 392 | 4.88 s |
| `4bf180897cfb.wav` | 536 | 7.33 s |

Including one unchanged final certificate, the estimated total pass goes from
about 14.35→5.15 s for the first file and 21.43→7.73 s for the second: about
2.8× end-to-end. This estimate applies to the surrogate-pass-plus-final-scan
work unit; it does not claim to accelerate the certificate itself.

## Logit comparison and alignment investigation

Dense logits were compared at the 0.256-second grid starts against the exact
librosa/ONNX 10-second scorer, including a zero-padded diagnostic for the one
partial right-edge cell:

| file | logit correlation | max absolute error | mean absolute error |
| --- | ---: | ---: | ---: |
| `4d363ababbc6.m4a` | 0.871 | 9.54 | 2.47 |
| `4bf180897cfb.wav` | 0.891 | 6.97 | 1.27 |

To isolate whole-track pooling/boundary effects from the separate CQT
implementation difference, the dense path was also compared with the existing
nnAudio-based differentiable 10-second scorer. On complete windows the
correlations were 0.925 and 0.924, with max errors 9.63 and 6.47 logits. The
error stayed substantial after trimming 64 cells from both track ends, so this
is not only a first/last-track boundary issue.

The centered CQT gives 313 frames for a 10-second segment. A whole-track CQT
does not zero-pad at every 10-second window boundary; it supplies neighboring
track content instead. The CNN trunk also has a 32-CQT-frame temporal
receptive field before its 8-frame output stride. Consequently, each local
standalone score sees segment-local CQT/convolution boundary conditions that a
single whole-track trunk cannot reproduce. A small offset probe found no stable
correction: the first file preferred 0 samples, while the second preferred
about +2,048 samples on the 24-window probe, and the residual errors remained
large.

## Judgment

Exact boundary correction is not practical as a cheap post-processing step.
It would require recomputing a segment-aware CQT and CNN boundary context for
each candidate window (or building and validating a substantially different
streaming CQT with equivalent padding semantics). A constant offset or a
track-edge patch cannot recover the per-window zero-padding behavior.

The dense scorer is practical as a fast optimization surrogate with thorough
coverage, subject to its roughly 1.1–1.3 GB peak RSS and its known logit
transfer error. It must not certify a result. The final exact librosa/ONNX
0.5-second scan remains required and was kept intact.

## Verification

From `thefixer/`, with the project environment active:

```sh
python -m unittest discover -s tests -v
python tools/benchmark_cnn_wholetrack_dense.py \
  /path/to/4d363ababbc6.m4a /path/to/4bf180897cfb.wav
```

The full existing test suite passed: 22 tests.
