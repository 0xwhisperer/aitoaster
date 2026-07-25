# Parallel exact-window CNN optimizer

## Production failure used as the acceptance case

The 276.5-second reference job spent 1,390.9 seconds in its first Thorough
CNN pass and 1,112.1 seconds in a second full pass. The first correction
reached 7.82% after transfer, but later linear/timing processing moved the
post-chain CNN score to 99.742%. The second pass still delivered 85.1% CNN,
and the reported whole-chain SNR was -3.2 dB.

The false acceptance was reproducible: the old 0.5-second certificate omitted
the detector's fractional starts (for example, 69.137375 seconds). Nearby
grid points could pass while the omitted exact start scored almost 100%.

## Replacement

- Exact standalone 10-second gradients run in five persistent spawn-safe
  worker processes over shared audio/delta memory.
- The required set includes every production 0.5-second start plus each
  deployed detector start and its full ±0.5-second/0.1-second neighborhood.
- Exact scans initially cover the whole set. Once regions are safe, repeated
  gradients and intermediate scans use the active set plus deterministic
  whole-track sentinels. Full scans remain mandatory every 30 updates and at
  acceptance.
- The unchanged perceptual, frequency-band, tonality, and hard silence guards
  remain active. Constant quality terms are cached; loss and gradient equality
  are covered by an exact regression test.
- Native stereo transfer and PCM16 delivery verification occur while the
  optimizer, Adam moments, correction, weights, and workers are still alive.
  A failed delivery check reactivates its failed starts and continues rather
  than restarting the track.
- Temporal processing and the fast linear solve now precede CNN, removing the
  routine “certify, mutate, then redo the whole CNN track” path.

## Measured production-entry-point results

| Input | Duration | Runtime | Steps | Required windows | Native worst | Headline CNN | SNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `4d363ababbc6.m4a` | 205.2 s | 114.8 s | 30 | 441/441 pass | 0.172% | 0.000006% | 56.4 dB |
| failed-job pre-CNN signal | 276.5 s | 194.0 s | 60 | 584/584 pass | 2.495% | 0.0149% | 48.8 dB |

The long acceptance case is about 13× faster than the failed 42-minute CNN
work and about 7.2× faster than its original 23.2-minute first pass alone.
The remaining non-CNN mastering stages in the recorded job took about one
minute, so expected complete-job time is roughly four minutes rather than
42 minutes when no later emergency retry is needed.

## Rejected alternatives

- A shared feature-domain EQ could move model features quickly, but faithful
  waveform realization needed comb-like ±1.5–3 dB changes, produced only
  13–21 dB SNR, and still left failures.
- One whole-track dense CNN graph was 35–36× faster as a surrogate, but local
  CQT padding/receptive-field differences were too large for authoritative
  certification. It repairs an existing useful correction well but left
  148–149 required failures when asked to build the long-track correction
  from scratch.
- Apple MPS was dramatically slower for this CQT/autograd workload
  (approximately 90 seconds versus 0.6 seconds for a representative batch).
- FGSM/PGD/L-BFGS and low-iteration dense candidates did not meet the exact
  native certificate on both references.

## Correction listening artifacts

Completed jobs now expose linear, CNN, and combined correction-only WAVs.
Each has a true-level player and a separately labeled amplified preview. The
preview is for inspection only and is never mixed into the delivered master.
