# Measurements

Archived evidence for claims made elsewhere in this codebase. Each file here
is a real measurement, not an illustration - if a docstring or a UI popup
cites a number, this is where the number came from.

## watermark-per-cell-drop-measurement.png

Per-cell verification of the audio watermark, measured on `crazy2_test.mp3`
(178s, 48kHz) by re-analysing the **saved output** with a fresh STFT rather
than trusting the embed-time calculation. That distinction is the point of
the measurement: it proves what actually reached the file.

What it shows, per (time, frequency, bit-slot) position:

| | |
|---|---|
| average drop on a "1" bit | 22.7 dB |
| average change on a "0" bit | 1.85 dB |
| range on "1" bits | 12.8-26.2 dB |
| cells marked | 35 / 112 |

Eight frequency rows from 10,973 Hz to 16,067 Hz, checked twice per time
column (two bit-slots per window) for 16 bit positions total, matching the
16-bit payload.

The two results that matter:

- **"1" bits consistently measure a 12-26 dB drop.** The notch is really
  applied, at every marked location, and survives to the saved file.
- **"0" bits measure ~0 dB.** Nothing was changed where nothing was supposed
  to be changed - which is what makes the mark decodable rather than just a
  general dulling of the top end.

The shallower drops in the last two frequency rows are not a defect: those
bins had less pre-existing energy in this particular track, so there was less
headroom to drop before hitting the noise floor.

Related: the watermark's effect on the AI detectors is measured separately
and is inert - CNN delta exactly 0.00000000 pp, linear delta -0.0000060 pp,
sample count unchanged. See the comment above the `wm` stage in
`app/server.py`.
