"""The correction must stay masked by the source IN THE MODEL'S OWN BAND.

The CNN is a CQT-cepstrum model over 500Hz-8kHz.  A cepstral representation
is scale-invariant: it responds to spectral SHAPE, not absolute level.  So on
a passage that is loud broadband but nearly empty inside 500Hz-8kHz (a
bass-only intro, for example), the model's gradient explodes - measured at
1.58e+03 vs ~2-10 in the body of the track, a grad/source ratio of 2.6e+05
vs ~20-100 - because a tiny absolute change swings the cepstral shape a lot
when there is almost no in-band signal to begin with.

The optimizer happily spends that gradient, and the existing guards do not
stop it:

  * the silence guard is BROADBAND, so a -25dBFS bass-only intro reads as
    "loud" (gate = 1.000) even though 500Hz-8kHz is at -45 to -74dBFS;
  * perceptual_penalty's masking multiplier bottoms out at 0.05 for quiet
    bins, which makes injecting into an EMPTY band roughly 10x CHEAPER than
    injecting into a loud one - backwards, since unmasked energy is exactly
    what the ear picks out.

Measured on "North Star", the delivered correction reached -0.4dB relative
to the in-band source at t=0.60s, and 23 of the 30 worst blocks in a 277s
track fell inside the first 5 seconds.  That is the audible flutter.
"""
import unittest

import numpy as np

from app.cnn_fix import _apply_inband_audibility_ceiling


SR = 44100
BAND_LO, BAND_HI = 500, 8000


def _band_rms(x, sr, lo, hi, win):
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    spectrum[(freqs < lo) | (freqs >= hi)] = 0
    filtered = np.fft.irfft(spectrum, len(x))
    n_blocks = len(filtered) // win
    blocks = filtered[: n_blocks * win].reshape(n_blocks, win)
    return np.sqrt((blocks ** 2).mean(axis=1) + 1e-20)


def _bass_only_intro(sr=SR, intro_sec=2.0, total_sec=6.0):
    """Loud broadband, nearly empty in 500Hz-8kHz - the real failure case."""
    n = int(total_sec * sr)
    t = np.arange(n) / sr
    # strong 80Hz bass everywhere: broadband level stays high
    audio = 0.25 * np.sin(2 * np.pi * 80 * t).astype(np.float32)
    # in-band content only AFTER the intro
    n_intro = int(intro_sec * sr)
    rng = np.random.RandomState(0)
    band = rng.randn(n).astype(np.float32) * 0.15
    spectrum = np.fft.rfft(band)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spectrum[(freqs < BAND_LO) | (freqs >= BAND_HI)] = 0
    band = np.fft.irfft(spectrum, n).astype(np.float32)
    band[:n_intro] = 0.0
    return (audio + band).astype(np.float32)


class InBandAudibilityTests(unittest.TestCase):
    def test_correction_is_capped_relative_to_inband_source(self):
        mono = _bass_only_intro()
        rng = np.random.RandomState(1)
        # A correction concentrated in the model's band, flat across time -
        # what the level-invariant CNN gradient actually produces.
        delta = rng.randn(len(mono)).astype(np.float32) * 0.02
        spectrum = np.fft.rfft(delta)
        freqs = np.fft.rfftfreq(len(delta), 1 / SR)
        spectrum[(freqs < BAND_LO) | (freqs >= BAND_HI)] = 0
        delta = np.fft.irfft(spectrum, len(delta)).astype(np.float32)

        win = int(0.02 * SR)
        before = _band_rms(delta, SR, BAND_LO, BAND_HI, win) / (
            _band_rms(mono, SR, BAND_LO, BAND_HI, win) + 1e-12
        )
        # sanity: the unguarded correction really is audible in the intro
        n_intro_blocks = int(2.0 * SR) // win
        self.assertGreater(float(before[:n_intro_blocks].max()), 1.0)

        guarded = _apply_inband_audibility_ceiling(delta, mono, SR)
        after = _band_rms(guarded, SR, BAND_LO, BAND_HI, win) / (
            _band_rms(mono, SR, BAND_LO, BAND_HI, win) + 1e-12
        )

        # In the intro the correction must sit well below the in-band source.
        self.assertLess(
            float(after[:n_intro_blocks].max()),
            0.35,
            "correction still audible above the in-band source in the intro",
        )

    def test_loud_in_band_passages_are_left_alone(self):
        """The ceiling must not weaken the fix where the music masks it."""
        mono = _bass_only_intro()
        rng = np.random.RandomState(2)
        delta = rng.randn(len(mono)).astype(np.float32) * 0.002

        guarded = _apply_inband_audibility_ceiling(delta, mono, SR)

        body = slice(int(3.0 * SR), int(6.0 * SR))
        kept = np.sqrt(np.mean(guarded[body] ** 2)) / (
            np.sqrt(np.mean(delta[body] ** 2)) + 1e-12
        )
        self.assertGreater(
            kept,
            0.95,
            f"ceiling attenuated a well-masked correction to {kept:.3f}",
        )

    def test_gain_envelope_is_smooth_enough_not_to_flutter(self):
        """A hard per-block gain step is itself an audible modulation."""
        mono = _bass_only_intro()
        rng = np.random.RandomState(3)
        delta = rng.randn(len(mono)).astype(np.float32) * 0.02

        guarded = _apply_inband_audibility_ceiling(delta, mono, SR)
        gain = guarded / (delta + 1e-12)

        # Sample-to-sample gain change must stay small: no staircase edges.
        jumps = np.abs(np.diff(gain))
        self.assertLess(
            float(jumps.max()),
            0.02,
            f"gain envelope steps by {jumps.max():.4f} between samples",
        )


if __name__ == "__main__":
    unittest.main()
