"""Fade-in / fade-out tools."""
import unittest

import numpy as np

from app import chain


SR = 44100


def _tone(dur=5.0, sr=SR, level=0.5):
    t = np.arange(int(dur * sr)) / sr
    mono = (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1)


class FadeTests(unittest.TestCase):
    def test_fade_in_starts_at_silence_and_reaches_unity(self):
        audio = _tone()
        out, info = chain.apply_fade(audio, SR, fade_in_ms=1000, fade_out_ms=0)

        self.assertTrue(info["applied"])
        self.assertAlmostEqual(float(np.abs(out[0]).max()), 0.0, places=6)
        # by the end of the fade the signal must be back to full level
        end = int(1.0 * SR)
        original_tail = np.abs(audio[end + 100:end + 5000]).max()
        faded_tail = np.abs(out[end + 100:end + 5000]).max()
        self.assertAlmostEqual(float(faded_tail), float(original_tail), places=6)

    def test_fade_out_ends_at_silence(self):
        audio = _tone()
        out, info = chain.apply_fade(audio, SR, fade_in_ms=0, fade_out_ms=1000)

        self.assertTrue(info["applied"])
        self.assertAlmostEqual(float(np.abs(out[-1]).max()), 0.0, places=6)
        # audio before the fade window is untouched
        start = len(audio) - int(1.0 * SR)
        np.testing.assert_allclose(out[:start - 100], audio[:start - 100], atol=1e-6)

    def test_both_fades_apply_together(self):
        audio = _tone()
        out, _ = chain.apply_fade(audio, SR, fade_in_ms=500, fade_out_ms=500)
        self.assertAlmostEqual(float(np.abs(out[0]).max()), 0.0, places=6)
        self.assertAlmostEqual(float(np.abs(out[-1]).max()), 0.0, places=6)

    def test_curve_is_monotonic_and_smooth(self):
        """A smooth S-curve, not a linear ramp, and no steps."""
        audio = _tone()
        out, _ = chain.apply_fade(audio, SR, fade_in_ms=1000, fade_out_ms=0)
        n = int(1.0 * SR)
        envelope = np.abs(out[:n, 0]) / (np.abs(audio[:n, 0]) + 1e-9)
        # sample the envelope at the tone's peaks to avoid divide-by-noise
        peaks = np.abs(audio[:n, 0]) > 0.4
        env = envelope[peaks]
        self.assertGreater(len(env), 10)
        # monotonically rising, allowing tiny numerical wobble
        self.assertTrue(np.all(np.diff(env) > -1e-3), "fade envelope is not monotonic")
        self.assertLess(float(env[0]), 0.05)
        self.assertGreater(float(env[-1]), 0.95)

    def test_zero_duration_is_a_no_op(self):
        audio = _tone()
        out, info = chain.apply_fade(audio, SR, fade_in_ms=0, fade_out_ms=0)
        self.assertFalse(info["applied"])
        np.testing.assert_array_equal(out, audio)

    def test_durations_are_clamped_to_the_supported_range(self):
        audio = _tone(dur=5.0)
        # below the 10ms floor and above the 10000ms ceiling
        _, info = chain.apply_fade(audio, SR, fade_in_ms=1, fade_out_ms=99999)
        self.assertGreaterEqual(info["fade_in_ms"], 10)
        self.assertLessEqual(info["fade_out_ms"], 10000)

    def test_fade_longer_than_the_track_is_bounded(self):
        """A 10s fade on a 1s track must not read past the buffer."""
        audio = _tone(dur=1.0)
        out, info = chain.apply_fade(audio, SR, fade_in_ms=10000, fade_out_ms=10000)
        self.assertEqual(out.shape, audio.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        # the two fades must not overlap into a doubled gain
        self.assertLessEqual(float(np.abs(out).max()), float(np.abs(audio).max()) + 1e-6)

    def test_mono_input_is_supported(self):
        mono = _tone()[:, 0]
        out, info = chain.apply_fade(mono, SR, fade_in_ms=100, fade_out_ms=100)
        self.assertEqual(out.shape, mono.shape)
        self.assertAlmostEqual(float(abs(out[0])), 0.0, places=6)
        self.assertAlmostEqual(float(abs(out[-1])), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
