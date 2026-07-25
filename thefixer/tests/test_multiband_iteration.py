"""Multiband compression must converge on its own, not ask for re-uploads.

Reported directly: the UI told the user "Still 7.2dB over target in the
0-200Hz band - gentle by design, may take another pass or two to fully
clear", which reads as an instruction to run the file through the tool
repeatedly by hand.

The tool is deliberately gentle (ratio 1.3), so a single pass removes only
1 - 1/1.3 = 23.1% of the excess. Closing a 7.2dB overshoot to under 1dB
therefore needs eight passes:

    pass 0  7.20dB
    pass 1  5.54dB
    pass 4  2.52dB
    pass 8  0.88dB

That is the tool's own arithmetic, not a defect - but the iteration belongs
inside the tool, driven by the file's measured condition, rather than being
handed to the user as homework.
"""
import unittest

import numpy as np

from app import chain


SR = 44100


def _peaky_low_end(sr=SR, dur=6.0):
    """A track with genuinely peaky sub-200Hz dynamics.

    Tuned to land at ~7.2dB over target in the 0-200Hz band, reproducing the
    exact figure from the user's report. The band content must sit ABOVE the
    -12dBFS threshold for a meaningful fraction of the duration - a quieter
    signal measures as not peaky at all, however uneven its envelope.
    """
    n = int(dur * sr)
    t = np.arange(n) / sr
    # steady mid content
    mono = 0.15 * np.sin(2 * np.pi * 1000 * t)
    # low band with large, intermittent peaks - the imbalance to smooth
    envelope = np.ones(n) * 0.2
    for start in np.arange(0.3, dur, 0.9):
        i = int(start * sr)
        envelope[i:i + int(0.3 * sr)] = 1.0
    mono = mono + 0.75 * np.sin(2 * np.pi * 80 * t) * envelope
    return np.stack([mono, mono], axis=1).astype(np.float32)


class MultibandIterationTests(unittest.TestCase):
    def test_worst_band_converges_without_manual_re_runs(self):
        audio = _peaky_low_end()
        before = max(b["peak_over_db"] for b in chain.detect_band_peakiness(audio, SR))
        self.assertGreater(before, 1.0, "test setup: input must be genuinely peaky")

        out, info = chain.multiband_compress(audio, SR)
        after = max(b["peak_over_db"] for b in chain.detect_band_peakiness(out, SR))

        # The single-pass version left 5.76dB here. One call must now do the
        # bulk of the work rather than handing the rest back to the user.
        self.assertLess(
            after, before * 0.25,
            f"only got from {before:.2f}dB to {after:.2f}dB in one call - "
            "the user should not have to re-run it",
        )
        self.assertGreater(info["passes"], 1, "iteration did not happen")

        # And a SECOND call must find essentially nothing left to do, which is
        # the real definition of "converged": re-running by hand buys nothing.
        _, second = chain.multiband_compress(out, SR)
        self.assertLessEqual(
            second.get("passes", 0), 1,
            f"a manual re-run still ran {second.get('passes')} passes - "
            "the tool has not converged",
        )

    def test_reports_how_many_passes_it_ran(self):
        audio = _peaky_low_end()
        _, info = chain.multiband_compress(audio, SR)
        self.assertIn("passes", info)
        self.assertGreaterEqual(info["passes"], 1)

    def test_already_clean_audio_is_left_alone(self):
        """A file that is not peaky must not be processed at all."""
        t = np.arange(int(4.0 * SR)) / SR
        mono = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio = np.stack([mono, mono], axis=1)

        out, info = chain.multiband_compress(audio, SR)

        self.assertFalse(info["applied"])
        np.testing.assert_array_equal(out, audio)

    def test_iteration_is_bounded(self):
        """Pathological input must not loop forever."""
        rng = np.random.RandomState(0)
        n = int(4.0 * SR)
        mono = (rng.randn(n) * 0.4).astype(np.float32)
        mono[::5000] *= 8.0  # repeated extreme peaks
        audio = np.stack([mono, mono], axis=1)

        out, info = chain.multiband_compress(audio, SR)

        self.assertLessEqual(info.get("passes", 0), 12)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_does_not_introduce_clipping(self):
        audio = _peaky_low_end()
        out, _ = chain.multiband_compress(audio, SR)
        self.assertLessEqual(float(np.abs(out).max()), 0.98)


if __name__ == "__main__":
    unittest.main()
