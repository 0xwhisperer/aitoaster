"""Multiband compression must stay gentle.

CRITICAL finding from an adversarial mastering audit. This tool briefly
defaulted to iterating up to 12 passes (typically running 9), to stop the UI
telling users to re-run a file by hand. That silently turned a gentle
levelling tool into a multiband limiter.

The gain law is `gain_db = -over * (1 - 1/ratio)` applied to the ALREADY
COMPRESSED output each pass, so the excess decays as (1/ratio)^passes and
the effective ratio compounds:

    1 pass   ->  1.30:1        9 passes  -> 10.60:1
    3 passes ->  2.20:1       12 passes  -> 23.30:1

At a -12dB threshold, 10.6:1 is hard limiting on the BODY of a pop master,
not peak control - while the log still described it as "gentle". The tool
also has no attack or release (zero-phase sosfiltfilt with per-sample gain
from a 20ms median envelope), which is tolerable applied once and is exactly
how a master ends up flat and squeezed when applied nine times deep.

These tests pin the default at one pass and bound the gain reduction.
"""
import inspect
import unittest

import numpy as np

from app import chain


SR = 44100


def _peaky_low_end(sr=SR, dur=6.0):
    """Genuinely peaky sub-200Hz dynamics, ~7dB over target."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    mono = 0.15 * np.sin(2 * np.pi * 1000 * t)
    envelope = np.ones(n) * 0.2
    for start in np.arange(0.3, dur, 0.9):
        i = int(start * sr)
        envelope[i:i + int(0.3 * sr)] = 1.0
    mono = mono + 0.75 * np.sin(2 * np.pi * 80 * t) * envelope
    return np.stack([mono, mono], axis=1).astype(np.float32)


class MultibandGentlenessTests(unittest.TestCase):
    def test_default_is_a_single_pass(self):
        default = inspect.signature(chain.multiband_compress).parameters["max_passes"].default
        self.assertEqual(
            default, 1,
            "multiband must default to ONE pass - iterating compounds the "
            "effective ratio (9 passes of 1.3:1 is 10.6:1)",
        )

    def test_effective_ratio_stays_gentle_on_a_peaky_file(self):
        audio = _peaky_low_end()
        before = max(b["peak_over_db"] for b in chain.detect_band_peakiness(audio, SR))
        self.assertGreater(before, 1.0, "test setup: input must be peaky")

        out, info = chain.multiband_compress(audio, SR)
        after = max(b["peak_over_db"] for b in chain.detect_band_peakiness(out, SR))

        self.assertEqual(info.get("passes"), 1)
        # one pass of 1.3:1 removes ~23% of the excess - it must NOT close the
        # whole gap, because closing it means the ratio was effectively huge
        closed = (before - after) / before
        self.assertLess(
            closed, 0.40,
            f"one pass closed {closed:.0%} of the excess - that is not a "
            "1.3:1 ratio, the gain law is compounding again",
        )

    def test_total_gain_reduction_is_bounded(self):
        """Pop mastering wants <=2dB of multiband GR, not double digits."""
        audio = _peaky_low_end()
        _, info = chain.multiband_compress(audio, SR)
        worst = min(b["max_reduction_db"] for b in info["bands"])
        self.assertGreater(
            worst, -3.0,
            f"multiband pulled {abs(worst):.1f}dB in a single pass - far more "
            "than a gentle tonal-balance stage should",
        )

    def test_already_balanced_audio_is_untouched(self):
        t = np.arange(int(4.0 * SR)) / SR
        mono = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio = np.stack([mono, mono], axis=1)
        out, info = chain.multiband_compress(audio, SR)
        self.assertFalse(info["applied"])
        np.testing.assert_array_equal(out, audio)


if __name__ == "__main__":
    unittest.main()
