"""The delivered file must actually land on the requested LUFS target.

Direct user report: an export finished at -13.5 LUFS against a -14.0 target,
with no correction applied and no warning logged.

normalize_lufs cannot run last - it is a gain change, and a gain change can
push peaks over the ceiling, so the true-peak limiter has to come after it.
The limiter (and multiband, and the detector fixes) then move the loudness
back. A post-chain correction exists for exactly this, but its threshold was
0.5dB while the real drift measured 0.48dB - just under the bar, so it never
fired.

Measured through the chain on the reported track:

    after normalize_lufs   -14.00
    after multiband        -13.40    <- largest single contributor
    after true_peak_limit  -13.51
    after fade             -13.52

These tests pin the threshold low enough to catch that, and pin the
convergence property the correction depends on.
"""
import unittest

import numpy as np

from app import chain


SR = 44100


def _programme(dur=12.0, sr=SR):
    """Dense broadband material, loud enough to be limited."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.RandomState(0)
    mono = (0.35 * np.sin(2 * np.pi * 110 * t)
            + 0.20 * np.sin(2 * np.pi * 440 * t)
            + 0.10 * rng.randn(n))
    # intermittent peaks so the limiter has something to do
    for start in np.arange(0.5, dur, 1.7):
        i = int(start * sr)
        mono[i:i + int(0.05 * sr)] *= 2.4
    return np.stack([mono, mono], axis=1).astype(np.float32)


class LufsDriftThresholdTests(unittest.TestCase):
    def test_later_stages_do_drift_the_loudness(self):
        """The premise: the chain after normalize_lufs really does move it."""
        audio = _programme()
        norm, _ = chain.normalize_lufs(audio, SR, -14.0)
        self.assertAlmostEqual(chain.measure_lufs(norm, SR), -14.0, delta=0.05)

        after, _ = chain.multiband_compress(norm, SR)
        after, _ = chain.true_peak_limit(after, SR, ceiling_db=-1.0)
        drift = abs(chain.measure_lufs(after, SR) - (-14.0))
        self.assertGreater(
            drift, 0.1,
            "test premise failed: nothing drifted, so there is nothing to catch",
        )

    def test_correction_converges(self):
        """The correction must reach the target, not overshoot and stop.

        A single pass is not enough when the correction is upward: raising
        gain forces a re-limit, and that re-limit pulls loudness back down.
        Measured on this signal, one pass landed at -14.24.
        """
        audio = _programme()
        a, _ = chain.normalize_lufs(audio, SR, -14.0)
        a, _ = chain.multiband_compress(a, SR)
        a, _ = chain.true_peak_limit(a, SR, ceiling_db=-1.0)

        for _ in range(6):
            current = chain.measure_lufs(a, SR)
            step = -14.0 - current
            if abs(step) <= 0.1:
                break
            a = a * (10 ** (step / 20))
            peak = np.abs(a).max()
            if peak > 0.999:
                a = a * (0.999 / peak)
            if step > 0:
                a, _ = chain.true_peak_limit(a, SR, ceiling_db=-1.0)

        self.assertAlmostEqual(
            chain.measure_lufs(a, SR), -14.0, delta=0.1,
            msg="the corrective loop did not land on target",
        )

    def test_upward_correction_does_not_break_the_peak_ceiling(self):
        """Raising gain must not re-breach the ceiling the limiter enforced."""
        from scipy.signal import resample_poly
        audio = _programme()
        a, _ = chain.true_peak_limit(audio, SR, ceiling_db=-1.0)
        # force an upward correction
        raised = a * (10 ** (1.5 / 20))
        peak = np.abs(raised).max()
        if peak > 0.999:
            raised = raised * (0.999 / peak)
        relimited, _ = chain.true_peak_limit(raised, SR, ceiling_db=-1.0)

        tp = 20 * np.log10(np.abs(resample_poly(relimited, 4, 1, axis=0)).max() + 1e-12)
        self.assertLessEqual(
            tp, -1.0 + 0.05,
            f"true peak {tp:.3f}dBTP exceeds the -1.0dBTP ceiling after an "
            "upward loudness correction",
        )

    def test_threshold_is_tight_enough_to_catch_real_drift(self):
        """Guard the constant itself against being loosened back."""
        import re
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "app" / "server.py").read_text()
        m = re.search(r"abs\(final_lufs - target_lufs\) > ([\d.]+)", src)
        self.assertIsNotNone(m, "post-chain LUFS drift check not found")
        self.assertLessEqual(
            float(m.group(1)), 0.2,
            "the drift threshold is too loose to catch the ~0.5dB drift this "
            "chain actually produces",
        )


if __name__ == "__main__":
    unittest.main()
