import unittest
from unittest.mock import patch

import numpy as np

from app import watermark


class WatermarkFrequencyTests(unittest.TestCase):
    def test_v2_frequencies_stay_in_documented_band(self):
        for seed in range(10_000):
            freqs = watermark.derive_frequencies(seed, version=2)
            self.assertTrue(all(
                watermark.FREQ_BAND_LO_HZ <= freq <= watermark.FREQ_BAND_HI_HZ
                for freq in freqs
            ))

    def test_v2_frequencies_occupy_distinct_sub_bands(self):
        band_width = watermark.FREQ_BAND_HI_HZ - watermark.FREQ_BAND_LO_HZ
        for seed in range(1_000):
            for index, freq in enumerate(watermark.derive_frequencies(seed, version=2)):
                slot_lo = watermark.FREQ_BAND_LO_HZ + index * band_width // watermark.N_FREQ_BINS
                slot_hi = (
                    watermark.FREQ_BAND_HI_HZ
                    if index == watermark.N_FREQ_BINS - 1
                    else watermark.FREQ_BAND_LO_HZ
                    + (index + 1) * band_width // watermark.N_FREQ_BINS - 1
                )
                self.assertLessEqual(slot_lo, freq)
                self.assertLessEqual(freq, slot_hi)

    def test_v1_derivation_is_retained_for_legacy_detection(self):
        self.assertGreater(
            max(watermark.derive_frequencies(457, version=1)),
            watermark.FREQ_BAND_HI_HZ,
        )
        self.assertLessEqual(
            max(watermark.derive_frequencies(457, version=2)),
            watermark.FREQ_BAND_HI_HZ,
        )

    def test_v2_embed_detect_round_trip(self):
        rng = np.random.default_rng(7)
        audio = (rng.standard_normal(3 * 44_100) * 0.08).astype(np.float32)
        marked = watermark.embed_watermark(audio, 44_100, seed=3170)
        found, version, _ = watermark.detect_watermark(marked, 44_100, seed=3170)
        self.assertTrue(found)
        self.assertEqual(version, 2)

    def test_v1_mark_remains_detectable(self):
        rng = np.random.default_rng(11)
        audio = (rng.standard_normal(3 * 44_100) * 0.08).astype(np.float32)
        with patch.object(watermark, "WATERMARK_VERSION", 1):
            marked = watermark.embed_watermark(audio, 44_100, seed=457)
        found, version, _ = watermark.detect_watermark(marked, 44_100, seed=457)
        self.assertTrue(found)
        self.assertEqual(version, 1)


if __name__ == "__main__":
    unittest.main()
