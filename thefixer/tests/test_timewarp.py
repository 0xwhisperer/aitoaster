import unittest

import numpy as np

from app.timewarp import generate_warp_curve


class TimeWarpTests(unittest.TestCase):
    def test_curve_is_seed_reproducible_and_bounded(self):
        curve_a = generate_warp_curve(48_000, 48_000, seed=42, max_drift_ms=8)
        curve_b = generate_warp_curve(48_000, 48_000, seed=42, max_drift_ms=8)
        np.testing.assert_array_equal(curve_a, curve_b)
        self.assertLessEqual(np.abs(curve_a).max(), 0.008)

    def test_different_seeds_change_curve(self):
        curve_a = generate_warp_curve(48_000, 48_000, seed=1)
        curve_b = generate_warp_curve(48_000, 48_000, seed=2)
        self.assertFalse(np.array_equal(curve_a, curve_b))


if __name__ == "__main__":
    unittest.main()
