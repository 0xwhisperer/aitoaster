import unittest

import numpy as np

from app import chain


class QualityThresholdTests(unittest.TestCase):
    def test_phase_fix_targets_results_table_safety_bar(self):
        rng = np.random.default_rng(4)
        left = rng.standard_normal(50_000)
        independent = rng.standard_normal(50_000)
        right = 0.05 * left + np.sqrt(1 - 0.05 ** 2) * independent
        audio = np.stack([left, right], axis=1).astype(np.float32)

        self.assertLess(chain.stereo_correlation(audio), 0.1)
        fixed, info = chain.fix_phase_issues(
            audio, 44_100, min_correlation=0.1
        )

        self.assertTrue(info["applied"])
        self.assertGreaterEqual(chain.stereo_correlation(fixed), 0.1)


if __name__ == "__main__":
    unittest.main()
