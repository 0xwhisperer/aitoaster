import unittest

import numpy as np

from app.cnn_differentiable_v2 import get_real_score_segment
from app.cnn_real_scanner import ParallelRealScoreScanner


class ParallelRealScoreScannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1234)
        cls.audio = rng.normal(0, 0.05, 12 * 16000).astype(np.float32)
        cls.segment_length = 10 * 16000

    def setUp(self):
        self.scanner = ParallelRealScoreScanner(workers=2)

    def tearDown(self):
        self.scanner.close()

    def test_parallel_scores_match_sequential_and_preserve_order(self):
        positions = [2 * 16000, 0, 1 * 16000]
        expected = [
            get_real_score_segment(self.audio[pos:pos + self.segment_length])
            for pos in positions
        ]

        actual = self.scanner.scan(self.audio, positions, self.segment_length)

        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_short_window_ending_out_of_bounds_matches_sequential(self):
        positions = [3 * 16000]
        expected = [
            get_real_score_segment(self.audio[pos:pos + self.segment_length])
            for pos in positions
        ]

        actual = self.scanner.scan(self.audio, positions, self.segment_length)

        self.assertLess(len(self.audio[positions[0]:positions[0] + self.segment_length]), self.segment_length)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_repeat_calls_reuse_workers_and_refresh_shared_audio(self):
        positions = [0, 16000]
        first = self.scanner.scan(self.audio, positions, self.segment_length)
        executor = self.scanner._executor

        changed_audio = (self.audio * 0.25).copy()
        second = self.scanner.scan(changed_audio, positions, self.segment_length)
        expected = [
            get_real_score_segment(changed_audio[pos:pos + self.segment_length])
            for pos in positions
        ]

        self.assertIs(self.scanner._executor, executor)
        np.testing.assert_allclose(second, expected, rtol=0, atol=1e-6)
        self.assertEqual(len(first), len(second))

    def test_worker_failure_is_reported_and_next_call_recovers(self):
        bad_audio = np.array(["not-audio"], dtype="<U9")
        with self.assertRaises(RuntimeError):
            self.scanner.scan(bad_audio, [0], self.segment_length)

        actual = self.scanner.scan(self.audio, [0], self.segment_length)
        expected = [get_real_score_segment(self.audio[:self.segment_length])]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
