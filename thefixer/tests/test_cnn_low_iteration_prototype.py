import unittest
from unittest.mock import patch

import numpy as np
import torch

from app import cnn_low_iteration_prototype as prototype


class LowIterationPrototypeTests(unittest.TestCase):
    def test_gradient_sanitizer_accepts_only_measured_endpoint_bands(self):
        gradient = torch.ones(100)
        gradient[:2] = float("nan")
        gradient[-3:] = float("nan")
        with patch.object(prototype, "MAX_CQT_BOUNDARY_GRAD_SEC", 0.001):
            sanitized = prototype._sanitize_boundary_gradient(gradient)
        self.assertTrue(torch.isfinite(sanitized).all())
        self.assertTrue(torch.equal(sanitized[2:-3], torch.ones(95)))

        interior = torch.ones(100)
        interior[50] = float("nan")
        with patch.object(prototype, "MAX_CQT_BOUNDARY_GRAD_SEC", 0.001):
            with self.assertRaises(ValueError):
                prototype._sanitize_boundary_gradient(interior)

    def test_logical_union_deduplicates_absolute_eot_starts(self):
        positions = prototype.logical_union_positions(
            40_000,
            [10_000, 18_000],
            10_000,
            jitter_sec=0.5,
            step_sec=0.5,
        )
        self.assertEqual(positions, [2_000, 10_000, 18_000, 26_000])

    def test_certification_scores_each_union_start_once(self):
        audio = np.arange(8, dtype=np.float32)
        delta = np.zeros_like(audio)
        seen = []

        def score(segment):
            seen.append(segment.copy())
            return float(np.mean(segment))

        with (
            patch.object(prototype, "SEGMENT_SAMPLES", 4),
            patch.object(prototype, "dense_window_positions", return_value=([2, 4], 4)),
            patch.object(prototype, "apply_silence_guard_to_delta", lambda d, _a: d),
        ):
            _, result = prototype.certify_delta(
                audio,
                delta,
                jitter_sec=1 / prototype.SR,
                step_sec=1 / prototype.SR,
                score_fn=score,
            )

        self.assertEqual(result.dense_window_count, 2)
        self.assertEqual(result.union_window_count, 4)
        self.assertEqual(len(seen), result.union_window_count)
        self.assertEqual(result.certified_worst_score, 5.5)

    def test_adam_uses_one_whole_track_graph_per_step(self):
        calls = []

        def fake_dense(audio_batch):
            calls.append(tuple(audio_batch.shape))
            return audio_batch.mean(dim=1, keepdim=True)

        def penalty(delta, *_args, **_kwargs):
            return (delta ** 2).mean()

        with (
            patch.object(prototype, "SEGMENT_SAMPLES", 8),
            patch.object(prototype, "forward_dense_logit_grid", fake_dense),
            patch.object(prototype, "perceptual_penalty", penalty),
            patch.object(prototype, "band_limit_penalty", penalty),
            patch.object(prototype, "tonality_penalty", penalty),
            patch.object(prototype, "apply_silence_guard_to_delta", lambda d, _a: d),
        ):
            delta = prototype.optimize_low_iteration(
                np.zeros(8, dtype=np.float32), method="adam", steps=3,
            )

        self.assertEqual(delta.shape, (8,))
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(shape == (1, 8) for shape in calls))

    def test_all_candidate_methods_are_explicitly_supported(self):
        self.assertEqual(
            {"fgsm", "adam", "pgd", "lbfgs"},
            {"fgsm", "adam", "pgd", "lbfgs"},
        )
        with self.assertRaises(ValueError):
            prototype.optimize_low_iteration(
                np.zeros(8, dtype=np.float32), method="nope", steps=1,
            )


if __name__ == "__main__":
    unittest.main()
