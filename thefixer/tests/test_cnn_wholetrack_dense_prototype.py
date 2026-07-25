import unittest
from unittest.mock import patch

import numpy as np
import torch

from app import cnn_differentiable_v2 as base
from app import cnn_wholetrack_dense_prototype as dense


class DenseWholeTrackPrototypeTests(unittest.TestCase):
    def test_split_trunk_and_head_are_exactly_the_existing_graph(self):
        torch.manual_seed(7)
        audio = torch.randn(1, base.SEGMENT_SAMPLES)
        with torch.no_grad():
            cepstrum = base.differentiable_cepstrum(audio)
            split_logit = base.mlp_head_from_pooled(
                base.convolutional_trunk_from_cepstrum(cepstrum).mean((2, 3))
            )
            original_logit = base._model(cepstrum.unsqueeze(1))[:, 0]
        self.assertTrue(torch.equal(split_logit, original_logit))

    def test_model_aligned_grid_geometry(self):
        self.assertEqual(dense.GRID_SAMPLES, 4096)
        self.assertAlmostEqual(dense.GRID_SECONDS, 0.256)
        self.assertEqual(dense.MODEL_POOL_WIDTH, 39)
        positions = dense.dense_grid_positions(10 * base.SR + 3328)
        self.assertEqual(positions.tolist(), [0, 4096])
        self.assertEqual(
            dense.dense_grid_positions(10 * base.SR + 3328, False).tolist(),
            [0],
        )

    def test_first_dense_cell_matches_one_standalone_differentiable_window(self):
        torch.manual_seed(11)
        audio = torch.randn(1, base.SEGMENT_SAMPLES)
        with torch.no_grad():
            dense_logit = dense.forward_dense_logit_grid(audio)[0, 0]
            standalone_logit = base.forward_logit_differentiable(audio)
        self.assertLessEqual(float(torch.abs(dense_logit - standalone_logit)), 1e-5)

    def test_dense_grid_has_one_differentiable_output_per_model_window(self):
        torch.manual_seed(19)
        audio_leaf = torch.randn(12 * base.SR, requires_grad=True)
        audio = audio_leaf.unsqueeze(0)
        logits = dense.forward_dense_logit_grid(audio)
        expected = len(dense.dense_grid_positions(len(audio[0])))
        self.assertEqual(tuple(logits.shape), (1, expected))
        logits.sum().backward()
        self.assertIsNotNone(audio_leaf.grad)
        self.assertTrue(torch.isfinite(audio_leaf.grad).all().item())

    def test_certificate_scan_uses_exact_scorer_and_half_second_hop(self):
        audio = np.zeros(21 * base.SR, dtype=np.float32)
        calls = []

        def fake_exact(segment):
            calls.append(len(segment))
            return float(len(calls))

        with patch.object(dense, "get_real_logit_segment", fake_exact):
            positions, logits = dense.exact_certificate_scan(audio)

        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[1] - positions[0], base.SR // 2)
        self.assertEqual(positions[-1], len(audio) - base.SEGMENT_SAMPLES)
        self.assertEqual(len(calls), len(positions))
        self.assertTrue(np.all(logits == np.arange(1, len(calls) + 1)))


if __name__ == "__main__":
    unittest.main()
