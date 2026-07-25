import unittest
from unittest.mock import patch

import numpy as np
import torch

from app import cnn_adaptive_dense_prototype as adaptive
from app import cnn_parallel_optimizer as active
from app.cnn_adaptive_dense_prototype import _quality_loss
from app.cnn_low_iteration_prototype import (
    DEFAULT_LAMBDA_BAND,
    DEFAULT_LAMBDA_PERCEPTUAL,
    DEFAULT_LAMBDA_TONALITY,
)
from app.cnn_quality_context import CachedCNNQualityPenalty


class _RecordingScanner:
    def __init__(self, score=0.01):
        self.score = score
        self.calls = []

    def scan(self, _audio, positions, _segment_length):
        self.calls.append(list(positions))
        return [self.score] * len(positions)


class _SequencedScanner:
    scores = []
    calls = []

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def scan(self, _audio, positions, _segment_length):
        self.__class__.calls.append(list(positions))
        score = self.__class__.scores.pop(0)
        return [score] * len(positions)


class _FakeGradientPool:
    calls = 0

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def gradient(self, delta, _positions, _weights):
        self.__class__.calls += 1
        return np.ones_like(delta), 0.9, 1.0


class _FakeQuality:
    def __init__(self, _audio):
        pass

    def loss(self, delta, **_kwargs):
        return delta.square().mean()


class ParallelActivePrototypeTests(unittest.TestCase):
    def test_required_positions_add_fractional_evaluator_eot_neighborhood(self):
        audio = np.zeros(100, dtype=np.float32)
        with (
            patch.object(
                adaptive,
                "dense_window_positions",
                return_value=([0, 10, 20], 40),
            ),
            patch.object(
                adaptive,
                "get_real_evaluator_segments",
                return_value=[25],
            ),
            patch.object(adaptive, "EOT_OFFSETS", (-5, 0, 5)),
        ):
            positions, segment_length = adaptive.required_exact_positions(
                audio
            )
        self.assertEqual(segment_length, 40)
        self.assertEqual(positions, [0, 10, 20, 25, 30])

    def test_exact_certificate_scores_each_required_start_once(self):
        audio = torch.zeros(100)
        scanner = _RecordingScanner()
        with (
            patch.object(
                adaptive,
                "required_exact_positions",
                return_value=([0, 10, 20, 25], 40),
            ),
            patch.object(
                adaptive,
                "apply_silence_guard_to_delta",
                lambda delta, _audio: delta,
            ),
        ):
            _, result = adaptive.exact_union_scan(
                audio, torch.zeros_like(audio), scanner
            )
        self.assertTrue(result.passed)
        self.assertEqual(scanner.calls, [[0, 10, 20, 25]])

    def test_cached_quality_loss_and_gradient_are_identical(self):
        generator = torch.Generator().manual_seed(7)
        audio = torch.randn(32_000, generator=generator) * 0.1
        first = (
            torch.randn(32_000, generator=generator) * 1e-4
        ).requires_grad_(True)
        second = first.detach().clone().requires_grad_(True)
        expected = _quality_loss(first, audio)
        actual = CachedCNNQualityPenalty(audio).loss(
            second,
            lambda_perceptual=DEFAULT_LAMBDA_PERCEPTUAL,
            lambda_band=DEFAULT_LAMBDA_BAND,
            lambda_tonality=DEFAULT_LAMBDA_TONALITY,
        )
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            second.grad, first.grad, rtol=0, atol=0
        )

    def test_sentinels_are_deterministic_and_cover_endpoints(self):
        positions = list(range(100))
        first = active._sentinel_positions(positions, count=8)
        second = active._sentinel_positions(positions, count=8)
        self.assertEqual(first, second)
        self.assertEqual(first[0], positions[0])
        self.assertEqual(first[-1], positions[-1])
        self.assertEqual(len(first), 8)

    def test_invalid_partial_full_scan_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            active.optimize_parallel_active(
                np.zeros(160_000, dtype=np.float32),
                max_steps=1,
                real_check_interval=10,
                full_check_interval=25,
            )

    def test_failed_delivery_check_continues_same_optimizer_session(self):
        _SequencedScanner.scores = [0.9, 0.9, 0.9, 0.9, 0.01]
        _SequencedScanner.calls = []
        _FakeGradientPool.calls = 0
        delivered_deltas = []

        def delivery_transform(delta):
            delivered_deltas.append(delta.copy())
            return np.zeros(8, dtype=np.float32)

        with (
            patch.object(
                active,
                "_pad_for_detector",
                lambda audio: (audio, len(audio)),
            ),
            patch.object(
                active,
                "required_exact_positions",
                return_value=([0], 8),
            ),
            patch.object(active, "LocalGradientPool", _FakeGradientPool),
            patch.object(
                active,
                "ParallelRealScoreScanner",
                _SequencedScanner,
            ),
            patch.object(
                active, "CachedCNNQualityPenalty", _FakeQuality
            ),
            patch.object(
                active,
                "apply_silence_guard_to_delta",
                lambda delta, _audio: delta,
            ),
        ):
            _delta, scan, timing = active.optimize_parallel_active(
                np.zeros(8, dtype=np.float32),
                max_steps=2,
                min_steps=2,
                real_check_interval=1,
                full_check_interval=1,
                lr=0.01,
                max_repair_rounds=0,
                delivery_transform=delivery_transform,
                delivery_check_steps=(1, 2),
            )

        self.assertTrue(scan.passed)
        self.assertTrue(timing["accepted_delivery"])
        self.assertEqual(timing["delivery_checks"], 2)
        self.assertEqual(_FakeGradientPool.calls, 2)
        self.assertEqual(len(delivered_deltas), 2)
        self.assertGreater(
            np.linalg.norm(delivered_deltas[1]),
            np.linalg.norm(delivered_deltas[0]),
        )

if __name__ == "__main__":
    unittest.main()
