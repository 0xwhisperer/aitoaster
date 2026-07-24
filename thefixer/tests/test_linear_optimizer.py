import unittest
from unittest.mock import patch

import torch

from app import linear_gradient_optimizer as optimizer
from app.linear_fix import (
    REAL_TARGET_FLOOR,
    SURROGATE_TARGET_FLOOR,
    _tighten_retry_targets,
)


class LinearOptimizerControlFlowTests(unittest.TestCase):
    def _run_failed_real_checks(self, max_steps, interval):
        progress_steps = []

        def differentiable_logit(audio):
            return audio.sum() * 0 - 10

        def differentiable_score(_audio):
            return torch.tensor(0.0)

        def penalty(delta, *_args, **_kwargs):
            return (delta ** 2).mean()

        with (
            patch.object(optimizer, "forward_logit_differentiable", differentiable_logit),
            patch.object(optimizer, "forward_score_differentiable", differentiable_score),
            patch.object(optimizer, "compute_masking_mult", lambda _audio: torch.tensor(1.0)),
            patch.object(optimizer, "perceptual_penalty", penalty),
            patch.object(optimizer, "band_limit_penalty", penalty),
            patch.object(optimizer, "tonality_penalty", penalty),
            patch.object(optimizer, "_real_score_for_delta", return_value=0.9),
        ):
            delta, score = optimizer.optimize(
                torch.zeros(32),
                max_steps=max_steps,
                real_check_interval=interval,
                verbose=False,
                progress_cb=lambda step, _mx, _score: progress_steps.append(step),
            )
        return progress_steps, delta, score

    def test_unaligned_budget_extends_but_stops_at_fixed_absolute_ceiling(self):
        steps, delta, score = self._run_failed_real_checks(max_steps=3, interval=2)
        self.assertEqual(steps, list(range(12)))
        self.assertIsNotNone(delta)
        self.assertEqual(score, 0.9)

    def test_small_aligned_budget_terminates(self):
        steps, _, _ = self._run_failed_real_checks(max_steps=3, interval=1)
        self.assertEqual(steps, list(range(12)))

    def test_rejects_nonpositive_loop_parameters(self):
        with self.assertRaises(ValueError):
            optimizer.optimize(torch.zeros(8), max_steps=0, verbose=False)
        with self.assertRaises(ValueError):
            optimizer.optimize(
                torch.zeros(8), max_steps=1, real_check_interval=0, verbose=False
            )


class LinearRetryTargetTests(unittest.TestCase):
    def test_retry_targets_only_tighten_and_stop_at_role_specific_floors(self):
        real_target, surrogate_target = 0.00005, 0.01
        previous = (real_target, surrogate_target)
        for _ in range(20):
            current = _tighten_retry_targets(*previous)
            self.assertLessEqual(current[0], previous[0])
            self.assertLessEqual(current[1], previous[1])
            previous = current
        self.assertEqual(previous, (REAL_TARGET_FLOOR, SURROGATE_TARGET_FLOOR))


if __name__ == "__main__":
    unittest.main()
