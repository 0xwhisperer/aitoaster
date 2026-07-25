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

    def test_borderline_real_pass_does_not_break_before_min_steps(self):
        # BUG FIX (Grok audit, round 4, verified directly against the code
        # before fixing): the real-check break condition used to fire the
        # instant real_score dipped under real_target with NO stability
        # margin at all - unlike the analogous CNN optimizer, which only
        # trusts an early real-verified pass if it's comfortably clear
        # (<half the target), otherwise requiring min_steps of training
        # first. This drives a real_check_interval=1 run with a real_score
        # just barely under real_target (comfortably_clear is false) and
        # confirms the loop keeps running past the first several checks
        # instead of breaking on the very first one.
        def differentiable_logit(audio):
            return audio.sum() * 0 - 10

        def differentiable_score(_audio):
            return torch.tensor(0.0)

        def penalty(delta, *_args, **_kwargs):
            return (delta ** 2).mean()

        real_target = 0.05
        borderline_score = real_target * 0.9  # under target, but NOT comfortably clear (needs < 0.5x)
        budget = optimizer.MIN_STEPS_FOR_BORDERLINE_PASS + 5

        with (
            patch.object(optimizer, "forward_logit_differentiable", differentiable_logit),
            patch.object(optimizer, "forward_score_differentiable", differentiable_score),
            patch.object(optimizer, "compute_masking_mult", lambda _audio: torch.tensor(1.0)),
            patch.object(optimizer, "perceptual_penalty", penalty),
            patch.object(optimizer, "band_limit_penalty", penalty),
            patch.object(optimizer, "tonality_penalty", penalty),
            patch.object(
                optimizer, "_real_score_for_delta", return_value=borderline_score
            ) as real_check,
        ):
            optimizer.optimize(
                torch.zeros(32),
                max_steps=budget,
                real_target=real_target,
                real_check_interval=1,
                verbose=False,
            )

        # the key signal is HOW MANY TIMES the real model was actually
        # invoked before the loop gave up - breaking early on the first
        # borderline check (the bug) calls it once; running out the
        # min_steps floor (the fix) calls it once per step up to the
        # budget, since real_check_interval=1 here.
        self.assertGreaterEqual(
            real_check.call_count,
            optimizer.MIN_STEPS_FOR_BORDERLINE_PASS,
            "a borderline (not comfortably-clear) real-verified pass must not "
            "break out before MIN_STEPS_FOR_BORDERLINE_PASS real checks have "
            "run - it should keep re-checking until the floor is met.",
        )

    def test_comfortably_clear_real_pass_breaks_immediately(self):
        # the flip side of the fix above: a real_score well under half of
        # real_target should NOT be forced to wait out
        # MIN_STEPS_FOR_BORDERLINE_PASS - there's nothing left to gain by
        # continuing to grind on an already-decisive result.
        def differentiable_logit(audio):
            return audio.sum() * 0 - 10

        def differentiable_score(_audio):
            return torch.tensor(0.0)

        def penalty(delta, *_args, **_kwargs):
            return (delta ** 2).mean()

        real_target = 0.05
        clear_score = real_target * 0.1  # comfortably under half the target
        progress_steps = []

        with (
            patch.object(optimizer, "forward_logit_differentiable", differentiable_logit),
            patch.object(optimizer, "forward_score_differentiable", differentiable_score),
            patch.object(optimizer, "compute_masking_mult", lambda _audio: torch.tensor(1.0)),
            patch.object(optimizer, "perceptual_penalty", penalty),
            patch.object(optimizer, "band_limit_penalty", penalty),
            patch.object(optimizer, "tonality_penalty", penalty),
            patch.object(optimizer, "_real_score_for_delta", return_value=clear_score),
        ):
            optimizer.optimize(
                torch.zeros(32),
                max_steps=optimizer.MIN_STEPS_FOR_BORDERLINE_PASS + 50,
                real_target=real_target,
                real_check_interval=1,
                verbose=False,
                progress_cb=lambda step, _mx, _score: progress_steps.append(step),
            )

        self.assertLess(
            len(progress_steps), optimizer.MIN_STEPS_FOR_BORDERLINE_PASS,
            "a comfortably-clear real-verified pass should break out well "
            "before MIN_STEPS_FOR_BORDERLINE_PASS, not be forced to wait.",
        )

    def test_surrogate_convergence_check_is_edge_triggered_not_every_step(self):
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
            patch.object(
                optimizer, "_real_score_for_delta", return_value=0.9
            ) as real_check,
        ):
            optimizer.optimize(
                torch.zeros(32),
                max_steps=4,
                real_check_interval=100,
                verbose=False,
            )

        self.assertLessEqual(
            real_check.call_count,
            4,
            "Crossing the surrogate threshold should add one opportunistic "
            "real check, not turn every later optimizer step into a real check.",
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
