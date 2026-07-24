import unittest
from unittest.mock import patch

import numpy as np
import torch

from app import cnn_wholetrack_optimizer_v2 as optimizer


class CnnPostGuardVerificationTests(unittest.TestCase):
    def test_whole_track_reports_regression_introduced_by_guard(self):
        audio = np.zeros(optimizer.SEGMENT_SAMPLES, dtype=np.float32)

        def differentiable_logit(segment):
            return segment.sum() * 0 - 10

        def penalty(delta, *_args, **_kwargs):
            return (delta ** 2).mean()

        def real_score(segment):
            return 0.9 if abs(float(np.mean(segment))) > 0.0005 else 0.01

        with (
            patch.object(optimizer, "forward_logit_differentiable", differentiable_logit),
            patch.object(optimizer, "perceptual_penalty", penalty),
            patch.object(optimizer, "band_limit_penalty", penalty),
            patch.object(optimizer, "tonality_penalty", penalty),
            patch.object(optimizer, "get_real_score_segment", real_score),
            patch.object(
                optimizer,
                "apply_silence_guard_to_delta",
                lambda delta, _audio: delta + 0.001,
            ),
        ):
            guarded_delta, _, _, post_guard_worst = (
                optimizer.optimize_whole_track_verified(
                    audio,
                    max_steps=2,
                    min_steps=1,
                    real_check_interval=1,
                    verbose=False,
                    mode="simple",
                )
            )

        self.assertGreater(float(np.mean(guarded_delta)), 0.0005)
        self.assertEqual(post_guard_worst, 0.9)


if __name__ == "__main__":
    unittest.main()
