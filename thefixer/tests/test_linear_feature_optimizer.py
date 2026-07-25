import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from app import linear_fix
from app.detector import LinearDetector
from app.linear_differentiable import (
    BIAS,
    FREQ_MASK_IDX,
    N_FFT,
    SAMPLE_RATE,
    WEIGHTS,
)
from app.linear_feature_optimizer import (
    fakeprint_from_mean_db,
    optimize_feature_eq,
    score_from_mean_db,
)


class LinearFeatureAlgebraTests(unittest.TestCase):
    def test_feature_score_matches_direct_classifier_algebra(self):
        mean_db = torch.linspace(-50.0, -5.0, N_FFT // 2 + 1)
        fakeprint = fakeprint_from_mean_db(mean_db)
        direct = torch.sigmoid(torch.dot(WEIGHTS, fakeprint) + BIAS)
        self.assertAlmostEqual(
            float(score_from_mean_db(mean_db)), float(direct), places=7
        )
        self.assertEqual(fakeprint.shape[0], len(FREQ_MASK_IDX))

    def test_torch_sufficient_statistic_matches_exact_numpy_detector(self):
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(48000) * 0.1).astype(np.float32)
        window = torch.hann_window(N_FFT, periodic=True)
        stft = torch.stft(
            torch.from_numpy(audio),
            n_fft=N_FFT,
            hop_length=N_FFT // 4,
            win_length=N_FFT,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        mean_db = (
            10.0
            * torch.log10(
                torch.clamp(stft.abs().square(), min=1e-10, max=1e6)
            )
        ).mean(dim=1)

        feature_fakeprint = fakeprint_from_mean_db(mean_db).numpy()
        exact_fakeprint = LinearDetector().compute_fakeprint(audio, SAMPLE_RATE)
        np.testing.assert_allclose(
            feature_fakeprint, exact_fakeprint, rtol=1e-4, atol=5e-5
        )

    def test_rejects_invalid_audio_shape(self):
        with self.assertRaisesRegex(ValueError, "mono 1-D"):
            optimize_feature_eq(np.zeros((10000, 2), dtype=np.float32))

    def test_rejects_too_short_audio(self):
        with self.assertRaisesRegex(ValueError, "more than"):
            optimize_feature_eq(np.zeros(N_FFT // 2, dtype=np.float32))


class LinearFeatureIntegrationTests(unittest.TestCase):
    def _result(self, analysis, delta=1e-4, gain_peak=0.2):
        return SimpleNamespace(
            audio=analysis + delta,
            score=1e-6,
            elapsed_sec=0.25,
            gain_rms_db=0.1,
            gain_peak_db=gain_peak,
        )

    def test_verified_quality_guarded_feature_result_returns_without_waveform_search(self):
        stereo = np.full((N_FFT, 2), 0.1, dtype=np.float32)
        with (
            patch(
                "app.linear_feature_optimizer.optimize_feature_eq",
                side_effect=lambda analysis, **_kwargs: self._result(analysis),
            ),
            patch.object(linear_fix, "_score_stereo_array", return_value=1e-4),
            patch.object(linear_fix, "_optimize_linear") as waveform_optimizer,
        ):
            output, info = linear_fix.fix_linear(
                stereo, 16000, max_retries=0
            )

        self.assertEqual(info["method"], "feature_domain")
        self.assertTrue(info["applied"])
        self.assertLess(info["final_real_score"], linear_fix.ACCEPT_THRESHOLD)
        self.assertFalse(np.array_equal(output, stereo))
        waveform_optimizer.assert_not_called()

    def test_feature_quality_guard_uses_both_delivered_channels(self):
        stereo = np.empty((N_FFT, 2), dtype=np.float32)
        stereo[:, 0] = 1e-8
        stereo[:, 1] = 0.5
        with (
            patch(
                "app.linear_feature_optimizer.optimize_feature_eq",
                side_effect=lambda analysis, **_kwargs: self._result(analysis),
            ),
            patch.object(linear_fix, "_score_stereo_array", return_value=1e-4),
            patch.object(linear_fix, "_optimize_linear") as waveform_optimizer,
        ):
            _output, info = linear_fix.fix_linear(
                stereo, 16000, max_retries=0
            )

        self.assertEqual(info["method"], "feature_domain")
        self.assertGreater(info["snr_db"], 35.0)
        waveform_optimizer.assert_not_called()

    def test_feature_quality_failure_falls_back_to_waveform_optimizer(self):
        stereo = np.full((N_FFT, 2), 0.1, dtype=np.float32)
        with (
            patch(
                "app.linear_feature_optimizer.optimize_feature_eq",
                side_effect=lambda analysis, **_kwargs: self._result(
                    analysis, gain_peak=2.0
                ),
            ),
            patch.object(linear_fix, "_score_stereo_array", return_value=1e-4),
            patch.object(
                linear_fix, "_optimize_linear", return_value=(None, 1.0)
            ) as waveform_optimizer,
        ):
            _output, info = linear_fix.fix_linear(
                stereo, 16000, max_retries=0
            )

        self.assertFalse(info["applied"])
        waveform_optimizer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
