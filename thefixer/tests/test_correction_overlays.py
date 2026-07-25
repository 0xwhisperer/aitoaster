import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from app import server


class CorrectionOverlayTests(unittest.TestCase):
    def test_saves_true_level_amplified_and_combined_files(self):
        linear = np.full((100, 2), 0.001, dtype=np.float32)
        cnn = np.full((100, 2), -0.00025, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, "OUTPUT_DIR", Path(directory)
        ):
            info = server.save_correction_overlays(
                "abc123",
                {"linear": linear, "cnn": cnn},
                44100,
            )
            actual, _ = sf.read(
                Path(directory) / "abc123_overlay_linear.wav",
                dtype="float32",
                always_2d=True,
            )
            preview, _ = sf.read(
                Path(directory) / "abc123_overlay_linear_loud.wav",
                dtype="float32",
                always_2d=True,
            )
            combined, _ = sf.read(
                Path(directory) / "abc123_overlay_combined.wav",
                dtype="float32",
                always_2d=True,
            )

        np.testing.assert_allclose(actual, linear, rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            combined, linear + cnn, rtol=0, atol=1e-8
        )
        self.assertAlmostEqual(float(np.abs(preview).max()), 0.5, places=6)
        self.assertEqual(set(info), {"linear", "cnn", "combined"})
        self.assertGreater(info["linear"]["preview_gain_db"], 0)

    def test_zero_overlay_is_not_exposed(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            server, "OUTPUT_DIR", Path(directory)
        ):
            info = server.save_correction_overlays(
                "abc123",
                {"cnn": np.zeros((20, 2), dtype=np.float32)},
                44100,
            )
            self.assertEqual(info, {})
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
