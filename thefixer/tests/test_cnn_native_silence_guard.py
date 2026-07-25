"""The silence guard must survive the 16kHz -> native-rate transfer.

The optimizers gate the delta at the model's 16kHz rate, but the delivered
file is produced by resampling that delta up to the track's native rate and
mixing it into the stereo audio.  Polyphase resampling rings across the gate
boundary, so a delta that is a hard zero across a silent intro at 16kHz comes
back with real energy there at 44.1kHz.

On a near-silent track opening (-68dBFS) that leaked energy measured as loud
as the music itself (correction-to-source ratio ~1.0 where the guard intended
~0.002), which is audible as flutter at the start of the track.
"""
import unittest

import numpy as np

from app.cnn_fix import _transfer_delta_to_stereo


SR = 44100
CNN_SR = 16000


def _silent_intro_track(intro_sec=0.5, total_sec=3.0, sr=SR):
    """A track whose first `intro_sec` is near-silent, then loud."""
    n = int(total_sec * sr)
    n_intro = int(intro_sec * sr)
    rng = np.random.RandomState(0)
    mono = rng.randn(n).astype(np.float32) * 0.2
    # near-silence: about -68dBFS, matching the real track's opening
    mono[:n_intro] *= (10 ** (-68 / 20)) / 0.2
    return np.stack([mono, mono], axis=1)


class NativeSilenceGuardTests(unittest.TestCase):
    def test_transfer_does_not_leak_delta_into_guarded_silence(self):
        stereo = _silent_intro_track()
        n_intro_16k = int(0.5 * CNN_SR)
        n_16k = int(3.0 * CNN_SR)

        # A delta already gated to a hard zero across the silent intro,
        # exactly what apply_silence_guard_to_delta produces at 16kHz.
        rng = np.random.RandomState(1)
        delta_16k = rng.randn(n_16k).astype(np.float32) * 0.05
        delta_16k[:n_intro_16k] = 0.0
        self.assertEqual(float(np.abs(delta_16k[:n_intro_16k]).max()), 0.0)

        out = _transfer_delta_to_stereo(stereo, SR, delta_16k)
        applied = out[:, 0] - stereo[:, 0]

        n_intro = int(0.5 * SR)
        intro_leak = float(np.abs(applied[:n_intro]).max())
        active_rms = float(np.sqrt(np.mean(applied[n_intro:] ** 2)))

        # The guard zeroed this region; the delivered file must keep it
        # essentially zero rather than filling it with resampler ringing.
        self.assertLess(
            intro_leak,
            0.05 * active_rms,
            f"delta leaked {intro_leak:.3e} into guarded silence "
            f"(active rms {active_rms:.3e})",
        )

    def test_correction_stays_far_below_a_near_silent_opening(self):
        """End-to-end audibility check on the actual failure signature."""
        stereo = _silent_intro_track()
        n_16k = int(3.0 * CNN_SR)
        rng = np.random.RandomState(2)
        delta_16k = rng.randn(n_16k).astype(np.float32) * 0.05
        delta_16k[: int(0.5 * CNN_SR)] = 0.0

        out = _transfer_delta_to_stereo(stereo, SR, delta_16k)
        applied = out[:, 0] - stereo[:, 0]

        win = int(0.02 * SR)
        worst_ratio = 0.0
        for start in range(0, int(0.5 * SR), win):
            src = np.sqrt(np.mean(stereo[start:start + win, 0] ** 2) + 1e-12)
            cor = np.sqrt(np.mean(applied[start:start + win] ** 2) + 1e-12)
            worst_ratio = max(worst_ratio, cor / src)

        # Correction must never approach the level of the music it sits under.
        self.assertLess(
            worst_ratio,
            0.25,
            f"correction reached {worst_ratio:.3f}x the source level "
            "in the near-silent opening",
        )


if __name__ == "__main__":
    unittest.main()
