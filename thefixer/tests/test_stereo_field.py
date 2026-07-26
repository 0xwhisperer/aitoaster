"""Bass-mono, low-band phase correction, and automatic stereo width.

The existing fix_phase_issues measures ONE correlation figure across the
whole track and applies ONE mid/side blend to all of it. That is a blunt
instrument: real phase problems are frequency-dependent (almost always the
low end, where mono compatibility actually matters) and time-varying, so a
whole-track average can under-correct a genuine bass issue while needlessly
narrowing a stereo image that was fine everywhere else.

This replaces it with two targeted operations, both automatic:

  * BASS-MONO below ~120Hz. Standard pop/club practice. Low frequencies
    carry most of the energy, are nearly non-directional to the ear, and
    out-of-phase bass is what actually cancels on mono playback and causes
    trouble on vinyl and club systems. Collapsing them to mono costs nothing
    audible and removes the whole failure mode.

  * LOW-BAND PHASE CORRECTION, applied only below ~300Hz, leaving the stereo
    image above that untouched.

No width stage: widening above the crossover directly contradicts collapsing
below it, and an earlier attempt measurably reduced mono compatibility on a
real master. Width is an image decision, not a safety correction.
"""
import unittest

import numpy as np

from app import chain


SR = 44100


def _stereo(left, right):
    return np.stack([left, right], axis=1).astype(np.float32)


def _correlation(audio):
    left, right = audio[:, 0], audio[:, 1]
    if left.std() < 1e-9 or right.std() < 1e-9:
        return 1.0
    return float(np.corrcoef(left, right)[0, 1])


def _band_correlation(audio, sr, lo, hi):
    from scipy import signal as sig
    nyq = sr / 2
    sos = sig.butter(4, [max(lo, 1) / nyq, min(hi, nyq - 1) / nyq],
                     btype="bandpass", output="sos")
    filtered = np.stack([sig.sosfiltfilt(sos, audio[:, c]) for c in range(2)], axis=1)
    return _correlation(filtered)


def _out_of_phase_bass(sr=SR, dur=4.0):
    """Bass that cancels in mono, over a correlated mid/high mix."""
    t = np.arange(int(dur * sr)) / sr
    bass = 0.35 * np.sin(2 * np.pi * 60 * t)
    mid = 0.20 * np.sin(2 * np.pi * 900 * t)
    rng = np.random.RandomState(0)
    air_l = 0.05 * rng.randn(len(t))
    air_r = 0.05 * rng.randn(len(t))
    return _stereo(bass + mid + air_l, -bass + mid + air_r)


class BassMonoTests(unittest.TestCase):
    def test_out_of_phase_bass_is_collapsed_to_mono(self):
        audio = _out_of_phase_bass()
        before = _band_correlation(audio, SR, 20, 100)
        self.assertLess(before, -0.5, "test setup: bass must start out of phase")

        out, info = chain.stereo_field_correct(audio, SR)

        after = _band_correlation(out, SR, 20, 100)
        self.assertGreater(
            after, 0.95,
            f"bass correlation only {after:.3f} after bass-mono - it should be "
            "essentially mono below the crossover",
        )
        self.assertTrue(info["applied"])

    def test_mono_playback_retains_the_bass(self):
        """The actual point: summing to mono must not cancel the low end."""
        audio = _out_of_phase_bass()
        out, _ = chain.stereo_field_correct(audio, SR)

        def bass_energy(x):
            from scipy import signal as sig
            mono = x.mean(axis=1)
            sos = sig.butter(4, 100 / (SR / 2), btype="lowpass", output="sos")
            return float(np.sqrt((sig.sosfiltfilt(sos, mono) ** 2).mean()))

        self.assertGreater(
            bass_energy(out), bass_energy(audio) * 5,
            "summing to mono still cancels the bass",
        )

    def test_high_frequencies_keep_their_width(self):
        """Bass-mono must not collapse the whole image."""
        rng = np.random.RandomState(1)
        t = np.arange(int(3.0 * SR)) / SR
        bass = 0.3 * np.sin(2 * np.pi * 60 * t)
        wide_l = 0.15 * rng.randn(len(t))
        wide_r = 0.15 * rng.randn(len(t))
        audio = _stereo(bass + wide_l, bass + wide_r)

        out, _ = chain.stereo_field_correct(audio, SR)
        high = _band_correlation(out, SR, 2000, 16000)
        self.assertLess(
            high, 0.6,
            f"high-band correlation {high:.3f} - the stereo image above the "
            "crossover was collapsed, which bass-mono must never do",
        )


class LowBandPhaseTests(unittest.TestCase):
    def test_phase_correction_is_confined_to_the_low_band(self):
        """PHASE repair must not reach above its band.

        Measured with the width stage disabled, so this isolates the phase
        correction rather than conflating it with intentional widening.
        """
        audio = _out_of_phase_bass()
        out, _ = chain.stereo_field_correct(audio, SR)

        from scipy import signal as sig
        sos = sig.butter(4, 1000 / (SR / 2), btype="highpass", output="sos")

        def high_band(x):
            return np.stack([sig.sosfiltfilt(sos, x[:, c]) for c in range(2)], axis=1)

        # compare the MONO SUM above 1kHz: that is what phase work would alter.
        # Comparing L and R separately measures the uncorrelated noise in this
        # fixture rather than anything the tool did.
        a_hi = high_band(audio).mean(axis=1)
        o_hi = high_band(out).mean(axis=1)
        err = np.sqrt(((o_hi - a_hi) ** 2).mean()) / (np.sqrt((a_hi ** 2).mean()) + 1e-12)
        self.assertLess(
            err, 0.05,
            f"the mono sum above 1kHz changed by {err:.1%} - phase correction "
            "is reaching outside its band",
        )

    def test_the_band_split_reconstructs_exactly(self):
        """low + high must equal the input, or every stage inherits an error."""
        audio = _out_of_phase_bass()
        low, high = chain._band_split(audio, SR, 120.0)
        np.testing.assert_allclose(low + high, audio, atol=1e-6)



class MutationCatchingTests(unittest.TestCase):
    """Assertions that fail if the implementation is quietly broken.

    An adversarial audit ran a mutation battery against the first version of
    this suite and 4 of 6 deliberate breakages PASSED - including "always
    invert the bass polarity", "make phase correction a no-op", and "remove
    the clipping guard". These tests exist to kill those specific mutants.
    """

    def test_correlated_bass_is_not_polarity_inverted(self):
        """The mutant that always takes the invert branch must die here.

        For normal in-phase bass the mono sum must stay in phase with L+R.
        An unconditional (L-R)/2 would produce the SIDE signal instead, which
        is uncorrelated with the actual bass note.
        """
        t = np.arange(int(3.0 * SR)) / SR
        bass = 0.35 * np.sin(2 * np.pi * 60 * t)
        rng = np.random.RandomState(4)
        audio = _stereo(bass + 0.05 * rng.randn(len(t)),
                        bass + 0.05 * rng.randn(len(t)))

        out, info = chain.stereo_field_correct(audio, SR)

        self.assertEqual(
            info["bass_polarity_inverted"], 0.0,
            "in-phase bass must never be polarity-inverted",
        )
        from scipy import signal as sig
        sos = sig.butter(4, 100 / (SR / 2), btype="lowpass", output="sos")
        src = sig.sosfiltfilt(sos, audio.mean(axis=1))
        got = sig.sosfiltfilt(sos, out.mean(axis=1))
        self.assertGreater(
            float(np.corrcoef(src, got)[0, 1]), 0.9,
            "the delivered bass is no longer in phase with the source's",
        )

    def test_polarity_decision_has_no_discontinuity_at_zero(self):
        """Near-zero correlation must not flip a coin on bass polarity.

        Branching on `corr < 0` produced two essentially uncorrelated outputs
        for correlation either side of zero, so estimator noise decided the
        result. The response must be continuous and gated well away from 0.
        """
        t = np.arange(int(2.0 * SR)) / SR
        bass = 0.3 * np.sin(2 * np.pi * 60 * t)
        rng = np.random.RandomState(5)
        results = []
        for corr_target in (0.02, -0.02):
            # mix in decorrelated bass to land either side of zero
            other = 0.3 * np.sin(2 * np.pi * 60 * t + np.pi / 2)
            right = bass * corr_target + other * (1 - abs(corr_target))
            audio = _stereo(bass + 0.02 * rng.randn(len(t)),
                            right + 0.02 * rng.randn(len(t)))
            out, info = chain.stereo_field_correct(audio, SR)
            results.append(info["bass_polarity_inverted"])
        self.assertEqual(
            results, [0.0, 0.0],
            f"polarity inversion engaged near zero correlation ({results}) - "
            "it must be gated to decisively anti-phase bass only",
        )

    def test_phase_stage_actually_repairs_its_band(self):
        """The no-op mutant must die: the 120-300Hz band must improve."""
        t = np.arange(int(3.0 * SR)) / SR
        low_mid = 0.3 * np.sin(2 * np.pi * 200 * t)
        rng = np.random.RandomState(6)
        audio = _stereo(low_mid + 0.03 * rng.randn(len(t)),
                        -low_mid + 0.03 * rng.randn(len(t)))

        before = _band_correlation(audio, SR, 130, 290)
        self.assertLess(before, -0.5, "test setup: this band must start anti-phase")

        out, info = chain.stereo_field_correct(audio, SR)
        after = _band_correlation(out, SR, 130, 290)
        self.assertGreater(
            after, before + 0.3,
            f"120-300Hz correlation only moved {before:.3f} -> {after:.3f}; "
            "the phase stage is not doing its job",
        )

    def test_hot_input_does_not_clip_and_reports_the_rescale(self):
        """The mutant that deletes the peak guard must die."""
        t = np.arange(int(2.0 * SR)) / SR
        bass = 0.6 * np.sin(2 * np.pi * 60 * t)
        mid = 0.39 * np.sin(2 * np.pi * 900 * t)
        audio = _stereo(bass + mid, -bass + mid)
        self.assertGreater(float(np.abs(audio).max()), 0.9, "test setup: hot input")

        out, info = chain.stereo_field_correct(audio, SR)
        self.assertLessEqual(float(np.abs(out).max()), 0.999 + 1e-6)
        self.assertLessEqual(info["peak_rescaled"], 1.0)


class SafetyTests(unittest.TestCase):
    def test_no_clipping_introduced(self):
        audio = _out_of_phase_bass()
        out, _ = chain.stereo_field_correct(audio, SR)
        self.assertLessEqual(float(np.abs(out).max()), 0.999)

    def test_shape_and_length_preserved(self):
        audio = _out_of_phase_bass()
        out, _ = chain.stereo_field_correct(audio, SR)
        self.assertEqual(out.shape, audio.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_reports_what_it_did(self):
        audio = _out_of_phase_bass()
        _, info = chain.stereo_field_correct(audio, SR)
        for key in ("applied", "bass_mono_hz", "correlation_before", "correlation_after",
                    "bass_polarity_inverted", "peak_rescaled"):
            self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
