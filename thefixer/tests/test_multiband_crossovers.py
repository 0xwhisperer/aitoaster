"""4-band complementary crossovers with real attack and release.

Three defects in the previous 3-band implementation, all from the adversarial
mastering audit:

  1. CROSSOVER POINTS. The split was 200Hz / 2000Hz, so the top band ran from
     2kHz to Nyquist. That puts vocal presence (2-5kHz, the range that makes a
     lead vocal intelligible and forward), snare crack and guitar bite in the
     SAME band as cymbals and air, sharing one gain control. A loud cymbal or
     sibilant then ducks the lead vocal's presence along with it. Pop wants
     the presence region in its own band: 100 / 800 / 5000Hz.

  2. NOT COMPLEMENTARY. Bands were independent 4th-order Butterworth
     bandpasses summed back together. Butterworth sections do not sum flat -
     they bump at the crossover - which is why the code carried a 0.97 clamp
     patching "recombination exceeded input peak". Linkwitz-Riley crossovers
     sum flat by construction and remove the need for that patch.

  3. NO ATTACK OR RELEASE. Gain was computed per-sample from a 20ms median
     envelope with no time constants at all. That is spectral gain-riding,
     not compression: it cannot let a transient through, so drums lose punch,
     and it is a large part of why the tool sounded squeezed.
"""
import unittest

import numpy as np

from app import chain


SR = 44100


def _stereo(mono):
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _band_rms(audio, sr, lo, hi):
    from scipy import signal as sig
    nyq = sr / 2
    if lo <= 0:
        sos = sig.butter(4, min(hi, nyq - 1) / nyq, btype="lowpass", output="sos")
    elif hi >= nyq - 1:
        sos = sig.butter(4, lo / nyq, btype="highpass", output="sos")
    else:
        sos = sig.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    return float(np.sqrt((sig.sosfiltfilt(sos, mono) ** 2).mean()))


class CrossoverTests(unittest.TestCase):
    def test_four_bands_with_presence_separated(self):
        bands = chain.default_bands(SR)
        self.assertEqual(len(bands), 4, "pop wants 4 bands, not 3")
        edges = [b[1] for b in bands[:-1]]
        self.assertEqual([round(e) for e in edges], [100, 800, 5000])

    def test_vocal_presence_is_not_in_the_cymbal_band(self):
        """3kHz (vocal presence) and 10kHz (air) must be in different bands."""
        bands = chain.default_bands(SR)

        def band_of(freq):
            for i, (lo, hi) in enumerate(bands):
                if lo <= freq < hi:
                    return i
            return len(bands) - 1

        self.assertNotEqual(
            band_of(3000), band_of(10000),
            "vocal presence and cymbals share a gain control - reducing one "
            "ducks the other",
        )

    def test_crossovers_sum_flat(self):
        """Complementary bands must reconstruct the input exactly."""
        rng = np.random.RandomState(0)
        audio = _stereo(rng.randn(int(2.0 * SR)) * 0.1)
        split = chain.split_bands_complementary(audio, SR, chain.default_bands(SR))
        recombined = sum(split)
        err = np.sqrt(((recombined - audio) ** 2).mean()) / np.sqrt((audio ** 2).mean())
        self.assertLess(
            err, 0.02,
            f"bands sum with {err:.2%} error - they are not complementary, "
            "which is what forced the old 0.97 recombination clamp",
        )

    def test_no_level_bump_at_a_crossover_frequency(self):
        """A tone sitting exactly on a crossover must not gain level."""
        for freq in (100.0, 800.0, 5000.0):
            t = np.arange(int(1.0 * SR)) / SR
            audio = _stereo(0.3 * np.sin(2 * np.pi * freq * t))
            split = chain.split_bands_complementary(audio, SR, chain.default_bands(SR))
            recombined = sum(split)
            ratio = np.sqrt((recombined ** 2).mean()) / np.sqrt((audio ** 2).mean())
            self.assertAlmostEqual(
                float(ratio), 1.0, delta=0.05,
                msg=f"{freq}Hz tone changed level by {20*np.log10(ratio):+.2f}dB "
                    "through the crossover",
            )


class AttackReleaseTests(unittest.TestCase):
    def test_a_transient_is_not_instantly_flattened(self):
        """With real attack, the first moments of a hit pass through."""
        t = np.arange(int(2.0 * SR)) / SR
        mono = (0.08 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
        hit = int(1.0 * SR)
        mono[hit:hit + int(0.05 * SR)] += 0.55
        audio = _stereo(mono)

        out, info = chain.multiband_compress(audio, SR)
        self.assertTrue(info["applied"], "test setup: compressor must engage")

        # the very start of the transient should be far less reduced than
        # its sustained tail - that difference IS the attack time
        onset = float(np.abs(out[hit:hit + 32]).max() / np.abs(audio[hit:hit + 32]).max())
        tail = float(np.abs(out[hit + 1500:hit + 2000]).max()
                     / np.abs(audio[hit + 1500:hit + 2000]).max())
        self.assertGreater(
            onset, tail + 0.02,
            f"onset gain {onset:.3f} vs tail {tail:.3f} - no attack time, the "
            "compressor clamps instantly and kills transients",
        )

    def test_gain_recovers_after_the_loud_moment(self):
        """Release: gain must return toward unity once the peak passes."""
        t = np.arange(int(3.0 * SR)) / SR
        mono = (0.08 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
        hit = int(1.0 * SR)
        mono[hit:hit + int(0.05 * SR)] += 0.55
        audio = _stereo(mono)

        out, _ = chain.multiband_compress(audio, SR)
        late = int(2.5 * SR)
        recovered = float(np.abs(out[late:late + 4000]).max()
                          / np.abs(audio[late:late + 4000]).max())
        self.assertGreater(
            recovered, 0.9,
            f"gain still at {recovered:.3f} a second and a half later - the "
            "release is not recovering",
        )

    def test_gain_envelope_is_smooth(self):
        """No per-sample stepping - that is distortion, not compression."""
        t = np.arange(int(2.0 * SR)) / SR
        rng = np.random.RandomState(1)
        mono = (0.3 * np.sin(2 * np.pi * 120 * t) + 0.1 * rng.randn(len(t))).astype(np.float32)
        audio = _stereo(mono)
        out, _ = chain.multiband_compress(audio, SR)

        gain = np.abs(out[:, 0]) / (np.abs(audio[:, 0]) + 1e-6)
        # sample-to-sample gain movement should be tiny with real smoothing
        jumps = np.abs(np.diff(gain[int(0.5 * SR):int(1.5 * SR)]))
        self.assertLess(
            float(np.percentile(jumps, 99)), 0.5,
            "the gain envelope steps sharply between samples",
        )


class EnvelopeFollowerTests(unittest.TestCase):
    """Assert the follower DIRECTLY, not through the whole compressor.

    An adversarial audit injected 11 mutations and 9 survived the original
    suite - including deleting the attack entirely and REVERSING the per-band
    time constants. The cause was that every test asserted end-to-end through
    multiband_compress, where band-split settling and threshold behaviour
    swamp the thing being measured. These test _envelope_follower itself.
    """

    def _impulse(self, at=4410, n=None, width=10):
        n = n or int(0.5 * SR)
        level = np.zeros(n, dtype=np.float32)
        level[at:at + width] = 1.0
        return level

    def test_envelope_is_strictly_causal(self):
        """Nothing before the transient - a centred window is pre-echo."""
        level = self._impulse()
        env = chain._envelope_follower(level, SR, 8.0, 100.0)
        self.assertEqual(
            float(env[:4410].max()), 0.0,
            "the envelope rises BEFORE the transient - the attack window is "
            "looking into the future, which ducks the mix ahead of the beat",
        )

    def test_release_decays_at_the_stated_time_constant(self):
        """After the peak, the envelope must fall like exp(-t/tau)."""
        level = self._impulse()
        env = chain._envelope_follower(level, SR, 8.0, 100.0)
        peak_end = 4420
        at_release = float(env[peak_end + int(0.100 * SR)])
        # one time constant should leave roughly 1/e; allow generous slack for
        # the peak-hold offset, but a collapsed release reads ~0.07 here
        self.assertGreater(
            at_release, 0.25,
            f"envelope at one release constant is {at_release:.3f} - the "
            "release has collapsed (a broken follower measured 0.074)",
        )
        self.assertLess(at_release, 0.6, "the release is not decaying at all")

    def test_release_does_not_snap_back_in_a_few_samples(self):
        """The specific bug: a 22dB drop in two samples is a click."""
        level = self._impulse()
        env = chain._envelope_follower(level, SR, 8.0, 100.0)
        peak_end = 4420
        drop_db = 20 * np.log10(
            (env[peak_end + 2] + 1e-12) / (env[peak_end - 2] + 1e-12))
        self.assertGreater(
            drop_db, -3.0,
            f"envelope fell {abs(drop_db):.1f}dB in 4 samples - that is a step "
            "discontinuity in the gain, audible as a click on every transient",
        )

    def test_attack_ramps_rather_than_clamping_instantly(self):
        """A step input must not reach full envelope on its first sample."""
        n = int(0.2 * SR)
        level = np.zeros(n, dtype=np.float32)
        level[1000:] = 1.0
        env = chain._envelope_follower(level, SR, 30.0, 200.0)
        self.assertLess(
            float(env[1000]), 0.99,
            "the envelope hits full value on the first sample of a step - "
            "there is no attack time at all",
        )
        self.assertGreater(
            float(env[1000 + int(0.030 * SR)]), 0.9,
            "the envelope has not arrived by the end of the attack window",
        )

    def test_band_time_constants_actually_differ(self):
        """Slow and fast settings must produce different envelopes.

        Kills both 'delete the per-band constants' and 'reverse them'.
        """
        level = self._impulse()
        slow = chain._envelope_follower(level, SR, 30.0, 200.0)
        fast = chain._envelope_follower(level, SR, 3.0, 60.0)
        late = 4420 + int(0.100 * SR)
        self.assertGreater(
            float(slow[late]) - float(fast[late]), 0.1,
            "slow and fast time constants produce near-identical envelopes - "
            "the per-band settings are not being applied",
        )

    def test_low_band_is_slower_than_high_band(self):
        """Guards against the constants being swapped."""
        self.assertGreater(chain.BAND_ATTACK_MS[0], chain.BAND_ATTACK_MS[-1])
        self.assertGreater(chain.BAND_RELEASE_MS[0], chain.BAND_RELEASE_MS[-1])


class GentlenessTests(unittest.TestCase):
    def test_still_one_pass_by_default(self):
        import inspect
        default = inspect.signature(chain.multiband_compress).parameters["max_passes"].default
        self.assertEqual(default, 1)

    def test_reduction_stays_modest(self):
        rng = np.random.RandomState(2)
        t = np.arange(int(4.0 * SR)) / SR
        mono = 0.15 * np.sin(2 * np.pi * 1000 * t)
        env = np.ones(len(t)) * 0.2
        for s in np.arange(0.3, 4.0, 0.9):
            i = int(s * SR)
            env[i:i + int(0.3 * SR)] = 1.0
        mono = (mono + 0.75 * np.sin(2 * np.pi * 80 * t) * env).astype(np.float32)
        audio = _stereo(mono)

        _, info = chain.multiband_compress(audio, SR)
        worst = min(b["max_reduction_db"] for b in info["bands"])
        self.assertGreater(
            worst, -4.0,
            f"pulled {abs(worst):.1f}dB in one pass - too much for a gentle "
            "tonal-balance stage",
        )

    def test_already_balanced_audio_is_untouched(self):
        t = np.arange(int(3.0 * SR)) / SR
        audio = _stereo((0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
        out, info = chain.multiband_compress(audio, SR)
        self.assertFalse(info["applied"])
        np.testing.assert_array_equal(out, audio)

    def test_no_clipping_from_recombination(self):
        """Splitting and summing must not push a hot input over full scale.

        The fixture is normalised to 0.95 first: a raw randn fixture peaks
        above 1.0 on its own, so an un-normalised version tested nothing
        about this tool.
        """
        # dense tonal content, loud enough in-band to actually cross the
        # threshold, then normalised so the INPUT is not already clipping
        t = np.arange(int(2.0 * SR)) / SR
        mono = (0.8 * np.sin(2 * np.pi * 60 * t)
                + 0.5 * np.sin(2 * np.pi * 400 * t)
                + 0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
        mono = (mono / np.abs(mono).max() * 0.95).astype(np.float32)
        audio = _stereo(mono)
        self.assertLessEqual(float(np.abs(audio).max()), 0.96, "test setup")

        out, info = chain.multiband_compress(audio, SR)
        self.assertTrue(info["applied"], "test setup: compressor must engage")
        self.assertLessEqual(float(np.abs(out).max()), 0.999)


if __name__ == "__main__":
    unittest.main()
