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
        # a hit long enough for the attack to arrive, so "one time constant
        # later" is measured against a meaningful peak. A 10-sample impulse
        # against an 8ms attack only reaches 0.08, and 1/e of that is 0.03 -
        # which an earlier version of this test misread as a collapsed release.
        level = self._impulse(width=int(0.050 * SR))
        env = chain._envelope_follower(level, SR, 8.0, 100.0)
        peak = float(env.max())
        peak_end = 4410 + int(0.050 * SR)
        ratio = float(env[peak_end + int(0.100 * SR)]) / peak
        self.assertGreater(
            ratio, 0.25,
            f"envelope fell to {ratio:.3f} of peak after one release constant "
            "- the release has collapsed",
        )
        self.assertLess(ratio, 0.6, "the release is not decaying at all")

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
        level = self._impulse(width=int(0.050 * SR))
        slow = chain._envelope_follower(level, SR, 30.0, 200.0)
        fast = chain._envelope_follower(level, SR, 3.0, 60.0)
        late = 4410 + int(0.050 * SR) + int(0.100 * SR)
        self.assertGreater(
            float(slow[late]) - float(fast[late]), 0.1,
            "slow and fast time constants produce near-identical envelopes - "
            "the per-band settings are not being applied",
        )

    def test_gain_reduction_lands_on_the_transient_not_after_it(self):
        """The defect a second audit blocked: GR arriving too late.

        A dilation-then-ramp cascade delayed peak gain reduction to +40ms on
        the low band, delivering only 9.5% of the intended reduction while
        the transient was actually present and the rest onto whatever
        followed - gain reduction pointed at the wrong audio.
        """
        hit_len = int(0.010 * SR)
        level = np.zeros(int(0.3 * SR), dtype=np.float32)
        level[4410:4410 + hit_len] = 1.0
        env = chain._envelope_follower(level, SR, 30.0, 200.0)

        during = float(env[4410:4410 + hit_len].mean())
        self.assertGreater(
            during, 0.25,
            f"envelope averages {during:.3f} during the transient - the "
            "compressor is barely reacting while the peak is present",
        )
        peak_at_ms = (int(np.argmax(env)) - 4410) / SR * 1000
        self.assertLess(
            peak_at_ms, 35.0,
            f"peak gain reduction lands {peak_at_ms:.1f}ms after onset, well "
            "past the 30ms attack window - it is ducking the NEXT sound",
        )

    def test_attack_completes_within_its_window(self):
        """On a transient longer than the attack, the envelope must arrive."""
        level = np.zeros(int(0.5 * SR), dtype=np.float32)
        level[4410:4410 + int(0.060 * SR)] = 1.0
        env = chain._envelope_follower(level, SR, 30.0, 200.0)
        at_window_end = float(env[4410 + int(0.030 * SR)])
        self.assertGreater(
            at_window_end, 0.9,
            f"envelope only reached {at_window_end:.3f} by the end of its own "
            "30ms attack window",
        )

    def test_matches_the_documented_recurrence_exactly(self):
        """The JIT path must equal the recurrence the docstring states.

        Earlier versions composed filters instead, and the docstring described
        a recurrence the code never computed - a literal implementation
        deviated from the shipped one by 0.33.
        """
        rng = np.random.RandomState(7)
        level = np.abs(rng.randn(30000)).astype(np.float32)
        env = chain._envelope_follower(level, SR, 1.0, 0.1)

        attack_coeff = float(np.exp(-3.0 / max(1.0 * 0.001 * SR, 1.0)))
        release_coeff = float(np.exp(-1.0 / max(0.1 * 0.001 * SR, 1.0)))
        reference = np.empty(len(level), dtype=np.float64)
        prev = float(level[0])
        for i, x in enumerate(level):
            x = float(x)
            if x > prev:
                prev = prev + (1.0 - attack_coeff) * (x - prev)
            else:
                prev = max(x, prev * release_coeff)
            reference[i] = prev

        self.assertEqual(len(env), len(level))
        self.assertTrue(np.all(np.isfinite(env)))
        self.assertLess(
            float(np.abs(env - reference).max()), 1e-6,
            "the follower diverges from the recurrence its docstring states",
        )

    def test_compressor_never_boosts_below_threshold(self):
        """Removing the over-threshold clamp turns this into an EXPANDER.

        Asserts on the GAIN ITSELF (max_gain_db), not on a band's RMS. Two
        earlier versions of this test measured band RMS and both were
        structurally blind: the boost multiplies a near-silent band, so its
        absolute RMS barely moves while the actual gain reaches +22.5dB. The
        mutant survived a test written specifically to catch it, twice.
        """
        t = np.arange(int(1.0 * SR)) / SR
        loud_low = 0.7 * np.sin(2 * np.pi * 60 * t)
        quiet_high = 0.001 * np.sin(2 * np.pi * 9000 * t)
        audio = _stereo((loud_low + quiet_high).astype(np.float32))

        _, info = chain._multiband_compress_pass(audio, SR)
        for band in info["bands"]:
            self.assertLessEqual(
                band["max_gain_db"], 1e-9,
                f"band {band['range_hz']}Hz applied POSITIVE gain "
                f"({band['max_gain_db']:+.2f}dB) - the over-threshold clamp "
                "is missing, so signal UNDER the threshold gets boosted. "
                "This is an expander, not a compressor.",
            )

    def test_public_compressor_never_boosts_either(self):
        """Same guarantee through the public multi-pass entry point."""
        t = np.arange(int(1.0 * SR)) / SR
        rng = np.random.RandomState(7)
        mono = (0.6 * np.sin(2 * np.pi * 60 * t)
                + 0.002 * np.sin(2 * np.pi * 9000 * t)
                + 0.05 * rng.randn(len(t))).astype(np.float32)
        audio = _stereo(mono)

        out, info = chain.multiband_compress(audio, SR)
        for band in info["bands"]:
            self.assertLessEqual(
                band["max_gain_db"], 1e-9,
                f"band {band['range_hz']}Hz boosted by "
                f"{band['max_gain_db']:+.2f}dB",
            )
        self.assertLessEqual(
            float(np.abs(out).max()), float(np.abs(audio).max()) + 1e-6,
            "output peak exceeds input peak - a band was boosted",
        )

    def test_detection_uses_channel_PEAK_not_channel_average(self):
        """A hard-panned loud element must compress on its own level.

        Averaging the channels halves the detected level for anything panned
        wide, so a loud hard-panned bass or guitar sails under the threshold
        and never gets compressed - and the compression it DOES get depends
        on how wide the mix is, which is musically wrong.
        """
        t = np.arange(int(1.0 * SR)) / SR
        loud = (0.7 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
        silent = np.zeros_like(loud)
        panned = np.stack([loud, silent], axis=1)     # max=0.70, mean=0.35
        centred = np.stack([loud, loud], axis=1)      # max=0.70, mean=0.70

        _, panned_info = chain._multiband_compress_pass(panned, SR)
        _, centred_info = chain._multiband_compress_pass(centred, SR)

        p_red = panned_info["bands"][0]["max_reduction_db"]
        c_red = centred_info["bands"][0]["max_reduction_db"]
        self.assertAlmostEqual(
            p_red, c_red, delta=0.05,
            msg=f"hard-panned element reduced {p_red:.2f}dB but the same "
                f"element centred reduced {c_red:.2f}dB - detection is "
                "averaging the channels instead of taking their peak, so "
                "compression depends on stereo placement",
        )

    def test_gain_slope_follows_the_compression_ratio(self):
        """Reduction must be over_db * (1 - 1/ratio), the compressor law.

        Confusing this with (1/ratio) is a silent 3.3x over-compression at
        the default 1.3 ratio - the tool would sound crushed while still
        reporting ratio 1.3.
        """
        t = np.arange(int(1.0 * SR)) / SR
        mono = (0.9 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
        audio = _stereo(mono)

        reductions = {}
        for ratio in (1.3, 2.0, 4.0):
            _, info = chain._multiband_compress_pass(audio, SR, ratio=ratio)
            reductions[ratio] = -info["bands"][0]["max_reduction_db"]

        # over_db is fixed (same signal, same threshold), so the ratio of two
        # reductions must equal the ratio of their (1 - 1/r) slopes.
        for r in (2.0, 4.0):
            expected = (1 - 1 / r) / (1 - 1 / 1.3)
            actual = reductions[r] / reductions[1.3]
            self.assertAlmostEqual(
                actual, expected, delta=0.05,
                msg=f"ratio {r}: reduction scaled {actual:.3f}x vs 1.3 but "
                    f"the compressor law requires {expected:.3f}x - the gain "
                    "slope is not (1 - 1/ratio)",
            )

        # The ratio check above is SCALE-INVARIANT: any constant multiple
        # K*(1-1/r) cancels exactly, so a 3x over-compression (effective
        # 3.25:1 while the UI still says 1.3:1) satisfies it. Pin the
        # absolute values too. This signal sits 8.2992dB over threshold.
        for ratio, expected_db in ((1.3, 1.9152), (2.0, 4.1496), (4.0, 6.2244)):
            self.assertAlmostEqual(
                reductions[ratio], expected_db, delta=0.02,
                msg=f"ratio {ratio}: reduced {reductions[ratio]:.4f}dB but "
                    f"over_db*(1-1/ratio) requires {expected_db:.4f}dB - the "
                    "gain law is off by a constant factor, which the "
                    "scale-invariant ratio check above cannot see",
            )

    def test_max_gain_db_reflects_the_real_gain_path(self):
        """The expander sentinel must not be a constant.

        max_gain_db was added to catch a missing over-threshold clamp, but
        nothing asserted it tracked the actual gain - so hardcoding it to
        0.0, or setting it to gain_db.MIN, passed the whole suite. A guard
        that can be stubbed out silently is not a guard.
        """
        t = np.arange(int(1.0 * SR)) / SR
        mono = (0.9 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
        _, info = chain._multiband_compress_pass(_stereo(mono), SR)
        band = info["bands"][0]

        # this band is genuinely compressing, so its gain sweeps from the
        # peak reduction up to (but never past) 0dB
        self.assertLess(
            band["max_reduction_db"], -0.5,
            "test setup: band 0 must actually be reducing",
        )
        self.assertLess(
            band["max_gain_db"], 0.0,
            f"max_gain_db is {band['max_gain_db']:.4f} on a band that is "
            "actively reducing - it is not reading the gain path (a "
            "hardcoded 0.0 looks exactly like this)",
        )
        self.assertGreater(
            band["max_gain_db"], band["max_reduction_db"],
            f"max_gain_db ({band['max_gain_db']:.4f}) is not above "
            f"max_reduction_db ({band['max_reduction_db']:.4f}) - it is "
            "reporting the MINIMUM gain, so a boost could never surface",
        )

        # EXACT equality against an independently recomputed gain array.
        # Range checks alone still let the sentinel be stubbed to -1e-12,
        # or scaled by 0.5, or wrapped in min(..., 0.0) - all of which
        # blind the expander guard while looking healthy.
        bands_def = chain.default_bands(SR)
        band_audio = chain.split_bands_complementary(
            _stereo(mono), SR, bands_def)[0]
        level = np.abs(band_audio).max(axis=1)
        env = chain._envelope_follower(
            level, SR, chain.BAND_ATTACK_MS[0], chain.BAND_RELEASE_MS[0])
        env_db = 20 * np.log10(np.maximum(env, 1e-8))
        over = np.maximum(env_db - info["threshold_db"], 0)
        gain_db = -over * (1 - 1 / info["ratio"])

        self.assertAlmostEqual(
            band["max_gain_db"], float(gain_db.max()), delta=1e-9,
            msg=f"max_gain_db reports {band['max_gain_db']:.6g} but the "
                f"recomputed gain path peaks at {float(gain_db.max()):.6g} - "
                "the sentinel is not the real maximum gain",
        )
        self.assertAlmostEqual(
            band["max_reduction_db"], float(gain_db.min()), delta=1e-9,
            msg="max_reduction_db does not match the recomputed gain path",
        )

    def test_multipass_aggregation_surfaces_a_boost_in_any_pass(self):
        """Across passes the sentinel must aggregate by MAX, not min.

        Must run with max_passes>1: at the default of 1 pass, max() and
        min() over a single element are identical, so this defect is
        literally unobservable through a default-configured run.
        """
        t = np.arange(int(2.0 * SR)) / SR
        rng = np.random.RandomState(3)
        mono = (0.6 * np.sin(2 * np.pi * 60 * t)
                + 0.3 * np.sin(2 * np.pi * 3000 * t)
                + 0.05 * rng.randn(len(t))).astype(np.float32)
        audio = _stereo(mono)

        real_pass = chain._multiband_compress_pass
        state = {"n": 0}

        def boost_on_first_pass(a, sr, **kw):
            out, info = real_pass(a, sr, **kw)
            state["n"] += 1
            if state["n"] == 1:                      # simulate one bad pass
                info["bands"][0] = {**info["bands"][0], "max_gain_db": 6.0}
            return out, info

        chain._multiband_compress_pass = boost_on_first_pass
        try:
            _, info = chain.multiband_compress(audio, SR, max_passes=3)
        finally:
            chain._multiband_compress_pass = real_pass

        self.assertGreater(
            state["n"], 1,
            f"test setup: only {state['n']} pass ran, so max() and min() "
            "over the aggregate are indistinguishable",
        )
        self.assertGreaterEqual(
            info["bands"][0]["max_gain_db"], 6.0,
            f"a +6dB boost in pass 1 aggregated to "
            f"{info['bands'][0]['max_gain_db']:.2f}dB - the multi-pass "
            "reducer is taking min() instead of max(), so a boost in any "
            "single pass is invisible",
        )

    def test_db_to_linear_conversion_is_correct(self):
        """gain = 10**(dB/20) for AMPLITUDE. Using /10 is the power form.

        The /10 form exactly doubles every reduction in dB (-1 becomes -2,
        -3 becomes -6, -6 becomes -12) while every reported number stays
        identical, because the reporting reads gain_db, not gain. The tool
        would compress twice as hard as it says it does.

        Compares the SAMPLE-WISE delivered gain against the reported peak
        reduction at the same instant - not band RMS, which averages the
        gain envelope over time and reads ~0.71x even when correct.
        """
        t = np.arange(int(1.0 * SR)) / SR
        mono = (0.9 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
        audio = _stereo(mono)
        out, info = chain._multiband_compress_pass(audio, SR)
        reported_db = info["bands"][0]["max_reduction_db"]

        # Recover the gain envelope from the band itself. Comparing raw
        # samples is meaningless near a sine's zero crossings (the ratio
        # there reads -42dB on correct code); take the ratio of the two
        # signals' PEAK ENVELOPES instead, sampled per cycle.
        band_in = chain.split_bands_complementary(
            audio, SR, chain.default_bands(SR))[0][:, 0]
        band_out = chain.split_bands_complementary(
            out, SR, chain.default_bands(SR))[0][:, 0]
        cycle = int(SR / 60)
        n = (len(band_in) // cycle) * cycle
        pk_in = np.abs(band_in[:n]).reshape(-1, cycle).max(axis=1)
        pk_out = np.abs(band_out[:n]).reshape(-1, cycle).max(axis=1)
        keep = pk_in > 0.2 * pk_in.max()
        delivered_db = float((20 * np.log10(pk_out[keep] / pk_in[keep])).min())

        self.assertAlmostEqual(
            delivered_db, reported_db, delta=0.35,
            msg=f"reported {reported_db:.3f}dB of peak reduction but "
                f"delivered {delivered_db:.3f}dB - the dB-to-linear "
                "conversion is not 10**(dB/20); a /10 exponent doubles the "
                "real reduction while reporting identical numbers",
        )

    def test_jit_path_is_actually_exercised(self):
        """The fallback must be a fallback, not the everyday path.

        A previous version of _follow() called ITSELF instead of
        _follow_fast, so numba never ran: 987 frames of recursion, a
        swallowed RecursionError, and the plain-Python path every time.
        Output was bit-identical, so all 188 tests passed while the
        compressor ran ~49x slower than intended. No output-based
        assertion can catch that - this one watches the call itself.
        """
        if chain._follow_fast is chain._follow_scalar:
            self.skipTest("numba not installed; fallback is expected")

        level = np.abs(np.random.RandomState(5).randn(2048)).astype(np.float64)
        seen = {"fast": 0, "scalar": 0}
        real_fast, real_scalar = chain._follow_fast, chain._follow_scalar

        def spy_fast(*a):
            seen["fast"] += 1
            return real_fast(*a)

        def spy_scalar(*a):
            seen["scalar"] += 1
            return real_scalar(*a)

        chain._follow_fast, chain._follow_scalar = spy_fast, spy_scalar
        try:
            chain._follow(level, 0.99, 0.999)
        finally:
            chain._follow_fast, chain._follow_scalar = real_fast, real_scalar

        self.assertEqual(
            seen["fast"], 1,
            "the compiled follower was never called - _follow() is not "
            "reaching the JIT path (it may be calling itself, or the "
            "fallback unconditionally)",
        )
        self.assertEqual(
            seen["scalar"], 0,
            "the plain-Python fallback ran even though numba compiled "
            "successfully - this is a silent ~40x performance regression",
        )

    def test_follow_falls_back_correctly_on_a_compile_failure(self):
        """And when the JIT genuinely fails, output must be unchanged."""
        if chain._follow_fast is chain._follow_scalar:
            self.skipTest("numba not installed; fallback is expected")

        from numba.core import errors as numba_errors

        level = np.abs(np.random.RandomState(6).randn(4096)).astype(np.float64)
        expected = chain._follow(level, 0.99, 0.999)

        real_fast = chain._follow_fast

        def boom(*a):
            raise numba_errors.TypingError("simulated lazy compile failure")

        chain._follow_fast = boom
        try:
            got = chain._follow(level, 0.99, 0.999)
        finally:
            chain._follow_fast = real_fast

        self.assertEqual(
            float(np.abs(got - expected).max()), 0.0,
            "the fallback does not reproduce the compiled path exactly",
        )

    def test_follow_does_not_swallow_genuine_errors(self):
        """A real bug must raise, not silently degrade to the slow path."""
        if chain._follow_fast is chain._follow_scalar:
            self.skipTest("numba not installed; fallback is expected")

        level = np.abs(np.random.RandomState(7).randn(512)).astype(np.float64)
        real_fast = chain._follow_fast

        def genuine_bug(*a):
            raise ValueError("a real numerical bug, not a compile failure")

        chain._follow_fast = genuine_bug
        try:
            with self.assertRaises(ValueError):
                chain._follow(level, 0.99, 0.999)
        finally:
            chain._follow_fast = real_fast

    def test_band_time_constants_are_indexed_correctly(self):
        """Guards the INDEXING, not just the tuple.

        An earlier test asserted BAND_ATTACK_MS was ordered slow-to-fast, but
        a mutation that reversed the INDEX left the tuple pristine and
        survived. This checks what each band actually received.
        """
        t = np.arange(int(1.0 * SR)) / SR
        rng = np.random.RandomState(11)
        mono = (0.4 * np.sin(2 * np.pi * 60 * t)
                + 0.3 * np.sin(2 * np.pi * 400 * t)
                + 0.2 * np.sin(2 * np.pi * 2000 * t)
                + 0.15 * rng.randn(len(t))).astype(np.float32)
        _, info = chain._multiband_compress_pass(_stereo(mono), SR)

        reported = [(b["attack_ms"], b["release_ms"]) for b in info["bands"]]
        expected = list(zip(chain.BAND_ATTACK_MS, chain.BAND_RELEASE_MS))
        self.assertEqual(
            reported, expected,
            f"bands received {reported} but the constants are {expected} - "
            "the per-band lookup is misindexed",
        )

    def test_anti_clip_guard_is_present(self):
        """The recombination clamp is what makes short-spike escape safe."""
        t = np.arange(int(1.0 * SR)) / SR
        mono = (0.96 * np.sin(2 * np.pi * 80 * t)
                + 0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
        out, _ = chain._multiband_compress_pass(_stereo(mono), SR)
        peak = float(np.abs(out).max())
        # Two-sided. An upper bound alone lets the clamp be made arbitrarily
        # MORE aggressive invisibly - dropping it to 0.90 is -0.66dB of
        # unrequested broadband gain reduction on every track that touches
        # the ceiling, and no test would fail.
        self.assertAlmostEqual(
            peak, 0.97, delta=5e-5,
            msg=f"recombination peak landed at {peak:.4f}, not the 0.97 "
                "anti-clip ceiling - the guard is missing or its constant "
                "has moved",
        )

    def test_split_slope_is_as_documented(self):
        """Dropping the 2x cascade halves the slope and falsifies the docs."""
        t = np.arange(int(1.0 * SR)) / SR

        def band0_db(freq):
            audio = _stereo(np.sin(2 * np.pi * freq * t))
            low = chain.split_bands_complementary(audio, SR, chain.default_bands(SR))[0]
            return 20 * np.log10(np.sqrt((low ** 2).mean())
                                 / np.sqrt((audio ** 2).mean()) + 1e-12)

        # measured across the crossover itself, where the slope is real -
        # further out the response hits a numerical floor around -50dB and
        # flattens, so probing there measures nothing
        slope = band0_db(200.0) - band0_db(100.0)
        self.assertLess(
            slope, -20.0,
            f"measured {slope:.1f}dB/octave at the crossover - dropping the "
            "cascade halves the slope",
        )

    def test_envelope_start_is_seeded_not_ramping_from_zero(self):
        """A track that opens loud must not fade in for the first 30ms."""
        level = np.full(int(0.2 * SR), 0.8, dtype=np.float32)
        env = chain._envelope_follower(level, SR, 30.0, 200.0)
        self.assertGreater(
            float(env[0]), 0.7,
            f"envelope starts at {env[0]:.4f} on already-loud audio - it is "
            "ramping up from a false zero state",
        )

    def test_envelope_never_rises_while_the_input_is_silent(self):
        """The v3 defect: the envelope climbed 3.03dB over 21.9ms of silence.

        It was chasing a stale decayed target rather than the signal, so it
        peaked 22ms INTO silence - putting gain reduction on the audio after
        the transient instead of on the transient.
        """
        level = np.zeros(int(0.3 * SR), dtype=np.float32)
        level[4410:4410 + int(0.010 * SR)] = 1.0
        env = chain._envelope_follower(level, SR, 30.0, 200.0)

        after = env[4410 + int(0.010 * SR):4410 + int(0.060 * SR)]
        rise_db = 20 * np.log10((after.max() + 1e-12) / (after[0] + 1e-12))
        self.assertLessEqual(
            rise_db, 0.1,
            f"envelope rose {rise_db:+.2f}dB while the input was silent",
        )

    def test_peak_lands_no_later_than_the_transient_ends(self):
        """Gain reduction must peak ON the transient, not after it."""
        hit_len = int(0.010 * SR)
        level = np.zeros(int(0.3 * SR), dtype=np.float32)
        level[4410:4410 + hit_len] = 1.0
        env = chain._envelope_follower(level, SR, 30.0, 200.0)
        peak_at_ms = (int(np.argmax(env)) - 4410) / SR * 1000
        self.assertLessEqual(
            peak_at_ms, 10.5,
            f"envelope peaks {peak_at_ms:.1f}ms after onset, past the end of a "
            "10ms transient - the reduction is landing on the next sound",
        )

    def test_attack_constant_count_is_three_time_constants(self):
        """Changing the 3.0 changes how much of the window the ramp uses."""
        level = np.zeros(int(0.3 * SR), dtype=np.float32)
        level[1000:] = 1.0
        env = chain._envelope_follower(level, SR, 30.0, 200.0)
        at_end = float(env[1000 + int(0.030 * SR)])
        self.assertGreater(at_end, 0.93, f"only {at_end:.3f} by end of window")
        self.assertLess(at_end, 0.98, f"{at_end:.3f} - the ramp is too fast")

    def test_level_uses_the_louder_channel_not_the_average(self):
        """A peak in one channel only must still be caught."""
        t = np.arange(int(1.0 * SR)) / SR
        quiet = (0.02 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        loud = quiet.copy()
        loud[int(0.5 * SR):int(0.5 * SR) + int(0.05 * SR)] = 0.8
        audio = np.stack([loud, quiet], axis=1).astype(np.float32)

        out, info = chain.multiband_compress(audio, SR)
        self.assertTrue(info["applied"],
                        "a loud peak in ONE channel was missed - the level "
                        "detector is averaging the channels rather than "
                        "taking the louder")

    def test_start_seed_uses_the_first_sample(self):
        """A track opening loud must not fade in from a false zero state."""
        level = np.full(int(0.2 * SR), 0.8, dtype=np.float32)
        env = chain._envelope_follower(level, SR, 30.0, 200.0)
        self.assertGreater(
            float(env[0]), 0.7,
            f"envelope starts at {env[0]:.4f} on already-loud audio",
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
