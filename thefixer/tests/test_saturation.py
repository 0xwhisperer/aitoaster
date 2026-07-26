"""tanh saturation: level-independent drive, 4x oversampling, DC guard.

Three sibling features were rejected before this one shipped, and the reasons
shape these tests:

  - TONAL EQ was built and blocked: its detector fired on sustained musical
    notes and ignored the resonances it targeted.
  - STEREO WIDTH was rejected at design: correctly guarded, the maximum
    permitted change was inaudible - it would have shipped as a no-op with a
    UI toggle.
  - DE-ESSER was rejected at design: no reliable sibilant/cymbal
    discriminator exists (AUC 0.658), and any guard strict enough to protect
    a ride cymbal made it a no-op.

So this suite asserts the properties whose absence killed the others: the
tool must do something MEASURABLE on real-shaped material, its drive must not
depend on input level, and its guards must actually bind on degenerate input
rather than being decorative.

The three defects the design audit caught, each with a test here:
  1. naive fixed drive varied the distortion residual 240x across 18dB
  2. symmetric tanh still rectifies DC on asymmetric program material,
     landing 19x over this app's own lossy re-check floor
  3. the percentile RMS estimator blows up on very short or near-silent input
"""
import unittest

import numpy as np

from app import chain


SR = 44100


def _stereo(mono):
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _music(seconds=4.0, seed=0, scale=0.2):
    """Noise with a musical crest factor - peaky, not flat like white noise."""
    rng = np.random.RandomState(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    body = 0.6 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    transients = np.zeros(n)
    for onset in range(0, n, int(0.5 * SR)):
        transients[onset:onset + 200] += 3.0 * np.hanning(200)
    mix = body + transients + 0.2 * rng.randn(n)
    return (mix / np.abs(mix).max() * scale).astype(np.float32)


def _tone(freq=1000.0, seconds=2.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _thd_pct(x, f0=1000.0):
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(len(x), 1 / SR)

    def amp(f):
        i = int(np.argmin(np.abs(freqs - f)))
        return mag[max(0, i - 2):i + 3].max()

    fund = amp(f0)
    harm = np.sqrt(sum(amp(f0 * k) ** 2 for k in range(2, 10)))
    return 100.0 * harm / max(fund, 1e-20)


def _median_crest_db(x):
    block = int(0.4 * SR)
    n = (len(x) // block) * block
    blocks = x[:n].reshape(-1, block)
    peak = np.abs(blocks).max(axis=1)
    rms = np.sqrt((blocks ** 2).mean(axis=1))
    ok = rms > 1e-6
    return float(np.median(20 * np.log10(peak[ok] / rms[ok])))


class LevelIndependenceTests(unittest.TestCase):
    """Defect 1. A naive fixed drive distorts a loud track far harder than a
    quiet one; measured, the residual varied 240x across an 18dB swing."""

    def test_the_same_track_at_different_levels_gets_the_same_treatment(self):
        base = _music()
        results = {}
        for offset_db in (-12, -6, 0, 6):
            scaled = (base * 10 ** (offset_db / 20)).astype(np.float32)
            out, info = chain.saturate(_stereo(scaled), SR, "medium")
            self.assertTrue(info["applied"])
            # peak RATIO is the level-invariant measure of how hard it worked
            ratio = float(np.abs(out).max() / np.abs(_stereo(scaled)).max())
            results[offset_db] = (ratio, info["makeup_db"])

        ratios = [r for r, _ in results.values()]
        self.assertAlmostEqual(
            max(ratios), min(ratios), delta=0.01,
            msg=f"peak ratio varied across input levels: {results} - the "
                "drive is not normalised to the program level, so a loud "
                "track saturates harder than a quiet one on the same setting",
        )
        makeups = [m for _, m in results.values()]
        self.assertAlmostEqual(
            max(makeups), min(makeups), delta=0.05,
            msg=f"makeup gain varied across input levels: {results}",
        )

    def test_makeup_gain_does_not_sit_on_its_clamp(self):
        """A clamp that always binds is a bug, not a safety net.

        An earlier version normalised the curve by tanh(drive) instead of
        drive, which BOOSTS level (measured +8.59dB at drive 3.0) and left
        the makeup gain pinned at its -3dB limit in every single case.
        """
        for amount in ("light", "medium", "strong"):
            _, info = chain.saturate(_stereo(_music()), SR, amount)
            self.assertLess(
                abs(info["makeup_db"]), chain.SATURATION_MAX_MAKEUP_DB - 0.5,
                f"{amount}: makeup {info['makeup_db']:+.2f}dB is at the "
                f"+/-{chain.SATURATION_MAX_MAKEUP_DB}dB clamp - the curve "
                "normalisation is wrong, so auto-gain is fighting it",
            )

    def test_small_signal_gain_is_close_to_unity(self):
        """Quiet passages must pass through at their own level; only the loud
        parts should compress. That is what makes it a saturator."""
        quiet = _stereo((_music() * 0.05).astype(np.float32))
        out, info = chain.saturate(quiet, SR, "medium")
        self.assertTrue(info["applied"])
        gain_db = 20 * np.log10(
            np.sqrt((out.astype(np.float64) ** 2).mean())
            / np.sqrt((quiet.astype(np.float64) ** 2).mean()))
        self.assertLess(
            abs(gain_db), 1.0,
            f"a quiet signal changed level by {gain_db:+.2f}dB",
        )


class DCGuardTests(unittest.TestCase):
    """Defect 2. tanh is odd, so intuition says it cannot rectify DC - but
    real program material is asymmetric, so it does."""

    def test_output_dc_is_far_below_the_apps_own_recheck_floors(self):
        # deliberately asymmetric, like real program material
        mono = _music()
        mono = np.where(mono > 0, mono * 1.3, mono).astype(np.float32)
        LOSSLESS_FLOOR = 1e-5
        for amount in ("light", "medium", "strong"):
            out, info = chain.saturate(_stereo(mono), SR, amount)
            dc = float(np.abs(out.mean(axis=0)).max())
            self.assertLess(
                dc, LOSSLESS_FLOOR / 10,
                f"{amount}: output DC {dc:.2e} is not an order of magnitude "
                f"under the app's lossless re-check floor ({LOSSLESS_FLOOR:.0e}). "
                "dc_offset runs at step 3 and this runs at step 8, so nothing "
                "downstream would catch it and the delivered file would "
                "re-recommend dc_offset on re-upload.",
            )

    def test_the_guard_actually_had_something_to_remove(self):
        """Guards the guard: if dc_removed were always ~0 this suite would
        pass even with the mean-subtraction deleted."""
        mono = _music()
        mono = np.where(mono > 0, mono * 1.3, mono).astype(np.float32)
        _, info = chain.saturate(_stereo(mono), SR, "strong")
        self.assertGreater(
            info["dc_removed"], 1e-5,
            f"only {info['dc_removed']:.2e} of DC was present before the "
            "guard, so this test cannot prove the guard works",
        )

    def test_existing_input_dc_is_not_amplified(self):
        mono = (_music() + 0.02).astype(np.float32)
        out, _ = chain.saturate(_stereo(mono), SR, "medium")
        self.assertLess(float(np.abs(out.mean(axis=0)).max()), 1e-5)


class PerChannelTests(unittest.TestCase):
    """A GLOBAL mean subtraction passes every dual-mono test, because both
    channels share the same mean there. These use channels with DIFFERENT
    DC so the distinction is observable."""

    def test_dc_is_removed_per_channel_not_globally(self):
        mono = _music()
        left = np.where(mono > 0, mono * 1.4, mono).astype(np.float32)
        right = np.where(mono < 0, mono * 1.4, mono).astype(np.float32)
        audio = np.stack([left, right], axis=1).astype(np.float32)
        out, _ = chain.saturate(audio, SR, "strong")
        per_channel = np.abs(out.mean(axis=0))
        self.assertLess(
            float(per_channel.max()), 1e-6,
            f"per-channel DC is {per_channel} - a GLOBAL mean subtraction "
            "cancels only the average of the two, leaving each channel "
            "offset in opposite directions",
        )

    def test_channels_with_opposite_dc_are_each_corrected(self):
        mono = _music()
        audio = np.stack([mono + 0.03, mono - 0.03], axis=1).astype(np.float32)
        out, _ = chain.saturate(audio, SR, "medium")
        for ch in (0, 1):
            self.assertLess(
                abs(float(out[:, ch].mean())), 1e-6,
                f"channel {ch} retains DC {out[:, ch].mean():.2e}",
            )

    def test_the_dc_guard_runs_before_auto_gain(self):
        """Order matters: makeup gain is measured on the signal that will be
        delivered, so it must be computed after DC is removed. Moving the
        guard after auto-gain leaves the reported makeup describing a
        different signal than the one written to disk."""
        mono = np.where(_music() > 0, _music() * 1.4, _music()).astype(np.float32)
        out, info = chain.saturate(_stereo(mono), SR, "strong")
        delivered_rms = float(np.sqrt((out.astype(np.float64) ** 2).mean()))
        input_rms = float(np.sqrt((_stereo(mono).astype(np.float64) ** 2).mean()))
        actual_db = 20 * np.log10(delivered_rms / input_rms)
        # the reported makeup should describe the delivered signal to within
        # the difference between whole-file RMS and the percentile estimator
        self.assertLess(
            abs(actual_db - info["makeup_db"]), 1.5,
            f"reported makeup {info['makeup_db']:+.2f}dB but the delivered "
            f"signal moved {actual_db:+.2f}dB",
        )


class AutoGainTests(unittest.TestCase):
    def test_makeup_gain_is_clamped(self):
        """The clamp must exist even though it should rarely bind. Removing
        it entirely, or widening it to 30dB, passed the original suite."""
        self.assertLessEqual(chain.SATURATION_MAX_MAKEUP_DB, 3.0)
        self.assertGreater(chain.SATURATION_MAX_MAKEUP_DB, 0.0)

    def test_the_clamp_binds_on_pathological_input(self):
        """The clamp only matters on input that defeats the level estimator.

        A large DC offset dominates the percentile RMS, so the estimator
        reports a level the music does not have and auto-gain tries to
        correct by a huge amount. Without the clamp that becomes a silent
        volume change. dc_offset normally runs upstream at step 3, but it is
        user-optional, so this input is reachable.
        """
        mono = (_music() + 0.25).astype(np.float32)
        _, info = chain.saturate(_stereo(mono), SR, "medium")
        self.assertLessEqual(
            abs(info["makeup_db"]), chain.SATURATION_MAX_MAKEUP_DB + 1e-6,
            f"makeup reached {info['makeup_db']:+.2f}dB, past the "
            f"+/-{chain.SATURATION_MAX_MAKEUP_DB}dB clamp - an unclamped "
            "auto-gain becomes an unrequested volume change",
        )

    def test_makeup_actually_compensates_level(self):
        """Removing the makeup gain entirely passed the original suite."""
        mono = _music()
        out, info = chain.saturate(_stereo(mono), SR, "strong")
        in_rms = float(np.sqrt((_stereo(mono).astype(np.float64) ** 2).mean()))
        out_rms = float(np.sqrt((out.astype(np.float64) ** 2).mean()))
        drop_db = 20 * np.log10(out_rms / in_rms)
        self.assertGreater(
            drop_db, -1.5,
            f"delivered level fell {drop_db:.2f}dB - auto-gain is not "
            "compensating for the level the curve removes",
        )
        self.assertNotEqual(info["makeup_db"], 0.0,
                            "makeup gain was never applied")

    def test_level_estimator_uses_a_high_percentile(self):
        """A median (50th) estimator is dragged down by quiet passages, so
        loud sections saturate harder. Verified by giving the same music a
        long quiet tail: a percentile-based estimate must barely move."""
        loud = _music(seconds=4.0)
        quiet_tail = np.concatenate([loud, (loud * 0.02)[:int(4 * SR)]]).astype(np.float32)
        a = chain._program_rms(_stereo(loud), SR)
        b = chain._program_rms(_stereo(quiet_tail), SR)
        self.assertAlmostEqual(
            20 * np.log10(b / a), 0.0, delta=1.0,
            msg=f"level estimate moved {20 * np.log10(b / a):+.2f}dB when a "
                "quiet tail was appended - the estimator is not tracking the "
                "level the track plays at",
        )


class HarmonicsTests(unittest.TestCase):
    def test_thd_rises_monotonically_with_drive(self):
        thds = {}
        for amount in ("light", "medium", "strong"):
            out, _ = chain.saturate(_stereo(_tone()), SR, amount)
            thds[amount] = _thd_pct(out[:, 0])
        self.assertLess(thds["light"], thds["medium"])
        self.assertLess(thds["medium"], thds["strong"])

    def test_saturation_is_gentle_at_the_default(self):
        """Mastering-bus saturation, not a distortion box."""
        out, _ = chain.saturate(_stereo(_tone()), SR, "medium")
        thd = _thd_pct(out[:, 0])
        self.assertLess(
            thd, 5.0,
            f"medium measured {thd:.2f}% THD on a 1kHz tone - too much for a "
            "mastering bus",
        )

    def test_harmonics_are_odd_order(self):
        """tanh is an odd function: H2/H4 should be far below H3/H5. Even
        harmonics would mean an asymmetric curve, which was rejected because
        its H2 lands where pop vocal and snare content already lives."""
        out, _ = chain.saturate(_stereo(_tone()), SR, "strong")
        mag = np.abs(np.fft.rfft(out[:, 0] * np.hanning(len(out))))
        freqs = np.fft.rfftfreq(len(out), 1 / SR)

        def amp(f):
            i = int(np.argmin(np.abs(freqs - f)))
            return mag[max(0, i - 2):i + 3].max()

        h2, h3 = amp(2000.0), amp(3000.0)
        self.assertLess(
            h2, h3 * 0.5,
            f"H2 ({20 * np.log10(h2 / amp(1000)):.1f}dB) is not well below "
            f"H3 ({20 * np.log10(h3 / amp(1000)):.1f}dB) - the curve is not odd",
        )

    def test_the_default_oversample_factor_is_at_least_4x(self):
        """The constant itself must be pinned.

        Every other test here passes `oversample=` explicitly, so setting
        SATURATION_OVERSAMPLE = 1 - shipping the feature's own stated
        non-negotiable switched off - passed all 23 tests. Measured, 1x
        leaves in-band alias at -40dB where 4x gives -88dB.
        """
        self.assertGreaterEqual(
            chain.SATURATION_OVERSAMPLE, 4,
            "in-band aliasing lands below 8kHz, inside both detectors' "
            "analysis band, where nothing downstream can remove it",
        )

    def test_aliasing_stays_out_of_the_detector_band(self):
        """Nonlinear processing folds harmonics back below Nyquist as
        inharmonic content, and it lands under 8kHz - inside BOTH detectors'
        analysis band, where nothing downstream can remove it.

        Uses a 15kHz tone deliberately. An earlier version used 7kHz, whose
        3rd harmonic at 21kHz still sits below the 44.1k Nyquist and so never
        folds - that test measured -96dB even at 1x oversampling, 36dB of
        slack, and could not fail. At 15kHz the 3rd harmonic is at 45kHz and
        must alias: measured 1x -33.2dB vs 4x -88.2dB.
        """
        out, _ = chain.saturate(_stereo(_tone(freq=15000.0)), SR, "strong")
        mag = np.abs(np.fft.rfft(out[:, 0] * np.hanning(len(out))))
        freqs = np.fft.rfftfreq(len(out), 1 / SR)
        fund = mag[np.abs(freqs - 15000.0) < 30].max()
        alias = mag[(freqs > 100) & (freqs < 8000)].max()
        alias_db = 20 * np.log10(alias / fund)
        self.assertLess(
            alias_db, -60.0,
            f"in-band aliasing at {alias_db:.1f}dB below the fundamental",
        )

    def test_oversampling_is_actually_applied(self):
        """Guards against the oversample parameter being ignored."""
        tone = _stereo(_tone(freq=15000.0))
        at_1x, _ = chain.saturate(tone, SR, "strong", oversample=1)
        at_4x, _ = chain.saturate(tone, SR, "strong")   # the DEFAULT path
        mag1 = np.abs(np.fft.rfft(at_1x[:, 0] * np.hanning(len(at_1x))))
        mag4 = np.abs(np.fft.rfft(at_4x[:, 0] * np.hanning(len(at_4x))))
        freqs = np.fft.rfftfreq(len(at_1x), 1 / SR)
        band = (freqs > 100) & (freqs < 8000)
        self.assertLess(
            mag4[band].max(), mag1[band].max(),
            "4x oversampling produced no less in-band alias than 1x - the "
            "oversample argument is being ignored",
        )


class ItActuallyDoesSomethingTests(unittest.TestCase):
    """The lesson from stereo width and the de-esser: both measured fine
    unguarded and became no-ops once correctly guarded. A tool that does
    nothing audible should not ship."""

    def test_medium_reduces_short_term_crest_measurably(self):
        mono = _music()
        out, _ = chain.saturate(_stereo(mono), SR, "medium")
        delta = _median_crest_db(out[:, 0]) - _median_crest_db(mono)
        self.assertLess(
            delta, -0.2,
            f"median short-term crest moved only {delta:+.3f}dB - this is "
            "close enough to a no-op that the tool is not worth shipping",
        )

    def test_it_reduces_peak_level(self):
        """Soft clipping ahead of a limiter should lower the limiter's work."""
        mono = _music()
        out, _ = chain.saturate(_stereo(mono), SR, "medium")
        self.assertLess(
            float(np.abs(out).max()), float(np.abs(_stereo(mono)).max()),
            "saturation did not reduce peak level",
        )

    def test_stronger_settings_do_more(self):
        mono = _music()
        deltas = []
        for amount in ("light", "medium", "strong"):
            out, _ = chain.saturate(_stereo(mono), SR, amount)
            deltas.append(_median_crest_db(out[:, 0]) - _median_crest_db(mono))
        self.assertLess(deltas[1], deltas[0])
        self.assertLess(deltas[2], deltas[1])


class GuardTests(unittest.TestCase):
    """Defect 3. The percentile RMS estimator is unreliable on very short or
    near-silent input - measured, a 0.05s file blew the normalisation factor
    up to 112,948 and a 0.2s file overshot its THD target by 2.6x."""

    def test_short_files_are_skipped(self):
        for seconds in (0.05, 0.2, 1.0, 1.9):
            audio = _stereo(_music(seconds=seconds))
            out, info = chain.saturate(audio, SR, "medium")
            self.assertFalse(
                info["applied"],
                f"a {seconds}s file was processed; the level estimator needs "
                f"{chain.SATURATION_MIN_DURATION_SEC}s",
            )
            self.assertIs(out, audio)

    def test_a_long_enough_file_is_processed(self):
        _, info = chain.saturate(_stereo(_music(seconds=3.0)), SR, "medium")
        self.assertTrue(info["applied"], info.get("reason"))

    def test_near_silence_is_skipped(self):
        quiet = _stereo((_music() * 1e-5).astype(np.float32))
        _, info = chain.saturate(quiet, SR, "medium")
        self.assertFalse(info["applied"],
                         "near-silence measured 12.3% THD when not guarded")

    def test_digital_silence_does_not_divide_by_zero(self):
        audio = np.zeros((int(4 * SR), 2), dtype=np.float32)
        out, info = chain.saturate(audio, SR, "medium")
        self.assertFalse(info["applied"])
        self.assertTrue(np.array_equal(out, audio))
        self.assertTrue(np.isfinite(out).all())

    def test_an_unknown_amount_is_refused(self):
        audio = _stereo(_music())
        out, info = chain.saturate(audio, SR, "nonsense")
        self.assertFalse(info["applied"])
        self.assertIs(out, audio)

    def test_mono_content_stays_mono(self):
        """Dual-mono in must be dual-mono out - never fabricate width."""
        mono = _music()
        out, _ = chain.saturate(_stereo(mono), SR, "strong")
        self.assertEqual(
            float(np.abs(out[:, 0] - out[:, 1]).max()), 0.0,
            "the two channels diverged - saturation is not channel-symmetric",
        )

    def test_length_is_preserved(self):
        """resample_poly round-trips can change length by a sample."""
        for seconds in (2.5, 3.0, 4.7):
            audio = _stereo(_music(seconds=seconds))
            out, info = chain.saturate(audio, SR, "medium")
            self.assertEqual(
                out.shape, audio.shape,
                f"{seconds}s: shape changed {audio.shape} -> {out.shape}",
            )
            self.assertEqual(
                len(out), len(audio),
                f"{seconds}s: length changed by "
                f"{len(out) - len(audio)} samples - a resample_poly round "
                "trip does not preserve length exactly and the fix-up is "
                "missing or truncating",
            )

    def test_mono_1d_input_is_handled(self):
        mono = _music()
        out, info = chain.saturate(mono, SR, "medium")
        self.assertEqual(out.shape, mono.shape)
        self.assertTrue(info["applied"])

    def test_non_finite_input_is_refused_not_silently_destroyed(self):
        """One NaN sample became 176,400 NaN samples: resample_poly spreads it
        across its filter length and the DC guard's mean then spreads it
        across the whole channel - while the status line still said "pass"."""
        mono = _music()
        mono[1000] = np.nan
        out, info = chain.saturate(_stereo(mono), SR, "medium")
        self.assertFalse(info["applied"], "NaN input was processed")
        self.assertIn("non-finite", info["reason"])
        for bad in (np.inf, -np.inf):
            m2 = _music()
            m2[500] = bad
            _, i2 = chain.saturate(_stereo(m2), SR, "medium")
            self.assertFalse(i2["applied"], f"{bad} input was processed")

    def test_out_of_phase_stereo_is_not_mistaken_for_silence(self):
        """L = -R sums to exactly zero, so a mono-sum level estimator called
        it silent and skipped it with a FALSE reason - each channel was at
        0.20 RMS. A wrong reason in the log is worse than a right refusal."""
        mono = _music()
        audio = np.stack([mono, -mono], axis=1).astype(np.float32)
        out, info = chain.saturate(audio, SR, "medium")
        self.assertTrue(
            info["applied"],
            f"anti-phase material was skipped: {info.get('reason')} - but "
            f"each channel measures {np.sqrt((mono.astype(np.float64) ** 2).mean()):.3f} RMS",
        )
        # and it must stay anti-phase
        self.assertLess(
            float(np.abs(out[:, 0] + out[:, 1]).max()),
            float(np.abs(out).max()) * 0.05,
            "anti-phase relationship was not preserved",
        )

    def test_output_is_finite(self):
        for amount in ("light", "medium", "strong"):
            out, _ = chain.saturate(_stereo(_music(scale=0.99)), SR, amount)
            self.assertTrue(np.isfinite(out).all(), f"{amount} produced non-finite output")


if __name__ == "__main__":
    unittest.main()
