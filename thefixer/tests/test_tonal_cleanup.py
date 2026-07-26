"""Tonal cleanup: cut a resonance, never the music.

FIVE detectors were built and measured before this one. Every failure is a
test here, because each was a case the previous version got confidently
wrong:

  1. Whole-track average excess over neighbours. Cut a sustained bass note
     26dB BELOW the noise bed (+3.08dB excess) and ignored a genuine +7dB
     resonance (+2.80dB). On a time-averaged spectrum a narrowband detector
     cannot tell a room mode from a bass line - both are narrowband energy.
  2. Sideband MEAN as the baseline. A steep slope drags the mean below the
     region and invents a resonance: a 2kHz lowpass with none at all read
     +7.18dB and drew a full cut.
  3. Log-LINEAR interpolation between sidebands. Worse - a filter skirt is
     curved as well as steep, so a chord through two points sits below the
     true curve.
  4. Per-frame OCCUPANCY above a threshold. The frame-to-frame noise on
     plain pink noise has std 3.39dB, as large as the resonance being looked
     for, so a real +7dB resonance and a sustained note both scored ~0.57.
  5. p10 with a -99 sentinel for skirt-like frames. On real music 15.5% of
     frames legitimately look skirt-like, and those sentinels dragged the
     percentile to -99, vetoing the whole band.

What works: the 10th percentile of the per-frame excess, with skirt-like
frames EXCLUDED. A resonance rings whenever any content excites it, so even
its quietest frames sit above the local trend. A note is loud while played
and gone otherwise. A skirt is not a local peak at all.

    signal                    p10     median    IQR
    pink noise, no resonance -3.83     0.51     4.75
    REAL +7dB resonance      -0.88     3.42     4.73
    intermittent bass note   -3.34     1.68     6.95
"""
import unittest

import numpy as np
from scipy import signal as sig

from app import chain


SR = 44100
DUR = 60.0


def _pink(seed=0, seconds=DUR, scale=0.3):
    n = int(seconds * SR)
    rng = np.random.RandomState(seed)
    b, a = sig.butter(1, 0.02, btype="low")
    x = sig.lfilter(b, a, rng.randn(n))
    return (x / np.abs(x).max() * scale).astype(np.float32)


def _stereo(mono):
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _with_resonance(audio, freq_hz, gain_db, q=chain.TONAL_Q):
    sos = np.asarray([chain._peaking_sos(freq_hz, q, gain_db, SR)])
    return sig.sosfilt(sos, audio, axis=0).astype(np.float32)


def _intermittent_note(f0=246.94, seconds=DUR, every=4.0, held=1.5):
    """A note: loud while played, absent otherwise. This is what the
    detector must NOT cut, and what killed detectors 1 and 4."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for start in range(0, n, int(every * SR)):
        sl = slice(start, min(start + int(held * SR), n))
        length = sl.stop - sl.start
        if length <= 0:
            continue
        out[sl] += sum(0.18 / (k + 1) * np.sin(2 * np.pi * f0 * (k + 1) * t[sl])
                       for k in range(6)) * np.hanning(length)
    return out


def _cut_for(info, label):
    for band in info["bands"]:
        if band["label"] == label:
            return band["cut_db"]
    return 0.0


def _any_cut(info):
    return any(b["cut_db"] < 0 for b in info["bands"])


class DoesNotCutMusicTests(unittest.TestCase):
    """The failure that killed detector 1: cutting the song."""

    def test_a_sustained_musical_note_is_not_cut(self):
        audio = _stereo((_pink() + _intermittent_note()).astype(np.float32))
        _, info = chain.tonal_cleanup(audio, SR)
        self.assertEqual(
            _cut_for(info, "boxiness"), 0.0,
            "a bass note at 247Hz was corrected as if it were a resonance - "
            "this is the failure that killed the first detector, which cut a "
            "note sitting 26dB BELOW the noise bed",
        )

    def test_a_louder_note_is_still_not_cut(self):
        loud = _intermittent_note() * 3.0
        audio = _stereo((_pink() + loud).astype(np.float32))
        _, info = chain.tonal_cleanup(audio, SR)
        self.assertEqual(
            _cut_for(info, "boxiness"), 0.0,
            "loudness is not the discriminator - persistence is. A note is "
            "still a note however loud it is played.",
        )

    def test_material_with_no_resonance_is_untouched(self):
        for seed in (0, 1, 2, 3):
            audio = _stereo(_pink(seed=seed))
            out, info = chain.tonal_cleanup(audio, SR)
            self.assertFalse(
                _any_cut(info),
                f"seed {seed}: corrected material that has no resonance - "
                f"{[(b['label'], b['cut_db']) for b in info['bands']]}",
            )
            self.assertIs(out, audio, "untouched audio should be returned as-is")


class DoesNotCutFilterSkirtsTests(unittest.TestCase):
    """The failure that killed detectors 2, 3 and 5.

    A quadratic baseline cannot follow a filter knee. On an 8th-order lowpass
    at 2kHz the fit undershoots so badly that the 3150Hz region read p10 =
    16.73dB - HIGHER than a genuine +7dB resonance at 2.03dB. No threshold on
    the excess alone can separate them; the skirt scores higher than the
    thing being looked for. The monotonicity veto is what catches it.
    """

    def test_lowpass_skirts_are_not_cut_at_any_order(self):
        for order in (2, 4, 6, 8):
            sos = sig.butter(order, 2000 / (SR / 2), btype="low", output="sos")
            audio = sig.sosfilt(sos, _stereo(_pink()), axis=0).astype(np.float32)
            _, info = chain.tonal_cleanup(audio, SR)
            self.assertFalse(
                _any_cut(info),
                f"a {order}th-order lowpass with no resonance was corrected: "
                f"{[(b['label'], b['cut_db']) for b in info['bands']]}. An "
                "earlier version tested only order 2 - the one order that "
                "happened to pass.",
            )

    def test_a_lowpass_just_below_a_region_is_not_cut(self):
        """The hardest skirt case: the knee sits right under the band."""
        for cutoff in (1000, 2000, 2500):
            sos = sig.butter(8, cutoff / (SR / 2), btype="low", output="sos")
            audio = sig.sosfilt(sos, _stereo(_pink()), axis=0).astype(np.float32)
            _, info = chain.tonal_cleanup(audio, SR)
            self.assertFalse(
                _any_cut(info),
                f"an 8th-order lowpass at {cutoff}Hz was corrected",
            )

    def test_highpass_skirts_are_not_cut(self):
        sos = sig.butter(8, 5000 / (SR / 2), btype="high", output="sos")
        audio = sig.sosfilt(sos, _stereo(_pink()), axis=0).astype(np.float32)
        _, info = chain.tonal_cleanup(audio, SR)
        self.assertFalse(_any_cut(info), "a highpass skirt was corrected")

    def test_skirt_frames_are_excluded_not_scored_negative(self):
        """Detector 5 set skirt-like frames to -99 and then took the p10
        across ALL frames. On real music 15.5% of frames look skirt-like, so
        those sentinels dragged the percentile to -99 and vetoed the band
        outright. They must be dropped from the sample instead."""
        audio = _with_resonance(_stereo(_pink()), 250.0, 9.0)
        excess, active = chain._frame_excess_db(audio, SR, 250.0, chain.TONAL_Q)
        self.assertGreater(
            active.sum(), len(excess) * 0.5,
            "more than half the frames were vetoed on material with a real "
            "resonance - the veto is too broad",
        )
        self.assertTrue(
            np.all(excess[active] > -50),
            "an excluded frame is still carrying a huge negative sentinel "
            "into the statistic",
        )


class DoesCutRealResonancesTests(unittest.TestCase):
    def test_a_real_resonance_is_corrected(self):
        audio = _with_resonance(_stereo(_pink()), 250.0, 7.0)
        _, info = chain.tonal_cleanup(audio, SR)
        self.assertLess(
            _cut_for(info, "boxiness"), 0.0,
            "a genuine +7dB Q=1.4 resonance was not corrected - detectors 1 "
            "and 4 both missed exactly this case",
        )

    def test_the_cut_scales_with_severity(self):
        cuts = []
        for gain in (5.0, 7.0, 12.0):
            audio = _with_resonance(_stereo(_pink()), 250.0, gain)
            _, info = chain.tonal_cleanup(audio, SR)
            cuts.append(_cut_for(info, "boxiness"))
        self.assertLess(cuts[1], cuts[0] + 1e-9,
                        f"a 7dB resonance ({cuts[1]}) drew no more than a 5dB "
                        f"one ({cuts[0]})")
        self.assertLessEqual(cuts[2], cuts[1] + 1e-9)

    def test_both_regions_work_independently(self):
        audio = _with_resonance(_stereo(_pink()), 3150.0, 7.0)
        _, info = chain.tonal_cleanup(audio, SR)
        self.assertLess(_cut_for(info, "harshness"), 0.0,
                        "a 3.15kHz resonance was not corrected")
        self.assertEqual(_cut_for(info, "boxiness"), 0.0,
                         "a 3.15kHz resonance triggered a 250Hz cut")

    def test_the_cut_is_capped(self):
        audio = _with_resonance(_stereo(_pink()), 250.0, 24.0)
        _, info = chain.tonal_cleanup(audio, SR)
        self.assertGreaterEqual(
            _cut_for(info, "boxiness"), -chain.TONAL_MAX_CUT_DB - 1e-9,
            "a grotesque resonance drew more than the cap",
        )
        self.assertLess(_cut_for(info, "boxiness"), 0.0)

    def test_the_cut_actually_changes_the_spectrum(self):
        audio = _with_resonance(_stereo(_pink()), 250.0, 12.0)
        out, info = chain.tonal_cleanup(audio, SR)
        self.assertTrue(info["applied"])

        def band_db(x):
            f, p = sig.welch(x.mean(axis=1), SR, nperseg=16384)
            s = (f >= 200) & (f < 320)
            return 10 * np.log10(p[s].mean() + 1e-20)

        self.assertLess(
            band_db(out) - band_db(audio), -0.1,
            "reported a cut but the 200-320Hz band did not move",
        )


class NeverBoostsTests(unittest.TestCase):
    def test_no_band_ever_receives_positive_gain(self):
        cases = [
            _stereo(_pink()),
            _with_resonance(_stereo(_pink()), 250.0, 9.0),
            _with_resonance(_stereo(_pink()), 250.0, -12.0),   # a DIP
            _stereo((_pink() + _intermittent_note()).astype(np.float32)),
        ]
        for i, audio in enumerate(cases):
            _, info = chain.tonal_cleanup(audio, SR)
            for band in info["bands"]:
                self.assertLessEqual(
                    band["cut_db"], 0.0,
                    f"case {i} band {band['label']} applied "
                    f"{band['cut_db']:+.2f}dB - this tool must never boost",
                )

    def test_a_deficient_region_is_not_filled_in(self):
        audio = _with_resonance(_stereo(_pink()), 250.0, -12.0)
        _, info = chain.tonal_cleanup(audio, SR)
        self.assertEqual(
            _cut_for(info, "boxiness"), 0.0,
            "a 250Hz DIP was corrected - the tool is acting on downward "
            "deviations, which means it boosts",
        )

    def test_output_peak_never_exceeds_input(self):
        audio = _with_resonance(_stereo(_pink()), 250.0, 12.0)
        out, info = chain.tonal_cleanup(audio, SR)
        self.assertTrue(info["applied"])
        self.assertLessEqual(float(np.abs(out).max()),
                             float(np.abs(audio).max()) + 1e-6)


class IdempotenceTests(unittest.TestCase):
    def test_running_twice_does_not_ratchet(self):
        """A tool that re-corrects its own output keeps cutting forever."""
        audio = _with_resonance(_stereo(_pink()), 250.0, 12.0)
        out, first = chain.tonal_cleanup(audio, SR)
        self.assertTrue(first["applied"])
        _, second = chain.tonal_cleanup(out, SR)
        self.assertGreater(
            _cut_for(second, "boxiness"), _cut_for(first, "boxiness") - 1e-9,
            "the second pass cut at least as much as the first - this "
            "ratchets and would keep cutting on every re-run",
        )

    def test_it_converges(self):
        audio = _with_resonance(_stereo(_pink()), 250.0, 12.0)
        cuts = []
        for _ in range(4):
            audio, info = chain.tonal_cleanup(audio, SR)
            cuts.append(_cut_for(info, "boxiness"))
        self.assertGreater(
            cuts[-1], cuts[0] - 1e-9,
            f"cuts across four passes: {cuts} - not converging",
        )


class GuardTests(unittest.TestCase):
    def test_short_tracks_are_skipped(self):
        for seconds in (5.0, 15.0, 29.0):
            audio = _stereo(_pink(seconds=seconds))
            out, info = chain.tonal_cleanup(audio, SR)
            self.assertFalse(info["applied"])
            self.assertIn("needs", info["reason"])
            self.assertIs(out, audio)

    def test_mono_input_is_handled(self):
        mono = _with_resonance(_stereo(_pink()), 250.0, 12.0)[:, 0]
        out, info = chain.tonal_cleanup(mono, SR)
        self.assertEqual(out.shape, mono.shape)

    def test_silence_does_not_crash(self):
        audio = np.zeros((int(DUR * SR), 2), dtype=np.float32)
        out, info = chain.tonal_cleanup(audio, SR)
        self.assertFalse(info["applied"])
        self.assertTrue(np.isfinite(out).all())

    def test_output_is_finite(self):
        audio = _with_resonance(_stereo(_pink()), 250.0, 18.0)
        out, _ = chain.tonal_cleanup(audio, SR)
        self.assertTrue(np.isfinite(out).all())

    def test_level_independence(self):
        """The same track quieter must get the same correction."""
        loud = _with_resonance(_stereo(_pink()), 250.0, 9.0)
        quiet = (loud * 0.05).astype(np.float32)
        _, a = chain.tonal_cleanup(loud, SR)
        _, b = chain.tonal_cleanup(quiet, SR)
        self.assertAlmostEqual(
            _cut_for(a, "boxiness"), _cut_for(b, "boxiness"), delta=0.05,
            msg="the same resonance at a different level drew a different cut",
        )


class PerBandTriggerTests(unittest.TestCase):
    """The p10 noise floor differs by band: the analysis window holds far
    more cycles at 3150Hz than at 250Hz. A single global trigger cut pure
    noise by 1.33dB in the harshness band while missing real resonances in
    the boxiness band."""

    def test_each_region_carries_its_own_trigger(self):
        for region in chain.TONAL_REGIONS:
            self.assertEqual(len(region), 3,
                             "each region must be (freq, label, trigger)")

    def test_the_triggers_sit_above_their_own_noise_floors(self):
        for freq_hz, label, trigger in chain.TONAL_REGIONS:
            floors = []
            for seed in (0, 1, 2):
                audio = _stereo(_pink(seed=seed))
                excess, active = chain._frame_excess_db(
                    audio, SR, freq_hz, chain.TONAL_Q)
                floors.append(np.percentile(excess[active], 10))
            worst = max(floors)
            self.assertGreater(
                trigger, worst,
                f"{label}: trigger {trigger} is at or below the measured "
                f"noise floor {worst:.2f} - it will cut material that has no "
                "resonance at all",
            )


if __name__ == "__main__":
    unittest.main()
