"""detect_transients must not flag vocal consonants as digital clicks.

Reported directly on "Poster on the Wall [Jul 26 III].m4a": the transient
tool was "blowing out the t's in the vocal" around 36s.

Measured on that track, the burst at 35.894s has 115 consecutive
sample-to-sample jumps over the 0.35 threshold inside a 60ms window, and its
4-12kHz energy is 24x the surrounding vowels (HF/LF 2.408 during the burst
vs 0.101 before). That is the textbook signature of a plosive/fricative -
a "t", "s" or "k" - not a digital discontinuity.

A genuine click is a near-instantaneous discontinuity: one or two samples
cross the jump threshold and the waveform is continuous on either side. A
consonant is a SUSTAINED broadband burst lasting tens of milliseconds, so it
crosses the threshold hundreds of times in a row.

Because fix_transient repairs a click by DELETING it (replacing the region
with linear interpolation between clean neighbours), a false positive here
does not merely attenuate the consonant - it erases it. On the real track,
running the chain destroyed 42% of that consonant's 4-12kHz energy (-4.8dB),
and 19 of the 25 detections across the whole track were consonants rather
than clicks.
"""
import unittest

import numpy as np

from app.chain import detect_transients


SR = 44100


def _stereo(mono):
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _vocal_with_consonant(sr=SR, dur=1.0, burst_at=0.5, burst_ms=40):
    """A quiet sung passage followed by a broadband consonant burst.

    Reproduces the real failure's two ingredients, both measured on the
    reported track: the surrounding vocal is QUIET (so the 200ms envelope
    collapses and the burst clears the 8.0 envelope ratio - the real track's
    envelope fell to 0.0417 from 0.1410 half a second earlier), and the burst
    is a sustained fricative that repeatedly crosses the jump threshold.
    """
    n = int(dur * sr)
    t = np.arange(n) / sr
    # quiet voiced vowel - low level is what lets the burst clear the ratio
    mono = 0.02 * np.sin(2 * np.pi * 220 * t) + 0.01 * np.sin(2 * np.pi * 440 * t)
    # consonant: a sustained high-band noise burst, continuous throughout -
    # no single-sample discontinuity anywhere in it
    rng = np.random.RandomState(0)
    start = int(burst_at * sr)
    length = int(burst_ms / 1000 * sr)
    burst = rng.randn(length).astype(np.float64)
    spectrum = np.fft.rfft(burst)
    freqs = np.fft.rfftfreq(length, 1 / sr)
    spectrum[freqs < 3000] = 0
    burst = np.fft.irfft(spectrum, length)
    burst /= np.abs(burst).max() + 1e-12
    envelope = np.concatenate([
        np.linspace(0, 1, length // 8),
        np.ones(length - length // 8),
    ])[:length]
    mono[start:start + length] += 0.7 * burst * envelope
    return mono.astype(np.float32)


def _true_click(sr=SR, dur=1.0, click_at=0.5):
    """A clean tone interrupted by a genuine one-sample discontinuity."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    mono = 0.2 * np.sin(2 * np.pi * 220 * t)
    mono[int(click_at * sr)] += 1.2
    return mono.astype(np.float32)


class TransientConsonantGuardTests(unittest.TestCase):
    def test_vocal_consonant_is_not_flagged_as_a_click(self):
        audio = _stereo(_vocal_with_consonant())
        hits = detect_transients(audio, SR)
        in_burst = [
            h for h in hits if 0.49 <= h["time_sec"] <= 0.56
        ]
        self.assertEqual(
            in_burst,
            [],
            f"a sustained vocal consonant was flagged as {len(in_burst)} "
            "click(s); fix_transient would interpolate straight through it "
            "and erase the consonant",
        )

    def test_genuine_click_is_still_detected(self):
        """The guard must not disarm the tool for real discontinuities."""
        audio = _stereo(_true_click())
        hits = detect_transients(audio, SR)
        near = [h for h in hits if abs(h["time_sec"] - 0.5) < 0.01]
        self.assertTrue(
            near,
            "a genuine single-sample discontinuity must still be detected",
        )

    def test_click_riding_on_top_of_a_vocal_is_still_detected(self):
        """A real click during singing must not be excused as a consonant."""
        mono = _vocal_with_consonant()
        mono[int(0.2 * SR)] += 1.5  # genuine discontinuity, away from burst
        hits = detect_transients(_stereo(mono), SR)
        near = [h for h in hits if abs(h["time_sec"] - 0.2) < 0.01]
        self.assertTrue(
            near,
            "a genuine click elsewhere in a vocal track must still be found",
        )


if __name__ == "__main__":
    unittest.main()
