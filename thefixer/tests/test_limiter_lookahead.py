"""The true-peak limiter must look ahead before reducing gain.

Finding from an adversarial mastering audit. The limiter had no lookahead -
the code said so itself: "catch the peak before it happens isn't possible
without lookahead, so this favors fast reduction". Without it the gain ramp
begins AT the peak, so the peak's leading edge passes through unreduced and
the gain envelope has to move abruptly underneath it. That abrupt movement
is itself distortion.

Measured on a pure 1kHz tone with a single smooth 40ms swell over the
ceiling - a signal with no harmonic content of its own, so any harmonics in
the output are the limiter's doing:

    before limiting:  -121.0 dB  (numerically clean)
    after limiting:    -51.1 dB
    limiter added:      69.9 dB of harmonic distortion

Every professional limiter has 1-5ms of lookahead for exactly this reason:
the gain reduction is ramped in BEFORE the peak arrives, so the envelope is
already at the right value when the transient hits and never has to jump.
"""
import unittest

import numpy as np
import numpy.fft as fft

from app import chain


SR = 44100


def _tone_with_swell(sr=SR, dur=1.0, freq=1000.0, at=0.5, swell_ms=20.0):
    """A pure tone with one smooth level swell over the ceiling.

    Deliberately harmonically clean: the swell is a Hann-windowed amplitude
    envelope, which adds no harmonics of its own, so anything harmonic in
    the limited output came from the limiter.
    """
    n = int(dur * sr)
    t = np.arange(n) / sr
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    env = np.ones(n, dtype=np.float32)
    c = int(at * sr)
    w = int(swell_ms / 1000 * sr)
    env[c - w:c + w] = 1.0 + 0.9 * np.hanning(2 * w)
    return np.stack([mono * env, mono * env], axis=1).astype(np.float32), c


def _harmonic_distortion_db(audio, centre, sr=SR, freq=1000.0, win=4096):
    seg = audio[centre - win // 2:centre + win // 2, 0] * np.hanning(win)
    spectrum = np.abs(fft.rfft(seg))
    freqs = fft.rfftfreq(win, 1 / sr)
    fundamental = spectrum[(freqs > freq - 50) & (freqs < freq + 50)].max()
    harmonics = max(
        spectrum[(freqs > 2 * freq - 50) & (freqs < 2 * freq + 50)].max(),
        spectrum[(freqs > 3 * freq - 50) & (freqs < 3 * freq + 50)].max(),
    )
    return 20 * np.log10(harmonics / (fundamental + 1e-12))


def _true_peak_db(audio, sr=SR, oversample=4):
    from scipy.signal import resample_poly
    up = resample_poly(audio, oversample, 1, axis=0)
    return 20 * np.log10(np.abs(up).max() + 1e-12)


class LimiterLookaheadTests(unittest.TestCase):
    def test_lookahead_reduces_harmonic_distortion(self):
        audio, centre = _tone_with_swell()
        out, info = chain.true_peak_limit(audio, SR, ceiling_db=-1.0)
        self.assertTrue(info.get("applied"), "test setup: limiter must engage")

        added = _harmonic_distortion_db(out, centre) - _harmonic_distortion_db(audio, centre)
        self.assertLess(
            added, 50.0,
            f"the limiter added {added:.1f}dB of harmonic distortion - without "
            "lookahead this measured 69.9dB",
        )

    def test_ceiling_is_still_held(self):
        """Lookahead must not cost the guarantee the limiter exists for."""
        audio, _ = _tone_with_swell()
        out, _ = chain.true_peak_limit(audio, SR, ceiling_db=-1.0)
        self.assertLessEqual(
            _true_peak_db(out), -1.0 + 0.05,
            "true peak exceeds the ceiling",
        )

    def test_output_length_and_alignment_are_preserved(self):
        """A lookahead delay must be compensated, not left in the output."""
        audio, centre = _tone_with_swell()
        out, _ = chain.true_peak_limit(audio, SR, ceiling_db=-1.0)
        self.assertEqual(out.shape, audio.shape)

        # the swell must still sit where it did - cross-correlate to confirm
        seg = int(0.2 * SR)
        a = audio[centre - seg:centre + seg, 0]
        b = out[centre - seg:centre + seg, 0]
        n = 1 << int(np.ceil(np.log2(4 * len(a))))
        cc = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
        lag = int(np.argmax(np.abs(cc)))
        if lag > n // 2:
            lag -= n
        self.assertLessEqual(
            abs(lag), 2,
            f"the limiter shifted the audio by {lag} samples - lookahead delay "
            "was not compensated",
        )

    def test_quiet_audio_is_untouched(self):
        t = np.arange(int(0.5 * SR)) / SR
        mono = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio = np.stack([mono, mono], axis=1)
        out, info = chain.true_peak_limit(audio, SR, ceiling_db=-1.0)
        self.assertFalse(info["applied"])
        np.testing.assert_array_equal(out, audio)

    def test_gain_reduction_begins_before_the_peak(self):
        """The defining property: attenuation must precede the transient."""
        audio, centre = _tone_with_swell()
        out, _ = chain.true_peak_limit(audio, SR, ceiling_db=-1.0)
        gain = np.abs(out[:, 0]) / (np.abs(audio[:, 0]) + 1e-9)
        # 1ms before the peak the limiter should already be reducing
        pre = gain[centre - int(0.001 * SR)]
        self.assertLess(
            pre, 0.999,
            "no gain reduction 1ms before the peak - the limiter is still "
            "reacting at the peak rather than ahead of it",
        )


if __name__ == "__main__":
    unittest.main()
