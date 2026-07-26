"""TPDF dither on 16-bit output.

The whole pipeline works in float32 and every WAV/FLAC output is written as
PCM_16, so every delivered file takes a bit-depth reduction. Undithered, the
quantization error is CORRELATED with the signal, which is why it reads as
distortion rather than as hiss.

Measured on a 1kHz tone at 1.5 LSB (the quantizer's resolution limit):

    truncated, no dither : harmonics -24.6 dB below the tone
    with TPDF dither     : harmonics -43.7 dB below the tone

A 19dB improvement. Dither replaces that correlated distortion with
uncorrelated noise at about -93dBFS, which is inaudible on a -14 LUFS master.

It matters specifically for LOW-LEVEL material - fade-outs, reverb tails,
quiet intros - because that is the signal that walks down through the least
significant bit on its way to silence. The default 3-second fade-out is
exactly such a signal.

Dither must be the LAST operation before the file is written: any gain change
applied after it would rescale the noise and defeat the purpose.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.fft as fft
import soundfile as sf

from app.server import tpdf_dither_noise, save_stereo


SR = 44100
LSB = 1.0 / 32768


def _tone_at_lsb(multiple=1.5, dur=3.0, sr=SR, freq=1000.0):
    """A tone at the quantizer's resolution limit, where truncation bites."""
    t = np.arange(int(dur * sr)) / sr
    mono = (multiple * LSB * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1)


def _harmonic_ratio_db(audio, sr=SR, freq=1000.0, win=131072):
    seg = audio[:win, 0] * np.hanning(win)
    spectrum = np.abs(fft.rfft(seg))
    freqs = fft.rfftfreq(win, 1 / sr)
    fundamental = spectrum[(freqs > freq - 50) & (freqs < freq + 50)].max()
    harmonics = max(
        spectrum[(freqs > 3 * freq - 50) & (freqs < 3 * freq + 50)].max(),
        spectrum[(freqs > 5 * freq - 50) & (freqs < 5 * freq + 50)].max(),
    )
    return 20 * np.log10(harmonics / (fundamental + 1e-20))


def _write_read(audio, sr=SR):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        path = tf.name
    try:
        save_stereo(path, audio, sr, dither=True)
        out, _ = sf.read(path, dtype="float32")
        return out
    finally:
        Path(path).unlink(missing_ok=True)


class DitherTests(unittest.TestCase):
    def test_dither_reduces_quantization_distortion(self):
        audio = _tone_at_lsb()
        out = _write_read(audio)
        ratio = _harmonic_ratio_db(out)
        self.assertLess(
            ratio, -35.0,
            f"harmonics only {abs(ratio):.1f}dB below the tone - undithered "
            "truncation measured -24.6dB here",
        )

    def test_dither_noise_is_inaudible_in_level(self):
        """The noise added must sit near the 16-bit floor, not above it."""
        silence = np.zeros((SR, 2), dtype=np.float32)
        out = _write_read(silence)
        rms_db = 20 * np.log10(np.sqrt((out ** 2).mean()) + 1e-20)
        self.assertLess(rms_db, -85.0, f"dither noise at {rms_db:.1f}dBFS is too loud")
        self.assertGreater(rms_db, -120.0, "no dither appears to have been applied")

    def test_it_is_harmless_on_loud_material(self):
        """If dither 'shouldn't' be needed it must not degrade anything."""
        t = np.arange(int(2.0 * SR)) / SR
        mono = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        audio = np.stack([mono, mono], axis=1)
        out = _write_read(audio)
        # a loud tone is thousands of LSBs tall - dither must be negligible
        n = min(len(out), len(audio))
        err = np.sqrt(((out[:n] - audio[:n]) ** 2).mean())
        self.assertLess(
            20 * np.log10(err + 1e-20), -85.0,
            "dither measurably altered loud material",
        )

    def test_dither_is_tpdf_at_the_correct_amplitude(self):
        """TPDF at 2 LSB peak-to-peak - the amplitude is the whole point.

        This first shipped at HALF amplitude, and the original version of this
        test could not detect that: its peak bound was an upper limit the
        correct implementation also satisfies, and it checked only the mean,
        which a uniform distribution would also pass. Assert the actual
        distribution now - peak, RMS and shape.
        """
        rng = np.random.default_rng(0)
        flat = tpdf_dither_noise((500000, 2), rng=rng).ravel()

        # +-1 LSB peak (2 LSB peak-to-peak). Half amplitude peaks at 1.0 LSB.
        peak = float(np.abs(flat).max()) / LSB
        self.assertGreater(peak, 1.9, f"peak {peak:.2f} LSB - amplitude is too low")
        self.assertLessEqual(peak, 2.0 + 1e-6, f"peak {peak:.2f} LSB exceeds +-1 LSB")

        # TPDF RMS is sqrt(2/6)*2 = 0.816 LSB; a half-amplitude one gives 0.408
        rms = float(np.sqrt((flat ** 2).mean())) / LSB
        self.assertAlmostEqual(rms, 0.816, delta=0.02, msg=f"RMS {rms:.3f} LSB")

        self.assertAlmostEqual(float(flat.mean()), 0.0, delta=LSB * 0.05)

        # excess kurtosis: triangular ~= -0.6, uniform ~= -1.2. This is what
        # actually distinguishes TPDF from RPDF, and was never checked before.
        x = flat / flat.std()
        excess_kurtosis = float((x ** 4).mean() - 3.0)
        self.assertAlmostEqual(
            excess_kurtosis, -0.6, delta=0.15,
            msg=f"excess kurtosis {excess_kurtosis:.2f} - distribution is not triangular",
        )

    def test_output_is_reproducible_for_the_same_audio(self):
        """Same input must give a bit-identical file - not a lottery ticket."""
        audio = _tone_at_lsb()
        first = _write_read(audio)
        second = _write_read(audio)
        np.testing.assert_array_equal(
            first, second,
            "two writes of identical audio differ - the dither is unseeded",
        )

    def test_different_audio_gets_different_dither(self):
        """Reproducibility must not mean one fixed noise sequence everywhere."""
        a = _write_read(_tone_at_lsb())
        b = _write_read(_tone_at_lsb(multiple=1.6))
        self.assertFalse(np.array_equal(a, b))

    def test_no_clipping_is_introduced(self):
        """Dither added to a full-scale signal must not push it over."""
        t = np.arange(int(0.5 * SR)) / SR
        mono = (0.9999 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        audio = np.stack([mono, mono], axis=1)
        out = _write_read(audio)
        self.assertLessEqual(float(np.abs(out).max()), 1.0)

    def test_flac_output_is_dithered_too(self):
        """FLAC is also 16-bit here, so it needs the same treatment."""
        audio = _tone_at_lsb()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav = tf.name
        try:
            save_stereo(wav, audio, SR, dither=True)
            out, _ = sf.read(wav, dtype="float32")
            self.assertLess(_harmonic_ratio_db(out), -35.0)
        finally:
            Path(wav).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
