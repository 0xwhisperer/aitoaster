"""The delivered file must match the format the user picked, and when it
deliberately does not, the app must SAY SO somewhere the user can see.

Reported as "if I select mp3, I expect mp3 and not a flac". The encoder was
never the problem - resolve_output_format() overrides a lossy request to FLAC
when a detector fix is selected, because a lossy encode overwrites the
correction (a file certified at 0.001% scored 99.2% after AAC encoding).

That override is correct and must stay. What was missing is that its
explanation existed ONLY as a job_log line, so the substitution looked
arbitrary. These tests pin both halves: the override fires exactly when it
should and never wider, and it always comes with a warning string that the
result payload can carry back to the UI.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.server import (  # noqa: E402
    DETECTOR_FIX_TOOLS,
    LOSSY_OUTPUT_FORMATS,
    encode_final_output,
    resolve_output_format,
)


def _codec_of(path):
    """What the file ACTUALLY contains, not what its extension claims."""
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()


class TestExplicitFormatIsHonored(unittest.TestCase):
    """An explicitly chosen format is delivered verbatim unless a detector
    fix forces the override."""

    def test_mp3_without_detector_fix_stays_mp3(self):
        # the exact reported complaint, in the configuration where the app
        # has no reason to override anything
        fmt, warning = resolve_output_format("mp3", "song.wav", ())
        self.assertEqual(fmt, "mp3")
        self.assertIsNone(warning)

    def test_mp3_with_unrelated_tools_stays_mp3(self):
        # only the two detector fixes may trigger the override - ordinary
        # chain stages must not drag a request to FLAC
        fmt, warning = resolve_output_format(
            "mp3", "song.wav", ("normalize_lufs", "strip_metadata", "limiter"))
        self.assertEqual(fmt, "mp3")
        self.assertIsNone(warning)

    def test_lossless_requests_are_never_overridden(self):
        for fmt_in in ("wav", "flac"):
            for tool in sorted(DETECTOR_FIX_TOOLS):
                fmt, warning = resolve_output_format(fmt_in, "song.wav", (tool,))
                self.assertEqual(fmt, fmt_in)
                self.assertIsNone(warning)


class TestDetectorFixOverride(unittest.TestCase):
    """The override itself: lossy + detector fix -> FLAC, always explained."""

    def test_each_detector_fix_overrides_each_lossy_format(self):
        for tool in sorted(DETECTOR_FIX_TOOLS):
            for fmt_in in sorted(LOSSY_OUTPUT_FORMATS):
                fmt, warning = resolve_output_format(fmt_in, "song.wav", (tool,))
                self.assertEqual(fmt, "flac")
                # the substitution must never be silent
                self.assertTrue(warning)
                self.assertIn(fmt_in, warning)

    def test_same_as_source_lossy_upload_also_overrides(self):
        # the original shipped-broken-file bug: .m4a in, .m4a out, correction
        # destroyed by the encode
        fmt, warning = resolve_output_format("same", "song.m4a", ("cnn_fix",))
        self.assertEqual(fmt, "flac")
        self.assertTrue(warning)


class TestSameAsSource(unittest.TestCase):
    def test_supported_extensions_round_trip(self):
        for ext, expected in (("mp3", "mp3"), ("flac", "flac"),
                              ("m4a", "m4a"), ("aac", "m4a"), ("wav", "wav")):
            fmt, warning = resolve_output_format("same", f"song.{ext}", ())
            self.assertEqual(fmt, expected)
            self.assertIsNone(warning)

    def test_unsupported_extension_falls_back_to_wav_with_warning(self):
        for ext in ("ogg", "opus", "aiff", "wma"):
            fmt, warning = resolve_output_format("same", f"song.{ext}", ())
            self.assertEqual(fmt, "wav")
            self.assertTrue(warning)


class TestEncoderWritesRequestedFormat(unittest.TestCase):
    """Guards the other reading of the report: that .mp3 might hold FLAC
    bytes. Verified by decoding, not by trusting the extension."""

    @classmethod
    def setUpClass(cls):
        sr = 44100
        t = np.arange(sr) / sr
        mono = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        cls.audio = np.stack([mono, mono], axis=1)
        cls.sr = sr

    def test_each_format_produces_matching_real_bytes(self):
        expected = {"mp3": "mp3", "flac": "flac",
                    "wav": "pcm_s16le", "m4a": "aac"}
        with tempfile.TemporaryDirectory() as d:
            for fmt, codec in expected.items():
                out = encode_final_output(
                    self.audio, self.sr, fmt, str(Path(d) / f"o_{fmt}"))
                self.assertEqual(Path(out).suffix, f".{fmt}")
                self.assertEqual(_codec_of(out), codec)

    def test_no_stray_temp_wav_left_behind(self):
        with tempfile.TemporaryDirectory() as d:
            encode_final_output(self.audio, self.sr, "mp3", str(Path(d) / "o"))
            self.assertEqual([p.name for p in Path(d).glob("*_tmp.wav")], [])


class TestWarningReachesTheUser(unittest.TestCase):
    """The actual defect behind the report: the reason was computed but
    never returned to the client."""

    def test_result_payload_carries_requested_format_and_warning(self):
        src = Path(__file__).resolve().parent.parent / "app" / "server.py"
        text = src.read_text()
        self.assertIn('"format_fallback_warning": format_fallback_warning', text)
        self.assertIn('"requested_format": output_format', text)

    def test_frontend_reads_the_warning(self):
        js = (Path(__file__).resolve().parent.parent
              / "static" / "app.js").read_text()
        self.assertIn("format_fallback_warning", js)
        # and explains the override at selection time, before the run
        self.assertIn("renderFormatOverrideNotice", js)


if __name__ == "__main__":
    unittest.main(verbosity=2)
