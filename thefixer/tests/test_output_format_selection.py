"""The selected output format is ALWAYS the delivered format.

Reported as "if I select mp3, I expect mp3 and not a flac". The app used to
override a lossy selection to FLAC whenever a detector fix was chosen. The
measurement behind that override is real - a lossy encode overwrites the
correction, and a file certified at 0.001% scored 99.2% after AAC encoding -
but silently substituting the format the user explicitly picked was the wrong
response to it. The interaction is now surfaced as a warning instead.

These tests pin the contract: pick mp3 and get .mp3, pick flac and get .flac,
pick wav and get .wav, with a detector fix selected or not. The advisory
warning must still be produced, and must never change the delivered format.
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
    lossy_detector_fix_warning,
    resolve_output_format,
)


def _codec_of(path):
    """What the file ACTUALLY contains, not what its extension claims."""
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()


class TestExplicitFormatIsAlwaysHonored(unittest.TestCase):
    """Pick a format, get that format. No exceptions."""

    ALL_TOOL_SETS = (
        (),
        ("normalize_lufs", "strip_metadata"),
        ("cnn_fix",),
        ("linear_fix",),
        ("cnn_fix", "linear_fix", "normalize_lufs"),
    )

    def test_every_format_survives_every_tool_combination(self):
        # the reported complaint, exhaustively: no tool selection may change
        # the delivered format away from what was picked
        for fmt_in in ("mp3", "flac", "wav", "m4a"):
            for tools in self.ALL_TOOL_SETS:
                fmt, warning = resolve_output_format(fmt_in, "song.wav", tools)
                self.assertEqual(
                    fmt, fmt_in,
                    f"selected {fmt_in} with tools {tools} but got {fmt}")
                # an explicit choice is never a "fallback"
                self.assertIsNone(warning)

    def test_mp3_with_detector_fix_is_still_mp3(self):
        # the precise case that used to yield FLAC
        for tool in sorted(DETECTOR_FIX_TOOLS):
            fmt, _ = resolve_output_format("mp3", "song.wav", (tool,))
            self.assertEqual(fmt, "mp3")


class TestLossyDetectorFixWarning(unittest.TestCase):
    """The interaction is now advisory - it warns without substituting."""

    def test_warns_for_each_lossy_format_and_detector_fix(self):
        for tool in sorted(DETECTOR_FIX_TOOLS):
            for fmt in sorted(LOSSY_OUTPUT_FORMATS):
                self.assertTrue(lossy_detector_fix_warning(fmt, (tool,)))

    def test_silent_without_a_detector_fix(self):
        for fmt in sorted(LOSSY_OUTPUT_FORMATS):
            self.assertIsNone(lossy_detector_fix_warning(fmt, ()))
            self.assertIsNone(
                lossy_detector_fix_warning(fmt, ("normalize_lufs",)))

    def test_silent_for_lossless_formats(self):
        for fmt in ("wav", "flac"):
            for tool in sorted(DETECTOR_FIX_TOOLS):
                self.assertIsNone(lossy_detector_fix_warning(fmt, (tool,)))

    def test_warning_does_not_promise_a_different_format(self):
        # regression guard: the advisory must not claim FLAC is being
        # delivered, which is what the old override text said
        w = lossy_detector_fix_warning("mp3", ("cnn_fix",))
        self.assertIn("mp3", w)
        self.assertNotIn("Delivering lossless", w)


class TestSameAsSourceLossyUpload(unittest.TestCase):
    def test_lossy_upload_with_detector_fix_still_matches_source(self):
        # "same as source" means same as source, even with a detector fix
        fmt, warning = resolve_output_format("same", "song.m4a", ("cnn_fix",))
        self.assertEqual(fmt, "m4a")
        self.assertIsNone(warning)
        # ...but the risk is still reported
        self.assertTrue(lossy_detector_fix_warning(fmt, ("cnn_fix",)))


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
        # and warns at selection time, before the run
        self.assertIn("renderFormatOverrideNotice", js)

    def test_frontend_never_rewrites_the_chosen_extension(self):
        # the filename field silently turning .mp3 into .flac was the most
        # visible symptom of the old override
        js = (Path(__file__).resolve().parent.parent
              / "static" / "app.js").read_text()
        self.assertNotIn('fmt = "flac"', js)


class TestM4aIsSelectable(unittest.TestCase):
    """M4A was encodable all along but had no button - reachable only via
    "same as source" from an m4a/aac upload."""

    def test_api_accepts_m4a(self):
        text = (Path(__file__).resolve().parent.parent
                / "app" / "server.py").read_text()
        self.assertIn('"m4a"', text)

    def test_ui_offers_an_m4a_button(self):
        html = (Path(__file__).resolve().parent.parent
                / "static" / "index.html").read_text()
        self.assertIn('data-format="m4a"', html)

    def test_m4a_resolves_and_encodes(self):
        fmt, warning = resolve_output_format("m4a", "song.wav", ())
        self.assertEqual(fmt, "m4a")
        self.assertIsNone(warning)


if __name__ == "__main__":
    unittest.main(verbosity=2)
