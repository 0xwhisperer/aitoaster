"""No source-platform metadata may survive into a delivered file.

The concern is not our own encoder tag - it is identifying material left by
the generating platform or upstream encoder. A real Suno export
(audio_archive/crazy2.mp3) carries, in its ID3:

    title   = "Velvet Harmony"
    artist  = "dvmusiclab"
    comment = "made with suno; created=2026-07-22T03:01:13.647Z;
               id=ec28b0b6-6e63-49c2-a3ae-e3972bab8b9d"

plus a second mjpeg stream holding embedded cover art, and a per-stream
"encoder: Lavc60.31" tag. The comment's `id=` field is a unique per-
generation identifier, which is the most sensitive item in the set.

Structurally this cannot survive: the pipeline decodes to raw samples and
re-encodes, so nothing container-level carries through. These tests pin that
guarantee so a future change to the encode path cannot quietly regress it -
for example by switching to a stream-copy fast path, or adding a
"preserve tags" convenience option.
"""
import subprocess
import unittest
from pathlib import Path

import numpy as np

from app.server import encode_final_output, load_stereo


def _find_fixture():
    """Locate the real Suno export used as the fixture.

    Checked against several roots rather than one hard-coded path: this file
    is run both from the main checkout and from git worktrees under
    .claude/worktrees/<name>/, where the repo-relative parent differs.
    Returning None here makes the tests SKIP, which would hide a real
    regression, so the search is deliberately generous.
    """
    name = Path("audio_archive") / "crazy2.mp3"
    here = Path(__file__).resolve()
    roots = [p / name for p in here.parents[:6]]
    # a worktree lives at <repo>/.claude/worktrees/<wt>/thefixer/tests, so the
    # real checkout is four levels above the worktree root
    for parent in here.parents:
        if parent.name == "worktrees":
            roots.append(parent.parent.parent / name)
    for candidate in roots:
        if candidate.exists():
            return candidate
    return None


ARCHIVE = _find_fixture()

# strings from the real file's own metadata, plus generic platform markers
FORBIDDEN = [
    b"suno", b"Suno", b"SUNO",
    b"Velvet Harmony",
    b"dvmusiclab",
    b"ec28b0b6",           # the per-generation tracking id
    b"created=",
    b"Lavc60",             # the SOURCE file's encoder tag
    b"JFIF", b"Exif",      # embedded cover-art image headers
]


class MetadataStrippingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ARCHIVE is None:
            raise unittest.SkipTest("audio_archive/crazy2.mp3 not found")
        cls.audio = load_stereo(str(ARCHIVE), 44100)

    def _encode(self, fmt, tmpdir):
        return encode_final_output(
            self.audio, 44100, fmt, Path(tmpdir) / f"out_{fmt}"
        )

    def test_no_platform_metadata_survives_in_any_format(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ("wav", "mp3", "flac", "m4a"):
                path = self._encode(fmt, tmp)
                raw = path.read_bytes()
                found = [s.decode("latin1") for s in FORBIDDEN if s in raw]
                self.assertEqual(
                    found, [],
                    f"{fmt}: source metadata leaked into the delivered file: {found}",
                )

    def test_embedded_cover_art_stream_is_dropped(self):
        """The source carries an mjpeg image stream; output must be audio only."""
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ("mp3", "flac", "m4a"):
                path = self._encode(fmt, tmp)
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_streams",
                     "-print_format", "json", str(path)],
                    capture_output=True, text=True,
                )
                streams = json.loads(probe.stdout).get("streams", [])
                kinds = [s.get("codec_type") for s in streams]
                self.assertEqual(
                    kinds, ["audio"],
                    f"{fmt}: expected audio only, got {kinds}",
                )

    def test_no_container_level_tags_are_written(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ("wav", "mp3", "flac"):
                path = self._encode(fmt, tmp)
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_format",
                     "-print_format", "json", str(path)],
                    capture_output=True, text=True,
                )
                tags = json.loads(probe.stdout).get("format", {}).get("tags", {})
                self.assertEqual(
                    tags, {},
                    f"{fmt}: unexpected container tags {tags}",
                )

    def test_mp3_carries_no_id3_or_ape_tag_blocks(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._encode("mp3", tmp).read_bytes()
            self.assertNotEqual(raw[:3], b"ID3", "MP3 still has an ID3v2 header")
            self.assertNotEqual(raw[-128:-125], b"TAG", "MP3 still has an ID3v1 footer")
            self.assertNotIn(b"APETAGEX", raw[-256:], "MP3 still has an APEv2 tag")

    def test_wav_has_only_fmt_and_data_chunks(self):
        """No LIST/INFO/id3/bext/iXML chunks, and nothing appended."""
        import struct
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._encode("wav", tmp).read_bytes()
            self.assertEqual(raw[:4], b"RIFF")
            self.assertEqual(raw[8:12], b"WAVE")
            chunks = []
            off = 12
            while off + 8 <= len(raw):
                cid = raw[off:off + 4]
                size = struct.unpack("<I", raw[off + 4:off + 8])[0]
                chunks.append(cid)
                if cid == b"data":
                    end = off + 8 + size + (size & 1)
                    self.assertEqual(
                        len(raw), end,
                        f"{len(raw) - end} trailing bytes after the audio payload",
                    )
                    break
                off += 8 + size + (size & 1)
            self.assertEqual(chunks, [b"fmt ", b"data"], f"unexpected chunks: {chunks}")


if __name__ == "__main__":
    unittest.main()
