"""The post-chain corrective pass must only fix what the CHAIN created.

Reported directly: an audible fast duck on the word "still" at 0:58 and
1:58 of "Poster on the Wall". Measured at -15.25dB and -14.24dB over ~4ms -
a deep, brief gain drop mid-word, exactly as described.

Traced to the post-chain corrective pass. The primary fix_transients pass
correctly skips these: in the SOURCE those consonants cross the jump
threshold 12-18 times in 30ms, so detect_transients' sustained-burst guard
rejects them as vocal material. But the chain's own compression and limiting
smooth them, and post-chain the same consonants cross only 4-6 times -
under the guard's 8-crossing bar. They then read as clicks, and
fix_transient repairs a click by DELETING it (interpolating across the
region), which punches the hole.

The discriminator is not spectral. Measured on the real file, genuine clicks
and these consonants overlap completely on crossings (2-8 vs 4-6), duration
(0.2-2.2ms vs 0.6-0.9ms) and HF/LF ratio (0.17-4.67 vs 1.25-2.44).

What DOES separate them is provenance: every one of these detections has a
matching large jump in the SOURCE audio (0.43-0.53), i.e. the chain did not
create them - they are pre-existing sharp edges the primary pass already
judged to be vocal. A genuinely chain-created artifact has no such source
counterpart. The corrective pass exists to clean up what later stages
introduce, so that is exactly what it should be limited to.
"""
import unittest

import numpy as np

from app import chain
from app.server import filter_chain_created_transients


SR = 44100


def _stereo(mono):
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _tone(dur=4.0, sr=SR, level=0.2, freq=220):
    t = np.arange(int(dur * sr)) / sr
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class PostChainTransientGuardTests(unittest.TestCase):
    def test_pre_existing_edge_is_not_corrected(self):
        """An edge already in the source must be left alone."""
        mono = _tone()
        c = int(2.0 * SR)
        mono[c:c + 6] += np.array([0.38, -0.41, 0.22, -0.35, 0.28, -0.21],
                                  dtype=np.float32)
        source = _stereo(mono)
        # the "processed" signal is the same audio, slightly gain-changed as
        # the real chain would leave it
        processed = (source * 0.9).astype(np.float32)

        hits = [{"time_sec": 2.0}]
        keep = filter_chain_created_transients(hits, processed, source, SR, 0)
        self.assertEqual(
            keep, [],
            "an edge present in the source was treated as chain-created",
        )

    def test_genuinely_new_artifact_is_still_corrected(self):
        """An artifact the chain introduced must still be fixed."""
        source = _stereo(_tone())
        mono = source[:, 0].copy()
        c = int(2.0 * SR)
        mono[c:c + 4] += np.array([0.9, -0.9, 0.8, -0.7], dtype=np.float32)
        processed = _stereo(mono)

        hits = [{"time_sec": 2.0}]
        keep = filter_chain_created_transients(hits, processed, source, SR, 0)
        self.assertEqual(
            [h["time_sec"] for h in keep], [2.0],
            "a chain-created artifact was skipped",
        )

    def test_lead_trim_offset_is_accounted_for(self):
        """trim_silence shifts the timeline; the lookup must follow it."""
        mono = _tone()
        c = int(2.0 * SR)
        mono[c:c + 6] += np.array([0.38, -0.41, 0.22, -0.35, 0.28, -0.21],
                                  dtype=np.float32)
        source = _stereo(mono)
        lead = int(0.108 * SR)          # the real file's 108ms head trim
        processed = source[lead:].copy()

        # in the processed timeline the edge now sits earlier by `lead`
        hits = [{"time_sec": 2.0 - lead / SR}]
        keep = filter_chain_created_transients(hits, processed, source, SR, lead)
        self.assertEqual(
            keep, [],
            "the source lookup ignored the trim offset and missed the edge",
        )

    def test_empty_input_is_handled(self):
        source = _stereo(_tone())
        self.assertEqual(
            filter_chain_created_transients([], source, source, SR, 0), []
        )


if __name__ == "__main__":
    unittest.main()
