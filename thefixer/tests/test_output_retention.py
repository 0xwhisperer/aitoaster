"""Finished jobs must not accumulate on disk forever.

Every job writes 8 files: the delivered output, the A/B original, and six
correction overlays that are each ~2x the size of the audio. Nothing ever
removed them, so outputs/ grew without bound - measured at 6.4GB, which
filled the disk to 99% and started making runs fail outright.

prune_old_outputs keeps the N most recent jobs completely intact (the UI's
A/B and overlay players read those paths, so the run the user is currently
looking at must survive) and removes older ones entirely.
"""
import unittest
from pathlib import Path
import tempfile
import time
import shutil

from app import server


SUFFIXES = (
    "",
    "_orig.wav",
    "_overlay_cnn.wav",
    "_overlay_cnn_loud.wav",
    "_overlay_combined.wav",
    "_overlay_combined_loud.wav",
    "_overlay_linear.wav",
    "_overlay_linear_loud.wav",
)


class OutputRetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._real_dir = server.OUTPUT_DIR
        server.OUTPUT_DIR = self.tmp

    def tearDown(self):
        server.OUTPUT_DIR = self._real_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_job(self, job_id, mtime):
        for suffix in SUFFIXES:
            name = f"{job_id}.wav" if suffix == "" else f"{job_id}{suffix}"
            path = self.tmp / name
            path.write_bytes(b"x" * 1024)
            import os
            os.utime(path, (mtime, mtime))

    def test_keeps_recent_jobs_and_removes_older_ones(self):
        base = time.time() - 10_000
        for i in range(8):
            self._make_job(f"job{i:02d}", base + i * 100)

        removed, freed = server.prune_old_outputs(keep=5)

        self.assertEqual(removed, 3)
        self.assertGreater(freed, 0)
        survivors = {p.name.split("_")[0].replace(".wav", "") for p in self.tmp.iterdir()}
        self.assertEqual(survivors, {"job03", "job04", "job05", "job06", "job07"})

    def test_surviving_jobs_keep_every_file(self):
        """A kept job must retain its overlays - the UI still plays them."""
        base = time.time() - 10_000
        for i in range(7):
            self._make_job(f"job{i:02d}", base + i * 100)

        server.prune_old_outputs(keep=5)

        newest = sorted(p.name for p in self.tmp.iterdir() if p.name.startswith("job06"))
        self.assertEqual(len(newest), len(SUFFIXES))

    def test_no_op_when_under_the_limit(self):
        base = time.time() - 1000
        for i in range(3):
            self._make_job(f"job{i:02d}", base + i * 100)
        before = sorted(p.name for p in self.tmp.iterdir())

        removed, freed = server.prune_old_outputs(keep=5)

        self.assertEqual((removed, freed), (0, 0))
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), before)

    def test_job_id_recovery_handles_every_artifact_suffix(self):
        for suffix in SUFFIXES:
            name = "abc123.wav" if suffix == "" else f"abc123{suffix}"
            self.assertEqual(
                server._job_id_from_output(Path(name)),
                "abc123",
                f"failed to recover job id from {name}",
            )


if __name__ == "__main__":
    unittest.main()
