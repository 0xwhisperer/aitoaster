"""The signal chain order, and the UI's depiction of it.

Findings from an adversarial mastering audit, all confirmed by measurement:

  * normalize_lufs ran at step 8, BEFORE multiband compression and the
    true-peak limiter - i.e. loudness was set and then the two stages that
    change loudness most ran afterwards. The whole 6-pass post-chain drift
    correction existed to paper over that. Loudness now runs second-to-last,
    limiter last.

  * high_pass ran at step 6, after transient repair and HF synthesis. Rumble
    removal must precede every level-dependent stage, since sub-30Hz energy
    inflates the envelope the multiband detector reads and eats limiter
    headroom.

  * fix_phase measured stereo correlation AFTER spectral_revive had
    synthesised new high-frequency content, so the metric was partly
    measuring fabricated material.

  * the webpage listed tools in the order the TOOLS array happened to be
    written, which no longer matched the real path - the page implied one
    chain while the audio got another.
"""
import re
import unittest
from pathlib import Path

from app.server import TOOL_ORDER


APP_JS = Path(__file__).resolve().parents[1] / "static" / "app.js"


class ChainOrderTests(unittest.TestCase):
    def _pos(self, tool):
        self.assertIn(tool, TOOL_ORDER, f"{tool} missing from TOOL_ORDER")
        return TOOL_ORDER.index(tool)

    def test_loudness_is_second_to_last_and_limiter_last(self):
        self.assertEqual(
            TOOL_ORDER[-1], "true_peak_limit",
            "the true-peak limiter must run last - it is the delivery ceiling",
        )
        self.assertEqual(
            TOOL_ORDER[-2], "normalize_lufs",
            "loudness must be set immediately before the limiter, so nothing "
            "after it can move the integrated level off target",
        )

    def test_loudness_runs_after_the_dynamics_stages(self):
        for earlier in ("multiband_compress", "fix_transients"):
            self.assertLess(
                self._pos(earlier), self._pos("normalize_lufs"),
                f"{earlier} changes level, so it must run BEFORE loudness is set",
            )

    def test_high_pass_precedes_every_level_dependent_stage(self):
        hp = self._pos("high_pass")
        for later in ("fix_transients", "spectral_revive", "multiband_compress",
                      "normalize_lufs", "true_peak_limit"):
            self.assertLess(
                hp, self._pos(later),
                f"high_pass must run before {later}, which measures or acts on level",
            )

    def test_phase_correction_precedes_hf_synthesis(self):
        self.assertLess(
            self._pos("fix_phase"), self._pos("spectral_revive"),
            "stereo correlation must be measured on real recorded content, "
            "not on synthesised high frequencies",
        )

    def test_detector_fixes_run_after_timing_changes(self):
        for earlier in ("temporal_normalize", "multiband_compress"):
            for fix in ("linear_fix", "cnn_fix"):
                self.assertLess(
                    self._pos(earlier), self._pos(fix),
                    f"{earlier} alters the signal the detector reads, so it must "
                    f"run before {fix}",
                )

    def test_ui_display_order_matches_the_real_chain(self):
        """The page must not imply a different chain than the one that runs."""
        js = APP_JS.read_text()
        m = re.search(r"const CHAIN_RUN_ORDER = \[(.*?)\];", js, re.S)
        self.assertIsNotNone(m, "CHAIN_RUN_ORDER not found in app.js")
        ui_order = re.findall(r'"([a-z_]+)"', m.group(1))

        # every tool the server runs must appear, in the same relative order
        ui_subset = [t for t in ui_order if t in TOOL_ORDER]
        self.assertEqual(
            ui_subset, list(TOOL_ORDER),
            "the UI's display order has drifted from the server's TOOL_ORDER",
        )

    def test_every_ui_tool_is_placed_in_the_run_order(self):
        js = APP_JS.read_text()
        declared = set(re.findall(r'\{ id: "([a-z_]+)"', js))
        m = re.search(r"const CHAIN_RUN_ORDER = \[(.*?)\];", js, re.S)
        ui_order = set(re.findall(r'"([a-z_]+)"', m.group(1)))
        missing = declared - ui_order
        self.assertEqual(
            missing, set(),
            f"these tools are shown in the UI but have no place in the run "
            f"order, so they sort to the end arbitrarily: {sorted(missing)}",
        )


if __name__ == "__main__":
    unittest.main()
