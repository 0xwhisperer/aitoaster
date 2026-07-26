"""THE TIMELINE INVARIANT: freeze the timeline before certifying.

A CNN certification is bound to the exact timeline it was made on.
CNNDetector.extract_segments derives every analysis window position from
len(audio), so changing the sample count slides every window and the
correction stops being where the detector looks.

This bug class has broken a delivered file seven times. The most recent: a
post-chain corrective trim cut 1.4ms lead and 311.5ms trail off a file
certified at 0.003%, which was delivered at 78.8%.

An earlier draft of this suite asserted the wrong rule - "nothing may mutate
the audio after cnn_fix". Measured against the real model, that rule is
backwards in both directions:

    normalize_lufs   -0.0002 / -0.0029pp   would have been guarded (safe)
    true_peak_limit  -0.0000 / +0.0000pp   would have been guarded (safe)
    fade, watermark  +0.0000 / +0.0000pp   would have been guarded (inert)
    temporal_normalize  +4.87 / +97.05pp   MISSED - it changes no length
    trail trim 311.5ms  +87.03 / +31.11pp  the actual bug

Amplitude changes at fixed positions are free. Timeline changes are fatal.
So these tests assert the length invariant, not a mutation ban.
"""
import inspect
import unittest

import numpy as np

from app import server


class TimelineStagesRunBeforeCertification(unittest.TestCase):
    def test_no_timeline_stage_runs_after_a_detector_fix(self):
        order = list(server.TOOL_ORDER)
        fixes = [t for t in ("linear_fix", "cnn_fix") if t in order]
        if not fixes:
            self.skipTest("no detector fix in TOOL_ORDER")
        first_fix = min(order.index(t) for t in fixes)
        for i, tool in enumerate(order):
            if tool in server.TIMELINE_STAGES and i > first_fix:
                self.fail(
                    f"'{tool}' changes the timeline but runs at position {i}, "
                    f"after '{order[first_fix]}' at {first_fix}. Every CNN "
                    "analysis window position derives from len(audio), so a "
                    "later timeline change invalidates the certification "
                    "wholesale - measured up to +99.5 percentage points.",
                )

    def test_only_measured_inert_stages_run_after_certification(self):
        order = list(server.TOOL_ORDER)
        if "cnn_fix" not in order:
            self.skipTest("cnn_fix not in TOOL_ORDER")
        for tool in order[order.index("cnn_fix") + 1:]:
            self.assertIn(
                tool, server.POST_CERTIFICATION_ALLOWED,
                f"'{tool}' runs after cnn_fix but has not been measured inert "
                "on a certified signal. Either move it before cnn_fix, or "
                "measure it and add it to POST_CERTIFICATION_ALLOWED with the "
                "number.",
            )

    def test_the_allow_list_is_amplitude_domain_only(self):
        """A stage that could ever change length must not be on the list."""
        overlap = server.POST_CERTIFICATION_ALLOWED & server.TIMELINE_STAGES
        self.assertEqual(
            overlap, frozenset(),
            f"{overlap} is on both lists - a timeline stage can never be "
            "safe after certification",
        )


class TheInvariantIsEnforcedAtRuntime(unittest.TestCase):
    def test_pipeline_asserts_the_length_it_certified(self):
        """The structural guard. A threshold re-score is not sufficient:
        fragility is CHAOTIC, not monotonic - a 10ms lead trim measured
        18.41% while 100ms measured 0.03% on the same track. A near-miss is
        therefore not a safe margin, and only an exact length assertion
        catches every case."""
        src = inspect.getsource(server.run_pipeline)
        self.assertIn(
            "_certified_length", src,
            "run_pipeline does not capture the length it certified at. "
            "Without it, any later stage that changes the sample count ships "
            "an invalid certificate silently.",
        )
        self.assertIn(
            "CERTIFIED TIMELINE CHANGED", src,
            "there is no assertion that the delivered length matches the "
            "certified length",
        )


class ExecutionOrderIsSingleSourced(unittest.TestCase):
    """The page's chain cards must show the order that actually runs."""

    def test_ui_order_matches_server_order(self):
        import pathlib
        repo = pathlib.Path(__file__).resolve().parent.parent
        app_js = (repo / "static" / "app.js").read_text()
        start = app_js.index("CHAIN_RUN_ORDER")
        block = app_js[start:app_js.index("]", start)]
        ui = [t.strip().strip('"') for t in block.split("[")[1].split(",")
              if t.strip().strip('"')]
        ui_chain = [t for t in ui if t in server.TOOL_ORDER]
        self.assertEqual(
            ui_chain, list(server.TOOL_ORDER),
            "the page shows a different signal chain than the server runs",
        )


if __name__ == "__main__":
    unittest.main()
