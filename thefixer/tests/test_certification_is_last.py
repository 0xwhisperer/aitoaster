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

    def test_rendered_card_order_matches_execution_order(self):
        """The cards as they actually RENDER, not the sort array.

        An earlier version of this test compared CHAIN_RUN_ORDER against
        TOOL_ORDER and passed while the page was visibly wrong. The cards
        render inside fixed visual groups and the sort only orders cards
        WITHIN a group, so group assignment overrides it entirely:
        temporal_normalize sat in the "AI" group next to linear_fix/cnn_fix
        while executing 5th, and fix_phase sat in "Mastering" while executing
        7th. Checking the sort array could never catch that.

        This reconstructs the real render order: group order from the HTML,
        then cards within each group by the sort array.
        """
        import pathlib
        import re
        repo = pathlib.Path(__file__).resolve().parent.parent
        app_js = (repo / "static" / "app.js").read_text()
        index_html = (repo / "static" / "index.html").read_text()

        run = [t.strip().strip('"') for t in
               re.search(r"CHAIN_RUN_ORDER = \[(.*?)\]", app_js, re.S).group(1).split(",")
               if t.strip().strip('"')]
        card_group = dict(re.findall(r'id: "([a-z_]+)", group: "(chainGroup[A-Za-z]+)"', app_js))
        group_order = re.findall(r'data-target="(chainGroup[A-Za-z]+)"', index_html)

        rendered = []
        for group in group_order:
            rendered += [t for t in run if card_group.get(t) == group]

        expected = list(server.TOOL_ORDER) + [server.FADE_TOOL]

        # DELIBERATE EXCEPTION, by product decision: temporal_normalize is
        # grouped with the AI-detector fixes because that is what it is FOR -
        # it displaces the low-frequency spectral peaks that fingerprint
        # matching uses as anchors. It EXECUTES 5th, with the other timeline
        # stages and well before certification, which is what actually
        # matters; only its card placement differs. Every other tool must
        # still render in execution order, so the comparison drops this one
        # tool from both sides rather than being relaxed.
        CARD_ORDER_EXCEPTIONS = {"temporal_normalize"}
        rendered_checked = [t for t in rendered if t not in CARD_ORDER_EXCEPTIONS]
        expected_checked = [t for t in expected if t not in CARD_ORDER_EXCEPTIONS]

        self.assertEqual(
            rendered_checked, expected_checked,
            "the cards RENDER in a different order than the chain executes. "
            "Check the group assignments in app.js TOOLS and the group order "
            "in index.html - sorting alone cannot fix a card in the wrong "
            "group.",
        )
        # the exception is a card-placement choice, NOT permission to execute
        # out of order: it must still run before the detector fixes
        order = list(server.TOOL_ORDER)
        for tool in CARD_ORDER_EXCEPTIONS:
            if tool in order and "cnn_fix" in order:
                self.assertLess(
                    order.index(tool), order.index("cnn_fix"),
                    f"{tool} is a card-order exception, but it must still "
                    "EXECUTE before certification",
                )


if __name__ == "__main__":
    unittest.main()


class DeliveredCertificationIsDenseTests(unittest.TestCase):
    """The delivered file must be scanned in full, not at five positions.

    scorer.score() samples 5 window positions derived from len(audio). That
    is the right instrument for "is this track AI-generated?" and the wrong
    one for "is this delivered file certified?": measured on a real delivered
    file the fixed-5 median read 0.00112% while a dense scan found 0.12442%
    at 95.0s - a 111x under-report, because the worst window fell between two
    sampled positions. On another track the gap reached 19x against the
    fixed-5 WORST.
    """

    def test_the_detector_can_scan_densely(self):
        from app.detector import CNNDetector
        self.assertTrue(
            hasattr(CNNDetector, "scan_dense"),
            "CNNDetector.scan_dense is gone - delivered certification would "
            "fall back to five fixed positions",
        )

    def test_the_pipeline_scans_the_delivered_file_densely(self):
        src = inspect.getsource(server.run_pipeline)
        self.assertIn(
            "scan_dense", src,
            "run_pipeline no longer scans the delivered file densely; a "
            "worst window between two sampled positions would go unreported",
        )
        self.assertIn(
            "delivered-file dense scan", src,
            "the dense scan result is not reported to the user",
        )


class FadesAreNotTrimmedTests(unittest.TestCase):
    """trim_silence must not delete a fade, whoever calls it.

    Its threshold is -66dBFS, not digital zero, so the tail of a fade-out
    reads as silence: measured, a 3-second fade-out passed straight in lost
    61ms (2458 samples). Making only the RECOMMENDER fade-aware left the tool
    itself still doing it.
    """

    SR = 44100

    def _faded_tone(self, seconds=10.0, fade_sec=3.0):
        n = int(seconds * self.SR)
        t = np.arange(n) / self.SR
        sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        env = np.ones(n, dtype=np.float32)
        f = int(fade_sec * self.SR)
        env[-f:] = np.cos(np.linspace(0, np.pi / 2, f)) ** 2
        mono = (sig * env).astype(np.float32)
        return np.stack([mono, mono], axis=1)

    def test_a_fade_out_is_not_trimmed(self):
        from app import chain
        audio = self._faded_tone()
        out, info = chain.trim_silence(audio, self.SR)
        self.assertEqual(
            len(out), len(audio),
            f"{len(audio) - len(out)} samples were cut off a deliberate "
            "3-second fade-out",
        )
        self.assertTrue(info.get("fade_protected"))

    def test_real_silence_is_still_trimmed(self):
        """The protection must not disable the tool."""
        from app import chain
        n = int(2.0 * self.SR)
        t = np.arange(n) / self.SR
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        pad = np.zeros(self.SR, dtype=np.float32)
        mono = np.concatenate([pad, tone, pad]).astype(np.float32)
        audio = np.stack([mono, mono], axis=1)
        out, info = chain.trim_silence(audio, self.SR)
        self.assertTrue(info["applied"], "flat digital silence was not trimmed")
        self.assertLess(len(out), len(audio))
        self.assertGreater(info["lead_ms"], 900)

    def test_the_helper_separates_a_fade_from_flat_silence(self):
        from app import chain
        audio = self._faded_tone()
        self.assertTrue(
            chain._tail_is_decaying(audio, int(3.0 * self.SR)),
            "a real fade was not recognised as decaying",
        )
        flat = np.zeros((int(3.0 * self.SR), 2), dtype=np.float32)
        tone = np.stack([np.full(self.SR, 0.5, dtype=np.float32)] * 2, axis=1)
        silent_tail = np.concatenate([tone, flat]).astype(np.float32)
        self.assertFalse(
            chain._tail_is_decaying(silent_tail, int(3.0 * self.SR)),
            "flat digital silence was mistaken for a fade - real silence "
            "would stop being trimmed",
        )
