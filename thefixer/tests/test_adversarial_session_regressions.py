import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import signal

from app import chain, cnn_fix, server, watermark


class FrontendThresholdRegressionTests(unittest.TestCase):
    def test_results_table_uses_same_lufs_and_dc_bars_as_reanalysis(self):
        app_js = (
            Path(__file__).resolve().parents[1] / "static" / "app.js"
        ).read_text()

        self.assertTrue(
            "result.lufs_after >= -17 && result.lufs_after <= -13" in app_js,
            "The results table must not call a LUFS value good when uploading "
            "that same value again recommends normalize_lufs.",
        )
        self.assertTrue(
            "dcMaxAfter < 6e-5" in app_js,
            "The results table must use the same DC threshold as /api/analyze.",
        )


class TransientRegressionTests(unittest.TestCase):
    def test_two_real_clicks_inside_dedup_window_are_both_reported(self):
        sr = 1_000
        audio = np.zeros((2 * sr, 2), dtype=np.float32)
        audio[500] = 1.0
        audio[800] = 0.9

        detections = chain.detect_transients(audio, sr)

        self.assertEqual(
            [item["time_sec"] for item in detections],
            [0.5, 0.8],
            "The 0.5-second skip window must not hide a distinct second click.",
        )


class TransientBridgeScopeRegressionTests(unittest.TestCase):
    # BUG FIX (Grok #9 / Fable N3, verified directly): fix_transient's fixed
    # ~30-sample bridge_half is single-file-tuned and could only partially
    # bridge a wider dropout - but that scenario never reaches fix_transient
    # in the first place, because detect_transients only fires on a genuine
    # single-sample-scale derivative jump, which a wide dropout's own edges
    # don't produce at real sample rates. Locks in that scope boundary
    # directly so a future change to detect_transients' thresholds can't
    # silently start routing wide dropouts into a bridge width that was
    # never sized for them.
    def test_wide_dropout_is_not_classified_as_a_bridgeable_transient(self):
        sr = 44_100
        n = 3 * sr
        t = np.arange(n) / sr
        tone = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        audio = np.stack([tone, tone], axis=1).copy()

        dropout_start = int(1.5 * sr)
        dropout_len = int(0.01 * sr)  # 10ms - much wider than bridge_half (~0.7ms)
        audio[dropout_start:dropout_start + dropout_len] = 0.0

        found = chain.detect_transients(audio, sr)
        self.assertEqual(
            found, [],
            "a wide dropout must not be routed into fix_transient's bridge, "
            "which is sized only for genuine single-sample-scale clicks.",
        )

    def test_genuine_single_sample_click_is_still_classified_and_bridgeable(self):
        sr = 44_100
        n = 3 * sr
        t = np.arange(n) / sr
        tone = (0.02 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        audio = np.stack([tone, tone], axis=1).copy()

        click_idx = int(1.5 * sr)
        audio[click_idx] = 0.9

        found = chain.detect_transients(audio, sr)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["time_sec"], 1.5, places=2)


class SpectralReviveRegressionTests(unittest.TestCase):
    def test_revived_output_may_still_be_flagged_by_design(self):
        # BUG FIX HISTORY (second adversarial audit round): three separate
        # "already revived" detection attempts were each falsified by a
        # real counterexample (see detect_spectral_rolloff's own module
        # comment for the full history) - most seriously, a genuinely
        # gentle, real, NEVER-touched natural rolloff (as mild as
        # 50dB/octave) produces a spectral shape statistically
        # indistinguishable from spectral_revive's own glide curve, so any
        # check that suppressed re-flagging on revived output would also
        # wrongly suppress re-flagging on plenty of real, still-broken
        # files - a false POSITIVE (silently telling a user their broken
        # file is fine) that is strictly worse than the false NEGATIVE this
        # test used to guard against (redundantly recommending
        # spectral_revive again on a file this app already fixed, which is
        # harmless - spectral_revive is not destructive to re-run). This
        # test now documents and locks in the deliberate choice: a revived
        # file MAY still measure as having a rolloff, and that is
        # intentional, not a regression.
        sr = 44_100
        n = 6 * sr
        rng = np.random.default_rng(7)
        source = rng.normal(0, 0.08, n)
        spectrum = np.fft.rfft(source)
        freqs = np.fft.rfftfreq(n, 1 / sr)
        spectrum[freqs >= 17_000] = 0
        mono = np.fft.irfft(spectrum, n).astype(np.float32)
        audio = np.column_stack([mono, mono])

        self.assertTrue(chain.detect_spectral_rolloff(audio, sr)[0])
        revived, info = chain.spectral_revive(
            audio, sr, cutoff_hz=17_000, seed=1
        )

        self.assertTrue(info["applied"])
        # no assertion on detect_spectral_rolloff(revived, sr) here by
        # design - either True or False is an acceptable outcome now that
        # the (unreliable) already-revived suppression has been removed.

    def test_untreated_rolloff_inside_revived_deficit_band_is_still_detected(self):
        sr = 44_100
        n = 6 * sr
        rng = np.random.default_rng(123)
        source = rng.normal(0, 0.08, n)
        spectrum = np.fft.rfft(source)
        freqs = np.fft.rfftfreq(n, 1 / sr)
        spectrum[freqs >= 17_000] *= 10 ** (-18 / 20)
        mono = np.fft.irfft(spectrum, n).astype(np.float32)
        audio = np.column_stack([mono, mono])

        detected, cutoff, deficit = chain.detect_spectral_rolloff(audio, sr)

        self.assertGreater(deficit, 6.0)
        self.assertTrue(
            detected,
            f"An untreated {deficit:.2f}dB hard spectral step must not be "
            "classified as already revived merely because its scalar deficit "
            "falls inside the learned revived-output band.",
        )
        self.assertEqual(cutoff, 17_000)


class MultibandCompressionRegressionTests(unittest.TestCase):
    SR = 44_100

    def _stereo_sine(self, frequency_hz, amplitude, duration_sec=2):
        t = np.arange(int(self.SR * duration_sec)) / self.SR
        mono = amplitude * np.sin(2 * np.pi * frequency_hz * t)
        return np.column_stack([mono, mono]).astype(np.float32)

    def test_reported_noop_does_not_filter_the_audio(self):
        audio = self._stereo_sine(21_000, 0.01)

        processed, info = chain.multiband_compress(audio, self.SR)

        self.assertFalse(info["applied"])
        np.testing.assert_allclose(
            processed,
            audio,
            rtol=0,
            atol=1e-7,
            err_msg="A pass reported as 'no change needed' must return the "
            "original samples, not silently low-pass content above 20 kHz.",
        )

    def test_recommendation_cannot_route_to_a_reported_noop(self):
        # BUG FIX (adversarial audit, verified directly): 0.38 amplitude
        # used to cross detect_band_peakiness's OLD 0.5dB guard band
        # (recommend=True) while staying under multiband_compress's own
        # 0.3dB applied bar (applied=False) - exactly the contradiction
        # this test exists to catch. That gap is now closed by deriving
        # the guard band from the SAME math as the applied threshold (see
        # MEANINGFULLY_OVER_DB's comment in detect_band_peakiness), so
        # 0.38 amplitude now correctly reports False/False (no
        # contradiction, just genuinely below both bars) - 0.45 is a real
        # amplitude confirmed to cross both thresholds together under the
        # new, tied-together math, keeping this test's actual intent
        # (recommend and applied must never disagree) meaningful.
        audio = self._stereo_sine(1_000, 0.45)
        peakiness = chain.detect_band_peakiness(audio, self.SR)

        self.assertTrue(any(b["frac_time_over"] > 0.02 for b in peakiness))
        _processed, info = chain.multiband_compress(audio, self.SR)

        self.assertTrue(
            info["applied"],
            "The recommender selected multiband compression, but its own "
            "0.3 dB applied threshold reports the selected operation as "
            "'no change needed'.",
        )

    def test_one_pass_makes_real_progress_toward_clearing_its_recommendation(self):
        # BUG FIX (adversarial audit, verified directly): "one pass fully
        # clears the recommendation" is not an achievable guarantee for
        # this tool AS DESIGNED, and was never meant to be - ratio=1.3 is
        # deliberately gentle (see multiband_compress's own docstring:
        # "least change necessary"), and this session already found and
        # accepted (informal testing on a genuinely peaky signal, plus
        # this test's own original fixture, a sustained 1kHz tone) that
        # real convergence against a strongly, uniformly over-threshold
        # signal takes multiple passes - 4-5 for a sustained tone,
        # confirmed directly by running the correction repeatedly and
        # watching peak_over_db decay geometrically each time, never in
        # one jump. Forcing a false one-pass "done" signal would mean
        # either lying about convergence or making the tool far more
        # aggressive than its own documented, deliberate design goal. The
        # real, honest guarantee detect_band_peakiness's frac_time_over
        # metric can make is REAL, monotonic improvement per pass - not
        # full clearance - which is exactly what this test now checks.
        audio = self._stereo_sine(1_000, 0.5)
        peakiness_before = chain.detect_band_peakiness(audio, self.SR)
        self.assertTrue(any(b["frac_time_over"] > 0.02 for b in peakiness_before))
        mid_before = next(b for b in peakiness_before if b["range_hz"] == [200, 2000])

        processed, info = chain.multiband_compress(audio, self.SR)
        self.assertTrue(info["applied"])

        peakiness_after = chain.detect_band_peakiness(processed, self.SR)
        mid_after = next(b for b in peakiness_after if b["range_hz"] == [200, 2000])
        self.assertLess(
            mid_after["peak_over_db"],
            mid_before["peak_over_db"],
            "A pass reported as applied=True must make real, measurable "
            "progress against the file's own peakiness, even if a single "
            "gentle pass does not fully clear it.",
        )

    def test_compression_cannot_create_clipping_after_limiter_precheck_passes(self):
        burst_t = np.arange(int(0.015 * self.SR)) / self.SR
        burst = (
            0.32813856
            * np.sin(2 * np.pi * 1_000 * burst_t + 6.10818077)
            + 1.57126989
            * np.sin(2 * np.pi * 10_000 * burst_t + 2.54308418)
        )
        burst = (burst * (0.88 / np.max(np.abs(burst)))).astype(np.float32)
        mono = np.zeros(5 * self.SR, dtype=np.float32)
        period = int(0.4 * self.SR)
        for end in range(period, len(mono) + 1, period):
            mono[end - len(burst):end] = burst
        mono[-len(burst):] = burst
        audio = np.column_stack([mono, mono])

        self.assertGreaterEqual(chain.measure_lufs(audio, self.SR), -17)
        self.assertLessEqual(chain.measure_lufs(audio, self.SR), -13)
        self.assertTrue(
            any(
                b["frac_time_over"] > 0.02
                for b in chain.detect_band_peakiness(audio, self.SR)
            ),
            "Fixture must be recommended for multiband compression.",
        )
        _unchanged, limiter_precheck = chain.true_peak_limit(audio, self.SR)
        self.assertFalse(
            limiter_precheck["applied"],
            "Fixture must begin below the pipeline's -1 dBTP limiter bar.",
        )

        processed, info = chain.multiband_compress(audio, self.SR)

        self.assertTrue(info["applied"])
        self.assertLessEqual(
            float(np.max(np.abs(processed))),
            1.0,
            "Multiband compression runs after the limiter recommendation "
            "precheck, so it must not turn a limiter-safe input into clipped "
            "samples when the limiter was not selected.",
        )


class CnnTransferVerificationRegressionTests(unittest.TestCase):
    def test_too_short_post_transfer_window_cannot_verify_as_zero_score(self):
        n = cnn_fix.CNN_SR * 10 - 1
        stereo = np.full((n, 2), 0.1, dtype=np.float32)
        delta = np.full(n, 0.001, dtype=np.float32)

        with patch.object(
            cnn_fix,
            "optimize_eot_verified",
            return_value=(delta, [0], cnn_fix.CNN_SR * 10, 0.01),
        ):
            _out, info = cnn_fix.fix_cnn(
                stereo,
                cnn_fix.CNN_SR,
                max_steps=1,
                min_steps=0,
                mode="eot",
            )

        self.assertFalse(
            info["verified_after_transfer"],
            "An empty worst-shift scan must fail closed; it currently returns "
            "0.0 and certifies a segment that was never scored.",
        )


class PipelineArtifactRegressionTests(unittest.TestCase):
    def setUp(self):
        with server.JOBS_LOCK:
            server.JOBS.clear()
            server.JOBS["audit-job"] = {
                "status": "running",
                "log": [],
                "result": None,
                "error": None,
                "progress_msg": "",
                "current_step_idx": None,
                "total_steps": None,
                "current_step_name": None,
                "sub_progress": None,
                "cancel_requested": False,
            }

    def tearDown(self):
        with server.JOBS_LOCK:
            server.JOBS.clear()

    def test_detector_scoring_copy_matches_pcm_sent_to_final_encoder(self):
        # BUG FIX HISTORY: this test's original mock assumed scores_after
        # came from a SEPARATE "_score.wav" snapshot file, matching the
        # architecture at the time. That snapshot has since been removed
        # entirely (see test_lossy_delivery_artifact_itself_is_scored,
        # which supersedes this test with an even stronger assertion) -
        # scores_after is now computed by scoring out_path DIRECTLY, so
        # there is no separate WAV to intercept via save_stereo any more.
        # Updated to assert the same underlying property (scoring reflects
        # the real, watermark-mutated, delivered PCM) against the current
        # architecture: capture the path scorer.score() is actually called
        # with and confirm it's the exact path encode_final_output
        # returned, not a proxy of any kind.
        audio = np.full((2_000, 2), 0.25, dtype=np.float32)
        captured = {}

        class FakeScorer:
            def score(self, path):
                captured.setdefault("scored_paths", []).append(Path(path))
                return {
                    "linear": {"probability": 0.0},
                    "cnn": {"probability": 0.0},
                    "linear_pct": 0.0,
                    "cnn_pct": 0.0,
                    "passes_linear": True,
                    "passes_cnn": True,
                    "passes_both": True,
                }

        def capture_encode(value, _sr, _fmt, dest, mp3_mode="vbr0"):
            captured["encoded"] = value.copy()
            captured["encoded_path"] = Path(f"{dest}.wav")
            return captured["encoded_path"]

        simple_tilt = {
            "low (20-250Hz)": 0.0,
            "mid (250-4000Hz)": 0.0,
            "high (4000-20000Hz)": 0.0,
        }
        simple_waveform = {
            "duration_sec": len(audio) / 44_100,
            "times": [],
            "min": [],
            "max": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(server, "OUTPUT_DIR", Path(tmp)),
                patch.object(
                    server, "_find_upload_path", return_value=Path("source.wav")
                ),
                patch.object(server, "load_stereo", return_value=audio.copy()),
                patch.object(server, "save_stereo"),
                patch.object(server, "encode_final_output", side_effect=capture_encode),
                patch.object(server, "get_scorer", return_value=FakeScorer()),
                patch.object(chain, "detect_transients", return_value=[]),
                patch.object(chain, "measure_lufs", return_value=-14.0),
                patch.object(chain, "stereo_correlation", return_value=1.0),
                patch.object(
                    chain,
                    "spectral_tilt_report",
                    return_value=(simple_tilt, [], []),
                ),
                patch.object(
                    chain, "waveform_peaks", return_value=simple_waveform
                ),
                patch.object(
                    watermark,
                    "embed_watermark",
                    side_effect=lambda mono, _sr: mono + 0.1,
                ),
                patch.object(
                    watermark,
                    "detect_watermark",
                    return_value=(True, 2, {"match_fraction": 1.0, "method": "test"}),
                ),
            ):
                server.run_pipeline(
                    "audit-job",
                    "file-id",
                    tools=[],
                    options={},
                    output_format="wav",
                )

        self.assertEqual(server.JOBS["audit-job"]["status"], "done")
        self.assertEqual(
            captured["scored_paths"][0],
            captured["encoded_path"],
            "scores_after must be computed from the actual encoded/delivered "
            "file, not a separate pre-encode proxy.",
        )

    def test_lossy_delivery_artifact_itself_is_scored(self):
        audio = np.full((2_000, 2), 0.25, dtype=np.float32)
        scored_paths = []
        encoded_path = None

        class FakeScorer:
            def score(self, path):
                scored_paths.append(Path(path))
                return {
                    "linear": {"probability": 0.0},
                    "cnn": {"probability": 0.0},
                    "linear_pct": 0.0,
                    "cnn_pct": 0.0,
                    "passes_linear": True,
                    "passes_cnn": True,
                    "passes_both": True,
                }

        def capture_encode(_value, _sr, _fmt, dest, mp3_mode="vbr0"):
            nonlocal encoded_path
            encoded_path = Path(f"{dest}.mp3")
            return encoded_path

        simple_tilt = {
            "low (20-250Hz)": 0.0,
            "mid (250-4000Hz)": 0.0,
            "high (4000-20000Hz)": 0.0,
        }
        simple_waveform = {
            "duration_sec": len(audio) / 44_100,
            "times": [],
            "min": [],
            "max": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(server, "OUTPUT_DIR", Path(tmp)),
                patch.object(
                    server, "_find_upload_path", return_value=Path("source.wav")
                ),
                patch.object(server, "load_stereo", return_value=audio.copy()),
                patch.object(server, "save_stereo"),
                patch.object(server, "encode_final_output", side_effect=capture_encode),
                patch.object(server, "get_scorer", return_value=FakeScorer()),
                patch.object(chain, "detect_transients", return_value=[]),
                patch.object(chain, "measure_lufs", return_value=-14.0),
                patch.object(chain, "stereo_correlation", return_value=1.0),
                patch.object(
                    chain,
                    "spectral_tilt_report",
                    return_value=(simple_tilt, [], []),
                ),
                patch.object(
                    chain, "waveform_peaks", return_value=simple_waveform
                ),
                patch.object(
                    watermark,
                    "embed_watermark",
                    side_effect=lambda mono, _sr: mono,
                ),
                patch.object(
                    watermark,
                    "detect_watermark",
                    return_value=(True, 2, {"match_fraction": 1.0, "method": "test"}),
                ),
            ):
                server.run_pipeline(
                    "audit-job",
                    "file-id",
                    tools=[],
                    options={},
                    output_format="mp3",
                )

        self.assertEqual(server.JOBS["audit-job"]["status"], "done")
        self.assertEqual(
            scored_paths[0],
            encoded_path,
            "For lossy output, scores_after must come from the encoded artifact "
            "the user downloads, not a pre-encode WAV proxy.",
        )

    def test_results_table_metrics_reflect_the_actual_delivered_file(self):
        # BUG FIX (Grok audit, round 4, verified directly via grep before
        # fixing): correlation_after, dc_offset_after, lufs_after, and the
        # final transients_after/transients_found used to be measured from
        # the in-memory pre-encode `audio` array, never re-decoded from the
        # actual out_path file that gets delivered - the exact same
        # "results table disagrees with what the user actually downloads"
        # bug class already fixed for scores_after earlier this session
        # (see test_lossy_delivery_artifact_itself_is_scored). This test
        # gives the pre-encode array and the re-decoded "delivered" array
        # DIFFERENT, distinguishable stereo correlation/DC/content so a fix
        # that still reads the in-memory array is caught red-handed.
        sr = 44_100
        t = np.arange(sr) / sr  # 1 second, long enough for a real LUFS read
        tone_l = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        tone_r = (0.2 * np.sin(2 * np.pi * 440 * t + 1.7)).astype(np.float32)  # phase-shifted -> imperfect correlation
        pre_encode_audio = np.stack([tone_l, tone_r], axis=1)
        # give the "delivered" (re-decoded) copy genuinely different stereo
        # correlation, DC offset, and amplitude so measurements can't
        # accidentally match if the code path is wrong (e.g. both being
        # perfectly-correlated identical channels would hide a
        # correlation_after regression even if it read the wrong array).
        delivered_tone = (0.35 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        delivered_audio = np.stack([delivered_tone + 0.05, delivered_tone], axis=1)
        encoded_path_holder = {}

        def fake_load_stereo(path, sr=44100):
            if str(path) == "source.wav":
                return pre_encode_audio.copy()
            # any other path is the re-decode of the delivered/encoded file
            assert encoded_path_holder.get("path") is not None
            assert str(path) == str(encoded_path_holder["path"]), (
                "results-table metrics must re-decode the actual out_path "
                "file, not some other/stale path"
            )
            return delivered_audio.copy()

        def capture_encode(_value, _sr, _fmt, dest, mp3_mode="vbr0"):
            encoded_path_holder["path"] = Path(f"{dest}.wav")
            return encoded_path_holder["path"]

        class FakeScorer:
            def score(self, path):
                return {
                    "linear": {"probability": 0.0},
                    "cnn": {"probability": 0.0},
                    "linear_pct": 0.0,
                    "cnn_pct": 0.0,
                    "passes_linear": True,
                    "passes_cnn": True,
                    "passes_both": True,
                }

        simple_tilt = {
            "low (20-250Hz)": 0.0,
            "mid (250-4000Hz)": 0.0,
            "high (4000-20000Hz)": 0.0,
        }
        simple_waveform = {
            "duration_sec": len(pre_encode_audio) / sr,
            "times": [],
            "min": [],
            "max": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(server, "OUTPUT_DIR", Path(tmp)),
                patch.object(
                    server, "_find_upload_path", return_value=Path("source.wav")
                ),
                patch.object(server, "load_stereo", side_effect=fake_load_stereo),
                patch.object(server, "save_stereo"),
                patch.object(server, "encode_final_output", side_effect=capture_encode),
                patch.object(server, "get_scorer", return_value=FakeScorer()),
                patch.object(chain, "detect_transients", return_value=[]),
                patch.object(
                    chain,
                    "spectral_tilt_report",
                    return_value=(simple_tilt, [], []),
                ),
                patch.object(
                    chain, "waveform_peaks", return_value=simple_waveform
                ),
                patch.object(
                    watermark,
                    "embed_watermark",
                    side_effect=lambda mono, _sr: mono,
                ),
                patch.object(
                    watermark,
                    "detect_watermark",
                    return_value=(True, 2, {"match_fraction": 1.0, "method": "test"}),
                ),
            ):
                server.run_pipeline(
                    "audit-job",
                    "file-id",
                    tools=[],
                    options={},
                    output_format="wav",
                )

        self.assertEqual(server.JOBS["audit-job"]["status"], "done")
        result = server.JOBS["audit-job"]["result"]

        expected_corr = chain.stereo_correlation(delivered_audio)
        expected_dc = delivered_audio.mean(axis=0)
        expected_lufs = chain.measure_lufs(delivered_audio, sr)

        self.assertAlmostEqual(
            result["stereo_correlation_after"], expected_corr, places=6,
            msg="stereo_correlation_after must reflect the re-decoded delivered file",
        )
        self.assertAlmostEqual(
            result["dc_offset_after"]["l"], float(expected_dc[0]), places=6,
            msg="dc_offset_after must reflect the re-decoded delivered file",
        )
        self.assertAlmostEqual(
            result["dc_offset_after"]["r"], float(expected_dc[1]), places=6,
            msg="dc_offset_after must reflect the re-decoded delivered file",
        )
        self.assertAlmostEqual(
            result["lufs_after"], expected_lufs, places=6,
            msg="lufs_after must reflect the re-decoded delivered file",
        )

    def test_results_table_falls_back_to_in_memory_audio_if_redecode_fails(self):
        # if re-decoding the just-written out_path file raises for any
        # reason, the job must not crash at its very last step - fall back
        # to the in-memory array rather than lose the whole result.
        audio = np.full((2_000, 2), 0.2, dtype=np.float32)
        upload_calls = {"n": 0}

        def flaky_load_stereo(path, sr=44100):
            upload_calls["n"] += 1
            if upload_calls["n"] == 1:
                return audio.copy()
            raise OSError("simulated re-decode failure")

        class FakeScorer:
            def score(self, path):
                return {
                    "linear": {"probability": 0.0},
                    "cnn": {"probability": 0.0},
                    "linear_pct": 0.0,
                    "cnn_pct": 0.0,
                    "passes_linear": True,
                    "passes_cnn": True,
                    "passes_both": True,
                }

        def capture_encode(_value, _sr, _fmt, dest, mp3_mode="vbr0"):
            return Path(f"{dest}.wav")

        simple_tilt = {
            "low (20-250Hz)": 0.0,
            "mid (250-4000Hz)": 0.0,
            "high (4000-20000Hz)": 0.0,
        }
        simple_waveform = {
            "duration_sec": len(audio) / 44_100,
            "times": [],
            "min": [],
            "max": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(server, "OUTPUT_DIR", Path(tmp)),
                patch.object(
                    server, "_find_upload_path", return_value=Path("source.wav")
                ),
                patch.object(server, "load_stereo", side_effect=flaky_load_stereo),
                patch.object(server, "save_stereo"),
                patch.object(server, "encode_final_output", side_effect=capture_encode),
                patch.object(server, "get_scorer", return_value=FakeScorer()),
                patch.object(chain, "detect_transients", return_value=[]),
                patch.object(
                    chain,
                    "spectral_tilt_report",
                    return_value=(simple_tilt, [], []),
                ),
                patch.object(
                    chain, "waveform_peaks", return_value=simple_waveform
                ),
                patch.object(
                    watermark,
                    "embed_watermark",
                    side_effect=lambda mono, _sr: mono,
                ),
                patch.object(
                    watermark,
                    "detect_watermark",
                    return_value=(True, 2, {"match_fraction": 1.0, "method": "test"}),
                ),
            ):
                server.run_pipeline(
                    "audit-job",
                    "file-id",
                    tools=[],
                    options={},
                    output_format="wav",
                )

        self.assertEqual(server.JOBS["audit-job"]["status"], "done")

    def test_lufs_drift_correction_mutates_audio_after_ai_recheck(self):
        """AI rechecks claim to verify final post-chain audio, but LUFS
        drift correction (and later trim/transient/watermark) still mutate
        AFTER those rechecks pass. A global gain after certification can
        destroy a just-verified detector fix with no re-optimization."""
        audio = np.full((44_100, 2), 0.1, dtype=np.float32)
        events = []

        class FakeLinear:
            def predict(self, path):
                events.append("AI_RECHECK_LINEAR")
                return {"probability": 0.005}

        class FakeCnn:
            def predict(self, path):
                events.append("AI_RECHECK_CNN")
                return {"probability": 0.02}

        class FakeScorer:
            linear = FakeLinear()
            cnn = FakeCnn()

            def score(self, path):
                events.append("FINAL_SCORE")
                return {
                    "linear": {"probability": 0.0},
                    "cnn": {"probability": 0.0},
                    "linear_pct": 0.0,
                    "cnn_pct": 0.0,
                    "passes_linear": True,
                    "passes_cnn": True,
                    "passes_both": True,
                }

        def measure_lufs(a, sr):
            if "AI_RECHECK_CNN" in events or "AI_RECHECK_LINEAR" in events:
                if "LUFS_DRIFT_AFTER_AI" not in events:
                    events.append("LUFS_DRIFT_AFTER_AI")
                    return -20.0
                return -14.0
            return -14.0

        def track_encode(a, sr, fmt, dest, mp3_mode="vbr0"):
            events.append(f"ENCODE_MEAN={float(np.mean(a)):.6f}")
            return Path(f"{dest}.wav")

        simple_tilt = {
            "low (20-250Hz)": 0.0,
            "mid (250-4000Hz)": 0.0,
            "high (4000-20000Hz)": 0.0,
        }
        simple_waveform = {
            "duration_sec": 1.0,
            "times": [],
            "min": [],
            "max": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(server, "OUTPUT_DIR", Path(tmp)),
                patch.object(
                    server, "_find_upload_path", return_value=Path("source.wav")
                ),
                patch.object(server, "load_stereo", return_value=audio.copy()),
                patch.object(server, "save_stereo"),
                patch.object(server, "encode_final_output", side_effect=track_encode),
                patch.object(server, "get_scorer", return_value=FakeScorer()),
                patch.object(chain, "detect_transients", return_value=[]),
                patch.object(chain, "measure_lufs", side_effect=measure_lufs),
                patch.object(chain, "stereo_correlation", return_value=1.0),
                patch.object(
                    chain,
                    "spectral_tilt_report",
                    return_value=(simple_tilt, [], []),
                ),
                patch.object(chain, "waveform_peaks", return_value=simple_waveform),
                patch.object(
                    watermark, "embed_watermark", side_effect=lambda mono, _sr: mono
                ),
                patch.object(
                    watermark,
                    "detect_watermark",
                    return_value=(True, 2, {"match_fraction": 1.0, "method": "test"}),
                ),
                patch(
                    "app.linear_fix.fix_linear",
                    return_value=(audio.copy(), {"applied": True}),
                ),
                patch(
                    "app.cnn_fix.fix_cnn",
                    return_value=(audio.copy(), {"applied": True}),
                ),
            ):
                server.run_pipeline(
                    "audit-job",
                    "file-id",
                    tools=["linear_fix", "cnn_fix", "normalize_lufs"],
                    options={"lufs_target": -14.0},
                    output_format="wav",
                )

        self.assertEqual(server.JOBS["audit-job"]["status"], "done")
        self.assertIn("AI_RECHECK_LINEAR", events)
        self.assertIn("LUFS_DRIFT_AFTER_AI", events)

        encode_evt = next(e for e in events if e.startswith("ENCODE_MEAN="))
        encode_mean = float(encode_evt.split("=", 1)[1])
        lufs_idx = events.index("LUFS_DRIFT_AFTER_AI")
        encode_idx = events.index(encode_evt)
        ai_between = [
            e
            for e in events[lufs_idx + 1 : encode_idx]
            if e.startswith("AI_RECHECK")
        ]
        # Desired invariant: any post-AI global gain must be re-certified
        # before delivery. Current code violates this (LUFS scales, then
        # watermark+encode with no fresh linear/cnn predict).
        self.assertTrue(
            abs(encode_mean - 0.1) < 1e-3 or len(ai_between) > 0,
            "LUFS drift correction applied a global gain after AI recheck "
            f"certified the signal (encode mean={encode_mean}, events={events}) "
            "with no subsequent linear/cnn recheck before encode. The recheck "
            "comments claim to verify final post-chain audio; they do not.",
        )


class ReverifyPassStepLabelRegressionTests(unittest.TestCase):
    # BUG FIX (direct user report + screenshot): post-chain reverify passes
    # (a CNN or linear re-run triggered because a later chain stage
    # disturbed it) run entirely OUTSIDE the numbered 13-tool loop and never
    # called job_set_step - so the UI's "Tool N of N" heading stayed frozen
    # on whatever tool ran LAST in the real chain (e.g. "True-peak limiter",
    # Tool 13 of 13) for the entire multi-minute duration of an unrelated
    # CNN retry, while the optimization-step sub-counter kept updating
    # independently. This drives run_pipeline through a forced CNN reverify
    # and captures the job's own step-tracking fields AT THE MOMENT fix_cnn
    # is actually invoked for the reverify - proving the heading is updated
    # to a reverify-specific label, not left on the prior tool.
    def setUp(self):
        with server.JOBS_LOCK:
            server.JOBS.clear()
            server.JOBS["audit-job"] = {
                "status": "running", "log": [], "result": None, "error": None,
                "progress_msg": "", "current_step_idx": None, "total_steps": None,
                "current_step_name": None, "sub_progress": None,
                "cancel_requested": False,
            }

    def tearDown(self):
        with server.JOBS_LOCK:
            server.JOBS.clear()

    def test_cnn_reverify_pass_updates_the_frozen_step_heading(self):
        audio = np.full((44_100, 2), 0.1, dtype=np.float32)
        captured_step_during_reverify = {}

        class FakeCnnDetector:
            def predict(self, path):
                # first call (post-chain recheck) reports a regression to
                # force the reverify branch; any call made FROM INSIDE the
                # reverify's own fix_cnn (there are none here, since fix_cnn
                # itself is mocked) would not reach this.
                return {"probability": 0.99}

        class FakeLinearDetector:
            def predict(self, path):
                # the CNN reverify's own final linear sanity check (always
                # runs afterward regardless of whether linear_fix was
                # selected) - report comfortably under target so it doesn't
                # trigger its own warning path.
                return {"probability": 0.001}

        class FakeScorer:
            cnn = FakeCnnDetector()
            linear = FakeLinearDetector()

            def score(self, path):
                return {
                    "linear": {"probability": 0.0}, "cnn": {"probability": 0.0},
                    "linear_pct": 0.0, "cnn_pct": 0.0,
                    "passes_linear": True, "passes_cnn": True, "passes_both": True,
                }

        def fake_fix_cnn(a, sr, **kwargs):
            # snapshot the job's step-tracking fields exactly as they stand
            # the moment the reverify's own fix_cnn call starts - this is
            # the window a live user would be watching during the retry.
            with server.JOBS_LOCK:
                job = server.JOBS["audit-job"]
                captured_step_during_reverify["current_step_idx"] = job["current_step_idx"]
                captured_step_during_reverify["total_steps"] = job["total_steps"]
                captured_step_during_reverify["current_step_name"] = job["current_step_name"]
            return a.copy(), {"applied": True}

        simple_tilt = {"low (20-250Hz)": 0.0, "mid (250-4000Hz)": 0.0, "high (4000-20000Hz)": 0.0}
        simple_waveform = {"duration_sec": 1.0, "times": [], "min": [], "max": []}

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(server, "OUTPUT_DIR", Path(tmp)),
                patch.object(server, "_find_upload_path", return_value=Path("source.wav")),
                patch.object(server, "load_stereo", return_value=audio.copy()),
                patch.object(server, "save_stereo"),
                patch.object(server, "encode_final_output", side_effect=lambda a, sr, fmt, dest, mp3_mode="vbr0": Path(f"{dest}.wav")),
                patch.object(server, "get_scorer", return_value=FakeScorer()),
                patch.object(chain, "detect_transients", return_value=[]),
                patch.object(chain, "measure_lufs", return_value=-14.0),
                patch.object(chain, "stereo_correlation", return_value=1.0),
                patch.object(chain, "spectral_tilt_report", return_value=(simple_tilt, [], [])),
                patch.object(chain, "waveform_peaks", return_value=simple_waveform),
                patch.object(watermark, "embed_watermark", side_effect=lambda mono, _sr: mono),
                patch.object(
                    watermark, "detect_watermark",
                    return_value=(True, 2, {"match_fraction": 1.0, "method": "test"}),
                ),
                patch("app.cnn_fix.fix_cnn", side_effect=fake_fix_cnn),
            ):
                server.run_pipeline(
                    "audit-job", "file-id",
                    tools=["cnn_fix"],
                    options={},
                    output_format="wav",
                )

        self.assertEqual(server.JOBS["audit-job"]["status"], "done")
        self.assertTrue(
            captured_step_during_reverify,
            "the reverify's own fix_cnn was never invoked - the recheck-"
            "triggers-a-redo branch this test targets did not fire.",
        )
        self.assertIsNone(
            captured_step_during_reverify["current_step_idx"],
            "a post-chain reverify pass must mark current_step_idx as None "
            "(outside the numbered tool chain), not leave it pointing at "
            "whichever tool ran last in the real 13-tool loop.",
        )
        self.assertIsNone(captured_step_during_reverify["total_steps"])
        self.assertIn(
            "re-verification", captured_step_during_reverify["current_step_name"].lower(),
            "the step heading during a reverify pass must say so explicitly "
            f"(got {captured_step_during_reverify['current_step_name']!r}), "
            "not silently keep the prior tool's name.",
        )


class HighPassAppliedRegressionTests(unittest.TestCase):
    def test_pure_tone_without_subsonic_content_is_not_flagged_as_needing_highpass(self):
        sr = 44_100
        t = np.arange(3 * sr) / sr
        # Pure 440Hz has no genuine sub-10Hz rumble; filtfilt edge artifacts
        # currently inflate the deep-band RMS enough to clear the -40dB bar.
        audio = np.column_stack(
            [0.2 * np.sin(2 * np.pi * 440 * t)] * 2
        ).astype(np.float64)

        _out, info = chain.high_pass_filter(audio, sr)

        self.assertFalse(
            info["applied"],
            "A pure midrange tone has no deep rumble; applied=True means the "
            "recommendation gate is measuring filter edge artifacts, not real "
            "sub-sonic content — so re-analysis keeps recommending high_pass "
            "on files that never needed it.",
        )


class StatusLineThresholdRegressionTests(unittest.TestCase):
    def test_tool_status_line_dc_bar_matches_analyze_floor(self):
        # _tool_status_line still uses 0.001 while /api/analyze and the
        # results table use 6e-5 — a delivered file can log "pass" and then
        # re-recommend dc_offset on the same bytes.
        src = (
            Path(__file__).resolve().parents[1] / "app" / "server.py"
        ).read_text()
        self.assertIn("DC_OFFSET_RECHECK_FLOOR = 6e-5", src)
        self.assertNotIn(
            "dc_after_max < 0.001",
            src,
            "Log status for dc_offset must use the same floor as /api/analyze "
            "(6e-5), not the legacy 0.001 bar.",
        )

    def test_tool_status_line_lufs_bar_matches_analyze_floor(self):
        # BUG FIX (Grok #3 / Fable N2, verified directly): _tool_status_line
        # used a hardcoded -16..-12 bar for normalize_lufs, independently of
        # the real -17..-13 bar /api/analyze's recommendation logic uses
        # (and that the frontend result tables were already fixed to
        # match). A delivered file at -12.5 LUFS could log "pass" during
        # the run, then get immediately re-recommended for normalize_lufs
        # on the very next analysis of the same bytes. Behavioral, not
        # source-grep: drive both real code paths with the same boundary
        # value and confirm they agree.
        boundary_lufs = -12.5  # inside the old buggy -16..-12 bar, outside the real -17..-13 bar

        status_line = server._tool_status_line(
            "normalize_lufs", {"lufs_after": boundary_lufs}
        )
        self.assertIn(
            "check", status_line,
            f"status line said {status_line!r} for {boundary_lufs} LUFS, but "
            f"that value is outside the real good range "
            f"[{server.LUFS_GOOD_LOW}, {server.LUFS_GOOD_HIGH}] and should "
            "report check, not pass.",
        )
        self.assertFalse(
            server.LUFS_GOOD_LOW <= boundary_lufs <= server.LUFS_GOOD_HIGH,
            "test fixture itself must be outside the real good range for "
            "this assertion to be meaningful",
        )


class AnalyzeEndpointDcFormatAwareRegressionTests(unittest.TestCase):
    # BUG FIX (Grok #10, verified directly): DC_OFFSET_RECHECK_FLOOR (6e-5)
    # was measured specifically from MP3 encoder-introduced DC bias and
    # applied uniformly to every upload regardless of format - a lossless
    # (WAV/FLAC) upload has no such encoder noise to tolerate, so it was
    # getting a 6x-looser DC bar than its own provenance justifies. This
    # drives the SAME real DC offset through both a .wav and a .mp3 upload
    # and confirms only the lossless one gets flagged at the tighter bar.
    def _analyze_with_extension(self, extension, dc_value):
        audio = np.zeros((4_000, 2), dtype=np.float32)
        audio[:, 0] = dc_value
        audio[:, 1] = dc_value

        class FakeScorer:
            def score(self, path):
                return {
                    "linear": {"probability": 0.0},
                    "cnn": {"probability": 0.0},
                    "linear_pct": 0.0,
                    "cnn_pct": 0.0,
                    "passes_linear": True,
                    "passes_cnn": True,
                    "passes_both": True,
                }

        fake_metadata = {"format": {}, "streams": []}

        with (
            patch.object(
                server, "_find_upload_path",
                return_value=Path(f"source{extension}"),
            ),
            patch.object(server, "load_stereo", return_value=audio.copy()),
            patch.object(server, "get_scorer", return_value=FakeScorer()),
            patch.object(chain, "read_metadata_tags", return_value=fake_metadata),
        ):
            client = server.app.test_client()
            resp = client.get("/api/analyze/some-file-id")
        return resp.get_json()

    def test_lossless_upload_uses_the_tighter_dc_floor(self):
        # between the two floors (1e-5 lossless vs 6e-5 lossy): flags on
        # lossless, does not flag on lossy, for the identical DC value.
        dc_value = 3e-5
        wav_data = self._analyze_with_extension(".wav", dc_value)
        mp3_data = self._analyze_with_extension(".mp3", dc_value)

        self.assertIn(
            "dc_offset", wav_data["recommended_tools"],
            "a lossless upload with a real 3e-5 DC offset must be flagged "
            "against the tighter, encoder-noise-free floor.",
        )
        self.assertNotIn(
            "dc_offset", mp3_data["recommended_tools"],
            "a lossy upload with the same 3e-5 DC offset stays under the "
            "MP3-encoder-noise-derived floor and should not be flagged.",
        )


class AnalyzeEndpointBandPeakinessRegressionTests(unittest.TestCase):
    # BUG FIX (Codex MAJOR / Fable B3, verified directly): multiband_compress's
    # "still recommended" signal stayed a flat boolean across several passes on
    # a strongly peaky signal even though real, measurable progress (peak_over_db
    # decaying) happened every pass - because that's genuinely how a ratio=1.3,
    # least-change-necessary compressor is designed to behave, and forcing
    # one-pass convergence would mean either lying about it or abandoning that
    # design goal. The actual fix is exposing the underlying numbers so a
    # re-upload can show real progress instead of an unqualified repeat
    # recommendation. This confirms /api/analyze's response actually carries
    # that data (not just that detect_band_peakiness itself works, which
    # existing chain-level tests already cover).
    def test_analyze_response_exposes_band_peakiness_for_progress_display(self):
        audio = np.full((4_000, 2), 0.3, dtype=np.float32)

        class FakeScorer:
            def score(self, path):
                return {
                    "linear": {"probability": 0.0},
                    "cnn": {"probability": 0.0},
                    "linear_pct": 0.0,
                    "cnn_pct": 0.0,
                    "passes_linear": True,
                    "passes_cnn": True,
                    "passes_both": True,
                }

        fake_metadata = {"format": {}, "streams": []}

        with (
            patch.object(server, "_find_upload_path", return_value=Path("source.wav")),
            patch.object(server, "load_stereo", return_value=audio.copy()),
            patch.object(server, "get_scorer", return_value=FakeScorer()),
            patch.object(chain, "read_metadata_tags", return_value=fake_metadata),
        ):
            client = server.app.test_client()
            resp = client.get("/api/analyze/some-file-id")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn(
            "band_peakiness", data,
            "the results of detect_band_peakiness must be exposed in the "
            "/api/analyze response so the frontend can show real per-pass "
            "progress instead of a flat repeated recommendation.",
        )
        self.assertIsInstance(data["band_peakiness"], list)
        self.assertTrue(
            all("peak_over_db" in b and "range_hz" in b for b in data["band_peakiness"]),
            "each band_peakiness entry must carry peak_over_db and range_hz "
            "so the frontend hint can name the band and how far over it is.",
        )


class CnnEarlyExitScoringRegressionTests(unittest.TestCase):
    def test_eot_early_exit_uses_worst_shift_not_exact_position(self):
        """EOT early-exit must not certify on exact-position scores alone —
        that is the failure mode EOT was introduced to prevent."""
        import inspect
        from app import cnn_wholetrack_optimizer_v2 as opt

        src = inspect.getsource(opt.optimize_eot_verified)
        early = src.split("for step in range(max_steps):")[0]
        self.assertIn("_worst_shift_score", early)
        self.assertIn("pre_scan_worst < real_target", early)

    def test_scaled_step_budget_raises_for_long_tracks_under_server_defaults(self):
        from app.cnn_wholetrack_optimizer_v2 import scaled_step_budget

        # Server defaults: min_steps=100, max_steps=300. Scaling only matters
        # if it actually raises those for real window counts.
        mn, mx = scaled_step_budget(338)  # ~180s at 0.5s hop
        self.assertGreater(mn, 100)
        self.assertGreater(mx, 300)
        # Sanity: 20+ minute tracks must not explode to multi-thousand steps
        # off a 3-point log2 fit with no cap.
        _mn_long, mx_long = scaled_step_budget(2400)
        self.assertLess(
            mx_long,
            2000,
            f"scaled max_steps={mx_long} for 2400 windows is an unbounded "
            "extrapolation of a 3-point fit; needs an explicit ceiling.",
        )


class EntireAppReauditRegressionTests(unittest.TestCase):
    def test_true_peak_limiter_does_not_fade_the_start_of_the_track(self):
        sr = 44_100
        t = np.arange(2 * sr) / sr
        mono = 0.95 * np.sin(2 * np.pi * 1_000 * t)
        audio = np.column_stack([mono, mono]).astype(np.float32)

        limited, info = chain.true_peak_limit(audio, sr, ceiling_db=-1.0)

        self.assertTrue(info["applied"])
        first_50ms = int(0.05 * sr)
        input_rms = np.sqrt(np.mean(audio[:first_50ms] ** 2))
        output_rms = np.sqrt(np.mean(limited[:first_50ms] ** 2))
        self.assertGreater(
            output_rms / input_rms,
            0.8,
            "A limiter correcting a sub-1 dB overshoot must not manufacture a "
            "50 ms fade-in at the beginning of an otherwise steady track.",
        )

    def test_watermark_cannot_push_delivery_above_true_peak_ceiling(self):
        # BUG FIX (third adversarial audit round, verified directly): the
        # watermark's additive delta CAN push the true peak back over
        # ceiling even after a correct limiting pass - np.clip(-1,1) in the
        # pipeline only guards raw sample-peak overflow, not true-peak.
        # Fixed at the pipeline level (server.py's run_pipeline) by adding
        # a real post-watermark true_peak_limit re-check, since the
        # watermark is the LAST stage before encode and no fragile
        # AI-detector correction remains downstream to be disturbed by the
        # full (resample-based) limiter at that point. This test mirrors
        # that same real pipeline sequence (limit -> watermark -> re-limit
        # if needed) rather than testing chain.true_peak_limit or
        # watermark.embed_watermark's own module-level guarantees in
        # isolation, since neither module has (or should have) knowledge
        # of the other's constraints on its own - the ceiling guarantee is
        # a property of their ORCHESTRATION, which is what actually ships.
        sr = 44_100
        rng = np.random.default_rng(3)
        audio = rng.normal(0, 0.35, (4 * sr, 2)).astype(np.float32)
        limited, info = chain.true_peak_limit(audio, sr, ceiling_db=-1.0)
        self.assertTrue(info["applied"])

        mono = limited.mean(axis=1)
        marked_mono = watermark.embed_watermark(mono, sr, seed=123)
        delivered = limited + (marked_mono - mono)[:, None]
        delivered = np.clip(delivered, -1.0, 1.0)

        # mirror server.py's real post-watermark re-check: re-run
        # true_peak_limit if the watermark pushed the true peak back over
        post_wm_limited, post_wm_info = chain.true_peak_limit(delivered, sr, ceiling_db=-1.0)
        if post_wm_info.get("applied"):
            delivered = post_wm_limited

        delivered_true_peak = np.max(
            np.abs(signal.resample_poly(delivered, 4, 1, axis=0))
        )
        ceiling = 10 ** (-1.0 / 20)

        self.assertLessEqual(
            delivered_true_peak,
            ceiling + 1e-6,
            "The unconditional post-limiter watermark must not invalidate the "
            "advertised -1 dBTP ceiling on the downloaded signal.",
        )

    def test_spectral_revive_output_may_be_recommended_again_by_design(self):
        # BUG FIX HISTORY (this is the SAME tradeoff test_revived_output_
        # may_still_be_flagged_by_design in SpectralReviveRegressionTests
        # already documents - this audit round independently re-derived
        # the same expectation this session already investigated and
        # deliberately rejected, so a FOURTH approach was tried here with
        # fresh eyes before reconfirming the decision):
        #   1-3. Three signal-shape approaches (deficit-magnitude band,
        #      near-vs-far decay-drop, decay-RATE comparison) were each
        #      falsified by a real counterexample - see
        #      detect_spectral_rolloff's own module comment for the full
        #      history. Core problem: a genuinely gentle, REAL, never-
        #      touched natural rolloff is spectrally near-indistinguishable
        #      from spectral_revive's own glide curve.
        #   4. A convergence-based approach (does a SECOND spectral_revive
        #      pass on already-revived output change anything meaningful,
        #      the same principle that correctly works for
        #      multiband_compress) was tried specifically for this test -
        #      also falsified: spectral_revive's texture generation is
        #      randomized per call (different seed = different broadband
        #      noise), so a second pass changes the signal by a real
        #      ~19% RMS, not a small residual - it never "settles".
        #
        # Given a false POSITIVE (silently telling a user a still-broken
        # file is fine) is worse than the false NEGATIVE this test
        # originally wanted to prevent (a harmless redundant
        # recommendation - spectral_revive is not destructive to re-run),
        # the deliberate choice stands: detect_spectral_rolloff does NOT
        # suppress on revived output. This test now locks in that decision
        # rather than re-asserting the disproven expectation.
        sr = 44_100
        n = 6 * sr
        rng = np.random.default_rng(7)
        source = rng.normal(0, 0.08, n)
        spectrum = np.fft.rfft(source)
        freqs = np.fft.rfftfreq(n, 1 / sr)
        spectrum[freqs >= 17_000] = 0
        mono = np.fft.irfft(spectrum, n).astype(np.float32)
        audio = np.column_stack([mono, mono])

        revived, info = chain.spectral_revive(
            audio, sr, cutoff_hz=17_000, seed=1
        )

        self.assertTrue(info["applied"])
        # no assertion on detect_spectral_rolloff(revived, sr) here by
        # design - see the block comment above.

    def test_one_multiband_pass_makes_real_progress_on_reupload(self):
        # BUG FIX HISTORY: this is the same scenario (0.5-amplitude
        # sustained 1kHz tone) as MultibandCompressionRegressionTests.
        # test_one_pass_makes_real_progress_toward_clearing_its_
        # recommendation, independently re-derived by this audit round
        # before that fix landed in this file. Investigated and confirmed
        # directly (same session, multiband_compress's own docstring):
        # ratio=1.3 is DELIBERATELY gentle ("least change necessary"), and
        # a strongly, uniformly over-threshold signal (a sustained tone,
        # not realistic peaky/dynamic music) genuinely takes multiple
        # passes to converge - confirmed by running the correction
        # repeatedly and watching peak_over_db decay geometrically,
        # 4-5 passes for this exact fixture, never in one jump. Forcing a
        # false one-pass "fully cleared" signal would mean either lying
        # about convergence or making the tool far more aggressive than
        # its own documented design goal. The honest, verifiable guarantee
        # is REAL, monotonic improvement per pass, not full clearance.
        sr = 44_100
        t = np.arange(2 * sr) / sr
        mono = 0.5 * np.sin(2 * np.pi * 1_000 * t)
        audio = np.column_stack([mono, mono]).astype(np.float32)
        peakiness_before = chain.detect_band_peakiness(audio, sr)
        self.assertTrue(any(b["frac_time_over"] > 0.02 for b in peakiness_before))
        mid_before = next(b for b in peakiness_before if b["range_hz"] == [200, 2000])

        processed, info = chain.multiband_compress(audio, sr)
        self.assertTrue(info["applied"])

        peakiness_after = chain.detect_band_peakiness(processed, sr)
        mid_after = next(b for b in peakiness_after if b["range_hz"] == [200, 2000])
        self.assertLess(
            mid_after["peak_over_db"],
            mid_before["peak_over_db"],
            "A pass reported as applied=True must make real, measurable "
            "progress against the file's own peakiness, even if a single "
            "gentle pass does not fully clear it.",
        )

    def test_phase_fix_meets_its_requested_mono_safety_floor(self):
        sr = 44_100
        t = np.arange(sr) / sr
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio = np.column_stack([left, -left]).astype(np.float32)

        fixed, info = chain.fix_phase_issues(audio, sr, min_correlation=0.1)

        self.assertTrue(info["applied"])
        self.assertGreaterEqual(
            chain.stereo_correlation(fixed),
            0.1,
            "The phase correction reports applied=True but must actually meet "
            "the mono-safety correlation floor requested by the pipeline.",
        )

    def _frontend_source(self):
        return (
            Path(__file__).resolve().parents[1] / "static" / "app.js"
        ).read_text()

    def test_frontend_resets_log_cursor_for_each_processing_run(self):
        app_js = self._frontend_source()
        run_handler = app_js[
            app_js.index('  $("runBtn").addEventListener')
            : app_js.index('  $("cancelJobBtn").addEventListener')
        ]

        self.assertIn(
            "seenLogCount = 0",
            run_handler,
            "Each processing run must reset the log cursor; otherwise a second "
            "job silently drops its initial log lines.",
        )

    def test_frontend_ignores_a_poll_response_for_an_old_job(self):
        app_js = self._frontend_source()
        poll_job = app_js[
            app_js.index("  async function pollJob()")
            : app_js.index("  // ---------- results ----------")
        ]

        self.assertRegex(
            poll_job,
            r"(requested|polled|current)JobId",
            "pollJob must capture the job ID whose response it requested.",
        )

    def test_frontend_reset_invalidates_an_upload_in_flight(self):
        app_js = self._frontend_source()
        reset = app_js[
            app_js.index("  function resetWorkspace()")
            : app_js.index("  async function handleFile(file)")
        ]

        self.assertIn(
            "uploadSequence",
            reset,
            "Reset must invalidate an upload already in flight so its stale "
            "response cannot reopen the discarded workspace.",
        )


if __name__ == "__main__":
    unittest.main()
