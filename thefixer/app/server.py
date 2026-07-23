"""
The Fixer - AI-music mastering & detector-evasion verification tool.

Flask backend: upload a file, get an AI-detector score + spectral report,
choose which mastering-chain tools to run (in a fixed sane order), get back
a processed file plus a full before/after report (scores, SNR, LUFS, spectral
tilt), and an A/B stream endpoint for real-time comparison in the browser.
"""
import json
import os
import re
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify, send_from_directory, Response

from . import chain
from .detector import Scorer


def _json_safe(value):
    """Recursively replace NaN/+-Infinity with None. Silent or near-silent
    audio can legitimately produce these (e.g. LUFS on true silence is
    mathematically -inf), but Python's json.dumps writes them as the bare
    tokens NaN/-Infinity/Infinity, which are NOT valid JSON and every
    browser's JSON.parse rejects outright - the analyze/job endpoints would
    otherwise return a response the frontend can't parse at all, silently
    breaking the whole page rather than just showing an odd number."""
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def safe_jsonify(payload):
    return jsonify(_json_safe(payload))

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=None)

_scorer = None
_scorer_lock = threading.Lock()

# in-memory job store: job_id -> {status, log, progress, result, error}
JOBS = {}
JOBS_LOCK = threading.Lock()


def get_scorer():
    global _scorer
    with _scorer_lock:
        if _scorer is None:
            _scorer = Scorer()
        return _scorer


def load_stereo(path, sr=44100):
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def save_stereo(path, audio, sr=44100):
    audio_clipped = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio_clipped, sr, subtype="PCM_16")


OUTPUT_FORMAT_EXTENSIONS = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac"}


def resolve_output_format(requested_format, original_upload_path):
    """"same" means match the original upload's container; anything else is
    taken literally. Falls back to wav if the source extension isn't one of
    the formats this app knows how to encode to."""
    if requested_format != "same":
        return requested_format
    ext = Path(original_upload_path).suffix.lower().lstrip(".")
    if ext in ("mp3",):
        return "mp3"
    if ext in ("flac",):
        return "flac"
    return "wav"


def encode_final_output(audio, sr, out_format, dest_path_no_ext, mp3_mode="vbr0"):
    """Write the final processed audio to disk in the requested format, with
    ALL container/ID3 metadata stripped regardless of format (WAV via
    soundfile carries none by default; MP3/FLAC are explicitly stripped via
    ffmpeg's -map_metadata -1 on top of that, so no encoder ever silently
    reintroduces a title/artist/comment field). Returns the actual path
    written (with the correct extension for the chosen format).

    mp3_mode selects between two different meanings of "highest quality":
    - "vbr0" (default): libmp3lame -q:a 0, LAME's highest VBR quality tier
      (~245kbps average). Considered transparent/near-lossless by most
      listeners, and doesn't waste bits on simple passages.
    - "cbr320": a flat 320kbps on every frame regardless of content - what
      most people mean literally by "highest bitrate MP3."
    """
    ext = OUTPUT_FORMAT_EXTENSIONS.get(out_format, ".wav")
    final_path = Path(f"{dest_path_no_ext}{ext}")

    if out_format == "wav":
        save_stereo(final_path, audio, sr)
        return final_path

    # MP3/FLAC: write a temporary WAV first (soundfile has no MP3/FLAC
    # writer of its own), then let ffmpeg do the real encode + explicit
    # metadata strip in one pass.
    tmp_wav = Path(f"{dest_path_no_ext}_tmp.wav")
    save_stereo(tmp_wav, audio, sr)
    try:
        if out_format == "mp3":
            if mp3_mode == "cbr320":
                mp3_args = ["-b:a", "320k"]
            else:
                mp3_args = ["-q:a", "0"]
            cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(tmp_wav),
                   "-map_metadata", "-1", "-codec:a", "libmp3lame", *mp3_args,
                   str(final_path)]
        elif out_format == "flac":
            cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(tmp_wav),
                   "-map_metadata", "-1", "-codec:a", "flac",
                   str(final_path)]
        else:
            raise ValueError(f"unknown output format: {out_format}")
        subprocess.run(cmd, check=True)
    finally:
        if tmp_wav.exists():
            tmp_wav.unlink()
    return final_path


def job_log(job_id, msg):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["log"].append({"t": time.time(), "msg": msg})
        job["progress_msg"] = msg
    print(f"[{job_id[:8]}] {msg}", flush=True)


def job_set_step(job_id, step_idx, total_steps, step_name):
    """Track which step in the tool chain is currently running, so the UI
    can show real step-aware progress ('step 3 of 6: CNN model fix') instead
    of a single undifferentiated percentage."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["current_step_idx"] = step_idx
        job["total_steps"] = total_steps
        job["current_step_name"] = step_name
        job["sub_progress"] = None


def job_set_sub_progress(job_id, current, total, extra=None):
    """Track fine-grained progress WITHIN one long-running step (the linear
    and CNN gradient optimizers, whose internal loops can run for several
    minutes with the step-level progress bar otherwise frozen the whole
    time). extra carries optional live detail - e.g. the current surrogate
    score, or which retry attempt is in progress - so the UI can show more
    than a bare step counter during a real slowdown."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        payload = {"current": current, "total": total}
        if extra:
            payload.update(extra)
        job["sub_progress"] = payload


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400

    file_id = uuid.uuid4().hex[:12]
    ext = Path(f.filename).suffix or ".wav"
    saved_path = UPLOAD_DIR / f"{file_id}{ext}"
    f.save(saved_path)

    try:
        audio = load_stereo(saved_path)
    except subprocess.CalledProcessError:
        return jsonify({"error": "could not decode audio file"}), 400

    duration_sec = len(audio) / 44100
    return jsonify({
        "file_id": file_id,
        "filename": f.filename,
        "duration_sec": round(duration_sec, 2),
        "samples": len(audio),
    })


_FILE_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _find_upload_path(file_id):
    # file_id must be exactly the 12-char hex ID this app itself generates
    # (uuid.uuid4().hex[:12] at upload time) - without this check, a glob
    # metacharacter like "*" as the literal file_id would match ANY file in
    # UPLOAD_DIR, letting a caller fetch/analyze an arbitrary previously-
    # uploaded file without knowing its real ID.
    if not _FILE_ID_RE.match(file_id):
        return None
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        return None
    return matches[0]


@app.route("/api/analyze/<file_id>", methods=["GET"])
def analyze(file_id):
    path = _find_upload_path(file_id)
    if path is None:
        return jsonify({"error": "unknown file_id"}), 404

    scorer = get_scorer()
    scores = scorer.score(str(path))

    audio = load_stereo(path)
    lufs = chain.measure_lufs(audio, 44100)
    correlation = chain.stereo_correlation(audio)
    dc = audio.mean(axis=0)
    transients = chain.detect_transients(audio, 44100)
    tilt_report, freqs, psd_db = chain.spectral_tilt_report(audio, 44100)

    _, silence_info = chain.trim_silence(audio, 44100)

    metadata = chain.read_metadata_tags(path)
    # flag anything that looks like it names a generation platform, tool, or
    # otherwise identifies provenance beyond what the user themselves set -
    # AI-generation platforms commonly embed this in comment/encoder fields
    PROVENANCE_KEYWORDS = ("suno", "udio", "made with", "generated", "ai-generated",
                           "riffusion", "mubert", "soundraw", "aiva", "boomy")
    all_tag_values = list(metadata["format"].items()) + [
        item for s in metadata["streams"] for item in s["tags"].items()
    ]
    provenance_hits = {
        k: v for k, v in all_tag_values
        if isinstance(v, str) and any(kw in v.lower() for kw in PROVENANCE_KEYWORDS)
    }
    has_embedded_images = any(s["is_attached_image"] for s in metadata["streams"])
    has_rolloff, rolloff_cutoff_hz, rolloff_deficit_db = chain.detect_spectral_rolloff(audio, 44100)

    recommendations = []
    if metadata["format"] or any(s["tags"] for s in metadata["streams"]) or has_embedded_images:
        recommendations.append("strip_metadata")
    if has_rolloff:
        recommendations.append("spectral_revive")
    if scores["linear"]["probability"] >= 0.01:
        recommendations.append("linear_fix")
    if scores["cnn"]["probability"] >= 0.5:
        recommendations.append("cnn_fix")
    if abs(dc[0]) > 1e-5 or abs(dc[1]) > 1e-5:
        recommendations.append("dc_offset")
    if silence_info.get("lead_ms", 0) > 20 or silence_info.get("trail_ms", 0) > 20:
        recommendations.append("trim_silence")
    if transients:
        recommendations.append("fix_transients")
    if correlation < 0.1:
        recommendations.append("fix_phase")
    if not (-15 <= lufs <= -13) and not (-17 <= lufs <= -15):
        recommendations.append("normalize_lufs")
    recommendations.append("high_pass")
    recommendations.append("multiband_compress")
    recommendations.append("true_peak_limit")

    return safe_jsonify({
        "file_id": file_id,
        "scores": scores,
        "lufs": lufs,
        "stereo_correlation": correlation,
        "dc_offset": {"l": float(dc[0]), "r": float(dc[1])},
        "silence": silence_info,
        "transients": transients,
        "spectral_tilt": tilt_report,
        "spectrum": {"freqs": freqs, "psd_db": psd_db},
        "metadata": metadata,
        "provenance_tags_found": provenance_hits,
        "has_embedded_images": has_embedded_images,
        "spectral_rolloff": {"detected": has_rolloff, "cutoff_hz": rolloff_cutoff_hz, "deficit_db": round(rolloff_deficit_db, 1)},
        "recommended_tools": recommendations,
        "passes_both": scores["passes_both"],
    })


# AI-detector fixes are extremely precisely-tuned adversarial corrections -
# they MUST run last (except the limiter, see below). Any gain/EQ/dynamics
# change applied after them (LUFS normalization, compression, phase
# adjustment) can perturb their exact spectral signature enough to undo the
# fix entirely, even though the fix passed real-model verification right
# after it ran. Confirmed directly: running linear_fix mid-chain scored <1%
# immediately after, but the SAME delta scored 16% once normalize_lufs/
# multiband_compress ran afterward and altered the signal the fix had been
# tuned against.
#
# true_peak_limit is the one exception: it's a safety net against clipping,
# not a musical/spectral-shaping step, and it must see the FINAL signal -
# including whatever small amount of energy the AI-detector fixes add - or
# it can't actually guarantee the delivered file stays under the true-peak
# ceiling. Both AI fixes already have their own last-resort peak clamp, but
# that's a blunt sample-peak scale-down, not an inter-sample-peak-aware
# limiter, so true_peak_limit still belongs after them as the real final stage.
TOOL_ORDER = [
    "strip_metadata", "trim_silence", "dc_offset", "fix_transients",
    "spectral_revive", "high_pass",
    "fix_phase", "normalize_lufs", "multiband_compress",
    "linear_fix", "cnn_fix",
    "true_peak_limit",
]

TOOL_LABELS = {
    "strip_metadata": "Strip metadata & embedded images",
    "trim_silence": "Trim leading/trailing silence",
    "dc_offset": "DC offset correction",
    "fix_transients": "Surgical transient/pop limiting",
    "spectral_revive": "High-frequency spectral fill-in (17kHz+)",
    "high_pass": "High-pass filter (rumble removal)",
    "linear_fix": "AI-detector fix: linear model",
    "cnn_fix": "AI-detector fix: CNN model",
    "fix_phase": "Stereo phase/correlation correction",
    "normalize_lufs": "LUFS loudness normalization",
    "multiband_compress": "Multiband tonal-balance compression",
    "true_peak_limit": "True-peak limiter",
}


def run_pipeline(job_id, file_id, tools, options, output_name=None, output_format="same", mp3_mode="vbr0"):
    try:
        path = _find_upload_path(file_id)
        if path is None:
            raise RuntimeError("unknown file_id")

        job_log(job_id, f"loading {path.name}")
        audio = load_stereo(path)
        original_audio = audio.copy()
        sr = 44100

        scorer = get_scorer()
        steps = []

        ordered_tools = [t for t in TOOL_ORDER if t in tools]
        lead_samples_trimmed = 0
        total_steps = len(ordered_tools)

        for step_idx, tool in enumerate(ordered_tools):
            job_set_step(job_id, step_idx, total_steps, TOOL_LABELS.get(tool, tool))
            job_log(job_id, f"running: {TOOL_LABELS.get(tool, tool)}")
            t0 = time.time()

            if tool == "strip_metadata":
                # nothing to do to the audio array itself here - every
                # output this pipeline writes is re-encoded from raw PCM
                # (WAV via soundfile carries no tags at all; MP3/FLAC are
                # re-encoded from a tagless temp WAV with an explicit
                # -map_metadata -1 pass in encode_final_output), so metadata
                # is dropped by construction regardless of this step. This
                # step exists to make that fact VISIBLE - report exactly
                # what tags/images were found on the original upload and
                # will not appear in the delivered file.
                metadata = chain.read_metadata_tags(path)
                all_tags = dict(metadata["format"])
                for s in metadata["streams"]:
                    all_tags.update(s["tags"])
                has_images = any(s["is_attached_image"] for s in metadata["streams"])
                info = {
                    "applied": bool(all_tags or has_images),
                    "tags_found": all_tags,
                    "has_embedded_images": has_images,
                }
            elif tool == "trim_silence":
                audio, info = chain.trim_silence(audio, sr)
                # track how much was cut from the FRONT specifically, so any
                # later before/after sample-domain comparison (SNR, delta)
                # can re-align on the same underlying content instead of
                # comparing silence against the first real transient
                lead_samples_trimmed = info.get("lead_samples", 0)
            elif tool == "dc_offset":
                audio, info = chain.fix_dc_offset(audio, sr)
            elif tool == "fix_transients":
                transients = chain.detect_transients(audio, sr)
                info = {"applied": len(transients) > 0, "count": len(transients), "details": []}
                for t in transients:
                    audio, tinfo = chain.fix_transient(audio, sr, t["time_sec"],
                                                         target_peak=options.get("transient_target_peak"))
                    info["details"].append(tinfo)
            elif tool == "spectral_revive":
                audio, info = chain.spectral_revive(audio, sr, cutoff_hz=options.get("spectral_revive_cutoff_hz"))
            elif tool == "high_pass":
                audio, info = chain.high_pass_filter(audio, sr, cutoff_hz=options.get("high_pass_hz", 30))
            elif tool == "linear_fix":
                from .linear_fix import fix_linear
                audio, info = fix_linear(
                    audio, sr, target=options.get("linear_target", 0.01),
                    progress_cb=lambda m: job_log(job_id, m),
                    step_progress_cb=lambda s, mx, score, att, mxatt: job_set_sub_progress(
                        job_id, s, mx, extra={"score_pct": round(score * 100, 4), "attempt": att, "max_attempts": mxatt}),
                )
            elif tool == "cnn_fix":
                from .cnn_fix import fix_cnn
                cnn_max_steps = options.get("cnn_max_steps", 300)
                audio, info = fix_cnn(audio, sr,
                                       max_steps=cnn_max_steps,
                                       min_steps=options.get("cnn_min_steps", 100),
                                       hop_sec=options.get("cnn_hop_sec", 2.5),
                                       progress_cb=lambda m: job_log(job_id, m),
                                       step_progress_cb=lambda s, mx: job_set_sub_progress(job_id, s, mx))
            elif tool == "fix_phase":
                audio, info = chain.fix_phase_issues(audio, sr)
            elif tool == "normalize_lufs":
                audio, info = chain.normalize_lufs(audio, sr, target_lufs=options.get("lufs_target", -14.0))
            elif tool == "multiband_compress":
                audio, info = chain.multiband_compress(audio, sr)
            elif tool == "true_peak_limit":
                audio, info = chain.true_peak_limit(audio, sr, ceiling_db=options.get("ceiling_db", -1.0))
            else:
                continue

            info["tool"] = tool
            info["label"] = TOOL_LABELS.get(tool, tool)
            info["elapsed_sec"] = round(time.time() - t0, 2)
            steps.append(info)
            job_log(job_id, f"  done ({info['elapsed_sec']}s)")

        # Final re-verification pass: cnn_fix's own correction can disturb
        # linear_fix's precise spectral tuning even when linear_fix ran first
        # and passed its own verification right after it ran (confirmed
        # directly: a verified 1.56% became 9.65% once cnn_fix ran on top of
        # it). Re-score both models on the actual post-chain audio and, if
        # the linear model is still above target, re-run linear_fix ONE more
        # time (cheap relative to cnn_fix) rather than silently shipping a
        # result that's worse than what linear_fix itself already achieved.
        if "linear_fix" in tools and "cnn_fix" in tools:
            job_log(job_id, "re-verifying linear model after full chain (cnn_fix can disturb it)")
            recheck_path = OUTPUT_DIR / f"_recheck_{uuid.uuid4().hex[:8]}.wav"
            try:
                save_stereo(recheck_path, audio, sr)
                recheck_score = scorer.linear.predict(str(recheck_path))["probability"]
                job_log(job_id, f"  post-chain linear score: {recheck_score * 100:.3f}%")
                if recheck_score >= 0.01:
                    job_log(job_id, "  above target - re-running linear_fix once more on the final signal")
                    from .linear_fix import fix_linear
                    t0 = time.time()
                    audio, reverify_info = fix_linear(
                        audio, sr, target=options.get("linear_target", 0.01),
                        progress_cb=lambda m: job_log(job_id, m),
                        step_progress_cb=lambda s, mx, score, att, mxatt: job_set_sub_progress(
                            job_id, s, mx, extra={"score_pct": round(score * 100, 4), "attempt": att, "max_attempts": mxatt}),
                    )
                    reverify_info["tool"] = "linear_fix_reverify"
                    reverify_info["label"] = "AI-detector fix: linear model (re-verification pass)"
                    reverify_info["elapsed_sec"] = round(time.time() - t0, 2)
                    reverify_info["triggered_by"] = f"post-chain recheck showed {recheck_score * 100:.3f}%"
                    steps.append(reverify_info)
                    job_log(job_id, f"  done ({reverify_info['elapsed_sec']}s)")

                    # the re-verification pass adds a fresh linear correction
                    # AFTER true_peak_limit already ran in the main loop above -
                    # if the limiter was selected, it must run again here so it
                    # remains the genuine last stage and the delivered file
                    # actually stays under its ceiling, rather than only having
                    # been checked against audio that predates this correction.
                    if "true_peak_limit" in tools:
                        job_log(job_id, "re-running true-peak limiter after re-verification pass")
                        t0 = time.time()
                        audio, limiter_info = chain.true_peak_limit(
                            audio, sr, ceiling_db=options.get("ceiling_db", -1.0))
                        limiter_info["tool"] = "true_peak_limit_reverify"
                        limiter_info["label"] = "True-peak limiter (post-reverification safety pass)"
                        limiter_info["elapsed_sec"] = round(time.time() - t0, 2)
                        steps.append(limiter_info)
                        job_log(job_id, f"  done ({limiter_info['elapsed_sec']}s)")
            finally:
                if recheck_path.exists():
                    recheck_path.unlink()

        # Final LUFS re-verification pass: normalize_lufs runs mid-chain, but
        # every stage after it (multiband compression, both AI-detector
        # fixes, the limiter) can shift overall loudness without anything
        # checking whether the FINAL delivered file still matches the target
        # - unlike linear_fix, which already gets a post-chain recheck. If
        # normalize_lufs was selected, measure the true final LUFS and, if it
        # has drifted meaningfully from target, apply one corrective gain
        # pass right before delivery (not a full re-run of normalize_lufs,
        # which would re-measure and target the same way - just a direct
        # final trim to the actual requested target).
        if "normalize_lufs" in tools:
            target_lufs = options.get("lufs_target", -14.0)
            final_lufs = chain.measure_lufs(audio, sr)
            if np.isfinite(final_lufs) and abs(final_lufs - target_lufs) > 0.5:
                job_log(job_id, f"post-chain LUFS check: {final_lufs:.1f} vs target {target_lufs:.1f} "
                                 f"- correcting drift introduced by later processing stages")
                t0 = time.time()
                gain_db = target_lufs - final_lufs
                gain_linear = 10 ** (gain_db / 20)
                audio = audio * gain_linear
                peak = np.abs(audio).max()
                if peak > 0.999:
                    audio = audio * (0.999 / peak)
                lufs_reverify_info = {
                    "tool": "normalize_lufs_reverify",
                    "label": "LUFS loudness normalization (post-chain drift correction)",
                    "applied": True,
                    "lufs_before": float(final_lufs),
                    "lufs_target": target_lufs,
                    "lufs_after": float(chain.measure_lufs(audio, sr)),
                    "elapsed_sec": round(time.time() - t0, 2),
                }
                steps.append(lufs_reverify_info)
                job_log(job_id, f"  corrected to {lufs_reverify_info['lufs_after']:.1f} LUFS "
                                 f"({lufs_reverify_info['elapsed_sec']}s)")

                # a late gain change here can push a peak back over the
                # limiter's ceiling - re-run it once more if it was selected,
                # for the same reason the linear-fix re-verification pass
                # re-runs the limiter above.
                if "true_peak_limit" in tools:
                    job_log(job_id, "re-running true-peak limiter after LUFS drift correction")
                    t0 = time.time()
                    audio, limiter_info2 = chain.true_peak_limit(
                        audio, sr, ceiling_db=options.get("ceiling_db", -1.0))
                    limiter_info2["tool"] = "true_peak_limit_reverify_lufs"
                    limiter_info2["label"] = "True-peak limiter (post-LUFS-correction safety pass)"
                    limiter_info2["elapsed_sec"] = round(time.time() - t0, 2)
                    steps.append(limiter_info2)
                    job_log(job_id, f"  done ({limiter_info2['elapsed_sec']}s)")

        out_id = uuid.uuid4().hex[:12]
        # always keep a WAV copy for detector re-scoring regardless of the
        # chosen delivery format, since the scorer needs a file ffmpeg/
        # librosa can decode and re-scoring must reflect the exact audio
        # that was actually processed
        scoring_wav_path = OUTPUT_DIR / f"{out_id}_score.wav"
        save_stereo(scoring_wav_path, audio, sr)

        resolved_format = resolve_output_format(output_format, path)
        job_log(job_id, f"saving output file (format: {resolved_format})")
        out_path = encode_final_output(audio, sr, resolved_format, OUTPUT_DIR / out_id, mp3_mode=mp3_mode)

        orig_path = OUTPUT_DIR / f"{out_id}_orig.wav"
        save_stereo(orig_path, original_audio, sr)

        job_log(job_id, "re-scoring with AI detectors")
        scores_after = scorer.score(str(scoring_wav_path))
        scores_before = scorer.score(str(path))
        if not scores_after["passes_both"]:
            failing = []
            if not scores_after["passes_linear"]:
                failing.append(f"linear={scores_after['linear_pct']:.2f}%")
            if not scores_after["passes_cnn"]:
                failing.append(f"cnn={scores_after['cnn_pct']:.1f}%")
            job_log(job_id, f"WARNING: final file still flagged by at least one model ({', '.join(failing)})")

        # align on the same underlying audio content before comparing samples:
        # if silence was trimmed from the front, "audio" starts later in the
        # timeline than "original_audio" by lead_samples_trimmed samples -
        # comparing raw [:n] slices without this offset would diff silence
        # against the first real transient and produce a meaningless SNR.
        aligned_original = original_audio[lead_samples_trimmed:]
        n = min(len(aligned_original), len(audio))
        delta = audio[:n, 0] - aligned_original[:n, 0]
        orig_rms = np.sqrt(np.mean(aligned_original[:n, 0] ** 2))
        delta_rms = np.sqrt(np.mean(delta ** 2))
        overall_snr = float(20 * np.log10(orig_rms / (delta_rms + 1e-12))) if delta_rms > 0 else None

        tilt_before, freqs_b, psd_b = chain.spectral_tilt_report(original_audio, sr)
        tilt_after, freqs_a, psd_a = chain.spectral_tilt_report(audio, sr)

        # the scoring WAV is redundant once re-scoring is done UNLESS the
        # delivered format IS wav (in which case out_path == scoring_wav_path
        # already covers it) - only remove the extra copy when they differ
        if scoring_wav_path != out_path and scoring_wav_path.exists():
            scoring_wav_path.unlink()

        default_output_name = f"{Path(path.name).stem}_fixed{out_path.suffix}"
        # the extension in output_name must always match what was ACTUALLY
        # encoded, regardless of what the user typed (e.g. leaving the
        # auto-filled "..._fixed.wav" in the field while choosing MP3 as the
        # output format would otherwise download real MP3 bytes under a
        # .wav name) - swap on whatever suffix is present, add one if absent.
        final_output_name = output_name or default_output_name
        if Path(final_output_name).suffix.lower() != out_path.suffix.lower():
            final_output_name = f"{Path(final_output_name).stem}{out_path.suffix}"

        result = {
            "out_id": out_id,
            "output_name": final_output_name,
            "output_format": resolved_format,
            "mp3_mode": mp3_mode if resolved_format == "mp3" else None,
            "output_ext": out_path.suffix,
            "steps": steps,
            "scores_before": scores_before,
            "scores_after": scores_after,
            "overall_snr_db": overall_snr,
            "lufs_before": chain.measure_lufs(original_audio, sr),
            "lufs_after": chain.measure_lufs(audio, sr),
            "spectrum_before": {"freqs": freqs_b, "psd_db": psd_b, "tilt": tilt_before},
            "spectrum_after": {"freqs": freqs_a, "psd_db": psd_a, "tilt": tilt_after},
            "passes_both_after": scores_after["passes_both"],
            "duration_sec": n / sr,
        }

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result"] = result
        job_log(job_id, "complete")

    except Exception as e:
        traceback.print_exc()
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
        job_log(job_id, f"ERROR: {e}")


@app.route("/api/process/<file_id>", methods=["POST"])
def process(file_id):
    path = _find_upload_path(file_id)
    if path is None:
        return jsonify({"error": "unknown file_id"}), 404

    body = request.get_json(force=True, silent=True) or {}
    tools = body.get("tools", [])
    options = body.get("options", {})
    output_name = _safe_download_name(body.get("output_name"), None)
    output_format = body.get("output_format", "same")
    if output_format not in ("same", "wav", "mp3", "flac"):
        return jsonify({"error": f"invalid output_format: {output_format!r} (expected same/wav/mp3/flac)"}), 400
    mp3_mode = body.get("mp3_mode", "vbr0")
    if mp3_mode not in ("vbr0", "cbr320"):
        return jsonify({"error": f"invalid mp3_mode: {mp3_mode!r} (expected vbr0/cbr320)"}), 400

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running", "log": [], "result": None, "error": None, "progress_msg": "",
            "current_step_idx": None, "total_steps": None, "current_step_name": None, "sub_progress": None,
        }

    thread = threading.Thread(target=run_pipeline, args=(job_id, file_id, tools, options, output_name, output_format, mp3_mode), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>", methods=["GET"])
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "unknown job_id"}), 404
        return safe_jsonify({
            "status": job["status"],
            "progress_msg": job.get("progress_msg", ""),
            "current_step_idx": job.get("current_step_idx"),
            "total_steps": job.get("total_steps"),
            "current_step_name": job.get("current_step_name"),
            "sub_progress": job.get("sub_progress"),
            "log": job["log"][-200:],
            "result": job["result"],
            "error": job["error"],
        })


def _safe_download_name(name, fallback, ext=".wav"):
    """Sanitize a user-supplied filename: strip any path components, keep
    only a safe character set, ensure it has SOME audio extension (defaults
    to `ext` if none was given so a bare name still downloads correctly),
    and fall back to a default if what's left is empty."""
    if not name:
        return fallback
    name = Path(name).name  # drop any directory components
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip()
    if not name or name in (".", ".."):
        return fallback
    if Path(name).suffix.lower() not in (".wav", ".mp3", ".flac"):
        name = f"{name}{ext}"
    return name


def _find_output_path(file_id, suffix=""):
    """Outputs can be .wav/.mp3/.flac depending on the output_format chosen
    at process time - glob for whichever extension was actually written
    instead of assuming .wav. file_id is validated the same way as
    _find_upload_path, for the same reason (a glob metacharacter must never
    be able to match an arbitrary file)."""
    if not _FILE_ID_RE.match(file_id):
        return None
    matches = list(OUTPUT_DIR.glob(f"{file_id}{suffix}.*"))
    return matches[0] if matches else None


@app.route("/api/audio/<kind>/<file_id>")
def serve_audio(kind, file_id):
    if kind == "upload":
        path = _find_upload_path(file_id)
        default_name = path.name if path else "audio.wav"
    elif kind == "output":
        path = _find_output_path(file_id)
        default_name = f"{file_id}_fixed{path.suffix if path else '.wav'}"
    elif kind == "output_orig":
        path = _find_output_path(file_id, suffix="_orig")
        default_name = f"{file_id}_original{path.suffix if path else '.wav'}"
    else:
        return jsonify({"error": "invalid kind"}), 400

    if path is None or not Path(path).exists():
        return jsonify({"error": "not found"}), 404

    download_name = _safe_download_name(request.args.get("name"), default_name, ext=path.suffix)
    return send_from_directory(path.parent, path.name, as_attachment=True, download_name=download_name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=False, threaded=True)
