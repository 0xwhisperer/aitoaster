"""
The Fixer - AI-music mastering & detector-evasion verification tool.

Flask backend: upload a file, get an AI-detector score + spectral report,
choose which mastering-chain tools to run (in a fixed sane order), get back
a processed file plus a full before/after report (scores, SNR, LUFS, spectral
tilt), and an A/B stream endpoint for real-time comparison in the browser.
"""
import json
import os
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


def job_log(job_id, msg):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["log"].append({"t": time.time(), "msg": msg})
        job["progress_msg"] = msg
    print(f"[{job_id[:8]}] {msg}", flush=True)


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


def _find_upload_path(file_id):
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

    recommendations = []
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

    return jsonify({
        "file_id": file_id,
        "scores": scores,
        "lufs": lufs,
        "stereo_correlation": correlation,
        "dc_offset": {"l": float(dc[0]), "r": float(dc[1])},
        "silence": silence_info,
        "transients": transients,
        "spectral_tilt": tilt_report,
        "spectrum": {"freqs": freqs, "psd_db": psd_db},
        "recommended_tools": recommendations,
        "passes_both": scores["passes_both"],
    })


# AI-detector fixes are extremely precisely-tuned adversarial corrections -
# they MUST run last. Any gain/EQ/dynamics change applied after them (LUFS
# normalization, compression, limiting, phase adjustment) can perturb their
# exact spectral signature enough to undo the fix entirely, even though the
# fix passed real-model verification right after it ran. Confirmed directly:
# running linear_fix mid-chain scored <1% immediately after, but the SAME
# delta scored 16% once normalize_lufs/multiband_compress/true_peak_limit
# ran afterward and altered the signal the fix had been tuned against.
TOOL_ORDER = [
    "trim_silence", "dc_offset", "fix_transients", "high_pass",
    "fix_phase", "normalize_lufs", "multiband_compress", "true_peak_limit",
    "linear_fix", "cnn_fix",
]

TOOL_LABELS = {
    "trim_silence": "Trim leading/trailing silence",
    "dc_offset": "DC offset correction",
    "fix_transients": "Surgical transient/pop limiting",
    "high_pass": "High-pass filter (rumble removal)",
    "linear_fix": "AI-detector fix: linear model",
    "cnn_fix": "AI-detector fix: CNN model",
    "fix_phase": "Stereo phase/correlation correction",
    "normalize_lufs": "LUFS loudness normalization",
    "multiband_compress": "Multiband tonal-balance compression",
    "true_peak_limit": "True-peak limiter",
}


def run_pipeline(job_id, file_id, tools, options):
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

        for tool in ordered_tools:
            job_log(job_id, f"running: {TOOL_LABELS.get(tool, tool)}")
            t0 = time.time()

            if tool == "trim_silence":
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
            elif tool == "high_pass":
                audio, info = chain.high_pass_filter(audio, sr, cutoff_hz=options.get("high_pass_hz", 30))
            elif tool == "linear_fix":
                from .linear_fix import fix_linear
                audio, info = fix_linear(audio, sr, target=options.get("linear_target", 0.01),
                                          progress_cb=lambda m: job_log(job_id, m))
            elif tool == "cnn_fix":
                from .cnn_fix import fix_cnn
                audio, info = fix_cnn(audio, sr,
                                       max_steps=options.get("cnn_max_steps", 300),
                                       min_steps=options.get("cnn_min_steps", 100),
                                       hop_sec=options.get("cnn_hop_sec", 2.5),
                                       progress_cb=lambda m: job_log(job_id, m))
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

        out_id = uuid.uuid4().hex[:12]
        out_path = OUTPUT_DIR / f"{out_id}.wav"
        job_log(job_id, "saving output file")
        save_stereo(out_path, audio, sr)

        orig_path = OUTPUT_DIR / f"{out_id}_orig.wav"
        save_stereo(orig_path, original_audio, sr)

        job_log(job_id, "re-scoring with AI detectors")
        scores_after = scorer.score(str(out_path))
        scores_before = scorer.score(str(path))

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

        result = {
            "out_id": out_id,
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

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "log": [], "result": None, "error": None, "progress_msg": ""}

    thread = threading.Thread(target=run_pipeline, args=(job_id, file_id, tools, options), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>", methods=["GET"])
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "unknown job_id"}), 404
        return jsonify({
            "status": job["status"],
            "progress_msg": job.get("progress_msg", ""),
            "log": job["log"][-200:],
            "result": job["result"],
            "error": job["error"],
        })


@app.route("/api/audio/<kind>/<file_id>")
def serve_audio(kind, file_id):
    if kind == "upload":
        path = _find_upload_path(file_id)
    elif kind == "output":
        path = OUTPUT_DIR / f"{file_id}.wav"
    elif kind == "output_orig":
        path = OUTPUT_DIR / f"{file_id}_orig.wav"
    else:
        return jsonify({"error": "invalid kind"}), 400

    if path is None or not Path(path).exists():
        return jsonify({"error": "not found"}), 404

    return send_from_directory(path.parent, path.name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=False, threaded=True)
