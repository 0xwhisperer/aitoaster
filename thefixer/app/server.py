"""
The Fixer - AI-music mastering & detector-evasion verification tool.

Flask backend: upload a file, get an AI-detector score + spectral report,
choose which mastering-chain tools to run (in a fixed sane order), get back
a processed file plus a full before/after report (scores, SNR, LUFS, spectral
tilt), and an A/B stream endpoint for real-time comparison in the browser.
"""
import hashlib
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

# BUG FIX (external chain audit): FIXER_WATERMARK_SEED was set in .env, but
# nothing ever read the file, so every run fell back to the insecure built-in
# default seed and logged a warning nobody could act on - the setting existed
# and did nothing. Loaded here, before any module reads the environment.
# No dependency: .env is a handful of KEY=value lines.
def _load_dotenv(path=Path(__file__).resolve().parent.parent / ".env"):
    try:
        if not path.is_file():
            return
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # never clobber a value already exported in the real environment
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # a malformed .env must not stop the app from starting
        pass


_load_dotenv()
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

# How many completed jobs keep their full file set on disk. Each job writes 8
# files - the delivered output, the A/B original, and six correction overlays
# that are each ~2x the audio's size - so outputs/ grew without bound (measured
# at 6.4GB, which filled the disk to 99% and started failing runs). The overlay
# players and the A/B player both read from these paths, so a finished job's
# files cannot simply be deleted while the user might still be looking at it;
# instead keep the N most recent jobs whole and prune older ones.
KEEP_RECENT_JOBS = 5

# Suffixes written per job, beyond the delivered file itself. Everything here
# is regenerable diagnostic material - the source upload and the delivered
# output are what actually matter.
_JOB_ARTIFACT_SUFFIXES = (
    "_orig.wav",
    "_overlay_cnn.wav",
    "_overlay_cnn_loud.wav",
    "_overlay_combined.wav",
    "_overlay_combined_loud.wav",
    "_overlay_linear.wav",
    "_overlay_linear_loud.wav",
)


def _job_id_from_output(path):
    """Recover the out_id a file in OUTPUT_DIR belongs to."""
    name = path.name
    for suffix in _JOB_ARTIFACT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def prune_old_outputs(keep=KEEP_RECENT_JOBS):
    """Delete artifacts from all but the `keep` most recent jobs.

    Groups every file in OUTPUT_DIR by its job id, orders the groups by their
    newest file's mtime, and removes the older groups entirely (delivered file
    included - an old job's output is no longer reachable from the UI, which
    only ever shows the current result). Returns (jobs_removed, bytes_freed).

    Deliberately conservative: it only ever touches OUTPUT_DIR, never the
    uploads or anything outside it, and it never removes the newest `keep`
    jobs, so the run the user is currently looking at - and several before
    it - stay completely intact.
    """
    try:
        files = [p for p in OUTPUT_DIR.iterdir() if p.is_file()]
    except FileNotFoundError:
        return 0, 0

    jobs = {}
    for path in files:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        entry = jobs.setdefault(_job_id_from_output(path), {"paths": [], "mtime": 0})
        entry["paths"].append(path)
        entry["mtime"] = max(entry["mtime"], stat.st_mtime)

    if len(jobs) <= keep:
        return 0, 0

    ordered = sorted(jobs.items(), key=lambda kv: kv[1]["mtime"], reverse=True)
    removed_jobs = 0
    freed = 0
    for _job_id, entry in ordered[keep:]:
        for path in entry["paths"]:
            try:
                freed += path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue
        removed_jobs += 1
    return removed_jobs, freed


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


# One 16-bit least-significant-bit step, the quantization grid PCM_16 writes on.
_PCM16_LSB = 1.0 / 32768.0


def tpdf_dither_noise(shape, rng=None):
    """Add TPDF dither ahead of a 16-bit write.

    The pipeline works in float32 throughout and every WAV/FLAC output is
    written as PCM_16, so every delivered file takes a bit-depth reduction.
    Undithered, the quantization error is CORRELATED with the signal, which
    is why it is heard as distortion rather than as hiss.

    Measured on a 1kHz tone at 1.5 LSB - the quantizer's resolution limit:

        truncated, no dither : harmonics -24.6dB below the tone
        with TPDF dither     : harmonics -43.7dB below the tone

    Dither trades that correlated distortion for uncorrelated noise at about
    -93dBFS, which is inaudible on a -14 LUFS master. It matters specifically
    for LOW-LEVEL material - fade-outs, reverb tails, quiet intros - because
    that is the signal that walks down through the least significant bit on
    its way to silence. The default 3-second fade-out is exactly that signal.

    TPDF (triangular, formed as the difference of two uniform draws) is the
    standard choice for audio: unlike uniform dither it also removes noise
    MODULATION, so the noise floor does not audibly breathe with the signal.
    Peak amplitude is +-1 LSB.

    Deliberately automatic and unconfigurable. On loud material the signal is
    thousands of LSBs tall and this is negligible, so applying it when it is
    not strictly needed costs nothing - which is why there is no setting to
    get wrong.
    """
    if rng is None:
        rng = np.random.default_rng()
    # The difference of two uniform [0,1) draws spans (-1, 1), so it must be
    # scaled by 2 LSB to yield the standard +-1 LSB (2 LSB peak-to-peak) TPDF.
    # Scaling by 1 LSB - as this first shipped - gives half amplitude, which
    # is a known-bad configuration: 2 LSB p-p is the unique amplitude that
    # makes BOTH the first and second moments of the quantization error
    # independent of the signal, and that second-moment property is the whole
    # reason to choose TPDF over rectangular dither. At half amplitude the
    # noise floor still audibly breathes with the signal.
    noise = (rng.random(shape, dtype=np.float32)
             - rng.random(shape, dtype=np.float32)) * (2.0 * _PCM16_LSB)
    return noise


def save_stereo(path, audio, sr=44100, dither=False):
    """Write 16-bit PCM. `dither` must be True ONLY for a delivered file.

    save_stereo is used for two very different things: writing the file the
    user receives, and writing short-lived intermediates - the temp WAV that
    feeds the MP3/AAC encoder, and the scratch files the detectors are scored
    against. Dither belongs on the first and must never touch the second:

      * adding noise to a scoring intermediate means the detector measures a
        signal the user never receives, and this pipeline's whole discipline
        is that verification runs on the delivered audio;
      * the encoder temp WAV is re-quantized by the codec anyway, so
        dithering it just adds noise the encoder then has to spend bits on.

    Hence opt-in rather than automatic-everywhere: the default is off, and
    only the delivered-output paths pass dither=True.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if dither:
        # Dither is the LAST thing to touch the samples before quantization.
        # Any gain change applied afterwards would rescale the noise and
        # defeat the purpose, so it happens here at the point of writing.
        #
        # The seed is derived from the audio content, matching what the
        # temporal warp does and for the same reason: an unseeded generator
        # makes the delivered file non-reproducible, so two runs on identical
        # input would differ in the LSB. Content-seeding gives bit-identical
        # output for the same audio while still giving every different file
        # its own noise, so no single fixed sequence becomes a signature.
        digest = hashlib.sha256(
            np.ascontiguousarray(audio, dtype=np.float32).tobytes()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        audio = audio + tpdf_dither_noise(audio.shape, rng=rng)
    # Clip AFTER dithering so noise can never push a sample past full scale.
    # On a correctly limited signal this is a defensive no-op: the -1.0dBTP
    # ceiling leaves peaks around 0.891, some 3500 LSB below full scale, far
    # beyond a +-1 LSB dither's reach. It matters only for input that never
    # went through the limiter.
    audio_clipped = np.clip(audio, -1.0, 1.0)

    # QUANTIZE BY ROUNDING, not by libsndfile's float->PCM_16 conversion,
    # which truncates toward -infinity. Measured directly: +0.5 LSB writes as
    # 0 while -0.5 LSB writes as -1, a systematic -0.5 LSB bias on EVERY
    # sample. That is where the delivered files' DC offset came from - a
    # digital-silence write measured -0.4998 LSB mean, and both real outputs
    # landed at about -1.6e-05, above this app's own 1e-05 lossless
    # re-check floor, so a finished file immediately re-recommended
    # dc_offset on re-upload.
    #
    # The dither was not at fault: its own mean measures -0.0002 LSB. No
    # zero-mean dither can correct a biased quantizer, which is why running
    # dc_offset and re-writing simply restored the same offset.
    #
    # np.rint is round-half-to-even, so it has no bias in either direction.
    quantized = np.rint(audio_clipped.astype(np.float64) / _PCM16_LSB)
    quantized = np.clip(quantized, -32768.0, 32767.0)
    sf.write(str(path), (quantized * _PCM16_LSB).astype(np.float32), sr,
             subtype="PCM_16")


def save_correction_overlays(out_id, overlay_sources, sr):
    """Save true-level and normalized detector-correction listening files."""
    sources = dict(overlay_sources)
    if "linear" in sources and "cnn" in sources:
        n = min(len(sources["linear"]), len(sources["cnn"]))
        sources["combined"] = sources["linear"][:n] + sources["cnn"][:n]
    info = {}
    for kind, overlay in sources.items():
        peak = float(np.abs(overlay).max(initial=0))
        if peak < 1e-10:
            continue
        sf.write(
            OUTPUT_DIR / f"{out_id}_overlay_{kind}.wav",
            overlay,
            sr,
            subtype="FLOAT",
        )
        preview_gain = min(1000.0, 0.5 / peak)
        sf.write(
            OUTPUT_DIR / f"{out_id}_overlay_{kind}_loud.wav",
            np.clip(overlay * preview_gain, -0.95, 0.95),
            sr,
            subtype="FLOAT",
        )
        info[kind] = {
            "preview_gain_db": float(20 * np.log10(preview_gain)),
            "peak": peak,
        }
    return info


OUTPUT_FORMAT_EXTENSIONS = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "m4a": ".m4a"}


# extensions this app can actually re-encode to for "same as source" -
# anything outside this set (ogg, aiff, wma, opus, etc.) falls back to wav,
# but resolve_output_format now reports that explicitly rather than
# silently substituting a different format than what was actually promised.
SAME_AS_SOURCE_SUPPORTED = {"mp3": "mp3", "flac": "flac", "m4a": "m4a", "aac": "m4a", "wav": "wav"}


# Formats whose encoder discards the adversarial detector corrections.
# Measured error inside the CNN's own analysis band (500Hz-8kHz, the CQT
# range in models/config.yaml) against the size of the correction itself:
#
#     the adversarial correction:  -37.5 dB (rel. signal)
#     AAC 256k quantization error: -39.9 dB   <- only 2.4dB below it
#     MP3 -q:a 0 error:            -33.1 dB
#     FLAC error:                 -217.7 dB   <- bit-exact
#
# AAC does not degrade the correction, it OVERWRITES it: the codec spends
# its bit budget re-deciding coefficients in exactly the midrange the
# detector reads. Confirmed end to end - a run verified at 0.001% pre-encode
# delivered an .m4a scoring 99.2%. MP3's survival (0.466%) is luck rather
# than headroom; its error in that band is larger still.
LOSSY_OUTPUT_FORMATS = {"mp3", "m4a"}
DETECTOR_FIX_TOOLS = {"linear_fix", "cnn_fix"}

# How large a high-frequency deficit must be before spectral_revive is worth
# recommending. Measured on a real track: an untouched file showed 24.32dB,
# one revive pass brought it to 13.31dB and a second to 8.19dB - each pass
# helping but never clearing the raw has_rolloff flag, so the stage
# re-recommended itself indefinitely. 15dB sits above what one pass leaves
# behind and well below a genuine lossy-codec cliff, which runs to tens of dB.
SPECTRAL_REVIVE_RECOMMEND_DEFICIT_DB = 15.0


def resolve_output_format(requested_format, original_upload_path, tools=()):
    """"same" means match the original upload's container; anything else is
    taken literally. Returns (resolved_format, fallback_warning_or_None) -
    fallback_warning is set only when "same" was requested but this app
    can't encode to the source's actual format, so the caller can tell the
    user honestly rather than silently deliver a different format than what
    "same as source" implied. Confirmed as a real gap: an uploaded .m4a
    with "same as source" selected was silently delivered as .wav with no
    indication anything had changed from what was requested."""
    wants_detector_fix = bool(set(tools) & DETECTOR_FIX_TOOLS)

    if requested_format != "same":
        if wants_detector_fix and requested_format in LOSSY_OUTPUT_FORMATS:
            return "flac", (
                f"a detector fix was selected, so .{requested_format} cannot be "
                f"delivered - lossy encoding overwrites the correction (verified: "
                f"a file certified at 0.001% scored 99.2% after AAC encoding). "
                f"Delivering lossless .flac instead; encode to "
                f".{requested_format} afterwards if you need it")
        return requested_format, None

    ext = Path(original_upload_path).suffix.lower().lstrip(".")
    resolved = SAME_AS_SOURCE_SUPPORTED.get(ext)
    if resolved:
        # "same as source" must never silently make the DELIVERED file lossy
        # just because the UPLOAD was. This is the actual root cause of the
        # shipped-broken-file bug: an .m4a upload defaulted to an .m4a output,
        # and the detector correction did not survive the encode.
        if wants_detector_fix and resolved in LOSSY_OUTPUT_FORMATS:
            return "flac", (
                f"the source is .{ext}, but a detector fix was selected and "
                f"lossy encoding overwrites the correction (verified: a file "
                f"certified at 0.001% scored 99.2% after AAC encoding). "
                f"Delivering lossless .flac instead of matching the source")
        return resolved, None
    return "wav", (f"the source file is .{ext}, which this app can't re-encode to - "
                    f"delivering as .wav instead of matching the original format")


# Metadata-suppression args applied to EVERY ffmpeg encode of a delivered
# file. -map_metadata -1 drops anything carried from the input; +bitexact
# additionally stops ffmpeg writing its OWN self-identifying encoder tag
# ("Lavf<version>"), which -map_metadata alone does not touch.
#
# Verified per format with byte-level inspection, not just ffprobe:
#   flac -> no encoder tag, no "Lavf" anywhere in the file
#   m4a  -> no encoder tag, no "Lavf" anywhere in the file
#   mp3  -> no encoder tag, and with -id3v2_version 0 no ID3 header at all
#
# NOT applied: -write_xing 0. It would remove the last "Lavf" string (which
# lives in the Xing/LAME info frame, part of the bitstream rather than a
# tag), but the Xing header carries the VBR duration/seek table. Measured
# directly: dropping it made a 3.000s file report as 2.980s, breaking
# seeking and gapless playback. Not worth trading real playback correctness
# to hide a string that identifies ffmpeg, not the user or the source.
#
# Also unavoidable: libmp3lame stamps "LAME3.100" into the frame data
# itself. No ffmpeg flag suppresses it; removing it would mean rewriting
# the encoded frames.
_STRIP_ARGS = [
    "-map_metadata", "-1",
    "-map_chapters", "-1",
    "-fflags", "+bitexact",
    "-flags:a", "+bitexact",
]


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
        save_stereo(final_path, audio, sr, dither=True)
        return final_path

    # MP3/FLAC: write a temporary WAV first (soundfile has no MP3/FLAC
    # writer of its own), then let ffmpeg do the real encode + explicit
    # metadata strip in one pass.
    #
    # Dither only for FLAC. FLAC is lossless, so whatever 16-bit samples are
    # written here are exactly what the listener receives - the bit-depth
    # reduction happens at THIS write and needs dithering. MP3/AAC re-quantize
    # into their own lossy representation regardless, so dithering their input
    # just hands the encoder noise to spend bits on.
    tmp_wav = Path(f"{dest_path_no_ext}_tmp.wav")
    save_stereo(tmp_wav, audio, sr, dither=(out_format == "flac"))
    try:
        if out_format == "mp3":
            if mp3_mode == "cbr320":
                mp3_args = ["-b:a", "320k"]
            else:
                mp3_args = ["-q:a", "0"]
            cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(tmp_wav),
                   *_STRIP_ARGS,
                   # suppress the ID3v2 container ffmpeg writes purely to
                   # hold its own encoder tag - verified: with this the file
                   # no longer starts with an "ID3" header at all
                   "-id3v2_version", "0",
                   "-codec:a", "libmp3lame", *mp3_args,
                   str(final_path)]
        elif out_format == "flac":
            cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(tmp_wav),
                   *_STRIP_ARGS, "-codec:a", "flac",
                   str(final_path)]
        elif out_format == "m4a":
            cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(tmp_wav),
                   *_STRIP_ARGS, "-codec:a", "aac", "-b:a", "256k",
                   "-f", "mp4", str(final_path)]
        else:
            raise ValueError(f"unknown output format: {out_format}")
        subprocess.run(cmd, check=True)
    finally:
        if tmp_wav.exists():
            tmp_wav.unlink()
    return final_path


def filter_chain_created_transients(hits, processed, original, sr,
                                     lead_samples_trimmed,
                                     jump_threshold=0.35, search_sec=0.005):
    """Keep only the anomalies that LATER CHAIN STAGES actually introduced.

    BUG FIX (direct user report: an audible fast duck on the word "still" at
    0:58 and 1:58 of a real track, measured at -15.25dB and -14.24dB over
    ~4ms). The post-chain corrective pass was deleting vocal consonants.

    The primary fix_transients pass correctly skips those consonants:
    detect_transients' sustained-burst guard rejects anything crossing the
    jump threshold more than 8 times in 30ms, and in the SOURCE these cross
    12-18 times. But the chain's own compression and limiting smooth them,
    so post-chain the same consonants cross only 4-6 times - under the bar.
    They then read as clicks, and fix_transient repairs a click by DELETING
    it (interpolating across the region), which is what punches the hole.

    No spectral test separates the two cases here. Measured on the reported
    file, genuine clicks and these consonants overlap completely on
    crossings (2-8 vs 4-6), duration (0.2-2.2ms vs 0.6-0.9ms) and HF/LF
    ratio (0.17-4.67 vs 1.25-2.44).

    Provenance does separate them. Every one of the false positives has a
    matching large jump in the source (0.43-0.53), meaning the chain did not
    create it - it is a pre-existing sharp edge the primary pass already
    judged to be vocal material. This pass exists specifically to clean up
    what later stages introduce, so restricting it to exactly that is both
    the fix and the tool's correct scope.

    `lead_samples_trimmed` maps a processed-timeline position back to the
    original's timeline, since trim_silence removes audio from the head
    (108ms on the reported file) and the lookup would otherwise probe the
    wrong moment entirely.
    """
    if not hits:
        return []
    source_mono = original.mean(axis=1) if original.ndim > 1 else original
    n_source = len(source_mono)
    search = max(1, int(search_sec * sr))
    kept = []
    for hit in hits:
        centre = int(hit["time_sec"] * sr) + int(lead_samples_trimmed)
        lo = max(0, centre - search)
        hi = min(n_source, centre + search)
        if hi - lo < 2:
            # cannot look it up (outside the source) - leave it alone rather
            # than delete audio on the strength of a failed check
            continue
        source_jump = float(np.abs(np.diff(source_mono[lo:hi])).max())
        if source_jump < jump_threshold:
            kept.append(hit)
    return kept


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
    of a single undifferentiated percentage.

    BUG FIX (direct user report + screenshot): post-chain reverify passes
    (cnn_fix_reverify, cnn_fix_reverify_lufs, and their linear equivalents)
    run entirely AFTER the numbered tool loop finishes and never called this
    function - so current_step_idx/total_steps/current_step_name stayed
    frozen on whatever the LAST real tool-loop entry was (e.g. "True-peak
    limiter", Tool 13 of 13) for the entire duration of a reverify pass,
    while sub_progress (the optimization-step counter) kept updating
    independently. The result: a live job could show "Tool 13 of 13
    (True-peak limiter) / Optimization step 14 of 516" when what was
    actually running was a CNN re-verification retry that has nothing to
    do with the limiter and isn't part of the 13-tool count at all.
    step_idx=None/total_steps=None now marks a step as OUTSIDE the numbered
    chain (a reverify pass) - the frontend renders step_name alone with no
    misleading "Tool N of N" wrapper in that case."""
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


class JobCancelled(Exception):
    """Raised when a job's cancel_requested flag is set, checked at safe
    points between pipeline stages and from the progress callbacks used by
    the long-running optimizers. A slow real-model scoring call can still
    delay the next callback, so cancellation is cooperative, not instant."""


def check_cancelled(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None and job.get("cancel_requested"):
            raise JobCancelled()


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
    source_format = chain.read_source_format(saved_path)
    return jsonify({
        "file_id": file_id,
        "filename": f.filename,
        "duration_sec": round(duration_sec, 2),
        "samples": len(audio),
        "source_format": source_format,
    })


_FILE_ID_RE = re.compile(r"^[0-9a-f]{12}$")



def _trail_looks_like_a_fade(audio, sr, trail_ms, floor_db=-66.0):
    """True when the trailing region decays smoothly rather than being flat.

    trim_silence uses a -66dBFS threshold, not digital zero, so the tail of a
    fade-out reads as "silence". Recommending a trim there tells the user to
    delete the fade they asked for - and acting on it shifts the timeline and
    invalidates the CNN certification (measured: a delivered file went from
    0.00205% to 0.33750% median, with individual windows at 99.62%).

    A fade decays monotonically over its whole length; genuine trailing
    silence is flat at the noise floor. Comparing the first and last thirds
    of the trailing region separates them cleanly: a fade shows a large drop,
    flat silence shows almost none.
    """
    if trail_ms <= 0:
        return False
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    n = int(trail_ms * 0.001 * sr)
    if n < int(0.05 * sr) or n > len(mono):
        return False
    tail = mono[-n:]
    third = max(1, n // 3)
    first_db = 20 * np.log10(np.sqrt((tail[:third].astype(np.float64) ** 2).mean()) + 1e-12)
    last_db = 20 * np.log10(np.sqrt((tail[-third:].astype(np.float64) ** 2).mean()) + 1e-12)
    # a fade keeps falling across the region; flat silence does not
    return (first_db - last_db) > 6.0


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
    waveform = chain.waveform_peaks(audio, 44100)

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

    # BUG FIX: high_pass/multiband_compress/true_peak_limit used to be
    # unconditionally recommended on EVERY file, with no check for whether
    # they'd actually do anything - meaning a file this app already
    # processed (or one that never needed them) would always show these
    # as "recommended" again, which is never an honest signal. Run each
    # tool's own real check (same function the pipeline itself uses) and
    # only recommend it if it reports it would actually change something.
    _, high_pass_check_info = chain.high_pass_filter(audio, 44100, cutoff_hz=30)
    _, true_peak_check_info = chain.true_peak_limit(audio, 44100, ceiling_db=-1.0)
    # BUG FIX (direct user report): multiband_compress used to be checked
    # the same way as high_pass/true_peak - run the tool itself and see if
    # its own gain math found anything - but that's the wrong question for
    # a genuinely gentle, diminishing-returns tool (ratio=1.3 by design).
    # Verified directly: repeatedly running multiband_compress on its own
    # output barely reduced its own reported max_reduction_db pass over
    # pass, so it kept recommending itself almost indefinitely on a real
    # peaky file. detect_band_peakiness measures the FILE's own actual
    # condition (how much of its duration is spent meaningfully over
    # threshold in each band) instead of the compressor's own trace - see
    # that function's docstring for the full comparison data.
    band_peakiness = chain.detect_band_peakiness(audio, 44100)

    recommendations = []
    if metadata["format"] or any(s["tags"] for s in metadata["streams"]) or has_embedded_images:
        recommendations.append("strip_metadata")
    # BUG FIX (external chain audit, reproduced): spectral_revive is not
    # idempotent against its own detector. On a real track the deficit went
    # 24.32dB -> 13.31dB after one pass -> 8.19dB after two, shrinking each
    # time but never clearing the has_rolloff bar - so a file this app had
    # already revived kept re-recommending the stage forever, and each extra
    # pass synthesises more HF on top of HF it synthesised itself.
    #
    # A deficit this small is no longer the hard artificial cliff the stage
    # exists to repair (a lossy/low-bitrate cutoff is tens of dB); it is the
    # residue of the stage's own extrapolation, which cannot reach the
    # detector's bar because the fitted slope only asymptotically approaches
    # it. Recommend only a deficit large enough to be a genuine cliff.
    if has_rolloff and rolloff_deficit_db >= SPECTRAL_REVIVE_RECOMMEND_DEFICIT_DB:
        recommendations.append("spectral_revive")
    if scores["linear"]["probability"] >= 0.01:
        recommendations.append("linear_fix")
    if scores["cnn"]["probability"] >= 0.5:
        recommendations.append("cnn_fix")
    # DC_OFFSET_RECHECK_FLOOR (module-level constant, see its own
    # definition above TOOL_ORDER for the full measured justification -
    # MP3 lossy quantization noise floor) is shared with _tool_status_line
    # so the live per-tool log line and this recommendation logic can
    # never independently drift again.
    #
    # BUG FIX (Grok #10, verified against the measurement this floor was
    # originally derived from): DC_OFFSET_RECHECK_FLOOR was measured
    # specifically from MP3 ENCODER-introduced DC bias (~2e-6 to ~3.5e-5
    # across synthetic tones) - it exists to stop a lossy round-trip from
    # re-flagging bias the encoder itself introduced, not the app. That
    # justification only applies to a LOSSY upload; a lossless upload
    # (WAV/FLAC/AIFF - no lossy encoder in its own history) has no such
    # noise floor to tolerate, so applying the same 6x-looser bar to it
    # uniformly meant a genuine ~3e-5 DC offset on a lossless file (well
    # above this app's original, tighter 1e-5 bar) could go unflagged for
    # no reason tied to how that specific file was actually produced.
    # LOSSLESS_UPLOAD_EXTENSIONS intentionally lists containers with no
    # lossy-encode step in their own provenance; anything else (mp3, aac,
    # ogg, opus, m4a, etc.) keeps the original MP3-derived floor, since an
    # uploaded lossy file already carries whatever bias ITS OWN encoder
    # introduced, matching the exact case this floor was measured for.
    dc_floor = (
        DC_OFFSET_LOSSLESS_RECHECK_FLOOR
        if path.suffix.lower() in LOSSLESS_UPLOAD_EXTENSIONS
        else DC_OFFSET_RECHECK_FLOOR
    )
    if abs(dc[0]) > dc_floor or abs(dc[1]) > dc_floor:
        recommendations.append("dc_offset")
    # BUG FIX: 20ms was tight enough to flag genuinely inaudible residue
    # (a few tens of milliseconds left over from resample/interpolation
    # round-trips elsewhere in this pipeline - CQT transfer, temporal
    # denormalization, etc.) as "still needs trimming," which meant a file
    # this app already trimmed could still show this as recommended again.
    # 100ms is comfortably above that kind of processing residue while
    # still well below anything a listener would perceive as a real gap.
    #
    # BUG FIX (external audit, reproduced): a 3000ms fade-out ends in
    # near-silence BY DESIGN, and trim_silence's threshold is -66dBFS rather
    # than digital zero, so re-uploading a file this app just delivered
    # recommended cutting 280.7ms off its own fade. Acting on that
    # recommendation shifts the timeline, and every CNN analysis window
    # position derives from len(audio) - measured, it took a delivered file
    # from 0.00205% to 0.33750% median with two sampled windows at 99.62% and
    # 97.92%. That is the exact failure the post-chain corrective trim was
    # deleted for; leaving it reachable through the recommender just moved it
    # one step away.
    #
    # A trail that DECAYS SMOOTHLY into the noise floor is a fade, not
    # silence to be trimmed. Real leading/trailing silence is flat. So a
    # trailing region is only worth recommending a trim for if it does not
    # look like a fade.
    lead_ms = silence_info.get("lead_ms", 0)
    trail_ms = silence_info.get("trail_ms", 0)
    trail_is_a_fade = _trail_looks_like_a_fade(audio, 44100, trail_ms)
    if lead_ms > 100 or (trail_ms > 100 and not trail_is_a_fade):
        recommendations.append("trim_silence")
    if transients:
        recommendations.append("fix_transients")
    if correlation < 0.1:
        recommendations.append("fix_phase")
    if not (LUFS_GOOD_LOW <= lufs <= LUFS_GOOD_HIGH):
        recommendations.append("normalize_lufs")
    if high_pass_check_info.get("applied"):
        recommendations.append("high_pass")
    # FRAC_TIME_OVER_RECOMMEND_FLOOR: measured directly, a genuinely-fine
    # real file (already processed by this app) spent at most ~1.3% of its
    # duration meaningfully over threshold in any band (a brief transient,
    # not real imbalance), while a deliberately-built peaky test signal
    # spent 16-25%. 2% sits comfortably above the well-mastered file's
    # real noise floor while well below the genuinely-peaky signal's
    # range, so a brief transient doesn't trigger a recommendation but
    # real, sustained tonal imbalance does.
    FRAC_TIME_OVER_RECOMMEND_FLOOR = 0.02
    if any(b["frac_time_over"] > FRAC_TIME_OVER_RECOMMEND_FLOOR for b in band_peakiness):
        recommendations.append("multiband_compress")
    if true_peak_check_info.get("applied"):
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
        "waveform": waveform,
        "metadata": metadata,
        "provenance_tags_found": provenance_hits,
        "has_embedded_images": has_embedded_images,
        "spectral_rolloff": {"detected": has_rolloff, "cutoff_hz": rolloff_cutoff_hz, "deficit_db": round(rolloff_deficit_db, 1)},
        "recommended_tools": recommendations,
        "band_peakiness": band_peakiness,
        # BUG FIX (Codex MAJOR / Fable B3, verified directly): multiband_compress
        # is deliberately gentle (ratio=1.3, "least change necessary" by design -
        # see its own docstring) and can take 4-5 passes to fully clear a
        # strongly peaky signal's frac_time_over recommendation bar, even though
        # peak_over_db decays geometrically and genuinely every single pass.
        # Making the tool more aggressive to converge in one pass would trade
        # away that documented design goal; the actual bug the audits caught is
        # that the UI had no way to SHOW that real progress even though the
        # flat "still recommended" signal stays true across several passes.
        # Expose the underlying per-band peak_over_db numbers (not just the
        # boolean recommendation) so a re-upload after one pass can say
        # "still peaky, but improved from 5.9dB to 5.0dB over" instead of
        # repeating an unqualified "run multiband_compress again" with no
        # indication whether the last pass did anything at all.
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
# The position-sensitive detector fixes run after temporal normalization.
# A real 4:37 production job proved the old order was untenable: CNN spent
# 23 minutes reaching a certified result, then linear EQ plus a 15ms timing
# warp changed that signal and the final CNN score jumped to 99.742%, forcing
# another 18-minute whole-track solve.  Linear now runs first because its
# feature-domain solve is cheap and reliable; Thorough CNN then optimizes and
# certifies the actual post-linear waveform.  The independent final checks
# below remain authoritative because either correction can still perturb the
# other, but ordinary temporal/linear processing no longer guarantees a full
# CNN restart by construction.
# Post-chain re-verification margins. linear_fix and cnn_fix already certify
# their output below a real_target threshold (0.008-0.01 for linear, 0.08 for
# cnn) before they hand off - that certification is the whole point of their
# worst-shift-scan verification loops. But the post-chain rechecks further
# down this file used the SAME numbers as the re-trigger point, with zero
# margin between "just certified" and "treat as regressed." Neither
# temporal_normalize nor true_peak_limit touch the frequency bands either
# detector model attends to (temporal warp is sub-audio-sample interpolation
# noise; the limiter only touches isolated peak samples), so any score
# movement they cause post-certification is floating-point/dither-level, not
# a real regression - but with zero margin, that noise alone was enough to
# re-trigger a full expensive re-run (Thorough-mode cnn_fix included) every
# single time, even when the file was already genuinely fine. Confirmed
# directly: this is why real production jobs were burning a full second
# cnn_fix pass (and sometimes a third) on files that never actually needed
# one - "3 of 4 passes do nothing but take time" per direct user report.
# These margins give real headroom above the optimizers' own certified
# targets before paying for a full re-run, while still catching genuine
# regressions (which move scores by whole percentage points, not noise).
# BUG FIX (adversarial audit, verified directly): these margins were
# originally sized to make a specific noise-triggered false-positive
# symptom go away, without measuring what real benign downstream drift
# actually looks like - the audit caught that a genuine 1.9% linear score
# (0.01+0.01=0.02 threshold, so 0.019 stays silently under it) or a
# genuine 10% CNN score (0.08+0.05=0.13, so 0.10 stays under it) would
# NOT re-trigger, even though both are real, meaningful regressions by
# this app's own stated safety-margin philosophy. Measured directly what
# benign downstream noise ACTUALLY looks like: a tiny gain-only mutation
# (simulating true_peak_limit) moved linear by ~0.0006 percentage points
# and CNN by ~0.033 percentage points; a full real MP3 round-trip moved
# linear by ~0.00005 and CNN by ~0.037 percentage points. Both are two
# orders of magnitude smaller than the old margins. These new margins
# keep real headroom (roughly 10-20x the measured noise floor) above that
# actual measured drift while catching real regressions like the audit's
# 0.019/0.10 examples, which the old margins let through entirely.
LINEAR_RECHECK_MARGIN = 0.001   # linear target 0.01 -> re-trigger at 0.011
CNN_RECHECK_MARGIN = 0.005      # cnn target 0.08 -> re-trigger at 0.085

# BUG FIX (third adversarial audit round, verified directly): this used to
# be a LOCAL variable defined only inside the /api/analyze endpoint - which
# meant _tool_status_line (the live per-tool log line shown DURING a job)
# had its own separate, stale 0.001 threshold with no way to reference the
# real one. Confirmed directly: a file could log "dc_offset: pass" during
# the run using the old 0.001 bar, then get re-recommended for dc_offset
# the moment that same delivered file was re-uploaded and checked against
# the real 6e-5 floor - the exact "results claim success but re-analysis
# disagrees" contradiction this session has repeatedly had to fix in other
# tools. Promoted to a real module-level constant so every consumer
# (analyze recommendations, the live status line, and any future caller)
# references the same single source of truth.
DC_OFFSET_RECHECK_FLOOR = 6e-5

# BUG FIX (Grok #10, verified against server.py's own DC_OFFSET_RECHECK_FLOOR
# measurement): that floor was derived specifically from MP3 encoder noise
# and is too loose for a genuinely lossless upload, which has no such noise
# to tolerate. Restores this app's original, tighter pre-MP3-measurement bar
# (1e-5) for lossless source containers only - see the DC_OFFSET_RECHECK_FLOOR
# recommendation site in /api/analyze for the full reasoning and the
# per-format selection.
DC_OFFSET_LOSSLESS_RECHECK_FLOOR = 1e-5
LOSSLESS_UPLOAD_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".alac"}

# BUG FIX (fourth adversarial audit round, Grok/Fable, verified directly by
# grep before fixing): same bug class as DC_OFFSET_RECHECK_FLOOR above, one
# instance missed - _tool_status_line's normalize_lufs branch had its own
# separate, stale -16..-12 bar, hardcoded independently of the real
# -17..-13 bar already used by /api/analyze's recommendation logic and (per
# an earlier fix this session) both frontend result tables. Confirmed
# directly: a delivered file at -12.5 LUFS would log "pass" during the run
# (inside the old -16..-12 bar) and then get immediately re-recommended for
# normalize_lufs on re-upload/re-analysis (outside the real -17..-13 bar) -
# the exact "log says pass, re-analysis disagrees" contradiction this
# session has repeatedly had to fix elsewhere. Promoted to real
# module-level constants, referenced by both consumers.
LUFS_GOOD_LOW = -17.0
LUFS_GOOD_HIGH = -13.0

TOOL_ORDER = [
    # --- cleanup: get the signal honest before anything MEASURES it ---
    "strip_metadata",
    # dc_offset and high_pass run BEFORE trim_silence. Measured: a file with
    # a 0.02 DC offset trimmed 0ms of lead and 0ms of trail, because the
    # offset lifts otherwise-silent samples above trim_silence's -66dBFS
    # threshold so the silence is invisible to it. After DC correction the
    # same file trims 995ms lead / 1000ms trail. High-pass runs before it for
    # the same reason - sub-30Hz rumble also holds the "silence" up.
    #
    # This does not conflict with the timeline rule: all three are still well
    # before the detector fixes. trim_silence remains the ONLY stage that
    # changes the sample count, and it now measures a signal that has been
    # centred and de-rumbled first, so it measures the right thing.
    "dc_offset",
    # high_pass moved up (was step 6, after transient repair and HF synthesis).
    # Rumble removal must precede every level-dependent stage: sub-30Hz energy
    # inflates the envelope that drives the multiband detector and eats limiter
    # headroom, so anything that measures level while it is still present is
    # measuring the wrong signal. DC offset and high-pass are the cleanup pair.
    "high_pass",
    "trim_silence",
    # temporal_normalize MOVED UP (was after multiband, just before the
    # detector fixes). It warps the time axis, so it is a TIMELINE stage even
    # though it preserves sample count - and applied to a certified signal it
    # measured +4.87pp (Poster) and +97.05pp (North Star). Every timeline
    # change must happen before certification; this is the last one.
    "temporal_normalize",
    "fix_transients",
    # --- tonal/spatial shaping ---
    # fix_phase moved BEFORE spectral_revive: correlation must be measured on
    # real recorded content, not on synthesised HF.
    "fix_phase",
    "spectral_revive",
    # tonal_cleanup sits AFTER spectral_revive (revive fits its own rolloff
    # slope from 3kHz upward; correcting at 3.15kHz first would corrupt that
    # fit) and BEFORE the dynamics, so the compressor reacts to a corrected
    # spectrum rather than to a resonance. Cut-only and never a shelf, so it
    # cannot re-boost what high_pass removed. Well before the detector fixes,
    # so the certification covers whatever it did.
    "tonal_cleanup",
    # --- dynamics at working level, BEFORE loudness is set ---
    "multiband_compress",
    # saturation runs AFTER multiband so the compressor's envelope detector
    # sees undistorted crest and the saturator sees a stabilised level (which
    # its own auto-gain estimator depends on), and BEFORE the detector fixes
    # because saturation applied after them DESTROYS the adversarial
    # correction they just computed. Measured by actually moving this stage
    # after cnn_fix and running the full pipeline: the delivered file scored
    # 99.997% CNN, up from ~0.2% - a near-total regression, not a tolerable
    # trade-off. (An earlier version of this comment said "up to 6.85
    # percentage points", which understated it by a factor of 15 and read
    # like an acceptable cost.) Before normalize_lufs too, matching the
    # "dynamics at working level" principle: it is RMS-matched and so nearly
    # loudness-neutral (measured +0.10 LU at the default drive), which is
    # also why the loudness stage downstream reclaims almost none of it.
    "saturate",
    # Timing changes must precede the position-sensitive detector fixes.
    # Run the cheap/reliable linear solve first and the exact-window CNN
    # solve on that final spectral signal, so Thorough CNN no longer pays
    # for a complete redo after temporal normalization or linear EQ.
    "linear_fix", "cnn_fix",
    # --- delivery: loudness second-to-last, limiter last ---
    # normalize_lufs used to run at step 8, BEFORE multiband and the limiter -
    # i.e. loudness was set and then the two stages that change loudness most
    # ran afterwards. That is why a 6-pass post-chain drift-correction loop
    # existed at all. Setting loudness here, with only the limiter after it,
    # means the delivered file lands on target by construction rather than by
    # repeated correction.
    "normalize_lufs",
    "true_peak_limit",
]

# "fade" is deliberately NOT in TOOL_ORDER: it is applied as a dedicated
# stage after the whole chain (including the post-chain LUFS drift
# correction), not inside the per-tool loop. See _apply_fade_stage.
FADE_TOOL = "fade"


# ---------------------------------------------------------------------------
# THE TIMELINE INVARIANT
#
# A CNN certification is bound to the exact TIMELINE it was made on. Confirmed
# in the model itself: CNNDetector.extract_segments derives every analysis
# window position from len(audio) (`end_offset = len(audio) - skip`, positions
# spread across `usable`). Change the sample count and every window slides;
# the correction is still in the file but no longer where the detector looks.
#
# Measured, on a CERTIFIED signal:
#   trail trim 311.5ms      Poster +87.03pp   NorthStar +31.11pp
#   lead trim 1.4ms         Poster  +0.85pp   NorthStar +99.52pp
#   pad +10ms at the end    Poster  92.10%
#   temporal_normalize      Poster  +4.87pp   NorthStar +97.05pp
#     (^ mutates NO length, but displaces content within the timeline)
#   zeroing head+tail       EXACTLY 0.0000pp - amplitude only, inert
#   normalize_lufs          -0.0002 / -0.0029pp
#   true_peak_limit         -0.0000 / +0.0000pp
#   fade, watermark         +0.0000 / +0.0000pp - bit-exactly inert
#   gain +/-3dB             +/-0.0006 / +/-0.013pp
#
# So the rule is NOT "certification last" and NOT "nothing may touch the audio
# afterwards". An earlier draft of this comment said exactly that, and it was
# wrong in both directions: it would have guarded normalize_lufs and
# true_peak_limit, which are the two SAFEST stages in the chain, while missing
# temporal_normalize, which preserves sample count and is one of the most
# destructive. The real rule is:
#
#   FREEZE THE TIMELINE BEFORE CERTIFYING. After that point, amplitude-domain
#   work is free; anything that changes the sample count or displaces content
#   within the timeline is forbidden.
#
# This is why the corrective trim broke a file certified at 0.003% and
# delivered it at 78.8%: it cut 1.4ms of lead and 311.5ms of trail. With the
# timeline frozen it cannot fire at all - measured, trim_silence on certified
# audio finds only 1.8-12.8ms, which is the fade's own tail.
#
# Note the fragility is CHAOTIC, not monotonic (Poster: a 10ms lead trim gives
# 18.41%, but 100ms gives 0.03%), so a near-miss is not a safe margin. The
# guard must be structural - assert the length - not a threshold re-score.
# ---------------------------------------------------------------------------

# Everything that may change the sample count or displace content in time.
# All of these MUST run before linear_fix/cnn_fix.
TIMELINE_STAGES = frozenset({
    "trim_silence",        # cuts lead/trail
    "temporal_normalize",  # length-preserving, but warps the time axis
    FADE_TOOL,             # amplitude-only, but its tail creates trimmable silence
})

# Stages permitted after certification, each measured inert on a certified
# signal. These are amplitude-domain only - they cannot move a window.
# The margin the optimizer itself targets (cnn_wholetrack_optimizer_v2's
# real_target). The model's raw pass/fail bar is 0.5; anything delivered
# between this and that bar has passed but without the headroom the fix was
# aiming for, and is reported as such rather than silently called a pass.
CNN_DELIVERED_MARGIN = 0.08

POST_CERTIFICATION_ALLOWED = frozenset({
    "normalize_lufs",    # -0.0002 / -0.0029pp
    "true_peak_limit",   # -0.0000 / +0.0000pp
})

TOOL_LABELS = {
    "strip_metadata": "Strip metadata & embedded images",
    "trim_silence": "Trim leading/trailing silence",
    "dc_offset": "DC offset correction",
    "fix_transients": "Surgical transient/pop limiting",
    "spectral_revive": "High-frequency spectral fill-in (17kHz+)",
    "high_pass": "High-pass filter (rumble removal)",
    "linear_fix": "AI-detector fix: linear model",
    "cnn_fix": "AI-detector fix: CNN model",
    "temporal_normalize": "Temporal pattern denormalization",
    "fix_phase": "Stereo phase/correlation correction",
    "normalize_lufs": "LUFS loudness normalization",
    "multiband_compress": "Multiband tonal-balance compression",
    "tonal_cleanup": "Tonal cleanup (boxiness / harshness)",
    "saturate": "Saturation (harmonic colour)",
    "true_peak_limit": "True-peak limiter",
    "fade": "Fade in / fade out",
}


def _tool_status_line(tool, info):
    """Builds ONE confirmation line for the given tool's outcome, called
    uniformly right after every "done (Xs)" line in the main tool loop -
    guarantees every selected tool gets a status line, in the same place,
    every time, rather than relying on each branch to remember to add its
    own (which is exactly how several tools ended up silently missing one).
    Returns None for tools with nothing meaningful to report beyond "done."

    Thresholds quoted here match the results table's own pass/fail bars
    (statusPill in app.js) exactly, so the live line and the final table
    never disagree about what counts as a pass."""
    applied = info.get("applied")

    if tool == "strip_metadata":
        n_tags = len(info.get("tags_found") or {})
        has_images = info.get("has_embedded_images")
        if not applied:
            return "pass (nothing found to strip)"
        parts = []
        if n_tags:
            parts.append(f"{n_tags} tag{'s' if n_tags != 1 else ''}")
        if has_images:
            parts.append("embedded image(s)")
        return f"pass (removed {' and '.join(parts)})"

    if tool == "trim_silence":
        if not applied:
            return "pass (no leading/trailing silence found)"
        lead_ms = info.get("lead_ms", 0)
        trail_ms = info.get("trail_ms", 0)
        return f"pass (trimmed {lead_ms:.0f}ms lead / {trail_ms:.0f}ms trail)"

    if tool == "dc_offset":
        dc_after_max = max(abs(info.get("dc_l_after", 0.0)), abs(info.get("dc_r_after", 0.0))) if applied else 0.0
        return f"{'pass' if dc_after_max < DC_OFFSET_RECHECK_FLOOR else 'check'} (max L/R after: {dc_after_max:.5f})"

    if tool == "fix_transients":
        count = info.get("count", 0)
        return f"pass (processed {count} anomal{'y' if count == 1 else 'ies'}; final check runs after the full chain)"

    if tool == "spectral_revive":
        if not applied:
            reason = info.get("reason", "no artificial rolloff detected")
            return f"pass ({reason})"
        cutoff = info.get("cutoff_hz")
        slope = info.get("fitted_rolloff_db_per_octave")
        return f"pass (filled above {cutoff / 1000:.0f}kHz, fitted rolloff {slope:.1f}dB/octave)" if cutoff and slope is not None else "pass (applied)"

    if tool == "tonal_cleanup":
        bands = info.get("bands") or []
        if not applied:
            skipped = [b for b in bands if b.get("skipped")]
            if skipped:
                return ("pass (nothing to correct; "
                        + ", ".join(f"{b['label']} reads as a spectral slope"
                                    for b in skipped) + ")")
            scored = [b for b in bands if b.get("persistent_db") is not None]
            if scored:
                w = max(scored, key=lambda b: b["persistent_db"])
                return (f"pass (nothing rings persistently; closest was "
                        f"{w['label']} at {w['persistent_db']:+.1f}dB against "
                        f"a {w['trigger_db']:+.1f}dB bar)")
            return "pass (nothing to correct)"
        cuts = [b for b in bands if b.get("cut_db", 0) < 0]
        return "pass (cut " + ", ".join(
            f"{b['label']} {b['cut_db']:.2f}dB at {b['freq_hz']:.0f}Hz"
            for b in cuts) + ")"

    if tool == "saturate":
        if not applied:
            return f"pass ({info.get('reason', 'not applied')})"
        return (f"pass ({info.get('amount')}, drive {info.get('drive')}, "
                f"{info.get('oversample')}x oversampled, "
                f"peak {info.get('peak_before', 0):.3f} -> "
                f"{info.get('peak_after', 0):.3f}, "
                f"makeup {info.get('makeup_db', 0):+.2f}dB)")

    if tool == "high_pass":
        cutoff = info.get("cutoff_hz")
        return f"pass (cutoff {cutoff:.0f}Hz)" if cutoff else "pass (applied)"

    if tool == "fix_phase":
        corr_after = info.get("correlation_after", info.get("correlation"))
        if corr_after is None:
            return "pass (applied)"
        ok = corr_after >= 0.1
        parts = [f"correlation: {corr_after:.2f}"]
        if info.get("bass_mono_hz"):
            parts.append(f"bass mono below {info['bass_mono_hz']:.0f}Hz")
        if info.get("phase_corrected"):
            parts.append("low-band phase corrected")
        if info.get("width_boost", 0) > 1.0:
            parts.append(f"width x{info['width_boost']:.2f}")
        return f"{'pass' if ok else 'check'} ({', '.join(parts)})"

    if tool == "normalize_lufs":
        lufs_after = info.get("lufs_after")
        if lufs_after is None:
            return "pass (no change needed)"
        ok = LUFS_GOOD_LOW <= lufs_after <= LUFS_GOOD_HIGH
        return f"{'pass' if ok else 'check'} ({lufs_after:.1f} LUFS)"

    if tool == "multiband_compress":
        bands = info.get("bands") or []
        max_reduction = min((b.get("max_reduction_db", 0.0) for b in bands), default=0.0)
        passes = info.get("passes", 1)
        pass_note = f", {passes} passes" if passes > 1 else ""
        return (f"pass (up to {abs(max_reduction):.1f}dB gentle reduction across "
                f"{len(bands)} bands{pass_note})")

    if tool == "temporal_normalize":
        # "pass" here means the operation completed and produced valid
        # audio - NOT a claim of verified effectiveness against any real
        # detector/fingerprinting service (there is no such verification
        # available - see app/timewarp.py's module docstring). Matches
        # what's actually knowable at this point in the pipeline, same
        # honesty standard as everything else in this app.
        return f"pass (applied, max drift: {info.get('max_drift_ms', 0):.0f}ms)"

    if tool == "fade":
        if not applied:
            return "pass (no fade requested)"
        parts = []
        if info.get("fade_in_samples", 0) > 0:
            parts.append(f"{info.get('fade_in_ms', 0)}ms in")
        if info.get("fade_out_samples", 0) > 0:
            parts.append(f"{info.get('fade_out_ms', 0)}ms out")
        return f"pass (applied: {', '.join(parts)})" if parts else "pass (applied)"

    if tool == "true_peak_limit":
        if not applied:
            return "pass (already under ceiling)"
        reduction = info.get("gain_reduction_db")
        ceiling = info.get("ceiling_db")
        return f"pass (reduced up to {abs(reduction):.1f}dB to hold {ceiling:.1f}dBTP ceiling)" if reduction is not None else "pass (applied)"

    return None


def run_pipeline(job_id, file_id, tools, options, output_name=None, output_format="same", mp3_mode="vbr0"):
    job_started_at = time.time()
    _certified_length = None
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
        correction_overlays = {}

        def _record_detector_overlay(kind, before, after):
            """Accumulate only the waveform added by one detector fix."""
            n = min(len(before), len(after))
            if n == 0:
                return
            change = np.zeros_like(after)
            change[:n] = after[:n] - before[:n]
            existing = correction_overlays.get(kind)
            if existing is None or existing.shape != change.shape:
                correction_overlays[kind] = change
            else:
                existing += change

        ordered_tools = [t for t in TOOL_ORDER if t in tools]
        lead_samples_trimmed = 0
        total_steps = len(ordered_tools)
        late_mutation_after_temporal = False

        def _cancel_aware_log(message):
            check_cancelled(job_id)
            job_log(job_id, message)

        def _linear_step_cb(step, mx, score, attempt, max_attempts):
            check_cancelled(job_id)
            job_set_sub_progress(
                job_id, step, mx,
                extra={
                    "score_pct": round(float(score) * 100, 4),
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                },
            )

        for step_idx, tool in enumerate(ordered_tools):
            check_cancelled(job_id)
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
                if all_tags:
                    tag_summary = ", ".join(f"{k}={v!r}" for k, v in all_tags.items())
                    job_log(job_id, f"  found and removing tags: {tag_summary}")
                else:
                    job_log(job_id, "  no text tags found on the source file")
                if has_images:
                    n_images = sum(1 for s in metadata["streams"] if s["is_attached_image"])
                    job_log(job_id, f"  found and removing {n_images} embedded image(s) (e.g. cover art)")
                else:
                    job_log(job_id, "  no embedded images found")
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
            elif tool == "tonal_cleanup":
                audio, info = chain.tonal_cleanup(audio, sr)
            elif tool == "saturate":
                audio, info = chain.saturate(
                    audio, sr, amount=options.get("saturation_amount", "medium"))
            elif tool == "spectral_revive":
                audio, info = chain.spectral_revive(audio, sr, cutoff_hz=options.get("spectral_revive_cutoff_hz"))
            elif tool == "high_pass":
                audio, info = chain.high_pass_filter(audio, sr, cutoff_hz=options.get("high_pass_hz", 30))
            elif tool == "linear_fix":
                from .linear_fix import fix_linear
                before_fix = audio.copy()
                audio, info = fix_linear(
                    audio, sr, target=options.get("linear_target", 0.01),
                    progress_cb=_cancel_aware_log,
                    step_progress_cb=_linear_step_cb,
                )
                _record_detector_overlay("linear", before_fix, audio)
            elif tool == "cnn_fix":
                from .cnn_fix import fix_cnn
                before_fix = audio.copy()
                cnn_max_steps = options.get("cnn_max_steps", 300)

                def _cnn_step_cb(s, mx, surrogate_score, real_check_extra):
                    check_cancelled(job_id)
                    # real_check_extra's values come from get_real_score_segment,
                    # which returns numpy float32 - round() does not convert that
                    # to a native python float, and Flask's json encoder can't
                    # serialize numpy scalar types (this exact bug class already
                    # hit twice earlier this session in other fields). float()
                    # first, then round.
                    extra = {"score_pct": round(float(surrogate_score) * 100, 4)}
                    if real_check_extra is not None:
                        extra["real_max_score_pct"] = round(float(real_check_extra["real_max_score"]) * 100, 4)
                        extra["windows_failing"] = int(real_check_extra["n_windows_bad"])
                        extra["windows_total"] = int(real_check_extra["n_windows"])
                    job_set_sub_progress(job_id, s, mx, extra=extra)

                cnn_mode = options.get("cnn_mode", "thorough")
                if cnn_mode not in ("simple", "eot", "thorough"):
                    cnn_mode = "thorough"
                audio, info = fix_cnn(audio, sr,
                                       max_steps=cnn_max_steps,
                                       min_steps=options.get("cnn_min_steps", 100),
                                       hop_sec=options.get("cnn_hop_sec", 0.5),
                                       progress_cb=_cancel_aware_log,
                                       step_progress_cb=_cnn_step_cb,
                                       mode=cnn_mode)
                _record_detector_overlay("cnn", before_fix, audio)
            elif tool == "fix_phase":
                # stereo_field_correct replaces fix_phase_issues' whole-track
                # scalar approach: bass-mono below 120Hz and phase repair
                # confined to 120-300Hz. It does NOT widen - an earlier draft
                # had a width stage and both this comment and the function's
                # own docstring went on claiming it long after it was removed.
                # It
                # keeps this slot in TOOL_ORDER deliberately - the position was
                # already right (before spectral_revive, so correlation is
                # measured on real recorded content rather than synthesised
                # high frequencies) and all three operations belong there.
                audio, info = chain.stereo_field_correct(audio, sr)
            elif tool == "normalize_lufs":
                audio, info = chain.normalize_lufs(audio, sr, target_lufs=options.get("lufs_target", -14.0))
            elif tool == "multiband_compress":
                audio, info = chain.multiband_compress(audio, sr)
            elif tool == "temporal_normalize":
                # EXPERIMENTAL - see app/timewarp.py's module docstring for
                # the full rationale and what has/hasn't been verified.
                # Must run BEFORE the unconditional watermark stage further
                # down this function. The measured five-seed comparison only
                # established that watermarking first and warping afterward
                # degrades mark recovery; it did not benchmark every possible
                # late re-verification combination. Linear/CNN/LUFS safety
                # passes can still run between this warp and the watermark,
                # but the watermark remains the final signal mutation, so
                # nothing ever warps an already-embedded mark.
                #
                # Applies the SAME warp curve identically to every channel
                # (never independent per-channel curves) - verified directly
                # that independent curves would desync L/R against each
                # other, a much larger and more obviously audible problem
                # than the warp itself.
                from .timewarp import generate_warp_curve
                from scipy.interpolate import interp1d
                n = len(audio)
                # 4ms default: landmark displacement saturates there (see the
                # slider's own note and generate_warp_curve's docstring).
                max_drift_ms = options.get("temporal_max_drift_ms", 4.0)
                # DETERMINISM (direct user report: "why is the CNN output
                # different every run - these are bits, not lottery tickets").
                # This used to pass seed=None, i.e. fresh OS entropy per run.
                # Because the warp runs immediately BEFORE the detector fixes,
                # the CNN optimised against different audio every time and
                # produced a different correction - so identical input, code
                # and settings gave a different delivered file and a different
                # final score on every run. Verified: seed=None produces
                # non-identical output, an explicit seed is bit-identical.
                #
                # The original reason for entropy was sound - a single FIXED
                # curve reused across every release would itself become a
                # repeated, detectable signature. Deriving the seed from the
                # AUDIO CONTENT keeps both properties: the same file always
                # warps the same way (reproducible), while different files
                # still get completely different curves (no shared signature).
                temporal_seed = options.get("temporal_seed")
                if temporal_seed is None:
                    import hashlib
                    digest = hashlib.sha256(
                        np.ascontiguousarray(audio, dtype=np.float32).tobytes()
                    ).digest()
                    temporal_seed = int.from_bytes(digest[:8], "big")
                offsets = generate_warp_curve(n, sr, seed=temporal_seed,
                                               max_drift_ms=max_drift_ms)
                original_t = np.arange(n) / sr
                warped_t = np.clip(original_t + offsets, original_t[0], original_t[-1])
                warped = np.zeros_like(audio)
                for ch in range(audio.shape[1]):
                    interpolator = interp1d(original_t, audio[:, ch], kind="cubic",
                                             bounds_error=False, fill_value=0.0)
                    warped[:, ch] = interpolator(warped_t)
                audio = warped.astype(np.float32)
                info = {
                    "applied": True,
                    "max_drift_ms": max_drift_ms,
                    "seed_mode": ("explicit" if options.get("temporal_seed") is not None
                                  else "content-derived"),
                    "note": "not verified against any real fingerprinting/pattern-matching service",
                }
            elif tool == "true_peak_limit":
                audio, info = chain.true_peak_limit(audio, sr, ceiling_db=options.get("ceiling_db", -1.0))
            else:
                continue

            info["tool"] = tool
            info["label"] = TOOL_LABELS.get(tool, tool)
            info["elapsed_sec"] = round(time.time() - t0, 2)
            steps.append(info)
            job_log(job_id, f"  done ({info['elapsed_sec']}s)")
            # ALWAYS logged right after "done", never before it, and never
            # skipped - every tool gets exactly one status line here (see
            # _tool_status_line's own docstring for why this is centralized
            # instead of scattered across each branch above).
            status_line = _tool_status_line(tool, info)
            if status_line is not None:
                job_log(job_id, f"  {tool}: {status_line}")

        # The blocks below sit outside TOOL_ORDER, so the loop's top-of-stage
        # checkpoint cannot cover them. Check here and again before every
        # potentially expensive re-verification call.
        check_cancelled(job_id)

        # Final re-verification pass: cnn_fix's own correction can disturb
        # linear_fix's precise spectral tuning even when linear_fix ran first
        # and passed its own verification right after it ran (confirmed
        # directly: a verified 1.56% became 9.65% once cnn_fix ran on top of
        # it). Re-score both models on the actual post-chain audio and, if
        # the linear model is still above target, re-run linear_fix ONE more
        # time (cheap relative to cnn_fix) rather than silently shipping a
        # result that's worse than what linear_fix itself already achieved.
        # BUG FIX (direct user report, real production job): this used to
        # require BOTH linear_fix AND cnn_fix to be selected before EITHER
        # re-verification pass below would run at all - so a job that only
        # selected linear_fix got NO post-chain safety net whatsoever.
        # Confirmed directly: linear_fix verified 0.006% mid-chain, but
        # temporal_normalize/true_peak_limit/watermark all ran after it with
        # nothing checking the actual delivered file, which shipped at
        # 9.02% - a large, completely unverified regression, on a job that
        # never even selected cnn_fix. Each recheck below now gates ONLY on
        # its OWN tool being selected, independently - linear_fix alone
        # still gets checked and can still trigger a redo even when cnn_fix
        # was never part of the run.
        if "linear_fix" in tools:
            job_log(job_id, "re-verifying linear model after full chain (later stages can disturb it)")
            recheck_path = OUTPUT_DIR / f"_recheck_{uuid.uuid4().hex[:8]}.wav"
            try:
                save_stereo(recheck_path, audio, sr)
                recheck_score = scorer.linear.predict(str(recheck_path))["probability"]
                job_log(job_id, f"  post-chain linear score: {recheck_score * 100:.3f}%")
                # LINEAR_RECHECK_MARGIN (see its definition above TOOL_ORDER):
                # linear_fix's own target is 0.01, so re-triggering at that
                # exact number meant benign downstream dither alone was
                # enough to force a full needless re-run every time.
                if recheck_score >= 0.01 + LINEAR_RECHECK_MARGIN:
                    check_cancelled(job_id)
                    job_log(job_id, "  above target - re-running linear_fix once more on the final signal")
                    job_set_step(job_id, None, None, "Linear re-verification pass")
                    from .linear_fix import fix_linear
                    t0 = time.time()
                    before_fix = audio.copy()
                    audio, reverify_info = fix_linear(
                        audio, sr, target=options.get("linear_target", 0.01),
                        progress_cb=_cancel_aware_log,
                        step_progress_cb=_linear_step_cb,
                    )
                    _record_detector_overlay(
                        "linear", before_fix, audio
                    )
                    late_mutation_after_temporal = (
                        late_mutation_after_temporal
                        or ("temporal_normalize" in tools and reverify_info.get("applied"))
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
                        # BUG FIX (direct user report, real production job):
                        # this used to call the FULL true_peak_limit, whose
                        # oversample/downsample round-trip (only triggered
                        # when actual limiting is needed) introduces real
                        # broadband reconstruction noise across the whole
                        # signal - measured directly at up to -46dB, exactly
                        # the magnitude this session has repeatedly found
                        # disturbs a fragile CNN/linear-optimized correction.
                        # A re-verification safety pass only needs to
                        # guarantee no raw digital clipping ships, not full
                        # inter-sample-peak-aware, dynamics-preserving
                        # limiting - sample_peak_safety_clamp achieves that
                        # with a flat scale (no resampling at all), and is a
                        # true no-op (byte-identical output) when nothing
                        # actually needs clamping, unlike true_peak_limit's
                        # resample-based approach.
                        job_log(job_id, "re-running peak safety clamp after re-verification pass")
                        t0 = time.time()
                        audio, limiter_info = chain.sample_peak_safety_clamp(
                            audio, ceiling_db=options.get("ceiling_db", -1.0))
                        limiter_info["tool"] = "true_peak_limit_reverify"
                        limiter_info["label"] = "Peak safety clamp (post-reverification safety pass)"
                        limiter_info["elapsed_sec"] = round(time.time() - t0, 2)
                        steps.append(limiter_info)
                        job_log(job_id, f"  done ({limiter_info['elapsed_sec']}s)")
            finally:
                if recheck_path.exists():
                    recheck_path.unlink()

        # CNN re-check must run whenever cnn_fix was selected, not only
        # inside the "linear needed a redo" branch above - confirmed
        # directly on a real production run where linear was ALREADY
        # fine post-chain (0.155%, no redo triggered) and this whole
        # check was skipped entirely, so nothing ever re-verified cnn
        # after true_peak_limit ran. cnn_fix's own internal check had
        # reached a genuine pass (1.09%), but the delivered file scored
        # 99.7% - the limiter alone was enough to disturb it,
        # independent of whether linear needed anything. This is the
        # SAME class of bug fixed once before (a 48%-internal result
        # shipping as 99.9% when linear DID need a redo) - that fix was
        # scoped too narrowly and only closed half the gap. Always check
        # the truly final audio, full stop - but only when cnn_fix was
        # actually part of this run (the outer if above now also covers
        # linear_fix-only jobs, which have nothing CNN-related to
        # recheck here).
        if "cnn_fix" in tools:
            check_cancelled(job_id)
            # THE TIMELINE INVARIANT. Every CNN analysis window position
            # derives from len(audio), so this length IS the certificate.
            # Asserted again at the encode below. A threshold re-score is not
            # sufficient: the fragility is chaotic, not monotonic - a 10ms
            # lead trim measured 18.41% where 100ms measured 0.03%.
            _certified_length = len(audio)
            cnn_recheck_path = OUTPUT_DIR / f"_cnn_recheck_{uuid.uuid4().hex[:8]}.wav"
            try:
                save_stereo(cnn_recheck_path, audio, sr)
                cnn_recheck_score = scorer.cnn.predict(str(cnn_recheck_path))["probability"]
                job_log(job_id, f"  post-chain cnn score: {cnn_recheck_score * 100:.3f}%")
                # re-trigger at 0.08 + CNN_RECHECK_MARGIN, not the raw 0.08
                # real_target itself (see the margin's definition above
                # TOOL_ORDER) - 0.08 alone left zero headroom above what
                # cnn_fix already certifies, so downstream dither from
                # temporal_normalize/true_peak_limit was enough to force a
                # full expensive Thorough-mode re-run on files that never
                # actually regressed. Still well below 0.5 (the model's raw
                # pass/fail boundary), so a genuine loss of safety margin -
                # not just a full regression back to "flagged" - still
                # catches and re-runs; only noise-level movement is ignored.
                if cnn_recheck_score >= 0.08 + CNN_RECHECK_MARGIN:
                    check_cancelled(job_id)
                    job_log(job_id, "  cnn lost its safety margin (or regressed) after later chain stages - re-running cnn_fix once more on the final signal")
                    job_set_step(job_id, None, None, "CNN re-verification pass")
                    from .cnn_fix import fix_cnn
                    t0 = time.time()
                    before_fix = audio.copy()

                    def _cnn_reverify_step_cb(s, mx, surrogate_score, real_check_extra):
                        check_cancelled(job_id)
                        extra = {"score_pct": round(float(surrogate_score) * 100, 4)}
                        if real_check_extra is not None:
                            extra["real_max_score_pct"] = round(float(real_check_extra["real_max_score"]) * 100, 4)
                            extra["windows_failing"] = int(real_check_extra["n_windows_bad"])
                            extra["windows_total"] = int(real_check_extra["n_windows"])
                        job_set_sub_progress(job_id, s, mx, extra=extra)

                    _cnn_reverify_mode = options.get("cnn_mode", "thorough")
                    if _cnn_reverify_mode not in ("simple", "eot", "thorough"):
                        _cnn_reverify_mode = "thorough"
                    audio, cnn_reverify_info = fix_cnn(
                        audio, sr,
                        max_steps=options.get("cnn_max_steps", 300),
                        min_steps=options.get("cnn_min_steps", 100),
                        hop_sec=options.get("cnn_hop_sec", 0.5),
                        progress_cb=_cancel_aware_log,
                        step_progress_cb=_cnn_reverify_step_cb,
                        mode=_cnn_reverify_mode,
                    )
                    _record_detector_overlay(
                        "cnn", before_fix, audio
                    )
                    late_mutation_after_temporal = (
                        late_mutation_after_temporal
                        or ("temporal_normalize" in tools and cnn_reverify_info.get("applied"))
                    )
                    cnn_reverify_info["tool"] = "cnn_fix_reverify"
                    cnn_reverify_info["label"] = "AI-detector fix: CNN model (re-verification pass)"
                    cnn_reverify_info["elapsed_sec"] = round(time.time() - t0, 2)
                    cnn_reverify_info["triggered_by"] = f"post-chain recheck showed {cnn_recheck_score * 100:.3f}%"
                    steps.append(cnn_reverify_info)
                    job_log(job_id, f"  done ({cnn_reverify_info['elapsed_sec']}s)")

                    if "true_peak_limit" in tools:
                        # same fix as the linear-reverify limiter re-run
                        # above - see that comment for the full mechanism
                        # this was found and fixed against (real production
                        # job: a verified 14.17% CNN result regressed to
                        # 77.147% with the OLD full true_peak_limit re-run as
                        # the only step in between).
                        job_log(job_id, "re-running peak safety clamp after cnn re-verification pass")
                        t0 = time.time()
                        audio, limiter_info2 = chain.sample_peak_safety_clamp(
                            audio, ceiling_db=options.get("ceiling_db", -1.0))
                        limiter_info2["tool"] = "true_peak_limit_reverify_cnn"
                        limiter_info2["label"] = "Peak safety clamp (post-cnn-reverification safety pass)"
                        limiter_info2["elapsed_sec"] = round(time.time() - t0, 2)
                        steps.append(limiter_info2)
                        job_log(job_id, f"  done ({limiter_info2['elapsed_sec']}s)")

                        # BUG FIX (found via direct testing, real file):
                        # this second limiter re-run is exactly one more
                        # place that can disturb the just-reverified CNN
                        # result, and nothing checked for it - confirmed
                        # directly: a genuinely verified 2.02% CNN result
                        # regressed to 66.5% by final delivery, with this
                        # limiter re-run as the only step in between.
                        # Same "certify then mutate with no re-check" bug
                        # class already fixed once inside the CNN optimizer
                        # itself (silence guard) and once at the pipeline
                        # level (this very re-verification block) - it just
                        # kept recurring one layer further out each time.
                        # Check honestly here too, and ship what's actually
                        # true rather than a stale "verified" claim.
                        check_cancelled(job_id)
                        final_cnn_check_path = OUTPUT_DIR / f"_final_cnncheck_{uuid.uuid4().hex[:8]}.wav"
                        try:
                            save_stereo(final_cnn_check_path, audio, sr)
                            final_cnn_score = scorer.cnn.predict(str(final_cnn_check_path))["probability"]
                            if final_cnn_score >= 0.08:
                                job_log(job_id, f"  WARNING: cnn regressed to {final_cnn_score * 100:.3f}% "
                                                 f"after the post-cnn-reverification limiter pass - not "
                                                 f"re-running cnn again to avoid an unbounded cnn/limiter "
                                                 f"ping-pong; the delivered file may still be flagged by "
                                                 f"the cnn model")
                        finally:
                            if final_cnn_check_path.exists():
                                final_cnn_check_path.unlink()

                    # this cnn re-run could, in turn, disturb linear again the
                    # same way the ORIGINAL cnn_fix pass did (documented
                    # above: a verified 1.56% became 9.65%) - rather than
                    # ping-pong indefinitely between the two, do ONE final
                    # cheap linear check (no re-optimization) and log honestly
                    # if it regressed rather than silently shipping a result
                    # nothing verified.
                    check_cancelled(job_id)
                    final_linear_check_path = OUTPUT_DIR / f"_final_lincheck_{uuid.uuid4().hex[:8]}.wav"
                    try:
                        save_stereo(final_linear_check_path, audio, sr)
                        final_linear_score = scorer.linear.predict(str(final_linear_check_path))["probability"]
                        if final_linear_score >= 0.01:
                            job_log(job_id, f"  WARNING: linear regressed to {final_linear_score * 100:.3f}% "
                                             f"after the cnn re-verification pass - not re-running again to "
                                             f"avoid an unbounded linear/cnn ping-pong; the delivered file may "
                                             f"still be flagged by the linear model")
                    finally:
                        if final_linear_check_path.exists():
                            final_linear_check_path.unlink()
            finally:
                if cnn_recheck_path.exists():
                    cnn_recheck_path.unlink()

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
        check_cancelled(job_id)
        if "normalize_lufs" in tools:
            target_lufs = options.get("lufs_target", -14.0)
            final_lufs = chain.measure_lufs(audio, sr)
            # BUG FIX (direct user report: an export landed at -13.5 against a
            # -14.0 target with no correction and no warning). The threshold
            # was 0.5dB, but the drift these later stages actually introduce
            # measured 0.48dB on a real track - just under the bar, so the
            # guard never fired and the file shipped half a dB off target.
            #
            # Measured drift through the chain on that file:
            #     after normalize_lufs  -14.00
            #     after multiband       -13.40   (the largest single step)
            #     after true_peak_limit -13.51
            #     after fade            -13.52
            #
            # 0.1dB is below the ~1dB level difference a listener can detect
            # on programme material, so anything above it is worth correcting;
            # the correction itself is one gain multiply plus a peak re-check.
            if np.isfinite(final_lufs) and abs(final_lufs - target_lufs) > 0.1:
                job_log(job_id, f"post-chain LUFS check: {final_lufs:.1f} vs target {target_lufs:.1f} "
                                 f"- correcting drift introduced by later processing stages")
                t0 = time.time()
                late_mutation_after_temporal = (
                    late_mutation_after_temporal or "temporal_normalize" in tools
                )
                # Iterate: an UPWARD correction has to be re-limited (raising
                # gain can push the true peak back over the ceiling the
                # limiter just enforced - the sample-peak clamp below only
                # guards raw digital overflow, not inter-sample peaks), and
                # that re-limiting pulls the loudness back down again.
                # Measured on a dense test signal, a single pass overshot to
                # -14.24 for exactly this reason. Loop until the target is met
                # or an extra pass stops helping, bounded so a track that
                # genuinely cannot hit both targets terminates instead of
                # spinning.
                gain_db = target_lufs - final_lufs
                for _lufs_pass in range(6):
                    current = chain.measure_lufs(audio, sr)
                    if not np.isfinite(current):
                        break
                    step_db = target_lufs - current
                    if abs(step_db) <= 0.1:
                        break
                    audio = audio * (10 ** (step_db / 20))
                    peak = np.abs(audio).max()
                    if peak > 0.999:
                        audio = audio * (0.999 / peak)
                    if step_db > 0 and "true_peak_limit" in tools:
                        audio, _tp_info = chain.true_peak_limit(
                            audio, sr, ceiling_db=options.get("ceiling_db", -1.0)
                        )
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
                    # same fix as the other post-reverification limiter
                    # re-runs - see the linear-reverify comment above for
                    # the full mechanism (real broadband resample noise
                    # disturbing a fragile AI-detector-fix correction).
                    # LUFS correction runs before the AI-detector-fix
                    # reverify passes in TOOL_ORDER, so this specific call
                    # site is lower-risk than the others, but there's no
                    # reason to accept ANY unnecessary resample-noise
                    # injection this late in the chain when a flat clamp
                    # achieves the same real safety guarantee.
                    job_log(job_id, "re-running peak safety clamp after LUFS drift correction")
                    t0 = time.time()
                    audio, limiter_info2 = chain.sample_peak_safety_clamp(
                        audio, ceiling_db=options.get("ceiling_db", -1.0))
                    limiter_info2["tool"] = "true_peak_limit_reverify_lufs"
                    limiter_info2["label"] = "Peak safety clamp (post-LUFS-correction safety pass)"
                    limiter_info2["elapsed_sec"] = round(time.time() - t0, 2)
                    steps.append(limiter_info2)
                    job_log(job_id, f"  done ({limiter_info2['elapsed_sec']}s)")

                # BUG FIX (third adversarial audit round, verified
                # directly): this LUFS drift correction applies a REAL
                # whole-signal gain scale (gain_linear above) - not
                # dither-level noise - after both the linear_fix and
                # cnn_fix post-chain rechecks already ran and certified
                # the signal. Nothing verified whether that gain change
                # itself disturbed either AI-detector score before this
                # fix - the exact "certify then mutate with no re-check"
                # bug class already fixed at several other layers this
                # session (CNN silence guard, the pipeline-level linear/
                # cnn rechecks, the post-CNN-reverification limiter),
                # recurring here at one more layer. A confirmed real
                # gain-only mutation (unlike the earlier resample-noise
                # cases) genuinely warrants a full check using the same
                # margins already established for the other rechecks.
                if "linear_fix" in tools or "cnn_fix" in tools:
                    check_cancelled(job_id)
                    post_lufs_check_path = OUTPUT_DIR / f"_post_lufs_check_{uuid.uuid4().hex[:8]}.wav"
                    try:
                        save_stereo(post_lufs_check_path, audio, sr)
                        if "linear_fix" in tools:
                            post_lufs_linear = scorer.linear.predict(str(post_lufs_check_path))["probability"]
                            job_log(job_id, f"  post-LUFS-correction linear score: {post_lufs_linear * 100:.3f}%")
                            if post_lufs_linear >= 0.01 + LINEAR_RECHECK_MARGIN:
                                check_cancelled(job_id)
                                job_log(job_id, "  LUFS gain change disturbed the linear model - re-running linear_fix once more")
                                job_set_step(job_id, None, None, "Linear re-verification pass")
                                from .linear_fix import fix_linear
                                t0 = time.time()
                                before_fix = audio.copy()

                                def _post_lufs_linear_step_cb(step, mx, score, attempt, max_attempts):
                                    check_cancelled(job_id)
                                    job_set_sub_progress(
                                        job_id, step, mx,
                                        extra={
                                            "score_pct": round(float(score) * 100, 4),
                                            "attempt": attempt,
                                            "max_attempts": max_attempts,
                                        },
                                    )

                                audio, post_lufs_linear_info = fix_linear(
                                    audio, sr, target=options.get("linear_target", 0.01),
                                    progress_cb=_cancel_aware_log,
                                    step_progress_cb=_post_lufs_linear_step_cb,
                                )
                                _record_detector_overlay(
                                    "linear", before_fix, audio
                                )
                                post_lufs_linear_info["tool"] = "linear_fix_reverify_lufs"
                                post_lufs_linear_info["label"] = "AI-detector fix: linear model (post-LUFS-correction pass)"
                                post_lufs_linear_info["elapsed_sec"] = round(time.time() - t0, 2)
                                post_lufs_linear_info["triggered_by"] = f"post-LUFS-correction recheck showed {post_lufs_linear * 100:.3f}%"
                                steps.append(post_lufs_linear_info)
                                job_log(job_id, f"  done ({post_lufs_linear_info['elapsed_sec']}s)")
                                save_stereo(post_lufs_check_path, audio, sr)
                        if "cnn_fix" in tools:
                            post_lufs_cnn = scorer.cnn.predict(str(post_lufs_check_path))["probability"]
                            job_log(job_id, f"  post-LUFS-correction cnn score: {post_lufs_cnn * 100:.3f}%")
                            if post_lufs_cnn >= 0.08 + CNN_RECHECK_MARGIN:
                                check_cancelled(job_id)
                                job_log(job_id, "  LUFS gain change disturbed the cnn model - re-running cnn_fix once more")
                                job_set_step(job_id, None, None, "CNN re-verification pass")
                                from .cnn_fix import fix_cnn
                                t0 = time.time()
                                before_fix = audio.copy()

                                def _post_lufs_cnn_step_cb(s, mx, surrogate_score, real_check_extra):
                                    check_cancelled(job_id)
                                    extra = {"score_pct": round(float(surrogate_score) * 100, 4)}
                                    if real_check_extra is not None:
                                        extra["real_max_score_pct"] = round(float(real_check_extra["real_max_score"]) * 100, 4)
                                        extra["windows_failing"] = int(real_check_extra["n_windows_bad"])
                                        extra["windows_total"] = int(real_check_extra["n_windows"])
                                    job_set_sub_progress(job_id, s, mx, extra=extra)

                                _post_lufs_cnn_mode = options.get("cnn_mode", "thorough")
                                if _post_lufs_cnn_mode not in ("simple", "eot", "thorough"):
                                    _post_lufs_cnn_mode = "thorough"
                                audio, post_lufs_cnn_info = fix_cnn(
                                    audio, sr,
                                    max_steps=options.get("cnn_max_steps", 300),
                                    min_steps=options.get("cnn_min_steps", 100),
                                    hop_sec=options.get("cnn_hop_sec", 0.5),
                                    progress_cb=_cancel_aware_log,
                                    step_progress_cb=_post_lufs_cnn_step_cb,
                                    mode=_post_lufs_cnn_mode,
                                )
                                _record_detector_overlay(
                                    "cnn", before_fix, audio
                                )
                                post_lufs_cnn_info["tool"] = "cnn_fix_reverify_lufs"
                                post_lufs_cnn_info["label"] = "AI-detector fix: CNN model (post-LUFS-correction pass)"
                                post_lufs_cnn_info["elapsed_sec"] = round(time.time() - t0, 2)
                                post_lufs_cnn_info["triggered_by"] = f"post-LUFS-correction recheck showed {post_lufs_cnn * 100:.3f}%"
                                steps.append(post_lufs_cnn_info)
                                job_log(job_id, f"  done ({post_lufs_cnn_info['elapsed_sec']}s)")
                    finally:
                        if post_lufs_check_path.exists():
                            post_lufs_check_path.unlink()

                # the limiter now does real dynamics limiting (only reduces
                # gain where peaks actually exceed ceiling), which preserves
                # LUFS far better than the old flat-scale approach - but on
                # an extremely peaky/already-loud track, hitting the target
                # LUFS AND the true-peak ceiling simultaneously can still be
                # genuinely impossible (there's no amount of limiting that
                # raises quiet passages without also raising the peaks that
                # are already at the ceiling). Rather than silently deliver
                # a file that's still off-target with no explanation, check
                # the truly final LUFS and say so honestly if it didn't
                # fully land on target.
                actual_final_lufs = chain.measure_lufs(audio, sr)
                if np.isfinite(actual_final_lufs) and abs(actual_final_lufs - target_lufs) > 0.5:
                    job_log(job_id, f"  WARNING: delivered file is {actual_final_lufs:.1f} LUFS, "
                                     f"still {abs(actual_final_lufs - target_lufs):.1f}dB from the "
                                     f"{target_lufs:.1f} LUFS target - the true-peak ceiling "
                                     f"({options.get('ceiling_db', -1.0):.1f}dBTP) doesn't leave enough "
                                     f"headroom to reach both targets on this track")

        # Fades are applied HERE - after every gain stage, including the
        # post-chain LUFS drift correction above - rather than as an ordinary
        # entry in TOOL_ORDER. Verified directly why this matters:
        #
        #   * a 3s fade-out shifts the measured loudness of a test tone by
        #     0.57dB (-6.97 -> -7.54 LUFS), so a fade applied earlier makes
        #     normalize_lufs measure a signal that is artificially quiet;
        #   * the post-chain drift correction then sees that same deficit and
        #     multiplies the WHOLE track by a makeup gain, which both partly
        #     undoes the fade and pushes the rest of the track louder than
        #     the user asked for.
        #
        # Running last means the delivered fade is exactly the requested
        # shape. The watermark still runs after this (it is additive and
        # tiny, and must remain the final signal mutation), and the true-peak
        # guard after the watermark only ever reduces gain, so neither can
        # reopen the faded edges.
        if FADE_TOOL in tools:
            check_cancelled(job_id)
            job_log(job_id, f"running: {TOOL_LABELS[FADE_TOOL]}")
            t0 = time.time()
            audio, fade_info = chain.apply_fade(
                audio, sr,
                fade_in_ms=options.get("fade_in_ms", 10),
                fade_out_ms=options.get("fade_out_ms", 3000),
            )
            fade_info["tool"] = FADE_TOOL
            fade_info["label"] = TOOL_LABELS[FADE_TOOL]
            fade_info["elapsed_sec"] = round(time.time() - t0, 2)
            steps.append(fade_info)
            job_log(job_id, f"  done ({fade_info['elapsed_sec']}s)")
            job_log(job_id, f"{FADE_TOOL}: {_tool_status_line(FADE_TOOL, fade_info)}")

        check_cancelled(job_id)
        if late_mutation_after_temporal:
            job_log(
                job_id,
                "temporal_normalize: note (a corrective post-chain stage ran after the warp; "
                "the watermark is still applied last, but this exact combination is unbenchmarked)",
            )

        # BUG FIX (direct user report, real production job): these
        # corrective re-checks used to run AFTER scoring_wav_path was
        # written, the watermark embedded, and out_path encoded - meaning
        # any correction they made here was applied to "audio" in memory
        # but NEVER reflected in either the scores or the actual delivered
        # file, both of which were already written from an earlier, stale
        # version of "audio". Confirmed directly: a job's own results table
        # showed "1 anomaly" while re-analyzing the DOWNLOADED file (built
        # from the stale pre-correction encode) showed 4, and separately
        # showed real leading/trailing silence the job's own trim_silence
        # step had supposedly already removed. Every mutating correction
        # below must run BEFORE scoring_wav_path/watermark/encode - this is
        # the same "certify then mutate with no re-check" bug class fixed
        # several times already this session (CNN silence guard, the
        # linear/cnn post-chain rechecks), just one layer further out:
        # right at the final write, not inside an earlier stage.
        # THE SECOND TRIM IS GONE, deliberately.
        #
        # There used to be a corrective trim_silence here, added on the theory
        # that later chain stages might introduce new silence worth removing.
        # It never had a real job. It runs after the fade, and a 3000ms
        # fade-out ends in silence BY DESIGN - so the only thing it ever found
        # was the fade's own tail, and it deleted it.
        #
        # That is what shipped a file certified at 0.003% as 78.8%: it cut
        # 1.4ms of lead and 311.5ms of trail, and because every CNN analysis
        # window position derives from len(audio), sliding the length slid
        # every window out from under the correction.
        #
        # Measured across real runs, the silence this check finds on a
        # finished file is 1.8-12.8ms, all of it the fade. No stage in this
        # chain adds meaningful silence: trim_silence runs once, at step 2,
        # and the length is fixed from that point forward. If a few
        # milliseconds of fade tail survive to the delivered file, that is
        # correct - it is the fade the user asked for.
        #
        # Do not add this back. If a future stage genuinely introduces
        # silence, fix that stage; do not trim underneath a certificate.

        transients_after = chain.detect_transients(audio, sr)
        late_transients_fixed = []
        if "fix_transients" in tools:
            # Only correct what the chain actually INTRODUCED. Anything with a
            # matching edge in the source is pre-existing material the primary
            # pass already judged (and, for vocal consonants, deliberately
            # skipped) - deleting it here produced an audible duck mid-word.
            n_detected = len(transients_after)
            transients_after = filter_chain_created_transients(
                transients_after, audio, original_audio, sr, lead_samples_trimmed
            )
            n_skipped = n_detected - len(transients_after)
            if n_skipped:
                job_log(job_id, f"  skipped {n_skipped} pre-existing anomal"
                                 f"{'y' if n_skipped == 1 else 'ies'} already present in the "
                                 f"source (not introduced by this chain)")
            _timeline_is_certified = ("cnn_fix" in tools
                                      and _certified_length is not None)
            if transients_after and _timeline_is_certified:
                # Same reasoning as the corrective trim above. fix_transient
                # repairs by interpolating ACROSS a discontinuity, which
                # rewrites a region of samples - and this pass runs after
                # cnn_fix has certified. It cannot be made safe by re-checking.
                #
                # It also has a bad history in its own right: this exact pass
                # was "deleting vocal consonants" (see the comment at the top
                # of filter_chain_created_transients), which is what produced
                # the reported blown-out "t" sounds in a vocal. It now only
                # ever runs on an uncertified timeline.
                job_log(job_id, f"fix_transients: final pass skipped - "
                                f"{len(transients_after)} anomal"
                                f"{'y' if len(transients_after) == 1 else 'ies'} detected, but "
                                f"the timeline is certified by the CNN fix and repairing "
                                f"would rewrite samples underneath it")
                transients_after = []
            elif transients_after:
                check_cancelled(job_id)
                job_log(job_id, f"  found {len(transients_after)} new anomal{'y' if len(transients_after) == 1 else 'ies'} "
                                 f"introduced by later chain stages - running one corrective pass")
                for t in transients_after:
                    audio, _ = chain.fix_transient(audio, sr, t["time_sec"],
                                                     target_peak=options.get("transient_target_peak"))
                    late_transients_fixed.append({"time_sec": t["time_sec"]})
                # Re-count against the same provenance rule the correction
                # used - otherwise the preserved source edges reappear here
                # and the "final" line reports a clean run as still failing.
                transients_after = filter_chain_created_transients(
                    chain.detect_transients(audio, sr), audio, original_audio,
                    sr, lead_samples_trimmed,
                )

            transient_status = "pass" if len(transients_after) == 0 else "check"
            job_log(
                job_id,
                f"fix_transients: final {transient_status} "
                f"({len(transients_after)} anomal{'y' if len(transients_after) == 1 else 'ies'} after full chain)",
            )

        out_id = uuid.uuid4().hex[:12]

        # product watermark: applied unconditionally to every delivered
        # file, not a user-selectable tool in TOOL_ORDER - this is
        # intentional (see app/watermark.py's module docstring for the
        # full design rationale).
        #
        # It runs after certification, which is only acceptable because it is
        # MEASURED INERT rather than assumed to be. On a real certified
        # output: CNN delta exactly 0.00000000pp (identical to 8 decimal
        # places), linear delta -0.0000060pp (down, i.e. the safe direction),
        # and sample count unchanged - so it cannot slide a CNN analysis
        # window, which is the thing that actually breaks a certificate.
        #
        # An earlier version of this comment justified the position by saying
        # it runs last "so nothing downstream can disturb it". That reasoning
        # is backwards: being last means it is the last thing to touch a
        # signal cnn_fix already certified. Do not use it as precedent for
        # placing anything else here - the bar for running after
        # certification is a measurement, not a convenience.
        job_log(job_id, "running: wm")
        # This stage runs OUTSIDE the per-tool loop, so it never reaches that
        # loop's centralized "done (Xs)" emission - it had no timing at all
        # and showed up in the live log as the only stage between two timed
        # ones with no Done line. Time it here so every stage reports
        # uniformly (same for the save and final re-score stages below).
        wm_t0 = time.time()
        try:
            from .watermark import embed_watermark, detect_watermark
            mono_for_mark = audio.mean(axis=1)
            marked_mono = embed_watermark(mono_for_mark, sr)
            mark_delta = marked_mono - mono_for_mark
            audio = audio.copy()
            audio[:, 0] += mark_delta
            audio[:, 1] += mark_delta
            audio = np.clip(audio, -1.0, 1.0)

            # verify it actually round-trips on the just-marked signal,
            # the same "verify, don't just trust the write path" pattern
            # already used for the linear/CNN detector fixes elsewhere in
            # this pipeline - proves the mark is really recoverable on
            # THIS file, not just in isolated module testing.
            wm_found, wm_version, wm_detail = detect_watermark(audio.mean(axis=1), sr)
            # "done" before the result line, matching the per-tool loop's own
            # ordering (done first, then the status line).
            job_log(job_id, f"  done ({round(time.time() - wm_t0, 2)}s)")
            if wm_found:
                job_log(job_id, f"wm: pass (version {wm_version}, "
                                 f"{wm_detail.get('match_fraction', 0) * 100:.0f}% confidence, "
                                 f"method={wm_detail.get('method')})")
            else:
                job_log(job_id, "wm: fail (embedded but could not be re-verified on this file - "
                                 "shipping anyway, footprint measurement only, not a delivery gate)")
        except Exception as e:
            # never let the watermark stage block a user's actual delivery -
            # this is a footprint-measurement feature, not a core function;
            # if it fails for any reason, log it and ship the file anyway.
            # Still emit "done" so the stage is not left dangling in the log
            # on the error path either.
            job_log(job_id, f"  done ({round(time.time() - wm_t0, 2)}s)")
            job_log(job_id, f"wm: error, shipping without it ({e})")

        # BUG FIX (third adversarial audit round, verified directly): the
        # watermark's additive delta can push the TRUE peak (inter-sample
        # reconstruction peak, what -1dBTP actually measures) above the
        # limiter's own advertised ceiling even while staying under 1.0
        # sample-peak - np.clip above only guards raw digital overflow, not
        # true-peak. Confirmed directly: a real limited signal at exactly
        # -1.0dBTP, watermarked, measured 0.10dB OVER that ceiling on the
        # actual delivered signal. If true_peak_limit was selected, its
        # whole advertised guarantee ("the delivered file stays under
        # ceiling_db") is false the moment the watermark (which always
        # runs, unconditionally) adds anything back on top. Unlike the
        # earlier post-CNN-reverification safety passes (where using the
        # FULL true_peak_limit risked destabilizing a still-fragile
        # adversarial correction with resample noise), the watermark is
        # the LAST stage before encode - all AI-detector-fix verification
        # is already complete and nothing downstream will re-check CNN/
        # linear again, so there's no fragile correction left to disturb.
        # The full, real true_peak_limit is the correct, safe tool here.
        if "true_peak_limit" in tools:
            check_cancelled(job_id)
            job_set_step(job_id, None, None, "True-peak limiter (post-watermark safety pass)")
            t0 = time.time()
            post_wm_audio, post_wm_limiter_info = chain.true_peak_limit(
                audio, sr, ceiling_db=options.get("ceiling_db", -1.0))
            if post_wm_limiter_info.get("applied"):
                job_log(job_id, "  watermark pushed the true peak back over ceiling - re-limiting")
                audio = post_wm_audio
                post_wm_limiter_info["tool"] = "true_peak_limit_reverify_watermark"
                post_wm_limiter_info["label"] = "True-peak limiter (post-watermark safety pass)"
                post_wm_limiter_info["elapsed_sec"] = round(time.time() - t0, 2)
                steps.append(post_wm_limiter_info)
                job_log(job_id, f"  done ({post_wm_limiter_info['elapsed_sec']}s)")

        # BUG FIX (adversarial audit, verified directly): the scoring WAV
        # used to be written BEFORE the watermark stage, on the reasoning
        # that the watermark's 10-16kHz notches sit outside both detector
        # models' analysis bands and therefore "shouldn't" move the score -
        # but the very next line of that old comment admitted the whole
        # point of re-scoring the delivered file was to remove doubt rather
        # than rely on that same assumption, while doing exactly the
        # opposite in practice. Confirmed directly with a test that mutates
        # audio inside embed_watermark (a null test purely to prove the
        # PRINCIPLE, not a claim about the real watermark's actual
        # magnitude): the old code reported a score from PCM that provably
        # differed from what was handed to the encoder, 100% of samples
        # mismatched. Encode the real delivered file FIRST, then score
        # THAT file directly - not a WAV proxy of any kind.
        #
        # BUG FIX (second adversarial audit round, verified directly): the
        # previous fix moved the WAV snapshot to after the watermark stage,
        # which fixed the watermark-mutation gap, but a WAV snapshot is
        # STILL not what the user actually receives for lossy formats -
        # for mp3/flac output, real encoder quantization happens AFTER any
        # WAV snapshot, so scores_after never reflected the genuine
        # lossy-encoded bytes. Confirmed directly: a real MP3 round-trip on
        # a 176,400-sample test signal changed 176,344 of those samples
        # (essentially all of them), with a max per-sample error of 0.018 -
        # far larger than DC-offset-level noise, easily enough to move a
        # borderline detector score. scorer.score() decodes via ffmpeg
        # (load_audio_mono), which handles wav/mp3/flac identically - there
        # is no reason to score a WAV proxy AT ALL when the real delivered
        # file can be scored directly, regardless of format.
        resolved_format, format_fallback_warning = resolve_output_format(
            output_format, path, tools)
        if format_fallback_warning:
            job_log(job_id, f"NOTE: {format_fallback_warning}")
        if "cnn_fix" in tools and _certified_length is not None and len(audio) != _certified_length:
            raise RuntimeError(
                f"CERTIFIED TIMELINE CHANGED: certified on {_certified_length} "
                f"samples, about to write {len(audio)} ({len(audio) - _certified_length:+d}). "
                f"Every CNN analysis window position derives from the length, so this "
                f"certificate no longer describes the delivered audio. A stage after "
                f"cnn_fix changed the timeline; it must run before certification."
            )

        job_log(job_id, f"saving output file (format: {resolved_format})")
        # Every output is written as 16-bit PCM, so every file takes a
        # bit-depth reduction whether the user asked for one or not. That is
        # why dither is unconditional and has no card: there is no choice to
        # offer. But it was also invisible - the log said only "saving output
        # file" while a stage that modifies every sample ran underneath it,
        # and while it was silently introducing DC (a truncating quantizer,
        # since fixed). State it.
        job_log(job_id, "  16-bit output: applying TPDF dither before "
                        "quantization (undithered truncation leaves harmonics "
                        "at -24.6dB against the tone; dithered, -43.7dB)")
        save_t0 = time.time()
        out_path = encode_final_output(audio, sr, resolved_format, OUTPUT_DIR / out_id, mp3_mode=mp3_mode)
        scoring_wav_path = out_path

        # align on the same underlying audio content before doing ANY
        # before/after comparison - if silence was trimmed from the front,
        # "audio" (the processed/delivered signal) starts later in the
        # timeline than "original_audio" by lead_samples_trimmed samples.
        # This must happen BEFORE saving the A/B playback "original" file
        # too, not just before the SNR math - otherwise the A/B player's
        # "A" (original) and "B" (fixed) sides are different lengths/offsets
        # from each other, so a listener scrubbing to e.g. 1:30 in each is
        # actually hearing two different moments in the song, and the
        # shared playhead/timeline drifts out of sync between them.
        aligned_original = original_audio[lead_samples_trimmed:]

        orig_path = OUTPUT_DIR / f"{out_id}_orig.wav"
        # The A/B "original" is a delivered artifact the user listens to,
        # and it takes the same 16-bit reduction. Dithering only the
        # processed side would bias the comparison in the product's
        # favour on exactly the quiet material where dither matters.
        save_stereo(orig_path, aligned_original, sr, dither=True)

        # Let the user hear the actual detector corrections in isolation.
        # Keep a floating-point true-level file (so a tiny correction is not
        # rounded away) and a clearly labeled normalized preview for each
        # selected detector plus their combined overlay.
        overlay_info = save_correction_overlays(
            out_id, correction_overlays, sr
        )
        # Covers the whole save stage the log announced: the final encode plus
        # the A/B original and the correction overlays, which is everything
        # written before scoring starts.
        job_log(job_id, f"  done ({round(time.time() - save_t0, 2)}s)")

        job_log(job_id, "re-scoring with AI detectors")
        rescore_t0 = time.time()
        scores_after = scorer.score(str(scoring_wav_path))
        scores_before = scorer.score(str(path))
        job_log(job_id, f"  done ({round(time.time() - rescore_t0, 2)}s)")
        if not scores_after["passes_both"]:
            failing = []
            if not scores_after["passes_linear"]:
                failing.append(f"linear={scores_after['linear_pct']:.2f}%")
            if not scores_after["passes_cnn"]:
                failing.append(f"cnn={scores_after['cnn_pct']:.1f}%")
            job_log(job_id, f"WARNING: final file still flagged by at least one model ({', '.join(failing)})")

        # CERTIFICATION IS THE SCORE ON THE DELIVERED BYTES, AND IT MUST HAVE
        # A MARGIN. scores_after already reads the real encoded output at
        # out_path rather than a pre-encode proxy, which is right - but a
        # score that merely scrapes under the bar is not a certificate.
        #
        # The optimizer targets real_target=0.08 while the pass bar is 0.5, so
        # a delivered file sitting between them has technically "passed" while
        # having almost none of the headroom the optimizer was aiming for. And
        # the sensitivity here is chaotic rather than monotonic - measured, a
        # 10ms trim moved one track to 18.41% while a 100ms trim left it at
        # 0.03% - so "just under 0.5" carries no guarantee that a trivial
        # downstream difference (a different decoder, a re-encode) stays under.
        #
        # So: warn explicitly whenever the delivered file passes without the
        # margin the optimizer was targeting. This is a report, not a silent
        # pass, and it is the difference between "we checked" and "we
        # certified".
        # DENSE CERTIFICATION OF THE DELIVERED BYTES. scorer.score() samples
        # five fixed positions derived from len(audio); a dense scan of every
        # window found 0.12442% at 95.0s on a file the fixed-5 reported as
        # 0.00112%, a 111x under-report, because the worst window fell
        # between two sampled positions. Certification has to measure what it
        # claims to, so the delivered file is scanned in full.
        if "cnn_fix" in tools:
            try:
                dense_worst, dense_at, dense_all = scorer.cnn.scan_dense(
                    str(scoring_wav_path))
                scores_after["cnn_dense_worst_pct"] = dense_worst * 100
                scores_after["cnn_dense_windows"] = len(dense_all)
                scores_after["passes_cnn_dense"] = dense_worst < 0.5
                job_log(job_id,
                        f"  delivered-file dense scan: {len(dense_all)} windows, "
                        f"worst {dense_worst * 100:.4f}% at {dense_at:.1f}s")
                if dense_worst >= 0.5:
                    job_log(job_id,
                            f"WARNING: the delivered file has a window at "
                            f"{dense_at:.1f}s scoring {dense_worst * 100:.1f}% - "
                            f"the five-position score did not sample it")
                elif dense_worst >= CNN_DELIVERED_MARGIN:
                    job_log(job_id,
                            f"NOTE: worst delivered window is "
                            f"{dense_worst * 100:.3f}% at {dense_at:.1f}s, above "
                            f"the {CNN_DELIVERED_MARGIN * 100:.0f}% the optimizer "
                            f"targets - it passes the 50% bar with less margin "
                            f"than the fix aimed for")
            except Exception as exc:
                job_log(job_id, f"  (dense delivered scan unavailable: {exc})")

        if not scores_after.get("passes_cnn_worst", True):
            worst = scores_after.get("cnn_worst_pct", 0.0)
            job_log(job_id,
                    f"WARNING: the delivered file passes on the median window "
                    f"({scores_after['cnn_pct']:.3f}%) but its WORST window "
                    f"scores {worst:.1f}% - the median can hide a failing "
                    f"window, so this file is not safely certified")
        elif scores_after["cnn"]["probability"] >= CNN_DELIVERED_MARGIN:
            job_log(job_id,
                    f"NOTE: delivered file passes but with little margin "
                    f"(cnn={scores_after['cnn_pct']:.3f}% median, "
                    f"{scores_after.get('cnn_worst_pct', 0):.3f}% worst window; "
                    f"bar is 50%, the "
                    f"optimizer targets {CNN_DELIVERED_MARGIN * 100:.0f}%) - "
                    f"it is certified, but a re-encode or a different decoder "
                    f"could move it")

        # BUG FIX (fourth adversarial audit round, verified directly):
        # scores_after was already correctly fixed (earlier this session)
        # to score the REAL delivered file at out_path rather than an
        # in-memory/pre-encode proxy - but every OTHER results-table metric
        # below (LUFS, stereo correlation, DC offset, spectral tilt,
        # waveform, transient count) still measured the in-memory `audio`
        # array directly, never re-decoding the actual encoded output.
        # For lossless (WAV) delivery these are identical, but for lossy
        # (MP3) delivery the real encoded bytes can differ measurably from
        # the pre-encode array (this session already measured real MP3
        # encoder artifacts elsewhere - DC bias, sample-level changes) -
        # meaning a user could see one set of numbers in the results table
        # and a different set on re-analyzing their own downloaded file,
        # the exact "results claim X, re-analysis says Y" contradiction
        # this session has repeatedly had to fix in other tools. Re-decode
        # the actual out_path file (via the same load_stereo used for the
        # original upload) and measure everything below from THAT, not
        # from `audio`.
        try:
            delivered_audio = load_stereo(str(out_path), sr)
        except Exception:
            # if re-decoding the just-written file somehow fails, fall back
            # to the in-memory array rather than crash the whole job at
            # its very last step - the AI scores (the actual safety-
            # critical numbers) are already correctly sourced from
            # out_path independently of this fallback.
            delivered_audio = audio

        n = min(len(aligned_original), len(delivered_audio))
        delta = delivered_audio[:n, 0] - aligned_original[:n, 0]
        orig_rms = np.sqrt(np.mean(aligned_original[:n, 0] ** 2))
        delta_rms = np.sqrt(np.mean(delta ** 2))
        overall_snr = float(20 * np.log10(orig_rms / (delta_rms + 1e-12))) if delta_rms > 0 else None

        tilt_before, freqs_b, psd_b = chain.spectral_tilt_report(aligned_original, sr)
        tilt_after, freqs_a, psd_a = chain.spectral_tilt_report(delivered_audio, sr)
        waveform_before = chain.waveform_peaks(aligned_original, sr)
        waveform_after = chain.waveform_peaks(delivered_audio, sr)

        # the results panel only ever showed a 5-row before/after table even
        # though the pre-processing analysis panel shows a much richer set
        # of stats (stereo correlation, DC offset, transients, spectral
        # tilt) - compute the equivalent AFTER-state numbers here (mirroring
        # exactly what /api/analyze already computes for the before state)
        # so the results panel can show the same depth, not a stripped-down
        # subset of it.
        correlation_before = chain.stereo_correlation(aligned_original)
        correlation_after = chain.stereo_correlation(delivered_audio)
        dc_before = aligned_original.mean(axis=0)
        dc_after = delivered_audio.mean(axis=0)
        # trim_silence/fix_transients post-chain corrections already ran
        # BEFORE scoring_wav_path/watermark/encode above (see the comment
        # there for why) - this final detect_transients call is read-only,
        # purely to report the actual final count for the results table
        # against audio that has already been corrected, not to correct
        # anything itself. Uses delivered_audio (the re-decoded actual
        # output file) for the same reason correlation_after/dc_after do -
        # see the delivered_audio comment above.
        transients_after = chain.detect_transients(delivered_audio, sr)
        # BUG FIX (direct user report: "now you're no longer fixing the
        # transients", results table showing "5 (fixed) / 3 flagged"): this
        # count must exclude the anomalies the corrective pass DELIBERATELY
        # preserved. Those are sharp edges that were already in the source -
        # vocal consonants that compression smoothed until they resembled
        # clicks - and leaving them intact is the correct behaviour, not a
        # failure to fix anything. Counting them here reported a successful
        # run as flagged, which reads as the tool having stopped working.
        if "fix_transients" in tools:
            transients_after = filter_chain_created_transients(
                transients_after, delivered_audio, original_audio, sr,
                lead_samples_trimmed,
            )

        # surface WHERE each fixed transient/pop was found, for the results
        # panel's waveform chart to mark - fix_transients' own step already
        # records each one's time_sec in its "details" list (from
        # chain.fix_transient's return dict), just not as a flat top-level
        # array the frontend can hand straight to the chart.
        transients_found = [
            {"time_sec": d["time_sec"]}
            for s in steps if s.get("tool") == "fix_transients"
            for d in s.get("details", []) if "time_sec" in d
        ] + late_transients_fixed

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
            "lufs_after": chain.measure_lufs(delivered_audio, sr),
            "spectrum_before": {"freqs": freqs_b, "psd_db": psd_b, "tilt": tilt_before},
            "spectrum_after": {"freqs": freqs_a, "psd_db": psd_a, "tilt": tilt_after},
            "waveform_before": waveform_before,
            "waveform_after": waveform_after,
            "transients_found": transients_found,
            "stereo_correlation_before": float(correlation_before),
            "stereo_correlation_after": float(correlation_after),
            "dc_offset_before": {"l": float(dc_before[0]), "r": float(dc_before[1])},
            "dc_offset_after": {"l": float(dc_after[0]), "r": float(dc_after[1])},
            "transients_after_count": len(transients_after),
            "passes_both_after": scores_after["passes_both"],
            "duration_sec": n / sr,
            "correction_overlays": overlay_info,
        }

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result"] = result

        # Prune older jobs' files now that this one is safely recorded. Runs
        # AFTER the result is stored so a cleanup failure can never cost the
        # user the run they just waited for - hence the broad except: this is
        # housekeeping, never a delivery gate.
        try:
            pruned_jobs, freed_bytes = prune_old_outputs()
            if pruned_jobs:
                job_log(job_id, f"cleanup: removed {pruned_jobs} older job(s), "
                                f"freed {freed_bytes / 1e9:.2f}GB")
        except Exception as e:
            job_log(job_id, f"cleanup: skipped ({e})")

        total = time.time() - job_started_at
        mins, secs = divmod(int(total), 60)
        job_log(job_id, f"total processing time: "
                        f"{f'{mins}m {secs}s' if mins else f'{total:.1f}s'}")
        job_log(job_id, "complete")

    except JobCancelled:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "cancelled"
        job_log(job_id, "cancelled by user")

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
            "cancel_requested": False,
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


@app.route("/api/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "unknown job_id"}), 404
        if job["status"] != "running":
            return jsonify({"error": f"job is already {job['status']}, cannot cancel"}), 400
        job["cancel_requested"] = True
    # Cancellation is cooperative, not instant. Pipeline boundaries and the
    # optimizers' progress callbacks observe this flag; a real-model scoring
    # call already in progress must return before its next callback can abort.
    return jsonify({"status": "cancel_requested"})


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
    elif kind.startswith("overlay_"):
        overlay_name = kind.removeprefix("overlay_")
        if overlay_name not in {
            "linear",
            "linear_loud",
            "cnn",
            "cnn_loud",
            "combined",
            "combined_loud",
        }:
            return jsonify({"error": "invalid overlay kind"}), 400
        path = _find_output_path(
            file_id, suffix=f"_overlay_{overlay_name}"
        )
        default_name = (
            f"{file_id}_{overlay_name}_correction.wav"
        )
    else:
        return jsonify({"error": "invalid kind"}), 400

    if path is None or not Path(path).exists():
        return jsonify({"error": "not found"}), 404

    download_name = _safe_download_name(request.args.get("name"), default_name, ext=path.suffix)
    return send_from_directory(path.parent, path.name, as_attachment=True, download_name=download_name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=False, threaded=True)
