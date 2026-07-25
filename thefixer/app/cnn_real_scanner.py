"""Parallel, exact CNN real-model scoring.

This module intentionally has no torch/onnx2torch/nnAudio imports.  That keeps
``spawn`` workers on macOS cheap and, more importantly, prevents a worker from
re-running the differentiable model's module-level initialization.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import yaml
from scipy.fft import dct


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _cnn_parameters():
    with open(MODELS_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)
    return (
        config["audio"]["sample_rate"],
        config["cqt"]["fmin"],
        config["cqt"]["n_bins"],
        config["cqt"]["bins_per_octave"],
        config["cqt"]["hop_length"],
        config["cepstrum"]["n_coeffs"],
    )


def score_real_segment_with_session(
    audio_np,
    session,
    input_name,
    *,
    sr,
    fmin,
    n_bins,
    bins_per_octave,
    hop_length,
    n_coeffs,
):
    """Score one segment using the exact librosa + ONNX certificate path."""
    cepstrum = extract_real_cepstrum(
        audio_np,
        sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        n_coeffs=n_coeffs,
    )
    batch = cepstrum[np.newaxis, np.newaxis, :, :].astype(np.float32)
    output = session.run(None, {input_name: batch})[0]
    return float(1 / (1 + np.exp(-output[0, 0])))


def extract_real_cepstrum(
    audio_np,
    *,
    sr,
    fmin,
    n_bins,
    bins_per_octave,
    hop_length,
    n_coeffs,
):
    """Extract the exact librosa CQT/DCT feature used by the certificate."""
    cqt = librosa.cqt(
        audio_np,
        sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
    )
    cqt_mag = np.abs(cqt)
    log_cqt = np.log(cqt_mag + 1e-6)
    return dct(log_cqt, type=2, axis=0, norm="ortho")[:n_coeffs, :]


_WORKER_AUDIO = None
_WORKER_SHM = None
_WORKER_SESSION = None
_WORKER_INPUT_NAME = None
_WORKER_PARAMS = None


def _worker_initializer(
    shm_name,
    n_samples,
    dtype_str,
    model_path,
    params,
    onnx_threads,
):
    """Attach shared audio and create one ONNX session in the worker."""
    global _WORKER_AUDIO, _WORKER_SHM, _WORKER_SESSION, _WORKER_INPUT_NAME, _WORKER_PARAMS

    _WORKER_SHM = shared_memory.SharedMemory(name=shm_name)
    _WORKER_AUDIO = np.ndarray(
        (n_samples,), dtype=np.dtype(dtype_str), buffer=_WORKER_SHM.buf
    )

    # The pool provides process-level parallelism.  One ONNX thread per
    # worker avoids oversubscription and is the selected deterministic setting
    # for both the parent certificate and these workers.
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(onnx_threads)
    session_options.inter_op_num_threads = int(onnx_threads)
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    _WORKER_SESSION = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    _WORKER_INPUT_NAME = _WORKER_SESSION.get_inputs()[0].name
    _WORKER_PARAMS = params


def _worker_score(task):
    index, start, stop = task
    # This is deliberately the same slice expression used by the sequential
    # callers.  It preserves short and out-of-bounds-window behavior exactly;
    # an invalid empty slice raises from librosa and is reported to the caller.
    segment = _WORKER_AUDIO[start:stop]
    score = score_real_segment_with_session(
        segment,
        _WORKER_SESSION,
        _WORKER_INPUT_NAME,
        sr=_WORKER_PARAMS[0],
        fmin=_WORKER_PARAMS[1],
        n_bins=_WORKER_PARAMS[2],
        bins_per_octave=_WORKER_PARAMS[3],
        hop_length=_WORKER_PARAMS[4],
        n_coeffs=_WORKER_PARAMS[5],
    )
    return index, score


class ParallelRealScoreScanner:
    """Reusable ordered process scanner for exact CNN real-model scores.

    A scanner instance is intentionally not used concurrently: each scan
    rewrites its shared audio buffer.  Calls may be repeated, and the worker
    pool/session stays alive when the audio shape and dtype stay the same.
    """

    def __init__(
        self,
        workers=None,
        *,
        start_method=None,
        model_path=None,
        params=None,
        onnx_threads=1,
    ):
        self.workers = int(workers or min(8, os.cpu_count() or 1))
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if start_method is None:
            # fork-after-import is unsafe for the native libraries used here;
            # spawn is also the default we need to exercise on macOS.
            start_method = "spawn" if sys.platform == "darwin" else "spawn"
        self.start_method = start_method
        self.model_path = str(model_path or MODELS_DIR / "cnn_detector.onnx")
        self.params = tuple(params or _cnn_parameters())
        self.onnx_threads = int(onnx_threads)
        if self.onnx_threads < 1:
            raise ValueError("onnx_threads must be at least 1")
        self._executor = None
        self._shared = None
        self._shared_shape = None
        self._shared_dtype = None
        self._lock = threading.RLock()

    def _shutdown_pool(self):
        executor, shm = self._executor, self._shared
        self._executor = None
        self._shared = None
        self._shared_shape = None
        self._shared_dtype = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if shm is not None:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def close(self):
        with self._lock:
            self._shutdown_pool()

    def _ensure_pool(self, audio):
        shape = audio.shape
        dtype = audio.dtype.str
        if (
            self._executor is not None
            and self._shared_shape == shape
            and self._shared_dtype == dtype
        ):
            return

        self._shutdown_pool()
        # SharedMemory does not accept a zero-byte allocation.  The ndarray
        # can still have shape (0,), and a zero-length window will fail in the
        # worker exactly as the sequential librosa call does.
        self._shared = shared_memory.SharedMemory(create=True, size=max(1, audio.nbytes))
        self._shared_shape = shape
        self._shared_dtype = dtype
        context = mp.get_context(self.start_method)
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_worker_initializer,
            initargs=(
                self._shared.name,
                len(audio),
                dtype,
                self.model_path,
                self.params,
                self.onnx_threads,
            ),
        )

    def scan(self, audio_np, positions, segment_length):
        """Return scores in the exact order of ``positions``.

        Each segment is extracted as ``audio_np[pos:pos + segment_length]``.
        No windows are dropped, clipped, padded, or reordered.
        """
        positions = [int(pos) for pos in positions]
        if not positions:
            return []
        audio = np.asarray(audio_np)
        if audio.ndim != 1:
            raise ValueError("audio_np must be a one-dimensional array")
        if not audio.flags.c_contiguous:
            audio = np.ascontiguousarray(audio)

        with self._lock:
            self._ensure_pool(audio)
            shared_audio = np.ndarray(audio.shape, dtype=audio.dtype, buffer=self._shared.buf)
            shared_audio[...] = audio
            tasks = [
                (index, pos, pos + int(segment_length))
                for index, pos in enumerate(positions)
            ]
            try:
                # executor.map is ordered even as workers finish out of order.
                results = list(self._executor.map(_worker_score, tasks, chunksize=1))
            except BaseException as exc:
                # A failed initializer or task can leave a broken pool.  Tear
                # it down so the same reusable scanner can recover on the next
                # valid call, then preserve the original cause for diagnosis.
                self._shutdown_pool()
                raise RuntimeError("parallel CNN real-score worker failed") from exc
        results.sort(key=lambda item: item[0])
        return [float(score) for _, score in results]

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()

    def __del__(self):
        try:
            self._shutdown_pool()
        except Exception:
            pass


_DEFAULT_SCANNER = None
_DEFAULT_SCANNER_LOCK = threading.Lock()


def get_default_real_score_scanner():
    global _DEFAULT_SCANNER
    with _DEFAULT_SCANNER_LOCK:
        if _DEFAULT_SCANNER is None:
            _DEFAULT_SCANNER = ParallelRealScoreScanner()
        return _DEFAULT_SCANNER


@atexit.register
def _close_default_scanner():
    if _DEFAULT_SCANNER is not None:
        _DEFAULT_SCANNER.close()
