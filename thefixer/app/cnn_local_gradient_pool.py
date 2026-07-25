"""Reusable process pool for exact local-window surrogate gradients.

Each worker evaluates standalone 10-second windows with the existing nnAudio
CQT and converted CNN, preserving the segment-local boundary conditions that
the whole-track dense surrogate cannot reproduce.  Workers accumulate into
independent shared gradient rows; the parent sums those rows deterministically.
"""

from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing import shared_memory
import threading

import numpy as np
import torch
import torch.nn.functional as F

from .cnn_differentiable_v2 import (
    SEGMENT_SAMPLES,
    forward_logit_differentiable,
)


_WORKER_AUDIO_SHM = None
_WORKER_DELTA_SHM = None
_WORKER_GRAD_SHM = None
_WORKER_AUDIO = None
_WORKER_DELTA = None
_WORKER_GRADS = None
_WORKER_LOGIT_TARGET = None


def _close_worker_shared() -> None:
    for shared in (
        _WORKER_AUDIO_SHM,
        _WORKER_DELTA_SHM,
        _WORKER_GRAD_SHM,
    ):
        if shared is not None:
            shared.close()


def _worker_initializer(
    audio_name: str,
    delta_name: str,
    gradient_name: str,
    n_samples: int,
    n_rows: int,
    torch_threads: int,
    target: float,
) -> None:
    global _WORKER_AUDIO_SHM, _WORKER_DELTA_SHM, _WORKER_GRAD_SHM
    global _WORKER_AUDIO, _WORKER_DELTA, _WORKER_GRADS
    global _WORKER_LOGIT_TARGET
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    _WORKER_AUDIO_SHM = shared_memory.SharedMemory(name=audio_name)
    _WORKER_DELTA_SHM = shared_memory.SharedMemory(name=delta_name)
    _WORKER_GRAD_SHM = shared_memory.SharedMemory(name=gradient_name)
    _WORKER_AUDIO = np.ndarray(
        (n_samples,), np.float32, buffer=_WORKER_AUDIO_SHM.buf
    )
    _WORKER_DELTA = np.ndarray(
        (n_samples,), np.float32, buffer=_WORKER_DELTA_SHM.buf
    )
    _WORKER_GRADS = np.ndarray(
        (n_rows, n_samples), np.float32, buffer=_WORKER_GRAD_SHM.buf
    )
    _WORKER_LOGIT_TARGET = torch.logit(
        torch.tensor(target), eps=1e-6
    )
    atexit.register(_close_worker_shared)


def _worker_gradient_chunk(task) -> tuple[float, float]:
    row, work = task
    output = _WORKER_GRADS[row]
    output.fill(0)
    max_score = 0.0
    total_loss = 0.0
    for position, weight in work:
        stop = position + SEGMENT_SAMPLES
        segment_np = (
            _WORKER_AUDIO[position:stop] + _WORKER_DELTA[position:stop]
        ).copy()
        if len(segment_np) != SEGMENT_SAMPLES:
            raise ValueError(f"incomplete CNN gradient window at {position}")
        segment = torch.from_numpy(segment_np).requires_grad_(True)
        logit = forward_logit_differentiable(segment.unsqueeze(0))
        loss = float(weight) * F.leaky_relu(
            logit - _WORKER_LOGIT_TARGET + 1.0,
            negative_slope=0.02,
        )
        gradient = torch.autograd.grad(loss, segment)[0]
        if not torch.isfinite(gradient).all():
            raise ValueError(
                f"non-finite local CNN gradient at window {position}"
            )
        output[position:stop] += gradient.numpy()
        max_score = max(max_score, float(torch.sigmoid(logit.detach())))
        total_loss += float(loss.detach())
    return max_score, total_loss


class LocalGradientPool:
    """Persistent macOS-safe pool for local-window gradient accumulation."""

    def __init__(
        self,
        audio: np.ndarray,
        *,
        workers: int = 5,
        torch_threads: int = 2,
        target: float = 0.05,
        start_method: str = "spawn",
    ):
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be one-dimensional")
        if workers < 1 or torch_threads < 1:
            raise ValueError("worker and thread counts must be positive")
        self.audio = audio
        self.workers = int(workers)
        self._lock = threading.RLock()
        self._closed = False
        n = len(audio)
        self._audio_shm = shared_memory.SharedMemory(
            create=True, size=max(1, audio.nbytes)
        )
        self._delta_shm = shared_memory.SharedMemory(
            create=True, size=max(1, audio.nbytes)
        )
        self._gradient_shm = shared_memory.SharedMemory(
            create=True, size=max(1, self.workers * audio.nbytes)
        )
        self._shared_audio = np.ndarray(
            audio.shape, np.float32, buffer=self._audio_shm.buf
        )
        self._shared_delta = np.ndarray(
            audio.shape, np.float32, buffer=self._delta_shm.buf
        )
        self._shared_gradients = np.ndarray(
            (self.workers, n),
            np.float32,
            buffer=self._gradient_shm.buf,
        )
        self._shared_audio[:] = audio
        self._shared_delta.fill(0)
        self._shared_gradients.fill(0)
        context = mp.get_context(start_method)
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_worker_initializer,
            initargs=(
                self._audio_shm.name,
                self._delta_shm.name,
                self._gradient_shm.name,
                n,
                self.workers,
                int(torch_threads),
                float(target),
            ),
        )

    def gradient(
        self,
        delta: np.ndarray,
        positions: list[int],
        weights: list[float] | None = None,
    ) -> tuple[np.ndarray, float, float]:
        with self._lock:
            if self._closed:
                raise RuntimeError("local CNN gradient pool is closed")
            delta = np.asarray(delta, dtype=np.float32)
            if delta.shape != self.audio.shape:
                raise ValueError("delta shape does not match pool audio")
            if weights is None:
                weights = [1.0] * len(positions)
            if len(weights) != len(positions):
                raise ValueError("positions and weights must have equal length")
            self._shared_delta[:] = delta
            work = list(
                zip(
                    [int(position) for position in positions],
                    [float(weight) for weight in weights],
                )
            )
            chunks = [
                chunk.tolist()
                for chunk in np.array_split(
                    np.asarray(work, dtype=object), self.workers
                )
            ]
            # Converting an object-array row with tolist() yields a list, not
            # the desired tuple, so normalize explicitly for pickling.
            chunks = [
                [(int(item[0]), float(item[1])) for item in chunk]
                for chunk in chunks
            ]
            results = list(
                self._executor.map(
                    _worker_gradient_chunk,
                    list(enumerate(chunks)),
                    chunksize=1,
                )
            )
            gradient = self._shared_gradients.sum(
                axis=0, dtype=np.float32
            )
            return (
                gradient,
                max((result[0] for result in results), default=0.0),
                sum(result[1] for result in results),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            for shared in (
                self._audio_shm,
                self._delta_shm,
                self._gradient_shm,
            ):
                shared.close()
                try:
                    shared.unlink()
                except FileNotFoundError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
