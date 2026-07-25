"""Benchmark exact local-window CNN gradient accumulation across processes."""

from __future__ import annotations

import argparse
import atexit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import time

import numpy as np
import torch
import torch.nn.functional as F

from app.cnn_differentiable_v2 import (
    SEGMENT_SAMPLES,
    forward_logit_differentiable,
    load_audio_mono,
)
from app.cnn_wholetrack_optimizer_v2 import build_sliding_windows


_AUDIO_SHM = None
_DELTA_SHM = None
_GRAD_SHM = None
_AUDIO = None
_DELTA = None
_GRADS = None
_TARGET = None


def _close_worker_shared() -> None:
    for shared in (_AUDIO_SHM, _DELTA_SHM, _GRAD_SHM):
        if shared is not None:
            shared.close()


def _init_worker(
    audio_name: str,
    delta_name: str,
    grad_name: str,
    n_samples: int,
    n_rows: int,
    torch_threads: int,
    target: float,
) -> None:
    global _AUDIO_SHM, _DELTA_SHM, _GRAD_SHM
    global _AUDIO, _DELTA, _GRADS, _TARGET
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    _AUDIO_SHM = shared_memory.SharedMemory(name=audio_name)
    _DELTA_SHM = shared_memory.SharedMemory(name=delta_name)
    _GRAD_SHM = shared_memory.SharedMemory(name=grad_name)
    _AUDIO = np.ndarray((n_samples,), np.float32, buffer=_AUDIO_SHM.buf)
    _DELTA = np.ndarray((n_samples,), np.float32, buffer=_DELTA_SHM.buf)
    _GRADS = np.ndarray((n_rows, n_samples), np.float32, buffer=_GRAD_SHM.buf)
    _TARGET = torch.logit(torch.tensor(target), eps=1e-6)
    atexit.register(_close_worker_shared)


def _gradient_chunk(task) -> tuple[float, float]:
    row, positions = task
    output = _GRADS[row]
    output.fill(0)
    max_score = 0.0
    total_loss = 0.0
    for position in positions:
        segment_np = (
            _AUDIO[position : position + SEGMENT_SAMPLES]
            + _DELTA[position : position + SEGMENT_SAMPLES]
        ).copy()
        segment = torch.from_numpy(segment_np).requires_grad_(True)
        logit = forward_logit_differentiable(segment.unsqueeze(0))
        loss = F.leaky_relu(logit - _TARGET + 1.0, negative_slope=0.02)
        gradient = torch.autograd.grad(loss, segment)[0]
        output[position : position + SEGMENT_SAMPLES] += gradient.numpy()
        max_score = max(max_score, float(torch.sigmoid(logit.detach())))
        total_loss += float(loss.detach())
    return max_score, total_loss


class GradientPool:
    def __init__(self, audio: np.ndarray, workers: int, torch_threads: int):
        self.audio = np.asarray(audio, dtype=np.float32)
        self.workers = workers
        n = len(self.audio)
        self.audio_shm = shared_memory.SharedMemory(create=True, size=self.audio.nbytes)
        self.delta_shm = shared_memory.SharedMemory(create=True, size=self.audio.nbytes)
        self.grad_shm = shared_memory.SharedMemory(
            create=True, size=workers * self.audio.nbytes
        )
        self.shared_audio = np.ndarray(
            self.audio.shape, np.float32, buffer=self.audio_shm.buf
        )
        self.shared_delta = np.ndarray(
            self.audio.shape, np.float32, buffer=self.delta_shm.buf
        )
        self.shared_grads = np.ndarray(
            (workers, n), np.float32, buffer=self.grad_shm.buf
        )
        self.shared_audio[:] = self.audio
        self.shared_delta.fill(0)
        self.shared_grads.fill(0)
        context = mp.get_context("spawn")
        self.executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(
                self.audio_shm.name,
                self.delta_shm.name,
                self.grad_shm.name,
                n,
                workers,
                torch_threads,
                0.05,
            ),
        )

    def gradient(self, positions: list[int]) -> tuple[np.ndarray, float]:
        chunks = [
            [int(value) for value in chunk]
            for chunk in np.array_split(positions, self.workers)
        ]
        results = list(
            self.executor.map(
                _gradient_chunk,
                list(enumerate(chunks)),
                chunksize=1,
            )
        )
        return self.shared_grads.sum(axis=0), max(value[0] for value in results)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        for shared in (self.audio_shm, self.delta_shm, self.grad_shm):
            shared.close()
            shared.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    audio = load_audio_mono(args.audio)
    positions, _ = build_sliding_windows(len(audio), hop_sec=0.5)
    positions = positions[: args.windows]
    pool = GradientPool(audio, args.workers, args.torch_threads)
    try:
        for run in range(2):
            started = time.perf_counter()
            gradient, max_score = pool.gradient(positions)
            print(
                {
                    "run": run,
                    "seconds": time.perf_counter() - started,
                    "windows": len(positions),
                    "max_score": max_score,
                    "gradient_norm": float(np.linalg.norm(gradient)),
                },
                flush=True,
            )
    finally:
        pool.close()


if __name__ == "__main__":
    main()
