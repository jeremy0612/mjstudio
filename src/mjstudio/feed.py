"""Thread-safe latest-value feed between a worker (producer) and the viewer loop."""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np


class Feed:
    """A lock-guarded snapshot the worker writes and the render loop reads.

    Always holds only the LATEST value (no queue, no backlog): a slow viewer never
    stalls a fast producer, and the viewer always draws the freshest state.
    """

    def __init__(self, log_maxlen: int = 400):
        self._lock = threading.Lock()
        self._qpos: np.ndarray | None = None
        self._info: dict = {}
        self._logs: deque[str] = deque(maxlen=log_maxlen)
        self._frames: dict = {}
        self._t0 = time.time()

    def set_qpos(self, qpos) -> None:
        q = np.asarray(qpos, dtype=float).copy()
        with self._lock:
            self._qpos = q

    def set_frames(self, frames: dict) -> None:
        with self._lock:
            self._frames = dict(frames)

    def info(self, **kw) -> None:
        with self._lock:
            self._info.update(kw)

    def log(self, msg: str) -> None:
        line = f"{time.time() - self._t0:7.2f}s  {msg}"
        with self._lock:
            self._logs.append(line)

    def snapshot(self):
        """Return (qpos|None, info dict, logs list, frames dict) — all copies."""
        with self._lock:
            q = None if self._qpos is None else self._qpos.copy()
            return q, dict(self._info), list(self._logs), dict(self._frames)
