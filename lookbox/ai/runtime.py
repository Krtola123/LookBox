"""ONNX Runtime sessions (ARCHITECTURE §10.1). Qt-free.

GPU first through DirectML (any DX12 GPU: RTX 4060, RX 570, integrated),
falling back to the CPU with a one-line notice, both when the session can't be
created and when the first GPU run fails (e.g. out of VRAM). Only one model is
kept loaded at a time, which suits a 4 GB card.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

GPU = "DmlExecutionProvider"
CPU = "CPUExecutionProvider"


def _ort():
    import onnxruntime  # imported lazily: the app starts fine without it

    return onnxruntime


def create_session(path: str, prefer_gpu: bool = True, ort: Any = None) -> tuple[Any, str, str | None]:
    """(session, provider actually used, notice for the user or None)."""
    ort = ort or _ort()
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False  # DirectML requirements (ORT docs)
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    notice = None
    if prefer_gpu:
        if GPU in ort.get_available_providers():
            try:
                s = ort.InferenceSession(path, sess_options=opts, providers=[GPU, CPU])
                return s, s.get_providers()[0], None
            except Exception as exc:  # reported, then CPU
                notice = f"GPU unavailable ({exc}); using the CPU, which is slower."
        else:
            notice = "No GPU runtime found (install onnxruntime-directml); using the CPU, which is slower."
    s = ort.InferenceSession(path, sess_options=opts, providers=[CPU])
    return s, CPU, notice


class ModelManager:
    """Holds at most one loaded model. Thread-safe; runs happen in worker threads."""

    def __init__(self, ort: Any = None) -> None:
        self._ort = ort
        self._lock = threading.Lock()
        self._key: tuple[str, str] | None = None  # (path, provider)
        self._session = None
        self.provider: str | None = None
        self.notice: str | None = None

    def unload(self) -> None:
        with self._lock:
            self._session, self._key, self.provider = None, None, None

    def _load(self, path: str, prefer_gpu: bool) -> None:
        if self._session is not None and self._key and self._key[0] == path and (prefer_gpu or self.provider == CPU):
            return
        self._session = None  # free the previous model before loading (low VRAM)
        self._session, self.provider, self.notice = create_session(path, prefer_gpu, self._ort)
        self._key = (path, self.provider)

    def run(self, path: str, tensor: np.ndarray) -> np.ndarray:
        """First output for a single-input model. A failed GPU run retries on the CPU."""
        with self._lock:
            self._load(path, prefer_gpu=True)
            name = self._session.get_inputs()[0].name
            try:
                return np.asarray(self._session.run(None, {name: tensor})[0])
            except Exception as exc:
                if self.provider == CPU:
                    raise
                self._load_cpu(path, f"The GPU run failed ({exc}); retried on the CPU.")
                return np.asarray(self._session.run(None, {name: tensor})[0])

    def input_size(self, path: str, default: int) -> int:
        """The model's fixed input side if it declares one (e.g. 1024), else `default`."""
        with self._lock:
            self._load(path, prefer_gpu=True)
            shape = self._session.get_inputs()[0].shape
        dims = [d for d in shape[-2:] if isinstance(d, int) and d > 0]
        return dims[0] if len(dims) == 2 and dims[0] == dims[1] else default

    def _load_cpu(self, path: str, notice: str) -> None:
        self._session = None
        self._session, self.provider, _ = create_session(path, prefer_gpu=False, ort=self._ort)
        self._key = (path, self.provider)
        self.notice = notice
