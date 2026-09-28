"""Shared test helpers."""

from __future__ import annotations

import os

import lookbox  # noqa: F401  (EXR env var before cv2)
import cv2
import numpy as np


def make_rgba(w: int, h: int, seed: int = 0, alpha: bool = True) -> np.ndarray:
    """Deterministic 8-bit-representable float32 RGBA test image."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 4)).astype(np.float32) / 255.0
    if not alpha:
        img[:, :, 3] = 1.0
    return img


def write_png(path: str, rgba: np.ndarray, bits: int = 8) -> str:
    scale = 255.0 if bits == 8 else 65535.0
    dtype = np.uint8 if bits == 8 else np.uint16
    q = np.round(np.clip(rgba, 0, 1) * scale).astype(dtype)
    assert cv2.imwrite(path, q[:, :, [2, 1, 0, 3]])
    return path


def tmp_png(folder, name: str, w: int, h: int, seed: int) -> str:
    return write_png(os.path.join(str(folder), name), make_rgba(w, h, seed))
