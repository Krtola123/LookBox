"""Blend modes (ARCHITECTURE §6.1 "composite"). M1: normal only."""

from __future__ import annotations

import numpy as np


def over(dst: np.ndarray, src: np.ndarray) -> None:
    """In place: premultiplied `src` over premultiplied `dst` (same shape)."""
    dst *= 1.0 - src[:, :, 3:4]
    dst += src


def composite(dst: np.ndarray, src: np.ndarray, mode: str) -> None:
    if mode == "normal":
        over(dst, src)
        return
    raise ValueError(f"Blend mode '{mode}' isn't implemented yet (arrives in M4).")
