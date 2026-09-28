"""numpy float32 RGBA → Qt images, with a small cache keyed by asset."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


def to_qimage(rgba: np.ndarray) -> QImage:
    """Straight-alpha float32 RGBA (0–1) → QImage (owns its memory)."""
    u8 = np.ascontiguousarray(np.round(np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8))
    h, w = u8.shape[:2]
    img = QImage(u8.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return img.copy()  # detach from the numpy buffer before it's freed


class PixmapCache:
    """One QPixmap per (asset id, crop). Assets are immutable, so entries never go stale."""

    def __init__(self) -> None:
        self._full: dict[tuple, QPixmap] = {}
        self._thumbs: dict[tuple, QPixmap] = {}

    def clear(self) -> None:
        self._full.clear()
        self._thumbs.clear()

    def full(self, key: tuple, pixels: np.ndarray) -> QPixmap:
        pm = self._full.get(key)
        if pm is None:
            pm = QPixmap.fromImage(to_qimage(pixels))
            self._full[key] = pm
        return pm

    def thumb(self, key: tuple, pixels: np.ndarray, size: int) -> QPixmap:
        k = key + (size,)
        pm = self._thumbs.get(k)
        if pm is None:
            pm = self.full(key, pixels).scaled(
                size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self._thumbs[k] = pm
        return pm
