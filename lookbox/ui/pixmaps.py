"""Layer thumbnails for the layers panel."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap

from lookbox.core.render.levels import thumbnail_array


def to_qimage(rgba: np.ndarray) -> QImage:
    """Straight-alpha float32 RGBA (0–1) → QImage (owns its memory)."""
    u8 = np.ascontiguousarray(np.round(np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8))
    h, w = u8.shape[:2]
    img = QImage(u8.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return img.copy()  # detach from the numpy buffer before it's freed


class ThumbCache:
    """One thumbnail per (asset id, crop, size). Assets are immutable, so entries never go stale."""

    def __init__(self) -> None:
        self._thumbs: dict[tuple, QPixmap] = {}

    def clear(self) -> None:
        self._thumbs.clear()

    def get(self, key: tuple, pixels: np.ndarray, size: int) -> QPixmap:
        k = key + (size,)
        pm = self._thumbs.get(k)
        if pm is None:
            pm = QPixmap.fromImage(to_qimage(thumbnail_array(pixels, size)))
            self._thumbs[k] = pm
        return pm
