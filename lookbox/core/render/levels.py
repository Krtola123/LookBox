"""Screen-resolution levels for the interactive preview (ARCHITECTURE §6.3).

Each layer is rendered at a power-of-two "level" just above the size it's
drawn at on screen, so the preview is sharp without shimmering (the display
never downsamples a level by more than 2×) and without rendering pixels
nobody can see.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from lookbox.core.model import Transform

MIN_LEVEL = 1.0 / 64.0


def on_screen_scale(t: Transform, zoom: float) -> float:
    """How many screen pixels one source pixel covers (worst axis)."""
    return max(t.scale_x, t.scale_y) * zoom


def choose_level(screen_scale: float) -> float:
    """Smallest power of two ≥ screen_scale, clamped to [MIN_LEVEL, 1]."""
    if screen_scale >= 1.0:
        return 1.0
    if screen_scale <= MIN_LEVEL:
        return MIN_LEVEL
    return 2.0 ** math.ceil(math.log2(screen_scale))


def to_display_bgra(premult: np.ndarray) -> np.ndarray:
    """Premultiplied float RGBA → uint8 BGRA (Qt's ARGB32_Premultiplied byte order)."""
    q = np.empty(premult.shape[:2] + (4,), dtype=np.uint8)
    # Rounding each channel independently can push a colour above its alpha; clamp.
    a = np.clip(np.rint(premult[:, :, 3] * 255.0), 0, 255)
    q[:, :, 3] = a
    for dst, src in ((0, 2), (1, 1), (2, 0)):
        q[:, :, dst] = np.minimum(np.clip(np.rint(premult[:, :, src] * 255.0), 0, 255), a)
    return q


def thumbnail_array(px: np.ndarray, size: int) -> np.ndarray:
    """Fit `px` inside size×size. Strides first so a 6000 px render takes ~2 ms, not 150."""
    h, w = px.shape[:2]
    s = min(size / w, size / h, 1.0)
    tw, th = max(1, round(w * s)), max(1, round(h * s))
    step = max(1, min(w // (tw * 4), h // (th * 4)))  # keep ≥4× headroom for the area pass
    if step > 1:
        px = np.ascontiguousarray(px[::step, ::step])
    return cv2.resize(px, (tw, th), interpolation=cv2.INTER_AREA)
