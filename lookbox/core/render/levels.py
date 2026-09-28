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


_NOISE_TILE = 64


def _tpdf_tile() -> np.ndarray:
    rng = np.random.default_rng(1234)
    shape = (_NOISE_TILE, _NOISE_TILE, 4)
    return (rng.random(shape, dtype=np.float32) - rng.random(shape, dtype=np.float32))


_TPDF = _tpdf_tile()  # triangular noise in (−1, 1) LSB, tiled: cheap and deterministic


_noise_cache: dict[tuple[int, int], np.ndarray] = {}


def _noise(h: int, w: int) -> np.ndarray:
    n = _noise_cache.get((h, w))
    if n is None:
        reps = ((h + _NOISE_TILE - 1) // _NOISE_TILE, (w + _NOISE_TILE - 1) // _NOISE_TILE, 1)
        n = np.ascontiguousarray(np.tile(_TPDF, reps)[:h, :w]) + np.float32(0.5)  # +0.5: rounding
        if len(_noise_cache) > 8:
            _noise_cache.clear()
        _noise_cache[(h, w)] = n
    return n


def to_display_bgra(premult: np.ndarray, dither: bool = True) -> np.ndarray:
    """Premultiplied float RGBA → uint8 BGRA (Qt's ARGB32_Premultiplied byte order).

    Dithered like the PNG export so gradients don't band on screen. Pass
    dither=False for untouched 8-bit sources shown 1:1, which should stay exact.
    """
    h, w = premult.shape[:2]
    v = premult * np.float32(255.0)
    if dither:
        v += _noise(h, w)
        np.floor(v, out=v)
    else:
        np.rint(v, out=v)
    np.clip(v, 0, 255, out=v)
    q = np.empty((h, w, 4), dtype=np.uint8)
    a = v[:, :, 3]
    q[:, :, 3] = a
    for dst, src in ((0, 2), (1, 1), (2, 0)):
        q[:, :, dst] = np.minimum(v[:, :, src], a)  # premultiplied colour can't exceed alpha
    return q


def needs_dither(layer, level: float) -> bool:
    """Everything except an untouched image shown at full resolution."""
    from lookbox.core.model import ImageLayer  # local: keeps this module's imports light

    return not (isinstance(layer, ImageLayer) and level >= 1.0 and layer.adjust.is_identity()
                and layer.fade is None and layer.mask is None)


def thumbnail_array(px: np.ndarray, size: int) -> np.ndarray:
    """Fit `px` inside size×size. Strides first so a 6000 px render takes ~2 ms, not 150."""
    h, w = px.shape[:2]
    s = min(size / w, size / h, 1.0)
    tw, th = max(1, round(w * s)), max(1, round(h * s))
    step = max(1, min(w // (tw * 4), h // (th * 4)))  # keep ≥4× headroom for the area pass
    if step > 1:
        px = np.ascontiguousarray(px[::step, ::step])
    return cv2.resize(px, (tw, th), interpolation=cv2.INTER_AREA)
