"""Mask operations (ARCHITECTURE §9). Qt-free.

A mask is float32 (H, W) in 0–1 at the layer's full source resolution.
Edge controls (shift, feather) are applied live at render time, scaled by the
preview level, so previews match the export.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from lookbox.core.io.images import ImageError


# ------------------------------------------------------------------ storage


def encode_mask_png(mask: np.ndarray) -> bytes:
    """16-bit greyscale PNG (smooth edges survive; fast compression: masks are big)."""
    q = np.round(np.clip(mask, 0.0, 1.0) * 65535.0).astype(np.uint16)
    ok, buf = cv2.imencode(".png", q, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ImageError("Couldn't encode the mask.")
    return buf.tobytes()


def mask_from_pixels(rgba: np.ndarray) -> np.ndarray:
    """A mask asset decodes like any image (grey in RGB); the mask is its first channel."""
    return rgba[:, :, 0]


# ------------------------------------------------------------------ edge controls


def grow_shrink(mask: np.ndarray, px: float) -> np.ndarray:
    """Move the edge outwards (+) or inwards (−) by `px` pixels."""
    r = int(round(abs(px)))
    if r < 1:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask, k) if px > 0 else cv2.erode(mask, k)


def feather(mask: np.ndarray, px: float) -> np.ndarray:
    """Soften the edge; `px` ≈ the width of the soft band (sigma = px / 2)."""
    sigma = px / 2.0
    if sigma < 0.3:
        return mask
    from lookbox.core.render.adjust import blur  # fast large-radius blur, shared

    return np.clip(blur(mask, sigma), 0.0, 1.0)


def mask_at_level(mask: np.ndarray, size: tuple[int, int], level: float, shift: float,
                  feather_px: float, invert: bool) -> np.ndarray:
    """Source-resolution mask → the size a preview level needs, with live edge controls.
    Pixel amounts scale with the level so every level looks like the downscaled export."""
    m = mask
    if (m.shape[1], m.shape[0]) != size:
        interp = cv2.INTER_AREA if size[0] < m.shape[1] else cv2.INTER_LINEAR
        m = cv2.resize(m, size, interpolation=interp)
    m = grow_shrink(m, shift * level)
    m = feather(m, feather_px * level)
    return (1.0 - m) if invert else m


# ------------------------------------------------------------------ refine edge


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. guided filter (grey guide): smooths `src` while snapping its edges to
    the guide's edges. Used to fit a model's coarse mask to real hair/fur edges."""
    def box(x):
        return cv2.boxFilter(x, -1, (2 * radius + 1, 2 * radius + 1), borderType=cv2.BORDER_REFLECT)

    mean_i, mean_p = box(guide), box(src)
    cov_ip = box(guide * src) - mean_i * mean_p
    var_i = box(guide * guide) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def refine_edge(mask: np.ndarray, rgb: np.ndarray, model_size: int = 1024) -> np.ndarray:
    """Snap an upsampled model mask to the photo's edges (optional: see ARCHITECTURE §10.2).

    Measured in M6: on clean edges with ordinary photo noise this makes masks slightly
    WORSE (it copies noise into the mask), so it's off by default and offered for hair/fur,
    where the model's low-res output misses fine strands. The radius follows how far the
    model's output was upscaled, since that's the scale of its softness."""
    h, w = mask.shape
    radius = max(2, int(round(2.0 * max(h, w) / model_size)))
    guide = np.ascontiguousarray(rgb[:, :, :3] @ np.array([0.299, 0.587, 0.114], np.float32))
    out = guided_filter(guide.astype(np.float32), mask.astype(np.float32), radius, 1e-3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ brush


def brush_weights(d: np.ndarray, radius: float, hardness: float) -> np.ndarray:
    """1 inside the hard core, smooth fall-off to 0 at `radius`."""
    inner = radius * min(max(hardness, 0.0), 0.99)
    t = np.clip((d - inner) / max(radius - inner, 1e-6), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def stamp(mask: np.ndarray, cx: float, cy: float, radius: float, hardness: float, restore: bool) -> tuple[int, int, int, int] | None:
    """Paint one round dab in place. Restore = towards 1 (show), else towards 0 (hide).
    Uses max/min so overlapping dabs in one stroke don't build up. Returns the dirty rect."""
    h, w = mask.shape
    x0, x1 = max(0, int(math.floor(cx - radius))), min(w, int(math.ceil(cx + radius)) + 1)
    y0, y1 = max(0, int(math.floor(cy - radius))), min(h, int(math.ceil(cy + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    ys = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
    xs = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5
    wgt = brush_weights(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2), radius, hardness).astype(np.float32)
    region = mask[y0:y1, x0:x1]
    if restore:
        np.maximum(region, wgt, out=region)
    else:
        np.minimum(region, 1.0 - wgt, out=region)
    return x0, y0, x1, y1


def stroke(mask: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], radius: float,
           hardness: float, restore: bool) -> tuple[int, int, int, int] | None:
    """Dabs along a segment, spaced a quarter of the radius apart (no gaps at speed)."""
    dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n = max(1, int(math.ceil(dist / max(radius * 0.25, 0.5))))
    rect = None
    for i in range(n + 1):
        f = i / n
        r = stamp(mask, p0[0] + (p1[0] - p0[0]) * f, p0[1] + (p1[1] - p0[1]) * f, radius, hardness, restore)
        if r is not None:
            rect = r if rect is None else (min(rect[0], r[0]), min(rect[1], r[1]),
                                           max(rect[2], r[2]), max(rect[3], r[3]))
    return rect


# ------------------------------------------------------------------ combining selections

COMBINE_MODES = ("replace", "add", "subtract")


def combine(current: np.ndarray, new: np.ndarray, mode: str) -> np.ndarray:
    """Merge a new selection into a mask: replace it, add to it (Shift), or take it
    away (Alt). Soft edges combine like Photoshop's (max / min with the inverse)."""
    if mode == "replace":
        return new.astype(np.float32, copy=True)
    if mode == "add":
        return np.maximum(current, new)
    if mode == "subtract":
        return np.minimum(current, 1.0 - new)
    raise ValueError(f"Unknown combine mode: {mode!r}")
