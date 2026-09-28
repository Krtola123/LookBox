"""Layer effects (ARCHITECTURE §8): layer blur, drop shadow (incl. floor/contact
squash), outer glow, outline.

Input is the layer's premultiplied pixels at a preview level; output is a
*padded* image (effects reach beyond the layer box) plus the padding in level
pixels. Effect sizes are canvas pixels and the shadow direction is canvas
space, so they're converted through the layer's scale/rotation/flip here.
That's why render_key includes those transform parts when effects are on.

Stack, bottom → top: shadow, glow, outside outline, layer, centred outline.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from lookbox.core.model import DropShadow, Effects, Glow, Outline, Transform
from lookbox.core.render.adjust import blur

SIGMA_PER_BLUR = 0.5  # "Blur 30 px" = Gaussian sigma 15 px: the visible soft edge ≈ the number
REACH_SIGMAS = 3.0


def layer_scale(t: Transform) -> float:
    """One number for 'how big is the layer on the canvas' (geometric mean of the axes)."""
    return math.sqrt(max(t.scale_x * t.scale_y, 1e-12))


def canvas_vec_to_local(t: Transform, vx: float, vy: float) -> tuple[float, float]:
    """A canvas-space vector expressed in the layer's own (pre-transform) pixels."""
    r = math.radians(t.rotation_deg)
    c, s = math.cos(r), math.sin(r)
    lx, ly = c * vx + s * vy, -s * vx + c * vy  # undo rotation (Rᵀ)
    fx = -1.0 if t.flip_h else 1.0
    fy = -1.0 if t.flip_v else 1.0
    return lx / (t.scale_x * fx), ly / (t.scale_y * fy)


def _shadow_offset(sh: DropShadow, t: Transform, level: float) -> tuple[float, float]:
    a = math.radians(sh.angle_deg)
    lx, ly = canvas_vec_to_local(t, math.cos(a) * sh.distance, math.sin(a) * sh.distance)
    return lx * level, ly * level


def padding(fx: Effects, t: Transform, level: float) -> int:
    """Level pixels needed on every side so nothing gets clipped."""
    k = level / layer_scale(t)
    reach = 0.0
    if fx.blur > 0:
        reach = max(reach, fx.blur * SIGMA_PER_BLUR * REACH_SIGMAS * k)
    if fx.shadow is not None:
        ox, oy = _shadow_offset(fx.shadow, t, level)
        soft = (fx.shadow.spread + fx.shadow.blur * SIGMA_PER_BLUR * REACH_SIGMAS) * k
        reach = max(reach, math.hypot(ox, oy) + soft)
    if fx.glow is not None:
        reach = max(reach, (fx.glow.spread + fx.glow.blur * SIGMA_PER_BLUR * REACH_SIGMAS) * k)
    if fx.outline is not None:
        reach = max(reach, fx.outline.width * k)
    return int(math.ceil(reach)) + 2 if reach > 0 else 0


def _dilate(alpha: np.ndarray, radius: float) -> np.ndarray:
    if radius < 0.5:
        return alpha
    r = int(round(radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(alpha, kernel)


def _tint(alpha: np.ndarray, color, opacity: float) -> np.ndarray:
    """Alpha mask → premultiplied RGBA in `color`."""
    a = alpha * np.float32(color[3] * opacity)
    out = np.empty(alpha.shape + (4,), np.float32)
    for i in range(3):
        out[:, :, i] = a * np.float32(color[i])
    out[:, :, 3] = a
    return out


def _over(dst: np.ndarray, src: np.ndarray) -> None:
    dst *= 1.0 - src[:, :, 3:4]
    dst += src


def shadow_alpha(alpha: np.ndarray, sh: DropShadow, t: Transform, level: float, k: float,
                 box_bottom: float) -> np.ndarray:
    a = _dilate(alpha, sh.spread * k)
    q = min(max(sh.squash, 0.0), 0.95)
    if q > 0:  # flatten towards the layer's bottom edge: y_src = bottom − (bottom − y)/(1 − q)
        s = 1.0 / (1.0 - q)
        m = np.float32([[1, 0, 0], [0, s, box_bottom * (1 - s)]])
        a = cv2.warpAffine(a, m, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderValue=0)
    a = blur(a, sh.blur * SIGMA_PER_BLUR * k)
    ox, oy = _shadow_offset(sh, t, level)
    if abs(ox) > 1e-3 or abs(oy) > 1e-3:
        a = cv2.warpAffine(a, np.float32([[1, 0, ox], [0, 1, oy]]), (a.shape[1], a.shape[0]),
                           flags=cv2.INTER_LINEAR, borderValue=0)
    return np.clip(a, 0.0, 1.0)


def glow_alpha(alpha: np.ndarray, g: Glow, k: float) -> np.ndarray:
    return np.clip(blur(_dilate(alpha, g.spread * k), g.blur * SIGMA_PER_BLUR * k), 0.0, 1.0)


def outline_alpha(alpha: np.ndarray, o: Outline, k: float) -> np.ndarray:
    """Anti-aliased ring from distance transforms (outside, or straddling the edge)."""
    inside = alpha >= 0.5
    dist_out = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # distanceTransform measures centre-to-centre; the edge sits half a pixel closer, so a
    # pixel's distance from the edge is d − 0.5 and its ring coverage is clip(w − (d − 0.5) + 0.5).
    w = o.width * k
    if o.position == "outside":
        return np.clip(w + 1.0 - dist_out, 0.0, 1.0) * (1.0 - alpha)
    dist_in = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    half = w / 2.0
    return np.where(inside, np.clip(half + 1.0 - dist_in, 0, 1), np.clip(half + 1.0 - dist_out, 0, 1))


def apply(premult: np.ndarray, fx: Effects, t: Transform, level: float) -> tuple[np.ndarray, int]:
    """Premultiplied layer pixels at `level` → (padded premultiplied result, padding)."""
    if not fx.active():
        return premult, 0
    k = level / layer_scale(t)  # canvas px → level px
    pad = padding(fx, t, level)
    h, w = premult.shape[:2]
    base = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.float32)
    base[pad:pad + h, pad:pad + w] = premult
    if fx.blur > 0:
        base = np.ascontiguousarray(blur(base, fx.blur * SIGMA_PER_BLUR * k))
    alpha = base[:, :, 3]

    out = np.zeros_like(base)
    if fx.shadow is not None:
        sa = shadow_alpha(alpha, fx.shadow, t, level, k, box_bottom=pad + h)
        _over(out, _tint(sa, fx.shadow.color, fx.shadow.opacity))
    if fx.glow is not None:
        _over(out, _tint(glow_alpha(alpha, fx.glow, k), fx.glow.color, fx.glow.opacity))
    centred = None
    if fx.outline is not None:
        ring = _tint(outline_alpha(alpha, fx.outline, k), fx.outline.color, fx.outline.opacity)
        if fx.outline.position == "outside":
            _over(out, ring)
        else:
            centred = ring
    _over(out, base)
    if centred is not None:
        _over(out, centred)
    return out, pad
