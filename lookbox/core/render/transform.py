"""Layer placement math (ARCHITECTURE §6.1 "transform").

One source of truth for where a layer sits: the UI builds its QTransform from
`layer_matrix`, and the exporter warps pixels with the same matrix. If preview
and export ever disagree about position, the bug is here.

Coordinate convention: "area" coordinates, where pixel i covers [i, i+1).
That's what Qt uses. OpenCV uses pixel centres, so we convert in `_to_cv`.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from lookbox.core.model import Transform


def _translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def _scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def _rotate(deg: float) -> np.ndarray:
    # y points down, so this standard matrix rotates clockwise on screen.
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def layer_matrix(t: Transform, w: int, h: int) -> np.ndarray:
    """3×3 matrix mapping layer area-coords (0..w, 0..h) → canvas area-coords."""
    fx = -1.0 if t.flip_h else 1.0
    fy = -1.0 if t.flip_v else 1.0
    return (
        _translate(t.x, t.y)
        @ _rotate(t.rotation_deg)
        @ _scale(t.scale_x * fx, t.scale_y * fy)
        @ _translate(-w / 2.0, -h / 2.0)
    )


def level_matrix(t: Transform, w: int, h: int, lw: int, lh: int, pad: int = 0) -> np.ndarray:
    """Like layer_matrix, for a level image whose lw × lh box stands in for the w × h
    layer, with `pad` extra level pixels on every side (effects reach past the box)."""
    return layer_matrix(t, w, h) @ _scale(w / lw, h / lh) @ _translate(-pad, -pad)


def corners(t: Transform, w: int, h: int) -> np.ndarray:
    """Canvas positions of the layer's corners: TL, TR, BR, BL (4×2)."""
    m = layer_matrix(t, w, h)
    pts = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64)
    return (m @ pts.T).T[:, :2]


def canvas_to_local(t: Transform, w: int, h: int, x: float, y: float) -> tuple[float, float]:
    """Inverse mapping: canvas point → layer area-coords (may be outside 0..w/0..h)."""
    inv = np.linalg.inv(layer_matrix(t, w, h))
    p = inv @ np.array([x, y, 1.0])
    return float(p[0]), float(p[1])


def _to_cv(m: np.ndarray) -> np.ndarray:
    """Area-coord matrix → OpenCV pixel-centre matrix."""
    return _translate(-0.5, -0.5) @ m @ _translate(0.5, 0.5)


def warp_to_canvas(
    premult: np.ndarray, t: Transform, out_w: int, out_h: int, out_scale: float = 1.0, pad: int = 0,
    box: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Warp a premultiplied RGBA layer into canvas space.

    Returns (patch, (x0, y0)): only the region the layer covers, to keep this
    fast. None if the layer lands entirely off-canvas. `out_scale` renders a
    scaled canvas (proxy or export scale); the canvas is out_w × out_h pixels.
    `box` = the layer's size at scale 1 when `premult` was rendered at another
    level (text above level 1); None = the pixels are the layer at level 1.
    """
    lh, lw = premult.shape[0] - 2 * pad, premult.shape[1] - 2 * pad  # the layer box, level px
    w, h = box if box is not None else (lw, lh)
    m = _scale(out_scale, out_scale) @ level_matrix(t, w, h, lw, lh, pad)
    h, w = premult.shape[:2]

    # Area-average first when shrinking (bilinear warps alias when downscaling), by
    # WHOLE-NUMBER factors only: OpenCV's area resize is an exact box filter then,
    # while fractional factors shift the image by a fraction of a pixel. The final
    # warp handles the remaining < 2× step.
    kx, ky = (box[0] / lw, box[1] / lh) if box is not None else (1.0, 1.0)
    fx = max(1, int(1.0 / max(t.scale_x * out_scale * kx, 1e-9)))
    fy = max(1, int(1.0 / max(t.scale_y * out_scale * ky, 1e-9)))
    src = premult
    if fx > 1 or fy > 1:
        # Grow the padding so the reduction blocks line up with the layer box (as they do
        # with no padding); otherwise blocks straddle the box edge and soften it.
        ex, ey = (-pad) % fx, (-pad) % fy  # extra left/top so that pad + extra ≡ 0
        ph, pw = -(-(h + ey) // fy) * fy, -(-(w + ex) // fx) * fx
        if (ex, ey) != (0, 0) or (ph, pw) != (h, w):
            src = np.zeros((ph, pw, 4), np.float32)
            src[ey:ey + h, ex:ex + w] = premult
            m = m @ _translate(-ex, -ey)
        src = cv2.resize(src, (pw // fx, ph // fy), interpolation=cv2.INTER_AREA)
        m = m @ _scale(fx, fy)

    sh, sw = src.shape[:2]
    pts = m @ np.array([[0, 0, 1], [sw, 0, 1], [sw, sh, 1], [0, sh, 1]], dtype=np.float64).T
    x0 = max(0, math.floor(pts[0].min()))
    y0 = max(0, math.floor(pts[1].min()))
    x1 = min(out_w, math.ceil(pts[0].max()))
    y1 = min(out_h, math.ceil(pts[1].max()))
    if x1 <= x0 or y1 <= y0:
        return None

    m_patch = _translate(-x0, -y0) @ m
    patch = cv2.warpAffine(
        src,
        _to_cv(m_patch)[:2],
        (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return patch, (x0, y0)
