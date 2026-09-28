"""Canvas interaction math: handles, dragging, hit testing.

Deliberately Qt-free, so the tricky geometry is unit-tested headless.
The canvas view (Qt) only converts mouse events and draws.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from lookbox.core.assets import AssetStore
from lookbox.core.model import Document, FillLayer, ImageLayer, Layer, TextLayer, Transform
from lookbox.core.render import text as text_render
from lookbox.core.render.pipeline import layer_source
from lookbox.core.render.transform import canvas_to_local, corners

# Handle name → (x sign, y sign) in layer space.
SCALE_HANDLES = {
    "nw": (-1, -1), "n": (0, -1), "ne": (1, -1), "e": (1, 0),
    "se": (1, 1), "s": (0, 1), "sw": (-1, 1), "w": (-1, 0),
}
CORNERS = {"nw", "ne", "se", "sw"}
ROTATE = "rot"
ROTATE_OFFSET_PX = 28.0  # screen pixels below the bottom edge
MIN_SIZE_PX = 4.0  # a layer can't be scaled below this many canvas pixels


def _rot(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s], [s, c]])


def _signed_scale(t: Transform) -> np.ndarray:
    return np.array([t.scale_x * (-1 if t.flip_h else 1), t.scale_y * (-1 if t.flip_v else 1)])


def local_to_canvas(t: Transform, p: np.ndarray) -> np.ndarray:
    """Point in centred layer coords (origin = layer centre, unscaled px) → canvas."""
    return np.array([t.x, t.y]) + _rot(t.rotation_deg) @ (_signed_scale(t) * p)


def handle_positions(t: Transform, w: int, h: int, zoom: float,
                     corners_only: bool = False) -> dict[str, np.ndarray]:
    """Canvas positions of all handles. `zoom` = screen px per canvas px.
    `corners_only`: text resizes by corners only (it scales the font; no stretching)."""
    half = np.array([w / 2.0, h / 2.0])
    out = {name: local_to_canvas(t, half * np.array(s)) for name, s in SCALE_HANDLES.items()}
    down = _rot(t.rotation_deg) @ np.array([0.0, 1.0])  # the layer's "down", ignoring flips
    # With flip_v the local "s" handle is on top, so pick whichever edge midpoint is lower.
    bottom = max((out["n"], out["s"]), key=lambda p: float(down @ (p - np.array([t.x, t.y]))))
    out[ROTATE] = bottom + down * (ROTATE_OFFSET_PX / zoom)
    if corners_only:
        out = {k: v for k, v in out.items() if k in CORNERS or k == ROTATE}
    return out


def handle_at(t: Transform, w: int, h: int, zoom: float, x: float, y: float,
              radius_px: float = 9.0, corners_only: bool = False) -> str | None:
    """Which handle (if any) is under canvas point (x, y)."""
    best, best_d = None, radius_px / zoom
    for name, p in handle_positions(t, w, h, zoom, corners_only).items():
        d = math.hypot(p[0] - x, p[1] - y)
        if d <= best_d:
            best, best_d = name, d
    return best


def point_in_layer(t: Transform, w: int, h: int, x: float, y: float) -> bool:
    lx, ly = canvas_to_local(t, w, h, x, y)
    return 0.0 <= lx <= w and 0.0 <= ly <= h


def drag_move(t0: Transform, press: tuple[float, float], cur: tuple[float, float],
              constrain: bool = False) -> Transform:
    dx, dy = cur[0] - press[0], cur[1] - press[1]
    if constrain:
        if abs(dx) >= abs(dy):
            dy = 0.0
        else:
            dx = 0.0
    return replace(t0, x=t0.x + dx, y=t0.y + dy)


def drag_scale(t0: Transform, w: int, h: int, handle: str, cur: tuple[float, float],
               free: bool = False, from_centre: bool = False) -> Transform:
    """Scale by dragging `handle` to canvas point `cur`.

    Corners scale uniformly (Canva behaviour) unless `free`. Edges scale one axis.
    The opposite handle stays put, or the centre if `from_centre`.
    """
    sign = np.array(SCALE_HANDLES[handle], dtype=float)
    half = np.array([w / 2.0, h / 2.0])
    h_local = sign * half
    a_local = np.zeros(2) if from_centre else -sign * half
    span = h_local - a_local  # local vector anchor → handle (0 on an axis we don't touch)

    rot = _rot(t0.rotation_deg)
    sf0 = _signed_scale(t0)
    anchor = local_to_canvas(t0, a_local)
    q = rot.T @ (np.array(cur, dtype=float) - anchor)  # in the layer's rotated frame

    min_scale = np.array([MIN_SIZE_PX / w, MIN_SIZE_PX / h])
    new_abs = np.array([t0.scale_x, t0.scale_y], dtype=float)
    if handle in CORNERS and not free:
        d = sf0 * span
        k = float(q @ d) / float(d @ d)
        k = max(k, float(np.max(min_scale / new_abs)))
        new_abs = new_abs * k
    else:
        for i in (0, 1):
            if span[i] != 0:
                sf = q[i] / span[i]  # signed scale the drag asks for
                new_abs[i] = max(sf * np.sign(sf0[i]), min_scale[i])  # never flip by dragging

    t1 = replace(t0, scale_x=float(new_abs[0]), scale_y=float(new_abs[1]))
    centre = anchor - rot @ (_signed_scale(t1) * a_local)
    return replace(t1, x=float(centre[0]), y=float(centre[1]))


def normalise_deg(deg: float) -> float:
    deg = math.fmod(deg, 360.0)
    if deg > 180.0:
        deg -= 360.0
    elif deg <= -180.0:
        deg += 360.0
    return deg


def drag_rotate(t0: Transform, press: tuple[float, float], cur: tuple[float, float],
                snap_15: bool = False) -> Transform:
    a0 = math.atan2(press[1] - t0.y, press[0] - t0.x)
    a1 = math.atan2(cur[1] - t0.y, cur[0] - t0.x)
    deg = t0.rotation_deg + math.degrees(a1 - a0)
    if snap_15:
        deg = round(deg / 15.0) * 15.0
    else:
        nearest_90 = round(deg / 90.0) * 90.0
        if abs(deg - nearest_90) < 2.0:  # soft snap to straight angles
            deg = nearest_90
    return replace(t0, rotation_deg=normalise_deg(deg))


def layer_size(doc: Document, layer: Layer) -> tuple[int, int]:
    if isinstance(layer, FillLayer):
        return layer.width, layer.height
    if isinstance(layer, TextLayer):
        return text_render.layer_box(layer)
    if layer.crop is not None:
        return layer.crop[2], layer.crop[3]
    info = doc.assets[layer.source]
    return info.width, info.height


def layer_at(doc: Document, store: AssetStore, x: float, y: float,
             alpha_threshold: float = 0.05) -> str | None:
    """Topmost visible, unlocked layer with visible pixels at canvas (x, y).

    Alpha-aware, so a full-canvas render with a transparent background doesn't
    block clicks on the layers under it.
    """
    for layer in reversed(doc.layers):
        if not layer.visible or layer.locked:
            continue
        w, h = layer_size(doc, layer)
        lx, ly = canvas_to_local(layer.transform, w, h, x, y)
        ix, iy = math.floor(lx), math.floor(ly)
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        if isinstance(layer, FillLayer):
            # Backdrops are hit anywhere inside their box (like Canva's background), unless invisible.
            if max(s.color[3] for s in layer.fill.stops) * layer.opacity >= alpha_threshold:
                return layer.id
        elif isinstance(layer, TextLayer):
            return layer.id  # anywhere in the text box: letters are too thin to aim at
        elif isinstance(layer, ImageLayer):
            if layer_source(layer, store)[iy, ix, 3] * layer.opacity >= alpha_threshold:
                return layer.id
    return None


def quad(t: Transform, w: int, h: int) -> np.ndarray:
    return corners(t, w, h)


# ------------------------------------------------------------------ snapping

SNAP_PX = 8.0  # screen pixels: consistent feel at any zoom


def bbox(t: Transform, w: int, h: int) -> tuple[float, float, float, float]:
    """Axis-aligned canvas bounds of the (possibly rotated) layer: x0, y0, x1, y1."""
    c = corners(t, w, h)
    return float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max())


def snap_targets(doc: Document, exclude: str | None) -> tuple[list[float], list[float]]:
    """Canvas edges + centre, plus every other visible layer's edges + centre."""
    xs = [0.0, doc.canvas.w / 2.0, float(doc.canvas.w)]
    ys = [0.0, doc.canvas.h / 2.0, float(doc.canvas.h)]
    for layer in doc.layers:
        if layer.id == exclude or not layer.visible:
            continue
        x0, y0, x1, y1 = bbox(layer.transform, *layer_size(doc, layer))
        xs += [x0, (x0 + x1) / 2.0, x1]
        ys += [y0, (y0 + y1) / 2.0, y1]
    return xs, ys


def _best(values: list[float], targets: list[float], thr: float) -> tuple[float, list[float]]:
    """Smallest correction that puts any of `values` on a target within `thr`, and every
    target that lines up after that correction (all of them get a guide)."""
    best = None
    for v in values:
        for t in targets:
            d = t - v
            if abs(d) <= thr and (best is None or abs(d) < abs(best)):
                best = d
    if best is None:
        return 0.0, []
    hits = sorted({t for v in values for t in targets if abs(v + best - t) < 1e-6})
    return best, hits


def snap_move(t: Transform, w: int, h: int, targets: tuple[list[float], list[float]],
              zoom: float) -> tuple[Transform, list[tuple[str, float]]]:
    """Nudge a moved layer so an edge or its centre lands on a target.
    Returns the snapped transform and the guides to draw: ("v", x) / ("h", y)."""
    thr = SNAP_PX / zoom
    x0, y0, x1, y1 = bbox(t, w, h)
    dx, tx = _best([x0, (x0 + x1) / 2.0, x1], targets[0], thr)
    dy, ty = _best([y0, (y0 + y1) / 2.0, y1], targets[1], thr)
    guides = [("v", x) for x in tx] + [("h", y) for y in ty]
    return replace(t, x=t.x + dx, y=t.y + dy), guides


def snap_point(t: Transform, handle: str, x: float, y: float, targets: tuple[list[float], list[float]],
               zoom: float) -> tuple[float, float, list[tuple[str, float]]]:
    """Snap the mouse while resizing, for layers that aren't rotated off-axis.
    Only the axes the handle actually moves are snapped."""
    if abs(normalise_deg(t.rotation_deg)) % 90.0 > 1e-6:
        return x, y, []
    thr = SNAP_PX / zoom
    sx, sy = SCALE_HANDLES[handle]
    if abs(normalise_deg(t.rotation_deg)) % 180.0 > 1e-6:  # 90° / 270°: the handle's axes swap
        sx, sy = sy, sx
    guides = []
    if sx != 0:
        dx, tx = _best([x], targets[0], thr)
        x += dx
        guides += [("v", v) for v in tx]
    if sy != 0:
        dy, ty = _best([y], targets[1], thr)
        y += dy
        guides += [("h", v) for v in ty]
    return x, y, guides


def canvas_to_source(doc: Document, layer: ImageLayer, x: float, y: float) -> tuple[float, float]:
    """Canvas point → pixel coordinates in the layer's full (uncropped) source image."""
    w, h = layer_size(doc, layer)
    lx, ly = canvas_to_local(layer.transform, w, h, x, y)
    if layer.crop is not None:
        lx, ly = lx + layer.crop[0], ly + layer.crop[1]
    return lx, ly
