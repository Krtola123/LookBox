"""Generated pixels: fill layers (solid/gradient backdrops) and gradient fade
(ARCHITECTURE §8, M4).

Everything is computed in coordinates relative to the layer box, so any
preview level is just a resampled export (resolution-independent by design).
Gradients interpolate in premultiplied sRGB (like CSS), so stops with
different alpha don't produce dark fringes.
"""

from __future__ import annotations

import math

import numpy as np

from lookbox.core.model import Fill, GradientFade


def _grid(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates relative to the box centre, in pixels."""
    x = np.arange(w, dtype=np.float32) + 0.5 - w / 2.0
    y = np.arange(h, dtype=np.float32) + 0.5 - h / 2.0
    return x[None, :], y[:, None]


def axis_t(angle_deg: float, w: int, h: int) -> np.ndarray:
    """0 → 1 along `angle` across the whole box (0° = left→right, 90° = top→bottom).

    Like CSS linear-gradient: the far corners land exactly on 0 and 1."""
    x, y = _grid(w, h)
    a = math.radians(angle_deg)
    c, s = np.float32(math.cos(a)), np.float32(math.sin(a))
    extent = abs(w / 2.0 * math.cos(a)) + abs(h / 2.0 * math.sin(a))
    return (x * c + y * s) / np.float32(2.0 * max(extent, 1e-6)) + np.float32(0.5)


def radial_t(cx: float, cy: float, w: int, h: int) -> np.ndarray:
    """Distance from (cx, cy) (box-relative 0–1) in units of the half-diagonal."""
    x, y = _grid(w, h)
    dx = x - np.float32((cx - 0.5) * w)
    dy = y - np.float32((cy - 0.5) * h)
    return np.sqrt(dx * dx + dy * dy) / np.float32(math.hypot(w, h) / 2.0)


def render_fill(fill: Fill, w: int, h: int) -> np.ndarray:
    """Straight float32 RGBA (h, w, 4)."""
    stops = sorted(fill.stops, key=lambda s: s.pos)
    if not stops:
        return np.zeros((h, w, 4), np.float32)
    if fill.kind == "solid" or len(stops) == 1:
        out = np.empty((h, w, 4), np.float32)
        out[:] = stops[0].color
        return out
    if fill.kind == "linear":
        t = axis_t(fill.angle_deg, w, h)
    elif fill.kind == "radial":
        t = radial_t(fill.cx, fill.cy, w, h) / np.float32(max(fill.radius, 1e-3))
    else:
        raise ValueError(f"Unknown fill kind: {fill.kind!r}")
    t = np.broadcast_to(np.clip(t, 0.0, 1.0), (h, w))

    pos = np.array([s.pos for s in stops], np.float32)
    cols = np.array([s.color for s in stops], np.float32)
    premult = cols.copy()
    premult[:, :3] *= premult[:, 3:4]
    out = np.empty((h, w, 4), np.float32)
    for ch in range(4):
        out[:, :, ch] = np.interp(t, pos, premult[:, ch]).astype(np.float32)
    a = out[:, :, 3:4]
    np.divide(out[:, :, :3], a, out=out[:, :, :3], where=a > 1e-8)
    return np.clip(out, 0.0, 1.0)


def _ramp(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """1 before start → 0 after end, smooth in between."""
    if abs(end - start) < 1e-6:
        return (t < start).astype(np.float32)
    u = np.clip((t - start) / (end - start), 0.0, 1.0)
    return 1.0 - u * u * (3.0 - 2.0 * u)


def fade_mask(fade: GradientFade, w: int, h: int) -> np.ndarray:
    """Alpha multiplier (h, w) for a gradient fade."""
    if fade.kind == "linear":
        t = axis_t(fade.angle_deg, w, h)
    elif fade.kind == "radial":
        t = radial_t(0.5, 0.5, w, h)
    else:
        raise ValueError(f"Unknown fade kind: {fade.kind!r}")
    f = _ramp(np.broadcast_to(t, (h, w)), fade.start, fade.end)
    return (1.0 - f if fade.invert else f).astype(np.float32)
