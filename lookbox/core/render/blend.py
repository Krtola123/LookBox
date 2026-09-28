"""Blend modes (ARCHITECTURE §6.1 "composite").

Separable modes follow the W3C Compositing and Blending spec, the same one Qt's
QPainter composition modes implement, so the canvas preview (Qt) and export
(this file) agree:

    co = cs·(1 − αb) + cb·(1 − αs) + αs·αb·B(Cb, Cs)      (premultiplied c, straight C)
    αo = αs + αb − αs·αb

`add` is Qt's CompositionMode_Plus: premultiplied sum, clamped.
"""

from __future__ import annotations

import numpy as np

from lookbox.core.model import BLEND_MODES


def over(dst: np.ndarray, src: np.ndarray) -> None:
    """In place: premultiplied `src` over premultiplied `dst` (same shape)."""
    dst *= 1.0 - src[:, :, 3:4]
    dst += src


def _straight(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(c)
    np.divide(c, a, out=out, where=a > 1e-8)
    return np.clip(out, 0.0, 1.0)


def _soft_light(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    d = np.where(cb <= 0.25, ((16.0 * cb - 12.0) * cb + 4.0) * cb, np.sqrt(cb))
    return np.where(cs <= 0.5, cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb), cb + (2.0 * cs - 1.0) * (d - cb))


def _hard_light(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return np.where(cs <= 0.5, 2.0 * cs * cb, 1.0 - 2.0 * (1.0 - cs) * (1.0 - cb))  # multiply / screen


BLEND_FUNCS = {
    "multiply": lambda cb, cs: cb * cs,
    "screen": lambda cb, cs: cb + cs - cb * cs,
    "overlay": lambda cb, cs: _hard_light(cs, cb),  # overlay = hard-light with layers swapped
    "soft_light": _soft_light,
}


def composite(dst: np.ndarray, src: np.ndarray, mode: str) -> None:
    """In place: premultiplied `src` onto premultiplied `dst` with `mode`."""
    if mode == "normal":
        over(dst, src)
        return
    if mode == "add":
        dst += src
        np.minimum(dst, 1.0, out=dst)
        return
    if mode not in BLEND_FUNCS:
        raise ValueError(f"Unknown blend mode '{mode}'. Known: {', '.join(BLEND_MODES)}")
    sa, da = src[:, :, 3:4], dst[:, :, 3:4]
    cs_p, cb_p = src[:, :, :3], dst[:, :, :3]
    b = BLEND_FUNCS[mode](_straight(cb_p, da), _straight(cs_p, sa))
    rgb = cs_p * (1.0 - da) + cb_p * (1.0 - sa) + sa * da * b
    alpha = sa + da - sa * da
    dst[:, :, :3] = rgb
    dst[:, :, 3:4] = alpha
