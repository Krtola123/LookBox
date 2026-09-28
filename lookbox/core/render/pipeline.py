"""Render orchestration (ARCHITECTURE §6.1).

M1 implements the subset of the pipeline that exists so far:
  source (crop) → opacity → transform → composite (normal)
The mask/adjust/LUT/blur/fade/effects stages slot in between, in §6.1 order,
in later milestones. Caching, proxy and the worker thread arrive in M2.
"""

from __future__ import annotations

import numpy as np

from lookbox.core.assets import AssetStore
from lookbox.core.model import Document, ImageLayer
from lookbox.core.render import blend
from lookbox.core.render.transform import warp_to_canvas


def premultiply(rgba: np.ndarray) -> np.ndarray:
    out = rgba.copy()
    out[:, :, :3] *= out[:, :, 3:4]
    return out


def unpremultiply(rgba: np.ndarray) -> np.ndarray:
    out = rgba.copy()
    a = out[:, :, 3:4]
    np.divide(out[:, :, :3], a, out=out[:, :, :3], where=a > 1e-8)
    out[:, :, :3] = np.where(a > 1e-8, out[:, :, :3], 0.0)
    return np.clip(out, 0.0, 1.0)


def layer_source(layer: ImageLayer, store: AssetStore) -> np.ndarray:
    """Source pixels with crop applied (read-only view is fine; callers copy)."""
    px = store.pixels(layer.source)
    if layer.crop is not None:
        x, y, w, h = layer.crop
        px = px[y : y + h, x : x + w]
    return px


def render(doc: Document, store: AssetStore, scale: float = 1.0) -> np.ndarray:
    """Flatten the document. Returns straight-alpha float32 RGBA, 0–1.

    Output size is the canvas × `scale` (rounded).
    """
    out_w = max(1, round(doc.canvas.w * scale))
    out_h = max(1, round(doc.canvas.h * scale))
    acc = np.zeros((out_h, out_w, 4), dtype=np.float32)
    if doc.background is not None:
        r, g, b, a = doc.background
        acc[:] = (r * a, g * a, b * a, a)

    for layer in doc.layers:
        if not layer.visible or layer.opacity <= 0.0:
            continue
        if not isinstance(layer, ImageLayer):
            raise TypeError(f"Can't render layer kind '{layer.kind}' yet.")
        src = premultiply(layer_source(layer, store))
        if layer.opacity < 1.0:
            src *= np.float32(layer.opacity)
        placed = warp_to_canvas(src, layer.transform, out_w, out_h, scale)
        if placed is None:
            continue
        patch, (x0, y0) = placed
        ph, pw = patch.shape[:2]
        blend.composite(acc[y0 : y0 + ph, x0 : x0 + pw], patch, layer.blend_mode)

    return unpremultiply(acc)
