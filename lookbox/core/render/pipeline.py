"""Render orchestration (ARCHITECTURE §6.1).

The pipeline has two halves:

1. `render_layer` — everything that happens to a layer *before* it's placed:
   source (crop) → [mask → adjust → LUT → blur → fade → effects, as milestones add them].
   The result is premultiplied, pre-transform, pre-opacity, and cacheable: it
   depends only on `render_key(layer)` and the resolution `level`.
2. Placement — opacity → transform → composite. Cheap. On screen, Qt does it
   (ui/canvas/layer_items.py); for export, `render` below does it in float.

Both halves use the same `layer_matrix`, so preview and export agree on geometry.
"""

from __future__ import annotations

import hashlib
import json
import threading

import cv2
import numpy as np

from lookbox.core.assets import AssetStore
from lookbox.core.model import Document, FillLayer, ImageLayer, Layer, TextLayer, layer_to_dict
from dataclasses import dataclass

from lookbox.core.masks import ops as mask_ops
from lookbox.core.render import adjust, blend, effects, fill, text
from lookbox.core.render.adjust import Cancelled  # noqa: F401  (re-exported: one Cancelled for the pipeline)
from lookbox.core.render.levels import export_level
from lookbox.core.render.transform import warp_to_canvas

# Layer fields that only affect placement, never the pre-transform pixels.
PLACEMENT_FIELDS = frozenset({"id", "name", "visible", "locked", "opacity", "blend_mode", "transform"})


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


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
    """Source pixels with crop applied (read-only; callers copy)."""
    px = store.pixels(layer.source)
    if layer.crop is not None:
        x, y, w, h = layer.crop
        px = px[y : y + h, x : x + w]
    return px


def render_key(layer: Layer) -> str:
    """Identity of a layer's pre-transform pixels.

    Built from every field except placement ones, so fields added by later
    milestones invalidate the cache automatically. Effects are specified in
    canvas space, so with effects on, scale/rotation/flip become part of the
    key (moving still never re-renders).
    """
    d = {k: v for k, v in layer_to_dict(layer).items() if k not in PLACEMENT_FIELDS}
    if layer.effects.active():
        t = layer.transform
        d["_fx_transform"] = [t.scale_x, t.scale_y, t.rotation_deg, t.flip_h, t.flip_v]
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()


@dataclass
class Rendered:
    """render_layer output: premultiplied pixels whose central box is the layer,
    with `pad` extra pixels on every side for effects."""

    pixels: np.ndarray
    pad: int = 0

    @property
    def nbytes(self) -> int:  # so the LRU cache can budget it
        return int(self.pixels.nbytes)


def level_size(w: int, h: int, level: float) -> tuple[int, int]:
    return max(1, round(w * level)), max(1, round(h * level))


def _base_pixels(layer: Layer, store: AssetStore, level: float) -> np.ndarray:
    """Straight float32 RGBA at `level`: the layer's own pixels before any stage.
    Images never go above level 1 (there's nothing more to show); text can."""
    if isinstance(layer, TextLayer):
        return text.render_text(layer, *level_size(*text.layer_box(layer), level))
    if isinstance(layer, FillLayer):
        return fill.render_fill(layer.fill, *level_size(layer.width, layer.height, level))
    if isinstance(layer, ImageLayer):
        src = layer_source(layer, store)
        if level < 1.0:
            h, w = src.shape[:2]
            # True area average (not striding): skipping pixels would bring back shimmer.
            src = cv2.resize(src, level_size(w, h, level), interpolation=cv2.INTER_AREA)
        if layer.mask is not None:  # §6.1: mask first, before adjust
            m = layer.mask
            full = mask_ops.mask_from_pixels(store.pixels(m.asset))
            if layer.crop is not None:
                x, y, cw, ch = layer.crop
                full = full[y:y + ch, x:x + cw]
            mask = mask_ops.mask_at_level(np.ascontiguousarray(full), (src.shape[1], src.shape[0]),
                                          level, m.shift, m.feather, m.invert)
            src = src.copy()
            src[:, :, 3] *= mask
        return src
    raise TypeError(f"Can't render layer kind '{layer.kind}'.")


def render_layer(layer: Layer, store: AssetStore, level: float = 1.0,
                 cancel: threading.Event | None = None) -> np.ndarray:
    """Pixels only (padding included if the layer has effects). See render_layer_full."""
    return render_layer_full(layer, store, level, cancel).pixels


def render_layer_full(layer: Layer, store: AssetStore, level: float = 1.0,
                      cancel: threading.Event | None = None) -> Rendered:
    """Pre-transform, pre-opacity layer pixels at `level` (1 = full res), premultiplied.

    Later stages must scale pixel radii by `level` (§6.3) so every level
    looks like a downscaled full-res render.
    """
    src = _base_pixels(layer, store, level)
    _check(cancel)
    # ---- §6.1 order: mask → adjust → LUT → blur → fade → effects (later milestones slot in) ----
    if not layer.adjust.is_identity():
        rgb = adjust.apply(src[:, :, :3], src[:, :, 3], layer.adjust, level, cancel)
        src = np.concatenate([rgb, src[:, :, 3:4]], axis=2)
    if layer.fade is not None:
        h, w = src.shape[:2]
        src = src if src.flags.writeable and src.base is None else src.copy()
        src[:, :, 3] *= fill.fade_mask(layer.fade, w, h)
    _check(cancel)
    premult = premultiply(src)
    if layer.effects.active():  # blur + shadow/glow/outline, padded (§6.1, §8)
        premult, pad = effects.apply(premult, layer.effects, layer.transform, level)
        _check(cancel)
        return Rendered(premult, pad)
    return Rendered(premult, 0)


def render(doc: Document, store: AssetStore, scale: float = 1.0,
           cancel: threading.Event | None = None, progress=None) -> np.ndarray:
    """Flatten the document in float. Returns straight-alpha float32 RGBA, 0–1.

    Output size is the canvas × `scale` (rounded). `progress(fraction)` is
    called after each layer; setting `cancel` raises Cancelled.
    """
    out_w = max(1, round(doc.canvas.w * scale))
    out_h = max(1, round(doc.canvas.h * scale))
    acc = np.zeros((out_h, out_w, 4), dtype=np.float32)
    if doc.background is not None:
        r, g, b, a = doc.background
        acc[:] = (r * a, g * a, b * a, a)

    n = max(1, len(doc.layers))
    for i, layer in enumerate(doc.layers):
        _check(cancel)
        if layer.visible and layer.opacity > 0.0:
            level = export_level(layer, scale)  # text renders sharp at the export size
            rendered = render_layer_full(layer, store, level, cancel)
            src = rendered.pixels
            if layer.opacity < 1.0:
                src = src * np.float32(layer.opacity)  # after effects: fades the shadow too
            box = text.layer_box(layer) if isinstance(layer, TextLayer) else None
            placed = warp_to_canvas(src, layer.transform, out_w, out_h, scale, pad=rendered.pad, box=box)
            if placed is not None:
                patch, (x0, y0) = placed
                ph, pw = patch.shape[:2]
                blend.composite(acc[y0 : y0 + ph, x0 : x0 + pw], patch, layer.blend_mode)
        if progress is not None:
            progress((i + 1) / n)

    return unpremultiply(acc)
