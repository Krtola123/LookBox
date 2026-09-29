"""Filling a selected area (ARCHITECTURE §9a, M13): "remove this and fill in behind".

Three ways to get the new pixels, one way to deliver them:

- **From a render** (clean plate): the same shot rendered without the object. Exact,
  and the right answer for renders. `plate_fill`.
- **AI fill** (LaMa, ai/lama.py): content-aware, for photos and anything without a plate.
- **Quick fill** (OpenCV Telea): no download; fine for small blemishes, smeary on big areas.

The result is never written into the original. `make_patch` makes the filled area (with
a soft edge) into an image of its own, which becomes a layer above the original (see
commands/mask_edits.fill_layer). Hide or delete it to get the original back.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

HOLE_THRESHOLD = 0.02  # mask values above this count as "to fill"
EDGE_GROW_PX = 3  # the fill reaches this far past the selection: covers its soft edge + halo
EDGE_SOFT_PX = 2.0  # then fades out over this, so the patch blends in


def hole_of(mask: np.ndarray) -> np.ndarray:
    """Binary uint8 hole (1 = fill) from a selection mask (0–1 float)."""
    return (mask > HOLE_THRESHOLD).astype(np.uint8)


def coverage(mask: np.ndarray) -> np.ndarray:
    """How much of the fill to use per pixel: the selection grown a little, then feathered."""
    hole = hole_of(mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_GROW_PX + 1, 2 * EDGE_GROW_PX + 1))
    grown = cv2.dilate(hole, k).astype(np.float32)
    soft = cv2.GaussianBlur(grown, (0, 0), EDGE_SOFT_PX)
    return np.maximum(soft, hole.astype(np.float32))  # fully covered inside the selection itself


def regions(mask: np.ndarray, min_context: int = 48, context_ratio: float = 0.6,
            max_regions: int = 16) -> list[tuple[int, int, int, int]]:
    """Boxes (x0, y0, x1, y1) around each area to fill, with surroundings for context
    (what the AI looks at). Boxes that overlap are merged; the biggest `max_regions` kept."""
    hole = hole_of(mask)
    h, w = hole.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(hole, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, _area = stats[i]
        side = max(bw, bh)
        pad = max(min_context, int(math.ceil(side * context_ratio)))
        boxes.append([max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)])
    merged = True
    while merged:  # merge until no two boxes overlap
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return [tuple(b) for b in boxes[:max_regions]]


def blend_in(rgb: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`rgb` with `filled` laid in where the selection (grown + feathered) is."""
    c = coverage(mask)[:, :, None]
    return (rgb + (filled - rgb) * c).astype(np.float32)


def plate_fill(rgb: np.ndarray, plate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """From a clean-plate render: the plate's pixels under the selection."""
    if plate.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"The render is {plate.shape[1]} × {plate.shape[0]}; it must be "
                         f"{rgb.shape[1]} × {rgb.shape[0]}, the same size as this image.")
    return blend_in(rgb, plate[:, :, :3].astype(np.float32), mask)


def quick_fill(rgb: np.ndarray, mask: np.ndarray, radius: int = 6) -> np.ndarray:
    """OpenCV's Telea inpainting, region by region (16-bit, so gradients stay smooth)."""
    out = rgb.astype(np.float32, copy=True)
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_GROW_PX + 1, 2 * EDGE_GROW_PX + 1))
    for x0, y0, x1, y1 in regions(mask, min_context=radius * 4):
        crop = np.clip(rgb[y0:y1, x0:x1], 0, 1)
        hole = cv2.dilate(hole_of(mask[y0:y1, x0:x1]), grow)
        # cv2.inpaint takes 8-bit or 16-bit-ish float; do it per channel in 16-bit for precision.
        u16 = np.round(crop * 65535).astype(np.uint16)
        res = np.stack([cv2.inpaint(u16[..., c], hole, radius, cv2.INPAINT_TELEA) for c in range(3)], axis=-1)
        out[y0:y1, x0:x1] = res.astype(np.float32) / 65535.0
    return blend_in(rgb, out, mask)


def on_transparency(alpha: np.ndarray, mask: np.ndarray, ring_px: int = 6) -> bool:
    """Is the selection surrounded mostly by transparency (an object on a transparent
    render)? Then "behind it" is nothing: hiding it (a mask) is the right removal, and
    painting an opaque fill there would be wrong."""
    hole = hole_of(mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    ring = cv2.dilate(hole, k).astype(bool) & ~hole.astype(bool)
    return bool(ring.any()) and float(alpha[ring].mean()) < 0.5


def make_patch(filled_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """The fill as an image the same size as the source: straight RGBA, transparent except
    over the (softened) selection. Same size on purpose: the fill layer is then a copy of
    the original layer with this as its source, so position, crop, mask, adjustments and
    filters (vignette included) all line up exactly. Mostly-empty PNGs compress to little.
    None if there's nothing to fill."""
    c = coverage(mask)
    if not (c > 1e-3).any():
        return None
    patch = np.zeros(filled_rgb.shape[:2] + (4,), np.float32)
    patch[..., :3] = np.clip(filled_rgb, 0, 1)
    patch[..., 3] = np.clip(c, 0, 1)
    return patch
