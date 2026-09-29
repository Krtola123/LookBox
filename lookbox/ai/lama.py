"""AI fill with LaMa (ARCHITECTURE §9a, §10.2). Qt-free; `run` is the model call.

The ONNX export (Carve/LaMa-ONNX, lama_fp32) takes a fixed 512 × 512 RGB image in 0–1
and a 0/1 mask (1 = fill), and returns RGB in 0–255 (checked, not assumed). Each area
to fill is cut out with context around it (core/masks/fill.regions), resized to 512,
filled, and resized back; only the selected area (softly) takes the result.
Big areas are filled at 512 px and scaled up, so they come out softer than small ones.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from lookbox.core.masks import fill as F

SIZE = 512


def _to_size(x: np.ndarray, size: int) -> np.ndarray:
    h, w = x.shape[:2]
    interp = cv2.INTER_AREA if (w > size or h > size) else cv2.INTER_CUBIC
    return cv2.resize(x, (size, size), interpolation=interp)


def fill(run: Callable[[dict], np.ndarray], rgb: np.ndarray, mask: np.ndarray, size: int = SIZE) -> np.ndarray:
    """`rgb` (straight float, H × W × 3) with the selected area filled in."""
    out = rgb.astype(np.float32, copy=True)
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for x0, y0, x1, y1 in F.regions(mask):
        crop = np.clip(rgb[y0:y1, x0:x1], 0, 1).astype(np.float32)
        h, w = crop.shape[:2]
        img = _to_size(crop, size)
        hole = cv2.resize(F.hole_of(mask[y0:y1, x0:x1]), (size, size), interpolation=cv2.INTER_NEAREST)
        hole = cv2.dilate(hole, grow)  # a little past the edge, so no halo of the old thing survives
        feeds = {"image": np.ascontiguousarray(img.transpose(2, 0, 1)[None], np.float32),
                 "mask": np.ascontiguousarray(hole[None, None].astype(np.float32))}
        res = np.asarray(run(feeds), np.float32)
        while res.ndim > 3:
            res = res[0]
        if res.shape[0] == 3:
            res = res.transpose(1, 2, 0)
        if res.max() > 2.0:  # 0–255 output
            res = res / 255.0
        back = cv2.resize(np.clip(res, 0, 1), (w, h), interpolation=cv2.INTER_CUBIC)
        out[y0:y1, x0:x1] = back
    return F.blend_in(rgb.astype(np.float32), out, mask)
