"""Lasso (ARCHITECTURE §9): a polygon in source pixels → an anti-aliased mask.

Freehand and point-by-point lassos both end up as a polygon. It's scan-converted
on a supersampled grid covering only its bounding box (a sample is inside if the
polygon contains its centre, even-odd rule), then area-averaged, so edges get
true coverage values. Big polygons use less supersampling to cap memory.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

MAX_SS = 4
MAX_WORK_SAMPLES = 100_000_000  # supersampled box cap; ~3 bytes a sample at peak (≈300 MB)


def _scan(pts: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Even-odd fill of polygon `pts` (area coords) on a rows × cols grid, by sample centres."""
    p0, p1 = pts, np.roll(pts, -1, axis=0)
    lo, hi = np.minimum(p0[:, 1], p1[:, 1]), np.maximum(p0[:, 1], p1[:, 1])
    # Rows whose centre (j + 0.5) an edge crosses: lo <= j + 0.5 < hi (half-open, so vertices count once).
    j0 = np.clip(np.ceil(lo - 0.5), 0, rows).astype(np.int64)
    j1 = np.clip(np.ceil(hi - 0.5), 0, rows).astype(np.int64)
    n = np.maximum(j1 - j0, 0)
    if n.sum() == 0:
        return np.zeros((rows, cols), np.uint8)
    edge = np.repeat(np.arange(len(pts)), n)
    row = np.concatenate([np.arange(a, b) for a, b in zip(j0[n > 0], j1[n > 0])])
    y = row + 0.5
    x0, y0, x1, y1 = p0[edge, 0], p0[edge, 1], p1[edge, 0], p1[edge, 1]
    x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    order = np.lexsort((x, row))  # by row, then x: crossings pair up left→right
    row, x = row[order], x[order]
    start, end = row[0::2], row[1::2]
    if len(start) != len(end) or not np.array_equal(start, end):
        raise ValueError("Polygon crossings don't pair up.")  # can't happen for a closed polygon
    xa = np.clip(np.ceil(x[0::2] - 0.5), 0, cols).astype(np.int64)  # first centre ≥ xa
    xb = np.clip(np.ceil(x[1::2] - 0.5), 0, cols).astype(np.int64)  # first centre ≥ xb (excluded)
    # Spans on a row never overlap, so the running sum is only ever 0 or 1: int8 is enough.
    diff = np.zeros((rows, cols + 1), np.int8)
    np.add.at(diff, (start, xa), 1)
    np.add.at(diff, (start, xb), -1)
    return np.cumsum(diff[:, :cols], axis=1, dtype=np.int8).view(np.uint8)


def lasso_mask(points, w: int, h: int) -> np.ndarray:
    """Coverage of the polygon `points` [(x, y), ...] (area coordinates: pixel i spans
    [i, i+1)) on a w × h mask. Fewer than 3 points → empty."""
    out = np.zeros((h, w), np.float32)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return out
    x0, y0 = max(0, math.floor(pts[:, 0].min())), max(0, math.floor(pts[:, 1].min()))
    x1, y1 = min(w, math.ceil(pts[:, 0].max())), min(h, math.ceil(pts[:, 1].max()))
    if x1 <= x0 or y1 <= y0:
        return out
    bw, bh = x1 - x0, y1 - y0
    ss = int(max(1, min(MAX_SS, math.floor(math.sqrt(MAX_WORK_SAMPLES / (bw * bh))))))
    inside = _scan((pts - (x0, y0)) * ss, bh * ss, bw * ss)
    if ss > 1:
        inside *= 255  # 0/255 in uint8: the area average stays uint8 (1/255 steps)
        cov = cv2.resize(inside, (bw, bh), interpolation=cv2.INTER_AREA).astype(np.float32) / np.float32(255.0)
    else:
        cov = inside.astype(np.float32)
    out[y0:y1, x0:x1] = cov
    return out
