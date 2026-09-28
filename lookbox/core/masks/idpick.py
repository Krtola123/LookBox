"""ID pick (ARCHITECTURE §9): click an object in a render's ID pass, get its mask.

ID passes paint every object (or material) in one flat colour. Renderers
anti-alias them like the beauty image, so edge pixels are *mixes* of IDs (or of
an ID and transparency). An exact colour match would give a jagged edge one
pixel inside the beauty's soft edge, so edge pixels get a coverage value instead,
by solving (premultiplied RGBA)

    p ≈ a·A + c·B1 + (1 − a − c)·B2

where A is the picked colour and B1, B2 the flat colours around the pixel that
aren't A (B2 = B1 on a plain two-colour edge; transparent if nothing is near).
Three colours cover corners where A meets another object *and* the background.
Renderers mix in linear light and some then encode the pass to sRGB, which
bends the mixes. Which one a pass is gets measured once, from all its edges: a
blend of two colours lies on the straight line between them only in the space it
was mixed in. The solve then runs in that space.

Never resampled: ID pixels are read exactly (§11). An `IdPass` is built once per
pass (16-bit colours, int64 keys, which pixels are flat); each click then costs
one comparison over the image plus work on the edge band only.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1.5 / 255.0  # one 8-bit step of slack for tolerance compares
EDGE_PX = 2  # how far outside the exact match edge pixels can be mixes
MIX_RESIDUAL = 0.3  # left-over error (relative to |A − B|) above which a pixel isn't a mix with A
_Q = 65535.0


def _quantise(rgba: np.ndarray) -> np.ndarray:
    """Straight float RGBA → premultiplied, 16 bits a channel (exact for 8/16-bit passes)."""
    pre = np.array(rgba, dtype=np.float32, copy=True)
    pre[:, :, :3] *= pre[:, :, 3:4]
    return np.round(np.clip(pre, 0.0, 1.0) * _Q).astype(np.uint16)


def _keys(q: np.ndarray) -> np.ndarray:
    """One int64 per pixel from 4 × 16 bits, so 'same ID' is one integer compare."""
    q = q.astype(np.uint64)
    k = (q[..., 0] << np.uint64(48)) | (q[..., 1] << np.uint64(32)) | (q[..., 2] << np.uint64(16)) | q[..., 3]
    return k.view(np.int64)


def _flat(keys: np.ndarray) -> np.ndarray:
    """Pixels equal to all four neighbours: the inside of a flat ID region."""
    pad = np.pad(keys, 1, mode="edge")
    c = pad[1:-1, 1:-1]
    return (pad[:-2, 1:-1] == c) & (pad[2:, 1:-1] == c) & (pad[1:-1, :-2] == c) & (pad[1:-1, 2:] == c)


def _linear(v: np.ndarray) -> np.ndarray:
    """Premultiplied sRGB-encoded → premultiplied linear (alpha untouched)."""
    out = v.copy()
    a = np.maximum(v[..., 3:4], 1e-8)
    s = np.clip(v[..., :3] / a, 0.0, 1.0)
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    out[..., :3] = lin * v[..., 3:4]
    return out


class IdPass:
    """An ID pass ready for picking. Build it once (off the UI thread for big renders)."""

    def __init__(self, rgba: np.ndarray) -> None:
        self.q = _quantise(rgba)
        self.keys = _keys(self.q)
        self.flat = _flat(self.keys)
        self.linear = self._mixed_in_linear()

    @property
    def shape(self) -> tuple[int, int]:
        return self.keys.shape

    def _vec(self, ys, xs) -> np.ndarray:
        return self.q[ys, xs].astype(np.float32) / np.float32(_Q)

    def sample(self, x: float, y: float) -> np.ndarray | None:
        """The ID colour (premultiplied) under source pixel (x, y). On an anti-aliased edge
        pixel, the most common flat colour in the 5×5 around it (you meant the object)."""
        h, w = self.shape
        ix, iy = int(np.floor(x)), int(np.floor(y))
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        if self.flat[iy, ix]:
            return self._vec(iy, ix)
        y0, x0 = max(0, iy - 2), max(0, ix - 2)
        flat = self.flat[y0:iy + 3, x0:ix + 3]
        if not flat.any():
            return self._vec(iy, ix)
        fy, fx = np.nonzero(flat)
        keys = self.keys[y0:iy + 3, x0:ix + 3][fy, fx]
        _, idx, counts = np.unique(keys, return_index=True, return_counts=True)
        best = idx[np.argmax(counts)]
        return self._vec(y0 + fy[best], x0 + fx[best])

    def mask(self, colour: np.ndarray, tolerance: float = 0.0, soft: bool = True) -> np.ndarray:
        """Mask (float32, 0–1, pass resolution) of everything with ID `colour` (premultiplied,
        as `sample` returns it). `tolerance` 0–1 widens the match (noisy or compressed passes)."""
        colour = np.asarray(colour, np.float32)
        cq = np.round(np.clip(colour, 0.0, 1.0) * _Q).astype(np.uint16)
        if tolerance <= 0:
            hard = self.keys == _keys(cq[None, :])[0]
        else:
            thr = int(round((float(tolerance) + EPS) * _Q))
            hard = np.abs(self.q.astype(np.int32) - cq.astype(np.int32)).max(axis=2) <= thr
        mask = hard.astype(np.float32)
        if not soft or not hard.any() or hard.all():
            return mask

        # Only pixels just outside the match can be edge mixes.
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * EDGE_PX + 1, 2 * EDGE_PX + 1))
        near = cv2.dilate(hard.view(np.uint8), k).astype(bool) & ~hard
        ys, xs = np.nonzero(near)
        if len(ys) == 0:
            return mask
        b1, b2 = self._neighbour_colours(hard, ys, xs)
        p = self._vec(ys, xs)
        if self.linear:
            p, colour, b1, b2 = _linear(p), _linear(colour[None, :])[0], _linear(b1), _linear(b2)
        mask[ys, xs] = _solve(p, colour, b1, b2)[0]
        return mask

    def _mixed_in_linear(self, samples: int = 20000) -> bool:
        """Were this pass's edges blended in linear light (then sRGB-encoded)? Fit edge pixels
        as mixes of their two most different flat neighbours in both spaces; the space with
        the straighter mixes wins. No evidence (no edges, or only mixes along an axis, which
        stay straight either way) → the stored values."""
        ys, xs = np.nonzero(~self.flat)
        if len(ys) == 0:
            return False
        if len(ys) > samples:
            pick = np.random.default_rng(0).choice(len(ys), samples, replace=False)
            ys, xs = ys[pick], xs[pick]
        b1, b2 = self._neighbour_colours(np.zeros(self.shape, bool), ys, xs)
        useful = np.abs(b1 - b2).max(axis=1) > 0.1
        if useful.sum() < 16:
            return False
        p, b1, b2 = self._vec(ys, xs)[useful], b1[useful], b2[useful]
        err = _line_error(p, b1, b2)
        err_lin = _line_error(_linear(p), _linear(b1), _linear(b2))
        return float(err_lin.mean()) < 0.5 * float(err.mean())

    def _neighbour_colours(self, hard: np.ndarray, ys: np.ndarray, xs: np.ndarray):
        """b1 = the nearest flat colour that isn't the pick; b2 = the one in the window most
        different from b1 (a third colour at a corner). Transparent where there's none."""
        r = EDGE_PX + 2
        offs = sorted(((dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)),
                      key=lambda o: o[0] * o[0] + o[1] * o[1])
        h, w = self.shape
        n = len(ys)

        def candidate(dy, dx):
            yy, xx = ys + dy, xs + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            iy, ix = yy[ok], xx[ok]
            ok[ok] = self.flat[iy, ix] & ~hard[iy, ix]  # flat, and another ID (or background)
            c = np.zeros((n, 4), np.float32)
            c[ok] = self._vec(yy[ok], xx[ok])
            return ok, c

        b1 = np.zeros((n, 4), np.float32)
        have = np.zeros(n, bool)
        for dy, dx in offs:  # nearest first
            ok, c = candidate(dy, dx)
            take = ok & ~have
            b1[take] = c[take]
            have |= take
        b2 = b1.copy()
        best = np.zeros(n, np.float32)
        for dy, dx in offs:  # recomputed, not kept: 81 candidates × edge pixels adds up
            ok, c = candidate(dy, dx)
            d = np.where(ok, np.abs(c - b1).max(axis=1), -1.0)
            better = d > best
            b2[better] = c[better]
            best = np.maximum(best, d)
        return b1, b2


def _line_error(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of each p from the segment a–b."""
    ab = a - b
    t = np.clip(((p - b) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-10), 0.0, 1.0)
    return np.linalg.norm(p - b - t[:, None] * ab, axis=1)


def _solve(p: np.ndarray, A: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage of A in each pixel, and the relative error of the fit.

    p ≈ a·A + c·b1 + (1 − a − c)·b2 by least squares, then clamped to a, c ≥ 0, a + c ≤ 1.
    Where b1 and b2 are the same colour, or the three are (nearly) collinear, it's the
    two-colour mix of A and b1."""
    A = A[None, :]
    u, v, d = A - b2, b1 - b2, p - b2
    uu, vv, uv = (u * u).sum(1), (v * v).sum(1), (u * v).sum(1)
    ud, vd = (u * d).sum(1), (v * d).sum(1)
    det = uu * vv - uv * uv
    three = (vv > 1e-8) & (det > 1e-4 * np.maximum(uu * vv, 1e-12))
    n = len(p)
    a, c = np.zeros(n, np.float32), np.zeros(n, np.float32)
    a[three] = (ud[three] * vv[three] - vd[three] * uv[three]) / det[three]
    c[three] = (vd[three] * uu[three] - ud[three] * uv[three]) / det[three]
    u1 = A - b1  # two-colour mix A–b1 where there's no usable third colour
    uu1 = (u1 * u1).sum(1)
    two = ~three
    a[two] = (u1[two] * (p[two] - b1[two])).sum(1) / np.maximum(uu1[two], 1e-10)
    a = np.clip(a, 0.0, 1.0)
    c[two] = 1.0 - a[two]  # no b2 term
    c = np.clip(c, 0.0, 1.0 - a)
    fit = a[:, None] * A + c[:, None] * b1 + (1.0 - a - c)[:, None] * b2
    err = np.linalg.norm(p - fit, axis=1) / np.maximum(np.sqrt(np.maximum(uu1, uu)), 1e-6)
    a[err > MIX_RESIDUAL] = 0.0
    return a.astype(np.float32), err
