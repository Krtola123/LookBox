"""LUT filters (ARCHITECTURE §7a, M10): Adobe/Resolve `.cube` files.

A 3D LUT maps each colour to a new one through a grid (e.g. 33 × 33 × 33 points),
interpolated tetrahedrally in between; a 1D LUT maps each channel through a curve.
Both are applied to the display-referred sRGB values the pipeline works in, which
is what creative .cube looks are made for. `strength` blends with the original.

.cube format (the parts that matter): `LUT_3D_SIZE n` or `LUT_1D_SIZE n`, optional
`DOMAIN_MIN r g b` / `DOMAIN_MAX r g b` (default 0 / 1), optional `TITLE "..."`,
`#` comments, then n³ (or n) lines of "r g b" with red changing fastest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

MAX_3D_SIZE = 129
MAX_1D_SIZE = 65536
_CHUNK = 1 << 18  # pixels per pass (and per thread): keeps temporaries small


class LutError(ValueError):
    """A .cube file we can't use; the message can be shown to the user as-is."""


@dataclass(frozen=True, eq=False)
class Lut:
    table: np.ndarray  # 3D: (n, n, n, 3) indexed [b, g, r]; 1D: (n, 3)
    dims: int  # 1 or 3
    domain_min: np.ndarray
    domain_max: np.ndarray
    title: str = ""

    @property
    def size(self) -> int:
        return self.table.shape[0]


def parse_cube(data: bytes | str) -> Lut:
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    size3 = size1 = None
    dmin, dmax, title = np.zeros(3, np.float32), np.ones(3, np.float32), ""
    rows: list[list[float]] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head = line.split(None, 1)
        key = head[0].upper()
        try:
            if key == "TITLE":
                title = head[1].strip().strip('"') if len(head) > 1 else ""
            elif key == "LUT_3D_SIZE":
                size3 = int(head[1])
            elif key == "LUT_1D_SIZE":
                size1 = int(head[1])
            elif key == "DOMAIN_MIN":
                dmin = np.array([float(v) for v in head[1].split()[:3]], np.float32)
            elif key == "DOMAIN_MAX":
                dmax = np.array([float(v) for v in head[1].split()[:3]], np.float32)
            elif key in ("LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE"):  # older Resolve syntax
                lo, hi = (float(v) for v in head[1].split()[:2])
                dmin, dmax = np.full(3, lo, np.float32), np.full(3, hi, np.float32)
            elif key[0].isdigit() or key[0] in "-+.":
                vals = [float(v) for v in line.split()]
                if len(vals) != 3:
                    raise LutError(f"Line {n} should have 3 numbers (red green blue).")
                rows.append(vals)
            # anything else (e.g. LUT_IN_VIDEO_RANGE, vendor keywords) is ignored
        except (ValueError, IndexError) as exc:
            if isinstance(exc, LutError):
                raise
            raise LutError(f"Line {n} can't be read: {raw.strip()[:40]!r}.") from exc
    if size3 is None and size1 is None:
        raise LutError("This isn't a .cube LUT (no LUT_3D_SIZE or LUT_1D_SIZE).")
    if np.any(dmax <= dmin):
        raise LutError("The LUT's DOMAIN_MAX must be above DOMAIN_MIN.")
    table = np.asarray(rows, np.float32)
    if size3 is not None:
        if not 2 <= size3 <= MAX_3D_SIZE:
            raise LutError(f"LUT size {size3} isn't supported (2–{MAX_3D_SIZE}).")
        if len(table) != size3 ** 3:
            raise LutError(f"The LUT should have {size3 ** 3} entries, it has {len(table)}.")
        return Lut(table.reshape(size3, size3, size3, 3), 3, dmin, dmax, title)
    if not 2 <= size1 <= MAX_1D_SIZE:
        raise LutError(f"LUT size {size1} isn't supported (2–{MAX_1D_SIZE}).")
    if len(table) != size1:
        raise LutError(f"The LUT should have {size1} entries, it has {len(table)}.")
    return Lut(table, 1, dmin, dmax, title)


def to_cube(table: np.ndarray, title: str = "") -> str:
    """Write a 3D table (n, n, n, 3) indexed [b, g, r] as .cube text."""
    n = table.shape[0]
    lines = [f'TITLE "{title}"'] if title else []
    lines.append(f"LUT_3D_SIZE {n}")
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in table.reshape(-1, 3)]
    return "\n".join(lines) + "\n"


def identity_table(n: int) -> np.ndarray:
    """(n, n, n, 3) [b, g, r] → the colour at that grid point."""
    v = np.linspace(0.0, 1.0, n, dtype=np.float32)
    b, g, r = np.meshgrid(v, v, v, indexing="ij")
    return np.stack([r, g, b], axis=-1)


def _apply3d(x: np.ndarray, lut: Lut) -> np.ndarray:
    """Tetrahedral interpolation (what Resolve uses): 4 grid points per pixel instead of
    trilinear's 8, and no hue shifts along the grey axis. Exact for linear maps."""
    n = lut.size
    flat = lut.table.reshape(-1, 3)
    u = np.clip((x - lut.domain_min) / (lut.domain_max - lut.domain_min), 0.0, 1.0) * (n - 1)
    i0 = np.minimum(u.astype(np.int32), n - 2)
    f = (u - i0).astype(np.float32)
    base = (i0[:, 2] * n + i0[:, 1]) * n + i0[:, 0]
    order = np.argsort(-f, axis=1, kind="stable")  # axes by fraction, largest first
    fs = np.take_along_axis(f, order, axis=1)
    step = np.array([1, n, n * n], np.int32)[order]  # grid offset along each of those axes
    v1 = base + step[:, 0]
    v2 = v1 + step[:, 1]
    a, b, c = fs[:, 0:1], fs[:, 1:2], fs[:, 2:3]
    return (1.0 - a) * flat[base] + (a - b) * flat[v1] + (b - c) * flat[v2] + c * flat[base + (1 + n + n * n)]


def _apply1d(x: np.ndarray, lut: Lut) -> np.ndarray:
    n = lut.size
    u = np.clip((x - lut.domain_min) / (lut.domain_max - lut.domain_min), 0.0, 1.0) * (n - 1)
    out = np.empty_like(x)
    grid = np.arange(n, dtype=np.float32)
    for ch in range(3):
        out[:, ch] = np.interp(u[:, ch], grid, lut.table[:, ch])
    return out


def apply(rgb: np.ndarray, lut: Lut, strength: float = 1.0) -> np.ndarray:
    """Straight float RGB (…, 3) through the LUT, blended by `strength` (0–1)."""
    if strength <= 0.0:
        return rgb
    shape = rgb.shape
    src = np.ascontiguousarray(rgb, np.float32).reshape(-1, 3)
    out = np.empty_like(src)
    fn = _apply3d if lut.dims == 3 else _apply1d
    k = np.float32(min(strength, 1.0))

    def run(s: int) -> None:
        part = src[s:s + _CHUNK]
        graded = fn(part, lut)
        out[s:s + _CHUNK] = graded if k >= 1.0 else part + (graded - part) * k

    starts = range(0, len(src), _CHUNK)
    if len(starts) > 1:  # numpy lets go of the GIL in these loops: use the cores
        list(_pool().map(run, starts))
    else:
        run(0)
    return out.reshape(shape)


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _POOL = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1), thread_name_prefix="lut")
    return _POOL
