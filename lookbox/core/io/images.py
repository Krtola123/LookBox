"""Image decode/encode (ARCHITECTURE §6.5, §11).

Everything returned from here is float32 RGBA, straight alpha, 0–1, sRGB-encoded.
"""

from __future__ import annotations

import os
import tempfile

import lookbox  # noqa: F401  (sets OPENCV_IO_ENABLE_OPENEXR before cv2 loads)
import cv2
import numpy as np

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".webp")


class ImageError(Exception):
    """Raised with a message that can be shown to the user as-is."""


def srgb_oetf(linear: np.ndarray) -> np.ndarray:
    """Linear light → sRGB-encoded values (standard piecewise curve)."""
    x = np.clip(linear, 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _to_rgba(img: np.ndarray) -> np.ndarray:
    """OpenCV channel layouts (gray, gray+alpha, BGR, BGRA) → RGBA, same dtype."""
    if img.ndim == 2:
        img = img[:, :, None]
    c = img.shape[2]
    if c == 1:
        rgb = np.repeat(img, 3, axis=2)
        alpha = None
    elif c == 2:
        rgb = np.repeat(img[:, :, :1], 3, axis=2)
        alpha = img[:, :, 1:2]
    elif c == 3:
        rgb = img[:, :, ::-1]
        alpha = None
    elif c == 4:
        rgb = img[:, :, 2::-1]
        alpha = img[:, :, 3:4]
    else:
        raise ImageError(f"Unsupported channel count: {c}.")
    if alpha is None:
        full = {np.uint8: 255, np.uint16: 65535}.get(img.dtype.type, 1.0)
        alpha = np.full(rgb.shape[:2] + (1,), full, dtype=img.dtype)
    return np.concatenate([rgb, alpha], axis=2)


def decode(data: bytes, ext: str, exr_exposure: float = 0.0) -> np.ndarray:
    """Decode encoded image bytes to float32 RGBA (H, W, 4)."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageError(f"Could not read this {ext or 'image'} file.")
    rgba = _to_rgba(img)
    if rgba.dtype == np.uint8:
        out = rgba.astype(np.float32) / 255.0
    elif rgba.dtype == np.uint16:
        out = rgba.astype(np.float32) / 65535.0
    elif rgba.dtype in (np.float32, np.float16, np.float64):
        # Scene-linear (EXR/HDR): exposure, then sRGB transfer, clip. Alpha is not curved.
        out = rgba.astype(np.float32)
        rgb = out[:, :, :3] * np.float32(2.0 ** exr_exposure)
        out[:, :, :3] = srgb_oetf(rgb)
        out = np.clip(out, 0.0, 1.0)
    else:
        raise ImageError(f"Unsupported pixel format: {rgba.dtype}.")
    return np.ascontiguousarray(out, dtype=np.float32)


def read_file(path: str) -> bytes:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ImageError(f"{os.path.basename(path)}: unsupported file type ({ext or 'none'}).")
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as exc:
        raise ImageError(f"Could not open {os.path.basename(path)}: {exc.strerror}.") from exc


# ------------------------------------------------------------------ encode


def quantize8(rgba: np.ndarray, seed: int = 0) -> np.ndarray:
    """float 0–1 → uint8 with triangular (TPDF) dither against gradient banding.

    Values already on the 8-bit grid are left undithered, so an untouched 8-bit
    source round-trips exactly. The RNG is seeded, so export is deterministic.
    """
    scaled = np.clip(rgba, 0.0, 1.0) * 255.0
    on_grid = np.abs(scaled - np.round(scaled)) < 1e-3
    rng = np.random.default_rng(seed)
    noise = rng.random(scaled.shape, dtype=np.float32) - rng.random(scaled.shape, dtype=np.float32)
    noise[on_grid] = 0.0
    return np.clip(np.floor(scaled + 0.5 + noise), 0, 255).astype(np.uint8)


def quantize16(rgba: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgba, 0.0, 1.0) * 65535.0).astype(np.uint16)


def encode_png(rgba: np.ndarray, bits: int = 8) -> bytes:
    if bits == 8:
        q = quantize8(rgba)
    elif bits == 16:
        q = quantize16(rgba)
    else:
        raise ValueError("bits must be 8 or 16")
    bgra = q[:, :, [2, 1, 0, 3]]
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise ImageError("PNG encoding failed.")
    return buf.tobytes()


def atomic_write(path: str, data: bytes) -> None:
    """Write via a temp file + rename, so a failed write never corrupts `path`."""
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".lookbox-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_png(path: str, rgba: np.ndarray, bits: int = 8) -> None:
    atomic_write(path, encode_png(rgba, bits))
