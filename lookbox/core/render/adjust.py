"""The Adjust panel's math (ARCHITECTURE §7). One function per control, applied
in `apply` in the fixed §7 order.

Works on straight (un-premultiplied) float32 RGB, sRGB-encoded, and only
clips at the very end. Pixel radii scale with `level` so a preview level
looks like a downscaled export (§6.3). Blurs are alpha-aware, so transparent
backgrounds of renders don't bleed dark halos into object edges.
"""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np

from lookbox.core.model import Adjustments, ColorBand

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec.709
BAND_HALF_WIDTH_DEG = 45.0  # colour-edit falloff: raised cosine, 0 at ±45°


class Cancelled(Exception):
    pass


# ------------------------------------------------------------------ helpers


def _mat(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Per-pixel colour matrix. cv2.transform is multithreaded and ~2.5× faster than rgb @ m.T."""
    return cv2.transform(np.ascontiguousarray(rgb, dtype=np.float32), m)


def luma(rgb: np.ndarray) -> np.ndarray:
    return _mat(rgb, LUMA[None, :])


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    """Hermite step; works with e0 > e1 too (falling edge)."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, fast for big radii: blur a downscaled copy and scale back.

    The downscale factor is proportional to sigma, so the same *relative*
    radius gives the same result at every preview level.
    """
    if sigma < 0.3:
        return x
    if sigma <= 6.0:
        return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    h, w = x.shape[:2]
    f = sigma / 3.0
    sw, sh = max(1, round(w / f)), max(1, round(h / f))
    small = cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3.0, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def masked_blur(x: np.ndarray, alpha: np.ndarray, sigma: float) -> np.ndarray:
    """Blur that ignores transparent pixels (normalised convolution)."""
    if sigma < 0.3:
        return x
    if alpha.min() >= 0.999:
        return blur(x, sigma)
    a = alpha if x.ndim == 2 else alpha[:, :, None]
    num = blur(x * a, sigma)
    den = blur(alpha, sigma)
    den = den if x.ndim == 2 else den[:, :, None]
    return np.where(den > 1e-4, num / np.maximum(den, 1e-4), x)


def _renorm_luma(rgb: np.ndarray, target_l: np.ndarray) -> np.ndarray:
    cur = luma(rgb)
    ratio = target_l / np.maximum(cur, 1e-6)
    ratio[cur <= 1e-6] = 1.0
    rgb *= ratio[:, :, None]
    return rgb


def _chroma_scale(rgb: np.ndarray, factor: np.ndarray | float) -> np.ndarray:
    """Scale saturation around the pixel's own grey (luma-preserving)."""
    lum = luma(rgb)[:, :, None]
    f = factor[:, :, None] if isinstance(factor, np.ndarray) else factor
    return lum + (rgb - lum) * f


# ------------------------------------------------------------------ controls (§7 order)


def _white_balance(rgb: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    # Channel gains then luma renormalisation, folded into one per-pixel ratio.
    g = np.asarray(gains, dtype=np.float32)
    target = luma(rgb)
    cur = _mat(rgb, (LUMA * g)[None, :])
    ratio = target / np.maximum(cur, 1e-6)
    ratio[cur <= 1e-6] = 1.0
    return rgb * (ratio[:, :, None] * g)


def temperature(rgb: np.ndarray, t: float) -> np.ndarray:
    return _white_balance(rgb, (1 + 0.3 * t, 1.0, 1 - 0.3 * t))


def tint(rgb: np.ndarray, t: float) -> np.ndarray:
    return _white_balance(rgb, (1.0, 1 - 0.3 * t, 1.0))  # + = magenta


def brightness(rgb: np.ndarray, t: float) -> np.ndarray:
    return np.power(np.maximum(rgb, 0.0), np.float32(2.0 ** -t))  # 0 and 1 stay put


def contrast(rgb: np.ndarray, t: float) -> np.ndarray:
    if t > 0:
        x = np.clip(rgb, 0.0, 1.0)
        s = x * x * (3.0 - 2.0 * x)  # smoothstep(0, 1, x), inlined
        return rgb + np.float32(t) * (s - rgb)
    return rgb + (0.5 - rgb) * (0.6 * -t)  # tuned: −100 flattens hard but isn't pure grey


def highlights(rgb: np.ndarray, t: float, l_blur: np.ndarray) -> np.ndarray:
    w = smoothstep(0.5, 1.0, l_blur)
    return rgb * (1.0 + 0.5 * t * w)[:, :, None]


def shadows(rgb: np.ndarray, t: float, l_blur: np.ndarray) -> np.ndarray:
    w = smoothstep(0.5, 0.0, l_blur)[:, :, None]
    if t > 0:
        return rgb + 0.5 * t * w * (1.0 - rgb)
    return rgb * (1.0 + 0.5 * t * w)


def whites(rgb: np.ndarray, t: float) -> np.ndarray:
    return rgb / np.float32(1.0 - 0.25 * t)


def blacks(rgb: np.ndarray, t: float) -> np.ndarray:
    black = np.float32(-0.25 * t)
    return (rgb - black) / (1.0 - black)


# CSS hue-rotate matrix split as M0 + cos·M1 + sin·M2 (luma-preserving).
_HUE_M0 = np.array([[0.213, 0.715, 0.072], [0.213, 0.715, 0.072], [0.213, 0.715, 0.072]], np.float32)
_HUE_M1 = np.array([[0.787, -0.715, -0.072], [-0.213, 0.285, -0.072], [-0.213, -0.715, 0.928]], np.float32)
_HUE_M2 = np.array([[-0.213, -0.715, 0.928], [0.143, 0.140, -0.283], [-0.787, 0.715, 0.072]], np.float32)


def _hue_rotate(rgb: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """Per-pixel hue rotation: three fixed colour transforms blended by cos/sin."""
    r = np.radians(deg).astype(np.float32, copy=False)
    c, s = np.cos(r)[:, :, None], np.sin(r)[:, :, None]
    return _mat(rgb, _HUE_M0) + c * _mat(rgb, _HUE_M1) + s * _mat(rgb, _HUE_M2)


def _hue_sat(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False), cv2.COLOR_RGB2HSV)
    return hsv[:, :, 0], smoothstep(0.05, 0.2, hsv[:, :, 1])


def band_weights(rgb: np.ndarray, band_hue: float, hue_sat=None) -> np.ndarray:
    """How much each pixel belongs to a colour band: hue closeness × 'has colour'."""
    hue, colourful = hue_sat if hue_sat is not None else _hue_sat(rgb)
    d = np.abs(((hue - band_hue) + 180.0) % 360.0 - 180.0)
    w = 0.5 * (1.0 + np.cos(np.minimum(d, BAND_HALF_WIDTH_DEG) * np.float32(np.pi / BAND_HALF_WIDTH_DEG)))
    return (w * colourful).astype(np.float32, copy=False)


def color_edit(rgb: np.ndarray, bands: list[ColorBand]) -> np.ndarray:
    active = [b for b in bands if b.hue_shift or b.saturation or b.lightness]
    if not active:
        return rgb
    shift = np.zeros(rgb.shape[:2], np.float32)
    sat = np.zeros_like(shift)
    light = np.zeros_like(shift)
    hs = _hue_sat(rgb)  # weights come from the colours *before* this step
    for b in active:
        w = band_weights(rgb, b.hue, hs)
        shift += w * (b.hue_shift / 100.0 * 30.0)
        sat += w * (b.saturation / 100.0)
        light += w * (b.lightness / 100.0)
    out = _hue_rotate(rgb, shift) if np.any(shift) else rgb
    out = _chroma_scale(out, 1.0 + sat)
    return out * (1.0 + 0.5 * light)[:, :, None]


def vibrance(rgb: np.ndarray, t: float) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)  # ~10× faster than rgb.max(axis=2)
    mn = np.minimum(np.minimum(r, g), b)
    s = np.clip((mx - mn) / np.maximum(mx, 1e-6), 0.0, 1.0)
    return _chroma_scale(rgb, 1.0 + t * (1.0 - s))  # muted colours move most; greys never


def saturation(rgb: np.ndarray, t: float) -> np.ndarray:
    return _chroma_scale(rgb, 1.0 + t)


def clarity(rgb: np.ndarray, t: float, alpha: np.ndarray, diag: float) -> np.ndarray:
    lum = luma(rgb)
    detail = lum - masked_blur(lum, alpha, 0.015 * diag)
    lc = np.clip(lum, 0.0, 1.0)
    return rgb + (0.6 * t * detail * 4.0 * lc * (1.0 - lc))[:, :, None]


def sharpness(rgb: np.ndarray, t: float, alpha: np.ndarray, level: float) -> np.ndarray:
    sigma = 1.0 * level  # 1 px at full resolution
    if t > 0:
        lum = luma(rgb)
        return rgb + (1.5 * t * (lum - masked_blur(lum, alpha, sigma)))[:, :, None]
    return rgb + (-t) * (masked_blur(rgb, alpha, sigma) - rgb)  # negative = slight softening


def vignette(rgb: np.ndarray, t: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    u = (np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0
    v = (np.arange(h, dtype=np.float32) + 0.5) / h * 2.0 - 1.0
    r = np.sqrt(u[None, :] ** 2 + v[:, None] ** 2) / math.sqrt(2.0)  # 1 at the corners
    return rgb * (1.0 + 0.8 * t * smoothstep(0.3, 1.0, r))[:, :, None]  # negative darkens


# ------------------------------------------------------------------ pipeline stage


def apply(rgb: np.ndarray, alpha: np.ndarray, adj: Adjustments, level: float = 1.0,
          cancel: threading.Event | None = None) -> np.ndarray:
    """Apply all adjustments in §7 order. `rgb` straight float32 (H, W, 3); returns clipped 0–1."""
    def step() -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    h, w = rgb.shape[:2]
    diag = math.hypot(w, h)
    v = {k: getattr(adj, k) / 100.0 for k in ("temperature", "tint", "brightness", "contrast",
                                              "highlights", "shadows", "whites", "blacks",
                                              "vibrance", "saturation", "clarity", "sharpness",
                                              "vignette")}
    out = rgb.astype(np.float32, copy=True)
    if v["temperature"]:
        out = temperature(out, v["temperature"])
    if v["tint"]:
        out = tint(out, v["tint"])
    if v["brightness"]:
        out = brightness(out, v["brightness"])
    if v["contrast"]:
        out = contrast(out, v["contrast"])
    step()
    if v["highlights"] or v["shadows"]:
        # One local-luma map feeds both (computed once: it's the expensive part).
        l_blur = masked_blur(luma(out), alpha, 0.01 * diag)
        if v["highlights"]:
            out = highlights(out, v["highlights"], l_blur)
        if v["shadows"]:
            out = shadows(out, v["shadows"], l_blur)
    if v["whites"]:
        out = whites(out, v["whites"])
    if v["blacks"]:
        out = blacks(out, v["blacks"])
    step()
    out = color_edit(out, adj.color_edit)
    if v["vibrance"]:
        out = vibrance(out, v["vibrance"])
    if v["saturation"]:
        out = saturation(out, v["saturation"])
    step()
    if v["clarity"]:
        out = clarity(out, v["clarity"], alpha, diag)
    if v["sharpness"]:
        out = sharpness(out, v["sharpness"], alpha, level)
    if v["vignette"]:
        out = vignette(out, v["vignette"])
    if adj.invert:
        out = 1.0 - out
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def suggest_swatch_hues(rgba: np.ndarray, k: int = 3) -> list[float]:
    """Dominant hues of an image (Canva's 'Color edit' swatches). Greys are ignored."""
    px = rgba.reshape(-1, 4)
    step = max(1, len(px) // 20000)
    px = px[::step]
    px = px[px[:, 3] > 0.5, :3]
    if len(px) == 0:
        return []
    hsv = cv2.cvtColor(np.clip(px, 0, 1).astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    colourful = hsv[hsv[:, 1] > 0.2]
    if len(colourful) < 20:
        return []
    ang = np.radians(colourful[:, 0])
    pts = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)  # hue is circular
    k = min(k, len(pts))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cv2.setRNGSeed(1)  # deterministic swatches
    _, labels, centres = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    hues = []
    for i in np.argsort(-counts):
        if counts[i] < 0.03 * len(pts):
            continue
        hue = math.degrees(math.atan2(centres[i, 1], centres[i, 0])) % 360.0
        if all(abs(((hue - h) + 180) % 360 - 180) > 20 for h in hues):  # skip near-duplicates
            hues.append(round(hue, 1))
    return hues
