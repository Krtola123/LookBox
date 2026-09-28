"""Deterministic test images for the adjust tests and golden files.

1. ramp   — luminance ramp × hue sweep, opaque (tone curves, colour).
2. render — shaded coloured "object" on a transparent background, like a
            Toolbag render (alpha-edge behaviour).
3. photo  — smooth colourful blobs + fine texture, opaque (local contrast, sharpening).
"""

from __future__ import annotations

import cv2
import numpy as np

W, H = 192, 128


def ramp() -> np.ndarray:
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    hue = np.linspace(0, 359, H, dtype=np.float32)[:, None]
    hsv = np.stack([np.broadcast_to(hue, (H, W)), np.full((H, W), 0.8, np.float32),
                    np.broadcast_to(x, (H, W))], axis=2).astype(np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[: H // 4] = x[:, :, None]  # a neutral grey ramp band on top
    return np.concatenate([rgb, np.ones((H, W, 1), np.float32)], axis=2)


def render(bg_rgb: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy, r = W * 0.5, H * 0.5, H * 0.38
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip((r - d) / 1.5 + 0.5, 0, 1)  # anti-aliased edge
    nz = np.sqrt(np.clip(1 - (d / r) ** 2, 0, 1))
    shade = np.clip(0.15 + 0.85 * (nz * 0.7 + (cx - xx) / r * 0.3), 0, 1)
    spec = np.exp(-(((xx - cx + r * 0.35) ** 2 + (yy - cy + r * 0.35) ** 2) / (r * 0.12) ** 2))
    rgb = np.stack([0.85 * shade, 0.35 * shade, 0.2 * shade], axis=2) + spec[:, :, None] * 0.9
    rgb = np.where(alpha[:, :, None] > 0, rgb, bg_rgb)
    return np.concatenate([rgb, alpha[:, :, None]], axis=2).astype(np.float32)


def photo() -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.random((H // 16, W // 16, 3)).astype(np.float32)
    base = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)
    tex = rng.random((H, W, 1)).astype(np.float32) * 0.15
    rgb = np.clip(base * 0.85 + tex, 0, 1)
    return np.concatenate([rgb, np.ones((H, W, 1), np.float32)], axis=2).astype(np.float32)


IMAGES = {"ramp": ramp, "render": render, "photo": photo}
