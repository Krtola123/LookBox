"""Background removal with BiRefNet (ARCHITECTURE §10.2). Qt-free.

Pre-processing follows the model's published config (onnx-community,
preprocessor_config.json): 1024 × 1024, RGB, /255, ImageNet mean/std, NCHW.
The output may be logits or already 0–1 depending on the export; it's detected
and a sigmoid applied only when needed.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from lookbox.ai.registry import ModelSpec
from lookbox.core.masks.ops import refine_edge


def preprocess(rgba: np.ndarray, size: int, mean, std) -> np.ndarray:
    """Straight float RGBA (H, W, 4) → (1, 3, size, size) float32.
    Transparent areas are composited over mid-grey so hidden colours don't confuse the model."""
    a = rgba[:, :, 3:4]
    rgb = rgba[:, :, :3] * a + 0.5 * (1.0 - a)
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if (w > size or h > size) else cv2.INTER_LINEAR
    x = cv2.resize(np.ascontiguousarray(rgb, dtype=np.float32), (size, size), interpolation=interp)
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def postprocess(raw: np.ndarray, h: int, w: int) -> np.ndarray:
    """Model output (any of (1,1,S,S), (1,S,S), (S,S)) → float32 mask (h, w) in 0–1."""
    arr = np.asarray(raw, np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.min() < -1e-3 or arr.max() > 1 + 1e-3:  # logits → probabilities
        arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -40, 40)))
    mask = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def remove_background(run: Callable[[np.ndarray], np.ndarray], rgba: np.ndarray, spec: ModelSpec,
                      size: int | None = None, refine: bool = False) -> np.ndarray:
    """Subject mask for the whole source image (H, W), 0–1.

    `run(tensor) → output` is the model call (ModelManager.run bound to a path), so this
    is testable with a stand-in. Pixels that were already transparent stay hidden."""
    h, w = rgba.shape[:2]
    s = size or spec.input_size
    mask = postprocess(run(preprocess(rgba, s, spec.mean, spec.std)), h, w)
    if refine:
        mask = refine_edge(mask, rgba[:, :, :3], model_size=s)
    return (mask * rgba[:, :, 3]).astype(np.float32)
