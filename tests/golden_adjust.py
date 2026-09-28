"""Golden-image data for the Adjust panel (ARCHITECTURE §14).

Each control at −100, −50, +50, +100 on the 3 test images, stored as float16
in tests/golden/adjust_<control>.npz. Regenerate only deliberately:
    pytest --update-golden          (or: python -m tests.golden_adjust)
and say in the commit which controls changed and why (§16.6).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from lookbox.core.model import ADJUST_SLIDERS, Adjustments, ColorBand
from lookbox.core.render import adjust as A
from tests import adjust_images as IM

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
VALUES = (-100, -50, 50, 100)
CONTROLS = ADJUST_SLIDERS + ("invert", "color_edit")
TOLERANCE = 2e-3  # float16 storage error is ~5e-4 near 1.0


def settings(control: str, value: int) -> Adjustments:
    if control == "invert":
        return Adjustments(invert=value > 0)
    if control == "color_edit":
        return Adjustments(color_edit=[ColorBand(hue=20.0, hue_shift=value, saturation=value / 2,
                                                 lightness=-value / 2)])
    return Adjustments(**{control: float(value)})


def golden_input(make) -> np.ndarray:
    """Half-size test image: plenty to catch a changed look, 4× smaller files."""
    img = make()
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)


def compute(control: str) -> dict[str, np.ndarray]:
    out = {}
    for name, make in IM.IMAGES.items():
        img = golden_input(make)
        for v in VALUES:
            out[f"{name}_{v}"] = A.apply(img[:, :, :3], img[:, :, 3], settings(control, v))
    return out


def path(control: str) -> str:
    return os.path.join(GOLDEN_DIR, f"adjust_{control}.npz")


def write_all() -> list[str]:
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for c in CONTROLS:
        np.savez_compressed(path(c), **{k: v.astype(np.float16) for k, v in compute(c).items()})
    return [path(c) for c in CONTROLS]


if __name__ == "__main__":
    for p in write_all():
        print("wrote", p)
