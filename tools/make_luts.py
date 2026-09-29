"""Dev-only: generate the bundled looks (lookbox/luts/*.cube). Run from the repo root:
python tools/make_luts.py. Pure numpy formulas on display-referred sRGB, so they suit
renders and photos alike; strength in the app blends them back toward the original."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lookbox.core.render.lut import identity_table, to_cube  # noqa: E402

N = 21
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def luma(c):
    return (c * LUMA).sum(-1, keepdims=True)


def s_curve(x, k):  # contrast around mid grey, keeps 0 and 1
    return np.clip(0.5 + (x - 0.5) * (1 + k) / (1 + k * np.abs(2 * x - 1)), 0, 1)


def sat(c, k):
    y = luma(c)
    return y + (c - y) * k


def lift_gain(c, lift, gain):
    return lift + c * (gain - lift)


def tint_by_luma(c, shadow_rgb, highlight_rgb, amount):
    y = luma(c)
    tone = np.asarray(shadow_rgb) * (1 - y) + np.asarray(highlight_rgb) * y
    return c + (tone - 0.5) * amount


LOOKS = {
    "Warm film": lambda c: s_curve(lift_gain(sat(c * [1.06, 1.0, 0.9], 0.92), 0.04, 0.98), 0.25),
    "Teal & orange": lambda c: s_curve(tint_by_luma(c, (0.35, 0.55, 0.62), (0.68, 0.52, 0.36), 0.35), 0.2),
    "Cool studio": lambda c: s_curve(c * [0.95, 1.0, 1.07], 0.3),
    "Bleach bypass": lambda c: s_curve(sat(c, 0.45), 0.55),
    "Soft matte": lambda c: lift_gain(s_curve(sat(c, 0.9), -0.15), 0.07, 0.93),
    "Punchy": lambda c: s_curve(sat(c, 1.3), 0.35),
    "Mono": lambda c: np.repeat(s_curve(luma(c), 0.25), 3, axis=-1),
    "Vintage fade": lambda c: lift_gain(tint_by_luma(sat(c, 0.75), (0.55, 0.45, 0.58), (0.6, 0.55, 0.4), 0.3),
                                        0.08, 0.95),
}


if __name__ == "__main__":
    out_dir = os.path.join("lookbox", "luts")
    os.makedirs(out_dir, exist_ok=True)
    grid = identity_table(N)
    for name, fn in LOOKS.items():
        table = np.clip(fn(grid.astype(np.float32)), 0, 1).astype(np.float32)
        path = os.path.join(out_dir, name.lower().replace(" & ", "_").replace(" ", "_") + ".cube")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(to_cube(table, name))
        print("wrote", path)
