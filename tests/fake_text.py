"""A deterministic stand-in for the Qt text engine (tests only).

Every character is 0.5 em wide (plus letter spacing); a non-space character is
drawn as a solid block from 0.7 em above the baseline to the baseline, 0.4 em
wide. Crude, but exact, so layout and rendering can be checked to the pixel.
"""

from __future__ import annotations

import numpy as np

from lookbox.core.render.text import FontMetrics, Layout

ADV_EM, INK_EM, CAP_EM = 0.5, 0.4, 0.7


class FakeTextEngine:
    def metrics(self, layer) -> FontMetrics:
        return FontMetrics(ascent=0.8 * layer.font_size, descent=0.2 * layer.font_size)

    def advance(self, layer, s: str) -> float:
        return len(s) * ADV_EM * layer.font_size  # letter spacing is added by the core

    def coverage(self, layer, layout: Layout, w: int, h: int) -> np.ndarray:
        # Supersampled 4×, then box-filtered: blocks land at fractional positions.
        ss = 4
        big = np.zeros((h * ss, w * ss), np.float32)
        kx, ky = w * ss / layout.width, h * ss / layout.height
        step = ADV_EM * layer.font_size + layer.letter_spacing
        for line in layout.lines:
            for i, ch in enumerate(line.text):
                if ch == " ":
                    continue
                x0 = line.x + i * step
                y0 = line.baseline - CAP_EM * layer.font_size
                big[round(y0 * ky):round(line.baseline * ky),
                    round(x0 * kx):round((x0 + INK_EM * layer.font_size) * kx)] = 1.0
        return big.reshape(h, ss, w, ss).mean(axis=(1, 3))
