"""Time the Adjust pipeline: `python -m tools.bench_adjust`.

Preview sizes are what a slider drag renders (a layer at screen resolution);
the 6000×4000 row is export. Run on the target machine; numbers vary by CPU.
"""

import time

import numpy as np

from lookbox.core.model import ADJUST_SLIDERS, Adjustments, ColorBand
from lookbox.core.render import adjust as A


def bench(w: int, h: int, adj: Adjustments, repeat: int = 3) -> float:
    rng = np.random.default_rng(0)
    img = rng.random((h, w, 4), dtype=np.float32)
    img[:, :, 3] = 1.0
    A.apply(img[:, :, :3], img[:, :, 3], adj)  # warm-up
    t = time.perf_counter()
    for _ in range(repeat):
        A.apply(img[:, :, :3], img[:, :, 3], adj)
    return (time.perf_counter() - t) / repeat * 1000.0


if __name__ == "__main__":
    everything = Adjustments(color_edit=[ColorBand(hue=20, hue_shift=30, saturation=20)],
                             **{k: 40.0 for k in ADJUST_SLIDERS})
    for label, (w, h) in [("preview 1024×683", (1024, 683)), ("preview 2048×1365", (2048, 1365)),
                          ("export 6000×4000", (6000, 4000))]:
        rep = 1 if w > 3000 else 3
        print(f"{label:20s}  one slider (brightness): {bench(w, h, Adjustments(brightness=40), rep):7.0f} ms"
              f"   all 15 on: {bench(w, h, everything, rep):7.0f} ms")
