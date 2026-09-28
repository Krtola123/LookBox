"""Drag frame timing for the M2 acceptance readout. Qt-free, so it's tested."""

from __future__ import annotations

import math

TARGET_FPS = 30.0


class FrameStats:
    def __init__(self, window: int = 120) -> None:
        self.window = window
        self.costs: list[float] = []  # seconds spent painting each frame
        self.stamps: list[float] = []  # when each frame finished

    def reset(self) -> None:
        self.costs.clear()
        self.stamps.clear()

    def add(self, cost: float, stamp: float) -> None:
        self.costs.append(cost)
        self.stamps.append(stamp)
        if len(self.costs) > self.window:
            del self.costs[0], self.stamps[0]

    def summary(self, final: bool = False) -> str | None:
        """Paint cost (95th percentile) is what the canvas can sustain; the rate is
        how often your mouse actually asked for a frame."""
        if len(self.costs) < 3:
            return None
        ordered = sorted(self.costs)
        cost_ms = ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)] * 1000.0  # nearest-rank p95
        span = self.stamps[-1] - self.stamps[0]
        rate = (len(self.stamps) - 1) / span if span > 0 else 0.0
        verdict = "OK" if cost_ms <= 1000.0 / TARGET_FPS else "SLOW"
        prefix = "Last drag" if final else "Dragging"
        return (f"{prefix}: {cost_ms:.1f} ms/frame (≈{1000.0 / max(cost_ms, 0.1):.0f} fps max, "
                f"{rate:.0f} fps shown) — {verdict}")
