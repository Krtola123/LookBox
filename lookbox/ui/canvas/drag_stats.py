"""Drag frame timing for the canvas (M2 acceptance: > 30 fps): paint cost per frame
while dragging, reported to the status bar. A mixin for CanvasView (kept out of it
for size); the view defines the `frame_stats` signal and `_drag`."""

from __future__ import annotations

import time

from PySide6.QtCore import QTimer

from lookbox.ui.canvas.frame_stats import FrameStats


class DragStatsMixin:
    def _init_stats(self) -> None:
        self._frames = FrameStats()
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(500)
        self._stats_timer.timeout.connect(self._emit_stats)

    def _stats_start(self) -> None:
        self._frames.reset()
        self._stats_timer.start()

    def _stats_stop(self, final: bool) -> None:
        self._stats_timer.stop()
        if final:
            self._emit_stats(final=True)

    def paintEvent(self, e) -> None:
        t0 = time.perf_counter()
        super().paintEvent(e)
        if self._drag is not None:
            now = time.perf_counter()
            self._frames.add(now - t0, now)

    def _emit_stats(self, final: bool = False) -> None:
        text = self._frames.summary(final)
        if text:
            self.frame_stats.emit(text)
