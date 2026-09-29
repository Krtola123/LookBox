"""Whole-design grade preview (ARCHITECTURE §6.3, M10).

The global adjust + LUT act on the flattened image, which the on-screen Qt
composite never has. So while a document has a global grade, a worker renders
the whole composite at screen resolution (`pipeline.render(preview=True)`,
graded) and it's shown as one item over the canvas, with the layer items hidden.

- Any change re-renders it (debounced); the last picture stays up meanwhile.
- While a layer is dragged, or a layer's sliders move, or a mask is being edited,
  the live ungraded layers show instead (instant feedback beats exact colour for
  those few seconds); the grade comes back when you let go.
- While the grade's own sliders move, it renders at half resolution to keep up.
"""

from __future__ import annotations

import copy
import math
import threading

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QGraphicsPixmapItem

from lookbox.core.render.levels import to_display_bgra
from lookbox.core.render.pipeline import Cancelled, premultiply, render
from lookbox.ui.canvas.layer_items import bgra_to_pixmap

MAX_PIXELS = 4_000_000  # the preview never renders more than this (fit-to-screen needs ~2 MP)
DEBOUNCE_MS = 60


class _Emitter(QObject):
    done = Signal(int, object, float)  # serial, BGRA array (or None on failure), scale
    failed = Signal(str)


class _Job(QRunnable):
    def __init__(self, serial, doc, store, scale, cancel, emitter) -> None:
        super().__init__()
        self.serial, self.doc, self.store, self.scale = serial, doc, store, scale
        self.cancel, self.emitter = cancel, emitter

    def run(self) -> None:
        try:
            out = render(self.doc, self.store, self.scale, self.cancel, preview=True)
            bgra = to_display_bgra(premultiply(out), dither=True)
        except Cancelled:
            return
        except Exception as exc:  # shown in the status bar, never swallowed (§16.5)
            self.emitter.failed.emit(f"Whole-design preview failed: {exc}")
            return
        self.emitter.done.emit(self.serial, bgra, self.scale)


class GradePreview(QObject):
    failed = Signal(str)

    def __init__(self, view, editor) -> None:
        super().__init__(view)
        self.view, self.editor = view, editor
        self.item = QGraphicsPixmapItem()
        self.item.setZValue(1e7)  # above every layer; handles are drawn in the foreground anyway
        self.item.setVisible(False)
        view.scene().addItem(self.item)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._emitter = _Emitter()
        self._emitter.done.connect(self._on_done)
        self._emitter.failed.connect(self.failed)
        self._serial = 0
        self._cancel: threading.Event | None = None
        self._suspended: set[str] = set()  # reasons the live layers must show ("drag", "slider", "mask")
        self._fast = False  # the grade's own sliders are moving: half resolution
        self._shown_serial = -1
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._start)
        editor.changed.connect(self.schedule)
        editor.document_replaced.connect(self._on_replaced)
        view.zoom_changed.connect(lambda _z: self.schedule())
        view.layers.live_changed.connect(lambda on: self.suspend("drag", on))

    # ---- control ----
    def suspend(self, reason: str, on: bool) -> None:
        (self._suspended.add if on else self._suspended.discard)(reason)
        self._apply_visibility()
        if not on:
            self.schedule()

    def set_fast(self, on: bool) -> None:
        self._fast = on
        if not on:
            self.schedule()  # full resolution when the slider is let go

    def schedule(self) -> None:
        if not self.editor.doc.has_global_grade():
            self._cancel_job()
            self._shown_serial = -1
            self._apply_visibility()
            return
        self._timer.start()

    # ---- internals ----
    def _active(self) -> bool:
        return self.editor.doc.has_global_grade() and self._shown_serial >= 0 and not self._suspended

    def _apply_visibility(self) -> None:
        on = self._active()
        self.item.setVisible(on)
        self.view.layers.set_covered(on)

    def _on_replaced(self) -> None:
        self._cancel_job()
        self._shown_serial = -1
        self._apply_visibility()
        self.schedule()

    def _cancel_job(self) -> None:
        self._timer.stop()
        if self._cancel is not None:
            self._cancel.set()
            self._cancel = None

    def _scale(self) -> float:
        c = self.editor.doc.canvas
        s = min(1.0, self.view.zoom() * self.view.devicePixelRatioF())
        s = min(s, math.sqrt(MAX_PIXELS / max(1, c.w * c.h)))
        return max(0.02, s / 2.0 if self._fast else s)

    def _start(self) -> None:
        self._cancel_job()
        self._serial += 1
        self._cancel = threading.Event()
        doc = copy.deepcopy(self.editor.doc)  # the worker never sees later edits
        self._pool.start(_Job(self._serial, doc, self.editor.store, self._scale(), self._cancel, self._emitter))

    def _on_done(self, serial: int, bgra, scale: float) -> None:
        if serial != self._serial:
            return  # a newer render is on its way
        self.item.setPixmap(bgra_to_pixmap(bgra))
        c = self.editor.doc.canvas
        h, w = bgra.shape[:2]
        self.item.setTransform(QTransform.fromScale(c.w / w, c.h / h))
        self._shown_serial = serial
        self._apply_visibility()

    def shutdown(self) -> None:
        self._cancel_job()
        self._pool.clear()
        self._pool.waitForDone(2000)
