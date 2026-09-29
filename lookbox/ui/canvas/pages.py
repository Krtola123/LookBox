"""Pages on the canvas (M14, §5a): every page of the project, stacked top to bottom
like Canva. The page being edited is live (its layers are the scene's items, at the
scene origin, exactly as before pages existed); the others are finished pictures
rendered in a worker at screen resolution, refreshed when they change. Click one to
edit it; "+ Add page" sits under the last page.

Coordinates: pages are laid out in *project* space (stacked, centred on one axis). The
scene is project space shifted so the active page's top-left is (0, 0), so nothing in
the layer/handle code knows pages exist. Switching page shifts the scene; the view is
scrolled by the same amount, so nothing jumps on screen.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QTransform
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem

from lookbox.core.model import Document, document_to_dict
from lookbox.core.render.levels import to_display_bgra
from lookbox.core.render.pipeline import premultiply, render
from lookbox.ui.canvas.layer_items import bgra_to_pixmap

GAP_FRACTION = 0.1  # space between pages (holds each page's label), of the tallest page…
GAP_MIN_PX = 44.0  # …but never less than this on screen, so labels don't crowd when zoomed out
MAX_PIXELS = 2_000_000  # per page picture
LABEL = QColor("#9fa0a6")
LABEL_ACTIVE = QColor("#c7a6ff")


def content_key(page: Document) -> str:
    return hashlib.sha1(json.dumps(document_to_dict(page), sort_keys=True).encode()).hexdigest()


class _Emitter(QObject):
    done = Signal(str, str, object, float)  # page id, content key, BGRA, scale


class _Job(QRunnable):
    def __init__(self, page, key, store, scale, emitter) -> None:
        super().__init__()
        self.page, self.key, self.store, self.scale, self.emitter = page, key, store, scale, emitter

    def run(self) -> None:
        try:
            bgra = to_display_bgra(premultiply(render(self.page, self.store, self.scale, preview=True)))
        except Exception:  # a page that can't render just keeps its last picture
            return
        self.emitter.done.emit(self.page.id, self.key, bgra, self.scale)


class _PageItems:
    def __init__(self, scene, checker) -> None:
        self.back = QGraphicsRectItem()
        self.back.setPen(Qt.PenStyle.NoPen)
        self.back.setBrush(checker)
        self.back.setZValue(-1e8)
        self.pix = QGraphicsPixmapItem()
        self.pix.setZValue(-1e8 + 1)
        self.pix.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        scene.addItem(self.back)
        scene.addItem(self.pix)
        self.key, self.scale = "", 0.0  # what the picture shows

    def remove(self, scene) -> None:
        scene.removeItem(self.back)
        scene.removeItem(self.pix)


class PageStrip(QObject):
    add_requested = Signal()  # "+ Add page" clicked

    def __init__(self, view, editor) -> None:
        super().__init__(view)
        self.view, self.editor = view, editor
        self.items: dict[str, _PageItems] = {}
        self._origin = QPointF(0, 0)  # project-space top-left of the active page
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._emitter = _Emitter()
        self._emitter.done.connect(self._on_done)
        self._pending: set[tuple[str, str]] = set()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._render_stale)
        self._sizes: tuple = ()
        editor.changed.connect(self._on_changed)
        editor.pages_changed.connect(self.relayout)
        editor.document_replaced.connect(self._on_replaced)
        view.zoom_changed.connect(self._on_zoom)

    # ---- layout ----
    def _gap(self) -> float:
        z = max(self.view.zoom(), 1e-3)
        return max(GAP_FRACTION * max(p.canvas.h for p in self.editor.project.pages), GAP_MIN_PX / z)

    def _project_rects(self) -> dict[str, QRectF]:
        pages = self.editor.project.pages
        max_w = max(p.canvas.w for p in pages)
        y, out = 0.0, {}
        for p in pages:
            out[p.id] = QRectF((max_w - p.canvas.w) / 2.0, y, p.canvas.w, p.canvas.h)
            y += p.canvas.h + self._gap()
        return out

    def _scene_rects(self) -> dict[str, QRectF]:
        rects = self._project_rects()
        o = rects[self.editor.active].topLeft()
        return {pid: r.translated(-o) for pid, r in rects.items()}

    def rects(self) -> list[QRectF]:
        return list(self._scene_rects().values())

    def add_rect(self) -> QRectF:
        """The "+ Add page" target under the last page."""
        last = self._scene_rects()[self.editor.project.pages[-1].id]
        z = max(self.view.zoom(), 1e-3)
        w, h = 150 / z, 34 / z  # a constant-size button on screen
        return QRectF(last.center().x() - w / 2, last.bottom() + 12 / z, w, h)

    def scene_rect(self) -> QRectF:
        u = QRectF()
        for r in self.rects() + [self.add_rect()]:
            u = u.united(r)
        m = max(u.width(), u.height()) * 0.5
        return u.adjusted(-m, -m, m, m)

    # ---- switching + relayout (the view stays put) ----
    def _shift_view(self, old_origin: QPointF) -> None:
        new_origin = self._project_rects()[self.editor.active].topLeft()
        delta = new_origin - old_origin  # the scene moves by −delta
        centre = self.view.mapToScene(self.view.viewport().rect().center())
        self._origin = new_origin
        self.view.sync()
        self.refresh()
        self.view.centerOn(QPointF(centre.x() - delta.x(), centre.y() - delta.y()))

    def switched(self) -> None:
        """The active page changed (called by the view instead of its usual reset)."""
        self._shift_view(self._origin)

    def relayout(self) -> None:
        self._shift_view(self._origin)

    def _on_replaced(self) -> None:
        if self.editor.page_switch:
            return  # handled by switched()
        for it in self.items.values():
            it.remove(self.view.scene())
        self.items.clear()
        self._origin = QPointF(0, 0)
        self.refresh()

    def _on_zoom(self, _z: float) -> None:
        if len(self.editor.project.pages) > 1:
            self.refresh()  # the gap is partly in screen pixels: other pages move a little
        self._timer.start()

    def _on_changed(self) -> None:
        sizes = tuple((p.id, p.canvas.w, p.canvas.h) for p in self.editor.project.pages)
        if sizes != self._sizes:  # a page was resized (or added/moved): lay out again
            self._sizes = sizes
            self.relayout()
        else:
            self._timer.start()

    # ---- items ----
    def refresh(self) -> None:
        scene, rects = self.view.scene(), self._scene_rects()
        for pid in list(self.items):
            if pid not in rects:
                self.items.pop(pid).remove(scene)
        for page in self.editor.project.pages:
            it = self.items.get(page.id)
            if it is None:
                it = self.items[page.id] = _PageItems(scene, self.view._checker)
            r = rects[page.id]
            active = page.id == self.editor.active
            it.back.setVisible(not active and page.background is None)
            it.back.setRect(r)
            it.pix.setVisible(not active and not it.pix.pixmap().isNull())
            if it.scale > 0:
                it.pix.setTransform(QTransform.fromScale(1 / it.scale, 1 / it.scale) * QTransform.fromTranslate(
                    r.left(), r.top()))
        self._timer.start()
        self.view.viewport().update()

    def _wanted_scale(self, page: Document) -> float:
        s = min(1.0, self.view.zoom() * self.view.devicePixelRatioF())
        return max(0.02, min(s, math.sqrt(MAX_PIXELS / max(1, page.canvas.w * page.canvas.h))))

    def _render_stale(self) -> None:
        for page in self.editor.project.pages:
            if page.id == self.editor.active:
                continue
            it = self.items.get(page.id)
            if it is None:
                continue
            key, scale = content_key(page), self._wanted_scale(page)
            if (it.key == key and it.scale >= scale * 0.7) or (page.id, key) in self._pending:
                continue
            self._pending.add((page.id, key))
            self._pool.start(_Job(copy.deepcopy(page), key, self.editor.store, scale, self._emitter))

    def _on_done(self, page_id: str, key: str, bgra, scale: float) -> None:
        self._pending.discard((page_id, key))
        it = self.items.get(page_id)
        if it is None or not self.editor.project.has_page(page_id):
            return
        if content_key(self.editor.project.page(page_id)) != key:
            return  # changed again meanwhile; a newer render is (or will be) on its way
        it.pix.setPixmap(bgra_to_pixmap(bgra))
        it.key, it.scale = key, scale
        self.refresh()

    # ---- labels + clicks ----
    def draw_labels(self, painter: QPainter) -> None:
        z = self.view.zoom()
        font = QFont()
        font.setPixelSize(12)
        rects = self._scene_rects()
        for i, page in enumerate(self.editor.project.pages, 1):
            r = rects[page.id]
            text = f"Page {i}" + (f" · {page.name}" if page.name else "") + f"   {page.canvas.w} × {page.canvas.h}"
            painter.save()
            painter.translate(r.left(), r.top())
            painter.scale(1 / z, 1 / z)  # constant size on screen
            painter.setFont(font)
            painter.setPen(LABEL_ACTIVE if page.id == self.editor.active else LABEL)
            painter.drawText(QPointF(0, -8), text)
            painter.restore()
        a = self.add_rect()
        painter.save()
        painter.setPen(LABEL)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = painter.pen()
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRoundedRect(a, 8 / z, 8 / z)
        painter.translate(a.center())
        painter.scale(1 / z, 1 / z)
        painter.setFont(font)
        painter.drawText(QRectF(-100, -12, 200, 24), Qt.AlignmentFlag.AlignCenter, "+  Add page")
        painter.restore()

    def activate_at(self, x: float, y: float) -> bool:
        """A click at scene (x, y) outside the active page's layers: another page → edit
        it; "+ Add page" → ask for one. True if the click was used."""
        if self.add_rect().contains(QPointF(x, y)):
            self.add_requested.emit()
            return True
        for pid, r in self._scene_rects().items():
            if pid != self.editor.active and r.contains(QPointF(x, y)):
                self.editor.set_active(pid)
                return True
        return False

    def shutdown(self) -> None:
        self._timer.stop()
        self._pool.clear()
        self._pool.waitForDone(2000)
