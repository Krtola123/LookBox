"""The canvas: shows layers, draws selection handles, handles mouse/keyboard.

Layer pixels come from LayerItems (rendered off-thread, cached per level).
Dragging only previews on the item; the document changes once, on release,
through a single SetTransform edit (§4.2).
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import QFrame, QGraphicsRectItem, QGraphicsScene, QGraphicsView

from lookbox.commands import edits
from lookbox.core.model import Layer, Transform
from lookbox.ui.canvas import handles as H
from lookbox.ui.canvas import overlay
from lookbox.ui.canvas.frame_stats import FrameStats
from lookbox.ui.canvas.layer_items import LayerItems
from lookbox.ui.editor import Editor
from lookbox.ui.render_service import RenderService

MIN_ZOOM, MAX_ZOOM = 0.02, 32.0


def _checker() -> QPixmap:
    pm = QPixmap(16, 16)
    pm.fill(QColor("#e6e6e6"))
    p = QPainter(pm)
    p.fillRect(0, 0, 8, 8, QColor("#c8c8c8"))
    p.fillRect(8, 8, 8, 8, QColor("#c8c8c8"))
    p.end()
    return pm


class _Drag:
    def __init__(self, mode: str, layer_id: str, handle: str | None, press: tuple[float, float],
                 t0: Transform, w: int, h: int) -> None:
        self.mode, self.layer_id, self.handle = mode, layer_id, handle
        self.press, self.t0, self.w, self.h = press, t0, w, h
        self.current = t0


class CanvasView(QGraphicsView):
    zoom_changed = Signal(float)
    files_dropped = Signal(list, object)  # paths, (x, y) canvas point
    frame_stats = Signal(str)  # drag performance readout for the status bar

    def __init__(self, editor: Editor, service: RenderService, parent=None) -> None:
        super().__init__(parent)
        self.editor = editor
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.layers = LayerItems(self._scene, service, editor)
        self.service = service
        self._canvas_item = QGraphicsRectItem()
        self._canvas_item.setPen(Qt.PenStyle.NoPen)
        self._canvas_item.setZValue(-1e9)
        self._scene.addItem(self._canvas_item)
        self._checker = QBrush(_checker())

        self._drag: _Drag | None = None
        self._pan_from: QPointF | None = None
        self._space = False
        self._hover: str | None = None
        self._guides: list[tuple[str, float]] = []  # snap guides shown during a drag
        self._targets: tuple[list[float], list[float]] = ([], [])
        # Level refresh after zooming, debounced so a wheel spin requests one render, not twenty.
        self._level_timer = QTimer(self)
        self._level_timer.setSingleShot(True)
        self._level_timer.setInterval(80)
        self._level_timer.timeout.connect(self._refresh_levels)
        self.zoom_changed.connect(lambda _z: self._level_timer.start())
        # Drag frame timing (M2 acceptance: >30 fps).
        self._frames = FrameStats()
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(500)
        self._stats_timer.timeout.connect(self._emit_stats)

        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setBackgroundBrush(QColor("#111214"))
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFrameShape(QFrame.Shape.NoFrame)

        editor.changed.connect(self.sync)
        editor.document_replaced.connect(self._on_replaced)
        editor.selection_changed.connect(self.viewport().update)

    # ------------------------------------------------------------ sync
    def _on_replaced(self) -> None:
        self._cancel_drag()
        self.layers.clear()
        self.service.clear()
        self.sync()
        self.fit()

    def _cancel_drag(self) -> None:
        d, self._drag = self._drag, None
        self._guides = []
        if d is not None:
            self.layers.end_preview(d.layer_id)
            self._stats_timer.stop()

    def _refresh_levels(self) -> None:
        self.layers.set_screen_scale(self.zoom() * self.devicePixelRatioF())

    def sync(self) -> None:
        """Bring scene items in line with the document."""
        doc = self.editor.doc
        self._cancel_drag()  # an undo mid-drag invalidates the drag
        canvas = QRectF(0, 0, doc.canvas.w, doc.canvas.h)
        self._canvas_item.setRect(canvas)
        if doc.background is None:
            self._canvas_item.setBrush(self._checker)
        else:
            r, g, b, a = doc.background
            self._canvas_item.setBrush(QColor.fromRgbF(r, g, b, a))
        m = max(doc.canvas.w, doc.canvas.h)
        self._scene.setSceneRect(canvas.adjusted(-m, -m, m, m))

        self.layers.screen_scale = self.zoom() * self.devicePixelRatioF()
        self.layers.sync()
        self.viewport().update()

    # ------------------------------------------------------------ zoom
    def zoom(self) -> float:
        return self.transform().m11()

    def set_zoom(self, z: float) -> None:
        z = max(MIN_ZOOM, min(MAX_ZOOM, z))
        centre = self.mapToScene(self.viewport().rect().center())
        self.setTransform(QTransform.fromScale(z, z))
        self.centerOn(centre)
        self.zoom_changed.emit(z)

    def fit(self) -> None:
        doc = self.editor.doc
        vw, vh = max(1, self.viewport().width()), max(1, self.viewport().height())
        z = 0.9 * min(vw / doc.canvas.w, vh / doc.canvas.h)
        self.setTransform(QTransform.fromScale(z, z))
        self.centerOn(doc.canvas.w / 2, doc.canvas.h / 2)
        self.zoom_changed.emit(self.zoom())

    def wheelEvent(self, e) -> None:
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            target = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom() * 1.0015 ** e.angleDelta().y()))
            f = target / self.zoom()
            self.scale(f, f)  # AnchorUnderMouse keeps the cursor point fixed
            self.zoom_changed.emit(self.zoom())
            e.accept()
        else:
            super().wheelEvent(e)

    # ------------------------------------------------------------ drawing
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        doc = self.editor.doc
        overlay.dim_outside(painter, rect, QRectF(0, 0, doc.canvas.w, doc.canvas.h))
        if self._hover and self._hover != self.editor.selected and self._drag is None:
            hl = self._layer(self._hover)
            if hl is not None:
                overlay.quad(painter, hl.transform, *self._size(hl), overlay.HOVER, 1.5)
        layer = self.editor.selected_layer()
        if layer is not None:
            live = self._drag is not None and self._drag.layer_id == layer.id
            t = self._drag.current if live else layer.transform
            overlay.selection(painter, t, *self._size(layer), self.zoom(), layer.locked)
        if self._guides:
            overlay.guides(painter, self._guides, QRectF(0, 0, doc.canvas.w, doc.canvas.h))

    # ------------------------------------------------------------ helpers
    def _layer(self, lid: str) -> Layer | None:
        doc = self.editor.doc
        return doc.layer(lid) if doc.has_layer(lid) else None

    def _size(self, layer: Layer) -> tuple[int, int]:
        return H.layer_size(self.editor.doc, layer)

    def _scene_pt(self, e) -> tuple[float, float]:
        p = self.mapToScene(e.position().toPoint())
        return p.x(), p.y()

    def _preview(self, t: Transform) -> None:
        d = self._drag
        self.layers.preview(d.layer_id, t)
        d.current = t
        self.viewport().update()

    def _cursor_for_handle(self, t: Transform, w: int, h: int, name: str) -> Qt.CursorShape:
        if name == H.ROTATE:
            return Qt.CursorShape.CrossCursor
        p = H.handle_positions(t, w, h, self.zoom())[name]
        ang = math.degrees(math.atan2(p[1] - t.y, p[0] - t.x)) % 180.0
        if ang < 22.5 or ang >= 157.5:
            return Qt.CursorShape.SizeHorCursor
        if ang < 67.5:
            return Qt.CursorShape.SizeFDiagCursor
        if ang < 112.5:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.SizeBDiagCursor

    # ------------------------------------------------------------ drag performance
    def _begin_drag(self, d: _Drag) -> None:
        self._drag = d
        self._targets = H.snap_targets(self.editor.doc, exclude=d.layer_id)  # fixed for this drag
        self._guides = []
        self._frames.reset()
        self._stats_timer.start()

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

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, e) -> None:
        self.setFocus()
        if e.button() == Qt.MouseButton.MiddleButton or (e.button() == Qt.MouseButton.LeftButton and self._space):
            self._pan_from = e.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        x, y = self._scene_pt(e)
        z = self.zoom()
        layer = self.editor.selected_layer()
        if layer is not None and not layer.locked:
            w, h = self._size(layer)
            handle = H.handle_at(layer.transform, w, h, z, x, y)
            if handle is not None:
                mode = "rotate" if handle == H.ROTATE else "scale"
                self._begin_drag(_Drag(mode, layer.id, handle, (x, y), layer.transform, w, h))
                return
            if H.point_in_layer(layer.transform, w, h, x, y):
                self._begin_drag(_Drag("move", layer.id, None, (x, y), layer.transform, w, h))
                return
        hit = H.layer_at(self.editor.doc, self.editor.store, x, y)
        self.editor.select(hit)
        if hit is not None:
            hl = self._layer(hit)
            self._begin_drag(_Drag("move", hit, None, (x, y), hl.transform, *self._size(hl)))

    def mouseMoveEvent(self, e) -> None:
        if self._pan_from is not None:
            delta = e.position() - self._pan_from
            self._pan_from = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(delta.y()))
            return
        x, y = self._scene_pt(e)
        mods = e.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        alt = bool(mods & Qt.KeyboardModifier.AltModifier)
        snap = not (mods & Qt.KeyboardModifier.ControlModifier)  # hold Ctrl to place freely
        d = self._drag
        if d is not None:
            self._guides = []
            if d.mode == "move":
                t = H.drag_move(d.t0, d.press, (x, y), constrain=shift)
                if snap:
                    t, self._guides = H.snap_move(t, d.w, d.h, self._targets, self.zoom())
                self._preview(t)
            elif d.mode == "scale":
                if snap:
                    x, y, self._guides = H.snap_point(d.t0, d.handle, x, y, self._targets, self.zoom())
                self._preview(H.drag_scale(d.t0, d.w, d.h, d.handle, (x, y), free=shift, from_centre=alt))
            else:
                self._preview(H.drag_rotate(d.t0, d.press, (x, y), snap_15=shift))
            return

        # Hover feedback: cursor over handles, outline on the layer under the mouse.
        cursor = Qt.CursorShape.ArrowCursor
        if self._space:
            cursor = Qt.CursorShape.OpenHandCursor
        else:
            layer = self.editor.selected_layer()
            if layer is not None and not layer.locked:
                w, h = self._size(layer)
                handle = H.handle_at(layer.transform, w, h, self.zoom(), x, y)
                if handle is not None:
                    cursor = self._cursor_for_handle(layer.transform, w, h, handle)
                elif H.point_in_layer(layer.transform, w, h, x, y):
                    cursor = Qt.CursorShape.SizeAllCursor
        self.viewport().setCursor(cursor)
        hover = H.layer_at(self.editor.doc, self.editor.store, x, y)
        if hover != self._hover:
            self._hover = hover
            self.viewport().update()

    def mouseReleaseEvent(self, e) -> None:
        if self._pan_from is not None:
            self._pan_from = None
            self.viewport().setCursor(
                Qt.CursorShape.OpenHandCursor if self._space else Qt.CursorShape.ArrowCursor)
            return
        d, self._drag = self._drag, None
        self._guides = []
        if d is not None:
            self._stats_timer.stop()
            self._emit_stats(final=True)
            if d.current != d.t0:
                text = {"move": "Move", "scale": "Resize", "rotate": "Rotate"}[d.mode]
                self.editor.push(edits.SetTransform(d.layer_id, d.t0, d.current, text=text))
            self.layers.end_preview(d.layer_id)
        self.viewport().update()

    def leaveEvent(self, e) -> None:
        if self._hover is not None:
            self._hover = None
            self.viewport().update()
        super().leaveEvent(e)

    # ------------------------------------------------------------ keys
    def keyPressEvent(self, e) -> None:
        key = e.key()
        if key == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return
        layer = self.editor.selected_layer()
        arrows = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0),
                  Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1)}
        if key in arrows and layer is not None and not layer.locked and self._drag is None:
            step = 10 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            dx, dy = arrows[key]
            t0 = layer.transform
            t1 = replace(t0, x=t0.x + dx * step, y=t0.y + dy * step)
            self.editor.push(edits.SetTransform(layer.id, t0, t1, text="Nudge", merge_key="nudge"))
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and layer is not None and self._drag is None:
            self.editor.push(edits.RemoveLayer(layer.id))
            return
        if key == Qt.Key.Key_Escape:
            if self._drag is not None:  # cancel the drag, restore the item
                self._cancel_drag()
                self.viewport().update()
            else:
                self.editor.select(None)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = False
            if self._pan_from is None:
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().keyReleaseEvent(e)

    # ------------------------------------------------------------ drag & drop from Explorer
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if not paths:
            super().dropEvent(e)
            return
        p = self.mapToScene(e.position().toPoint())
        e.acceptProposedAction()
        self.files_dropped.emit(paths, (p.x(), p.y()))
