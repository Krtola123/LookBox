"""The canvas: shows layers, draws selection handles, handles mouse/keyboard.

Placement comes from core.render.transform.layer_matrix (same math as export).
Dragging only previews on the item; the document changes once, on release,
through a single SetTransform edit (§4.2).
"""

from __future__ import annotations

import math
from dataclasses import replace

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap,
                           QPolygonF, QTransform)
from PySide6.QtWidgets import (QFrame, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsView)

from lookbox.commands import edits
from lookbox.core.model import ImageLayer, Transform
from lookbox.core.render.pipeline import layer_source
from lookbox.core.render.transform import layer_matrix
from lookbox.ui.canvas import handles as H
from lookbox.ui.editor import Editor
from lookbox.ui.pixmaps import PixmapCache

ACCENT = QColor("#8b3dff")
HOVER = QColor("#4aa3ff")
MIN_ZOOM, MAX_ZOOM = 0.02, 32.0


def qtransform(t: Transform, w: int, h: int) -> QTransform:
    m = layer_matrix(t, w, h)
    # Qt maps x' = m11·x + m21·y + dx, y' = m12·x + m22·y + dy.
    return QTransform(m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2])


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

    def __init__(self, editor: Editor, pixmaps: PixmapCache, parent=None) -> None:
        super().__init__(parent)
        self.editor = editor
        self.pixmaps = pixmaps
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._items: dict[str, QGraphicsPixmapItem] = {}
        self._canvas_item = QGraphicsRectItem()
        self._canvas_item.setPen(Qt.PenStyle.NoPen)
        self._canvas_item.setZValue(-1e9)
        self._scene.addItem(self._canvas_item)
        self._checker = QBrush(_checker())

        self._drag: _Drag | None = None
        self._pan_from: QPointF | None = None
        self._space = False
        self._hover: str | None = None

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
        for item in self._items.values():
            self._scene.removeItem(item)
        self._items.clear()
        self.pixmaps.clear()
        self.sync()
        self.fit()

    def sync(self) -> None:
        """Bring scene items in line with the document."""
        doc = self.editor.doc
        self._drag = None  # an undo mid-drag invalidates the drag
        canvas = QRectF(0, 0, doc.canvas.w, doc.canvas.h)
        self._canvas_item.setRect(canvas)
        if doc.background is None:
            self._canvas_item.setBrush(self._checker)
        else:
            r, g, b, a = doc.background
            self._canvas_item.setBrush(QColor.fromRgbF(r, g, b, a))
        m = max(doc.canvas.w, doc.canvas.h)
        self._scene.setSceneRect(canvas.adjusted(-m, -m, m, m))

        alive = set()
        for z, layer in enumerate(doc.layers):
            if not isinstance(layer, ImageLayer):
                continue
            alive.add(layer.id)
            item = self._items.get(layer.id)
            if item is None:
                item = QGraphicsPixmapItem()
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
                self._scene.addItem(item)
                self._items[layer.id] = item
            px = layer_source(layer, self.editor.store)
            pm = self.pixmaps.full((layer.source, layer.crop), px)
            if item.pixmap().cacheKey() != pm.cacheKey():
                item.setPixmap(pm)
            item.setTransform(qtransform(layer.transform, px.shape[1], px.shape[0]))
            item.setZValue(z)
            item.setVisible(layer.visible)
            item.setOpacity(layer.opacity)
        for lid in list(self._items):
            if lid not in alive:
                self._scene.removeItem(self._items.pop(lid))
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
        canvas = QRectF(0, 0, doc.canvas.w, doc.canvas.h)
        outside = QPainterPath()
        outside.addRect(rect)
        inside = QPainterPath()
        inside.addRect(canvas)
        painter.fillPath(outside.subtracted(inside), QColor(17, 18, 20, 200))

        z = self.zoom()
        if self._hover and self._hover != self.editor.selected and self._drag is None:
            hl = self._layer(self._hover)
            if hl is not None:
                self._draw_quad(painter, hl.transform, *self._size(hl), HOVER, 1.5)

        layer = self.editor.selected_layer()
        if layer is None:
            return
        w, h = self._size(layer)
        t = self._drag.current if self._drag and self._drag.layer_id == layer.id else layer.transform
        self._draw_quad(painter, t, w, h, ACCENT, 2.0)
        if layer.locked:
            return
        painter.setPen(QPen(ACCENT, 1.5 / z))
        painter.setBrush(QColor("white"))
        r = 5.5 / z
        for name, p in H.handle_positions(t, w, h, z).items():
            if name == H.ROTATE:
                painter.drawEllipse(QPointF(p[0], p[1]), r * 1.4, r * 1.4)
                painter.drawArc(QRectF(p[0] - r * 0.7, p[1] - r * 0.7, r * 1.4, r * 1.4), 30 * 16, 270 * 16)
            elif name in H.CORNERS:
                painter.drawEllipse(QPointF(p[0], p[1]), r, r)
            else:
                painter.drawRoundedRect(QRectF(p[0] - r, p[1] - r, 2 * r, 2 * r), r * 0.4, r * 0.4)

    def _draw_quad(self, painter: QPainter, t: Transform, w: int, h: int, color: QColor, width: float) -> None:
        pen = QPen(color, width)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in H.quad(t, w, h)]))

    # ------------------------------------------------------------ helpers
    def _layer(self, lid: str) -> ImageLayer | None:
        doc = self.editor.doc
        if not doc.has_layer(lid):
            return None
        layer = doc.layer(lid)
        return layer if isinstance(layer, ImageLayer) else None

    def _size(self, layer: ImageLayer) -> tuple[int, int]:
        return H.layer_size(self.editor.doc, layer)

    def _scene_pt(self, e) -> tuple[float, float]:
        p = self.mapToScene(e.position().toPoint())
        return p.x(), p.y()

    def _preview(self, t: Transform) -> None:
        d = self._drag
        item = self._items.get(d.layer_id)
        if item is not None:
            item.setTransform(qtransform(t, d.w, d.h))
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
                self._drag = _Drag(mode, layer.id, handle, (x, y), layer.transform, w, h)
                return
            if H.point_in_layer(layer.transform, w, h, x, y):
                self._drag = _Drag("move", layer.id, None, (x, y), layer.transform, w, h)
                return
        hit = H.layer_at(self.editor.doc, self.editor.store, x, y)
        self.editor.select(hit)
        if hit is not None:
            hl = self._layer(hit)
            self._drag = _Drag("move", hit, None, (x, y), hl.transform, *self._size(hl))

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
        d = self._drag
        if d is not None:
            if d.mode == "move":
                self._preview(H.drag_move(d.t0, d.press, (x, y), constrain=shift))
            elif d.mode == "scale":
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
        if d is not None and d.current != d.t0:
            text = {"move": "Move", "scale": "Resize", "rotate": "Rotate"}[d.mode]
            self.editor.push(edits.SetTransform(d.layer_id, d.t0, d.current, text=text))
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
                d, self._drag = self._drag, None
                item = self._items.get(d.layer_id)
                if item is not None:
                    item.setTransform(qtransform(d.t0, d.w, d.h))
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
