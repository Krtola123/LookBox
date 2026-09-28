"""Scene items for layers: one pixmap per layer at a screen-appropriate level.

Qt paints and places these (cheap, keeps dragging smooth); the pixels come
from the core pipeline via RenderService (expensive, cached, off-thread).
Placement uses core.render.transform.level_matrix, the same math as export.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene

from dataclasses import replace

from lookbox.core.model import Adjustments, Layer, Transform
from lookbox.core.render.levels import MIN_LEVEL, choose_level, on_screen_scale
from lookbox.core.render.pipeline import render_key
from lookbox.core.render.transform import level_matrix
from lookbox.ui.canvas.handles import layer_size
from lookbox.ui.editor import Editor
from lookbox.ui.render_service import RenderService


def to_qtransform(m) -> QTransform:
    # Qt maps x' = m11·x + m21·y + dx, y' = m12·x + m22·y + dy.
    return QTransform(m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2])


def bgra_to_pixmap(arr) -> QPixmap:
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 4 * w, QImage.Format.Format_ARGB32_Premultiplied)
    return QPixmap.fromImage(img)  # copies, so `arr` may be freed afterwards


# Qt's composition modes implement the same W3C formulas as core/render/blend.py.
_QT_MODES = {
    "normal": QPainter.CompositionMode.CompositionMode_SourceOver,
    "multiply": QPainter.CompositionMode.CompositionMode_Multiply,
    "screen": QPainter.CompositionMode.CompositionMode_Screen,
    "overlay": QPainter.CompositionMode.CompositionMode_Overlay,
    "add": QPainter.CompositionMode.CompositionMode_Plus,
    "soft_light": QPainter.CompositionMode.CompositionMode_SoftLight,
}


class LayerItem(QGraphicsPixmapItem):
    """A pixmap item that paints with its layer's blend mode."""

    def __init__(self) -> None:
        super().__init__()
        self.mode = _QT_MODES["normal"]

    def paint(self, painter, option, widget=None) -> None:
        if self.mode == _QT_MODES["normal"]:
            super().paint(painter, option, widget)
            return
        painter.save()
        painter.setCompositionMode(self.mode)
        super().paint(painter, option, widget)
        painter.restore()


class _Entry:
    def __init__(self, item: LayerItem) -> None:
        self.item = item
        self.target_key = ""  # render key the layer has now
        self.want = 1.0  # level the screen needs
        self.shown_key = ""  # render key of the pixmap on screen
        self.shown_level = 0.0
        self.lw = self.lh = 1  # size of the pixmap on screen
        self.w = self.h = 1  # full layer size


class LayerItems(QObject):
    def __init__(self, scene: QGraphicsScene, service: RenderService, editor: Editor) -> None:
        super().__init__(scene)
        self.scene, self.service, self.editor = scene, service, editor
        self.entries: dict[str, _Entry] = {}
        self.screen_scale = 1.0  # device pixels per canvas pixel
        self._live: dict[str, Transform] = {}  # transforms being dragged, not yet in the document
        self._interactive: set[str] = set()  # layers whose sliders are being dragged: half-res previews
        self._bypass: str | None = None  # layer shown "before" (adjustments off), view-only
        service.ready.connect(self._on_ready)

    # ---- public ----
    def clear(self) -> None:
        self._live.clear()
        self._interactive.clear()
        self._bypass = None
        for e in self.entries.values():
            self.scene.removeItem(e.item)
        self.entries.clear()

    def set_screen_scale(self, s: float) -> None:
        self.screen_scale = s
        self.refresh_levels()

    def sync(self) -> None:
        """Bring items in line with the document."""
        doc = self.editor.doc
        alive = set()
        for z, layer in enumerate(doc.layers):
            alive.add(layer.id)
            e = self.entries.get(layer.id)
            if e is None:
                item = LayerItem()
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
                self.scene.addItem(item)
                e = self.entries[layer.id] = _Entry(item)
            e.w, e.h = layer_size(doc, layer)
            e.target_key = render_key(self._effective(layer))
            e.item.setZValue(z)
            e.item.setVisible(layer.visible)
            e.item.setOpacity(layer.opacity)
            mode = _QT_MODES.get(layer.blend_mode, _QT_MODES["normal"])
            if e.item.mode != mode:
                e.item.mode = mode
                e.item.update()
            self._update_level(e, layer)
            self._place(e, self._transform_of(layer))
        for lid in list(self.entries):
            if lid not in alive:
                self.scene.removeItem(self.entries.pop(lid).item)

    def refresh_levels(self) -> None:
        doc = self.editor.doc
        for lid, e in self.entries.items():
            if doc.has_layer(lid):
                layer = doc.layer(lid)
                self._update_level(e, layer)
                self._place(e, self._transform_of(layer))

    def preview(self, layer_id: str, t: Transform) -> None:
        """Show a layer at `t` without touching the document (live drag)."""
        self._live[layer_id] = t
        e = self.entries.get(layer_id)
        if e is not None:
            self._place(e, t)

    def end_preview(self, layer_id: str) -> None:
        """Drop the live transform and show the document's again (release or cancel)."""
        self._live.pop(layer_id, None)
        doc = self.editor.doc
        e = self.entries.get(layer_id)
        if e is not None and doc.has_layer(layer_id):
            self._place(e, doc.layer(layer_id).transform)

    def set_interactive(self, layer_id: str, on: bool) -> None:
        """While a slider drags, render at half the level (4× fewer pixels) to keep up;
        the full level renders when it's released."""
        (self._interactive.add if on else self._interactive.discard)(layer_id)
        self._refresh_one(layer_id)

    def set_bypass(self, layer_id: str | None) -> None:
        """Before/after: show `layer_id` without its adjustments (None = normal)."""
        old, self._bypass = self._bypass, layer_id
        for lid in {old, layer_id} - {None}:
            self._refresh_one(lid)

    def _refresh_one(self, layer_id: str) -> None:
        doc = self.editor.doc
        e = self.entries.get(layer_id)
        if e is not None and doc.has_layer(layer_id):
            layer = doc.layer(layer_id)
            e.target_key = render_key(self._effective(layer))
            self._update_level(e, layer)
            self._place(e, self._transform_of(layer))

    # ---- internals ----
    def _effective(self, layer: Layer) -> Layer:
        """The layer as it should be *shown* (before/after bypass drops adjustments)."""
        if layer.id == self._bypass and not layer.adjust.is_identity():
            return replace(layer, adjust=Adjustments())
        return layer

    def _transform_of(self, layer: Layer) -> Transform:
        return self._live.get(layer.id, layer.transform)

    def _place(self, e: _Entry, t: Transform) -> None:
        e.item.setTransform(to_qtransform(level_matrix(t, e.w, e.h, e.lw, e.lh)))

    def _update_level(self, e: _Entry, layer: Layer) -> None:
        e.want = choose_level(on_screen_scale(layer.transform, self.screen_scale))
        if layer.id in self._interactive:
            e.want = max(MIN_LEVEL, e.want / 2.0)
        arr = self.service.request(self._effective(layer), self.editor.store, e.want, e.target_key)
        if arr is not None:
            self._show(e, e.target_key, e.want, arr)
            return
        # Not ready: keep what's on screen if it's the right content, else the best cached level.
        if e.shown_key != e.target_key:
            best = self._best_cached(e.target_key, e.want)
            if best is not None:
                self._show(e, e.target_key, *best)

    def _best_cached(self, key: str, want: float):
        """Closest cached level: sharper ones first (downsampling looks fine), then softer."""
        cache = self.service.cache
        level = want
        while level <= 1.0:
            arr = cache.get((key, level))
            if arr is not None:
                return level, arr
            level *= 2.0
        level = want / 2.0
        while level >= MIN_LEVEL:
            arr = cache.get((key, level))
            if arr is not None:
                return level, arr
            level /= 2.0
        return None

    def _show(self, e: _Entry, key: str, level: float, arr) -> None:
        if e.shown_key == key and e.shown_level == level:
            return
        e.item.setPixmap(bgra_to_pixmap(arr))
        e.shown_key, e.shown_level = key, level
        e.lh, e.lw = arr.shape[:2]

    def _on_ready(self, key: str, level: float) -> None:
        doc = self.editor.doc
        for lid, e in self.entries.items():
            if e.target_key != key or not doc.has_layer(lid):
                continue
            better = e.shown_key != key or level == e.want or (
                abs(level - e.want) < abs(e.shown_level - e.want))
            if better:
                arr = self.service.cache.get((key, level))
                if arr is not None:
                    self._show(e, key, level, arr)
                    self._place(e, self._transform_of(doc.layer(lid)))
