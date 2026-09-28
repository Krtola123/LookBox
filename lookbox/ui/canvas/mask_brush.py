"""Mask brush (ARCHITECTURE §9): paint a layer's mask on the canvas.

While active, the layer is shown without its mask and a red overlay marks what
is hidden (like Photoshop's quick mask). Painting edits a full-resolution
working copy; Done turns it into a new mask asset through one SetMask edit, so
a whole brush session is one undo step.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QGraphicsPixmapItem

from lookbox.commands import edits
from lookbox.core.masks import ops as M
from lookbox.core.model import ImageLayer, LayerMask
from lookbox.core.render.effects import layer_scale
from lookbox.core.render.transform import level_matrix
from lookbox.ui.canvas.handles import canvas_to_source, layer_size
from lookbox.ui.canvas.layer_items import bgra_to_pixmap, to_qtransform

OVERLAY_MAX_SIDE = 1600  # the red overlay is display-only: screen resolution is plenty
OVERLAY_RGB = (255, 60, 100)
OVERLAY_ALPHA = 0.55


class MaskBrush(QObject):
    changed = Signal()  # started / finished / settings changed

    def __init__(self, view) -> None:
        super().__init__(view)
        self.view = view
        self.editor = view.editor
        self.layer_id: str | None = None
        self.mask: np.ndarray | None = None  # full source resolution, 0–1
        self.size = 80.0  # brush diameter, canvas px
        self.hardness = 0.6
        self.restore = False  # False = erase (hide), True = restore (show)
        self._last: tuple[float, float] | None = None
        self._overlay: QGraphicsPixmapItem | None = None
        self._dirty = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)  # ≤ ~30 overlay refreshes a second while painting
        self._timer.timeout.connect(self._update_overlay)
        # End the session cleanly when the context changes under it.
        self.editor.selection_changed.connect(self._on_selection)
        self.editor.document_replaced.connect(lambda: self.finish(apply=False))
        self.editor.changed.connect(self._on_changed)

    def _on_selection(self) -> None:
        if self.active and self.editor.selected != self.layer_id:
            self.finish(apply=True)  # picking another layer keeps what you painted

    def _on_changed(self) -> None:
        if self.active and self._layer() is None:
            self.finish(apply=False)  # the layer was deleted (e.g. undo): nothing to apply to
        elif self.active:
            self._update_overlay()  # layer moved/cropped by undo: keep the overlay aligned

    @property
    def active(self) -> bool:
        return self.layer_id is not None

    def _layer(self) -> ImageLayer | None:
        doc = self.editor.doc
        if self.layer_id is None or not doc.has_layer(self.layer_id):
            return None
        layer = doc.layer(self.layer_id)
        return layer if isinstance(layer, ImageLayer) else None

    # ------------------------------------------------------------ session
    def start(self, layer_id: str) -> None:
        if self.active:
            self.finish(apply=True)
        doc = self.editor.doc
        layer = doc.layer(layer_id)
        if not isinstance(layer, ImageLayer):
            return
        src = self.editor.store.pixels(layer.source)
        if layer.mask is not None:
            base = M.mask_from_pixels(self.editor.store.pixels(layer.mask.asset))
        else:
            base = np.ones(src.shape[:2], np.float32)  # no mask yet: everything visible, erase to cut out
        self.mask = np.array(base, dtype=np.float32, copy=True)
        self.layer_id = layer_id
        self._overlay = QGraphicsPixmapItem()
        self._overlay.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._overlay.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.view.scene().addItem(self._overlay)
        self.view.layers.set_mask_bypass(layer_id)
        self._update_overlay()
        self.changed.emit()

    def finish(self, apply: bool) -> None:
        if not self.active:
            return
        layer = self._layer()
        if apply and layer is not None and self.mask is not None:
            old = layer.mask
            if old is not None or self.mask.min() < 1.0:  # an untouched full mask changes nothing
                info = self.editor.store.add_bytes(M.encode_mask_png(self.mask), ".png", "mask")
                new = LayerMask(asset=info.id, shift=old.shift if old else 0.0,
                                feather=old.feather if old else 0.0, invert=old.invert if old else False)
                if new != old:
                    self.editor.push(edits.SetMask(layer.id, old, new, asset=info, text="Brush mask"))
        if self._overlay is not None:
            self.view.scene().removeItem(self._overlay)
            self._overlay = None
        lid, self.layer_id, self.mask, self._last = self.layer_id, None, None, None
        self.view.layers.set_mask_bypass(None)
        self.changed.emit()

    # ------------------------------------------------------------ painting
    def paint(self, x: float, y: float, first: bool, flip_mode: bool) -> None:
        layer = self._layer()
        if layer is None or self.mask is None:
            return
        p = canvas_to_source(self.editor.doc, layer, x, y)
        radius = (self.size / 2.0) / layer_scale(layer.transform)  # canvas px → source px
        restore = self.restore != flip_mode
        if first or self._last is None:
            M.stamp(self.mask, p[0], p[1], radius, self.hardness, restore)
        else:
            M.stroke(self.mask, self._last, p, radius, self.hardness, restore)
        self._last = p
        if not self._timer.isActive():
            self._timer.start()

    def end_stroke(self) -> None:
        self._last = None
        self._update_overlay()

    def resize(self, factor: float) -> None:
        self.size = float(min(2000.0, max(2.0, self.size * factor)))
        self.changed.emit()

    # ------------------------------------------------------------ overlay
    def _update_overlay(self) -> None:
        layer = self._layer()
        if layer is None or self.mask is None or self._overlay is None:
            return
        m = self.mask
        if layer.crop is not None:
            cx, cy, cw, ch = layer.crop
            m = m[cy:cy + ch, cx:cx + cw]
        h, w = m.shape
        s = min(1.0, OVERLAY_MAX_SIDE / max(h, w))
        ow, oh = max(1, round(w * s)), max(1, round(h * s))
        small = cv2.resize(np.ascontiguousarray(m), (ow, oh), interpolation=cv2.INTER_AREA) if s < 1 else m
        a = np.clip((1.0 - small) * OVERLAY_ALPHA * 255.0, 0, 255).astype(np.uint8)
        bgra = np.empty((oh, ow, 4), np.uint8)
        for i, c in enumerate(reversed(OVERLAY_RGB)):  # premultiplied BGRA
            bgra[:, :, i] = (a.astype(np.uint16) * c // 255).astype(np.uint8)
        bgra[:, :, 3] = a
        self._overlay.setPixmap(bgra_to_pixmap(bgra))
        bw, bh = layer_size(self.editor.doc, layer)
        self._overlay.setTransform(to_qtransform(level_matrix(layer.transform, bw, bh, ow, oh)))
        entry = self.view.layers.entries.get(layer.id)
        self._overlay.setZValue((entry.item.zValue() if entry else 0) + 0.5)
