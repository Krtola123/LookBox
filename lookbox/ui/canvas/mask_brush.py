"""Mask editing session (ARCHITECTURE §9): brush, ID pick and lasso on the canvas.

While active, the layer is shown without its mask and a red overlay marks what
is hidden (like Photoshop's quick mask). Every tool edits one full-resolution
working copy; Done turns it into a new mask asset through one SetMask edit, so
a whole session is one undo step.

Tools: Brush (paint erase/restore), Pick (click an object in the ID pass) and
Lasso (drag = freehand; click, click, … Enter or double-click = polygon). Pick
and lasso replace the selection; Shift adds to it, Alt takes away.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem

from lookbox.commands import edits
from lookbox.core.masks import ops as M
from lookbox.core.masks.idpick import IdPass
from lookbox.core.masks.lasso import lasso_mask
from lookbox.core.model import ImageLayer, LayerMask
from lookbox.core.render.effects import layer_scale
from lookbox.core.render.transform import level_matrix
from lookbox.ui.canvas.handles import canvas_to_source, layer_size
from lookbox.ui.canvas.layer_items import bgra_to_pixmap, to_qtransform

OVERLAY_MAX_SIDE = 1600  # the red overlay is display-only: screen resolution is plenty
OVERLAY_RGB = (255, 60, 100)
OVERLAY_ALPHA = 0.55
TOOLS = ("brush", "pick", "lasso")
LASSO_MIN_STEP_PX = 3.0  # screen px between freehand points


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
        self.tool = "brush"
        self.pick_kind = "object_id"  # which ID pass Pick reads
        self.tolerance = 0  # 8-bit steps of colour slack for Pick
        self.lasso: list[tuple[float, float]] = []  # canvas points of the lasso being drawn
        self.polygon = False  # lasso: building a polygon click by click (vs a freehand drag)
        self._lasso_mode = "replace"
        self._ids: dict[str, Future] = {}  # ID passes being / already prepared, by asset id (≤ 2 kept)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="idpass")
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
        self.editor.about_to_switch.connect(lambda: self.finish(apply=True))  # keep painting done on this page
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
        if self.tool == "pick":
            self.prepare_ids()
        self.changed.emit()

    def finish(self, apply: bool) -> None:
        edit = self.take_edit() if apply else None
        if edit is not None:
            self.editor.push(edit)
        self.end()

    def take_edit(self) -> edits.SetMask | None:
        """The SetMask this session amounts to (None if it changes nothing). Doesn't push it."""
        layer = self._layer()
        if not self.active or layer is None or self.mask is None:
            return None
        old = layer.mask
        if old is None and self.mask.min() >= 1.0:  # an untouched full mask changes nothing
            return None
        info = self.editor.store.add_bytes(M.encode_mask_png(self.mask), ".png", "mask")
        new = LayerMask(asset=info.id, shift=old.shift if old else 0.0,
                        feather=old.feather if old else 0.0, invert=old.invert if old else False)
        return None if new == old else edits.SetMask(layer.id, old, new, asset=info, text="Edit cut-out")

    def end(self) -> None:
        """Close the session without applying anything."""
        if not self.active:
            return
        if self._overlay is not None:
            self.view.scene().removeItem(self._overlay)
            self._overlay = None
        self.layer_id, self.mask, self._last = None, None, None
        self.lasso, self.polygon = [], False
        self._ids.clear()  # a prepared 24 MP pass holds ~400 MB: let it go with the session
        self.view.layers.set_mask_bypass(None)
        self.changed.emit()

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(tool)
        self.tool = tool
        self.lasso, self.polygon = [], False
        if tool == "pick":
            self.prepare_ids()
        self.view.viewport().update()  # drop a half-drawn lasso from the screen
        self.changed.emit()

    def pick_kinds(self) -> list[str]:
        layer = self._layer()
        return [k for k in ("object_id", "material_id") if layer is not None and k in layer.passes]

    # ------------------------------------------------------------ pointer routing (from the view)
    @staticmethod
    def _mode(shift: bool, alt: bool) -> str:
        return "add" if shift else "subtract" if alt else "replace"

    def press(self, x: float, y: float, shift: bool, alt: bool) -> None:
        if self.tool == "brush":
            self.paint(x, y, first=True, flip_mode=alt)
        elif self.tool == "pick":
            self.pick(x, y, self._mode(shift, alt))
        elif self.polygon:
            self.lasso.append((x, y))  # next corner of the polygon
        else:
            self.lasso = [(x, y)]
            self._lasso_mode = self._mode(shift, alt)

    def move(self, x: float, y: float, dragging: bool, alt: bool) -> None:
        if self.tool == "brush" and dragging:
            self.paint(x, y, first=False, flip_mode=alt)
        elif self.tool == "lasso" and dragging and not self.polygon and self.lasso:
            lx, ly = self.lasso[-1]
            step = LASSO_MIN_STEP_PX / max(self.view.zoom(), 1e-6)
            if (x - lx) ** 2 + (y - ly) ** 2 >= step * step:
                self.lasso.append((x, y))

    def release(self) -> None:
        if self.tool == "brush":
            self.end_stroke()
        elif self.tool == "lasso" and not self.polygon and self.lasso:
            xs, ys = [p[0] for p in self.lasso], [p[1] for p in self.lasso]
            extent = max(max(xs) - min(xs), max(ys) - min(ys)) * self.view.zoom()
            if len(self.lasso) >= 3 and extent >= 3 * LASSO_MIN_STEP_PX:
                self.close_lasso()  # a freehand drag: done when you let go
            else:
                self.lasso = self.lasso[:1]  # a click (or a twitch): start a polygon, one corner per click
                self.polygon = True
        self.changed.emit()

    def key(self, key: int) -> bool:
        """Enter / Esc / [ ]. Returns True if handled."""
        if key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight):
            self.resize(0.8 if key == Qt.Key.Key_BracketLeft else 1.25)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.polygon:
                self.close_lasso()
            else:
                self.finish(apply=True)
        elif key == Qt.Key.Key_Escape:
            if self.lasso:
                self.lasso, self.polygon = [], False  # drop the lasso, keep the session
                self.changed.emit()
            else:
                self.finish(apply=False)
        else:
            return False
        return True

    # ------------------------------------------------------------ pick + lasso
    def _pass_id(self, layer: ImageLayer, kind: str | None = None) -> str | None:
        kind = kind or self.pick_kind
        return layer.passes.get(kind) or next(iter(
            layer.passes[k] for k in ("object_id", "material_id") if k in layer.passes), None)

    def prepare_ids(self) -> None:
        """Start getting the layer's ID passes ready in the background (~1–2 s for a
        24 MP pass), so the first click doesn't stall."""
        layer = self._layer()
        if layer is None:
            return
        for kind in ("object_id", "material_id"):
            aid = layer.passes.get(kind)
            if aid is not None and aid not in self._ids:
                if len(self._ids) >= 2:
                    self._ids.clear()
                pixels = self.editor.store.pixels(aid)
                self._ids[aid] = self._pool.submit(IdPass, pixels)

    def _id_pass(self, layer: ImageLayer) -> IdPass | None:
        aid = self._pass_id(layer)
        if aid is None:
            return None
        if aid not in self._ids:
            self.prepare_ids()
        try:
            return self._ids[aid].result()  # waits only if it's still being prepared
        except BaseException:
            self._ids.pop(aid, None)  # don't keep a failure: the next click tries again
            raise

    def shutdown(self) -> None:
        self.end()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def pick(self, x: float, y: float, mode: str) -> None:
        layer = self._layer()
        if layer is None or self.mask is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ids = self._id_pass(layer)
            if ids is None or ids.shape != self.mask.shape:
                return
            sx, sy = canvas_to_source(self.editor.doc, layer, x, y)
            colour = ids.sample(sx, sy)
            if colour is None:
                return
            self.mask = M.combine(self.mask, ids.mask(colour, self.tolerance / 255.0), mode)
        finally:
            QApplication.restoreOverrideCursor()
        self._update_overlay()

    def close_lasso(self) -> None:
        layer = self._layer()
        pts, mode = self.lasso, self._lasso_mode
        self.lasso, self.polygon = [], False
        if layer is not None and self.mask is not None and len(pts) >= 3:
            src = [canvas_to_source(self.editor.doc, layer, x, y) for x, y in pts]
            h, w = self.mask.shape
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.mask = M.combine(self.mask, lasso_mask(src, w, h), mode)
            finally:
                QApplication.restoreOverrideCursor()
            self._update_overlay()
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
