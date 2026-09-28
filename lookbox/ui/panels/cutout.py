"""Adjust tab → Cut-out (M6): remove background, edge controls, mask brush,
extract to a new layer, remove the cut-out."""

from __future__ import annotations

import copy

from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

from lookbox.commands import edits
from lookbox.core.model import ImageLayer
from lookbox.ui.canvas.mask_brush import MaskBrush
from lookbox.ui.cutout import CutoutController
from lookbox.ui.editor import Editor
from lookbox.ui.widgets.slider_row import SliderRow


def _title(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("sectionTitle")
    return lab


def _button(text: str, primary: bool = False) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("primaryPush" if primary else "")
    return b


class CutoutSection(QWidget):
    def __init__(self, editor: Editor, controller: CutoutController, brush: MaskBrush) -> None:
        super().__init__()
        self.editor, self.controller, self.brush = editor, controller, brush
        self._layer_id: str | None = None
        self._dragging = False
        self._serial = 0
        self._updating = False

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 8)
        col.setSpacing(6)
        col.addWidget(_title("Cut-out"))

        row = QHBoxLayout()
        self.remove_bg = _button("Remove background", primary=True)
        self.remove_bg.clicked.connect(self._remove_bg)
        self.model_btn = _button("Model…")
        self.model_btn.setToolTip("Choose Best quality or Fast")
        self.model_btn.clicked.connect(self._choose_model)
        row.addWidget(self.remove_bg, 1)
        row.addWidget(self.model_btn)
        col.addLayout(row)
        self.refine = QCheckBox("Refine edges (try for hair, fur)")
        self.refine.setObjectName("invertToggle")
        self.refine.setToolTip("Snaps the mask to the photo's edges. Helps fine strands; can add noise on clean edges.")
        col.addWidget(self.refine)

        # ---- when the layer has a mask ----
        self.mask_box = QWidget()
        mb = QVBoxLayout(self.mask_box)
        mb.setContentsMargins(0, 4, 0, 0)
        mb.setSpacing(4)
        self.shift = SliderRow("shift", "Edge shift", minimum=-30, maximum=30, default=0, suffix=" px")
        self.feather = SliderRow("feather", "Feather", minimum=0, maximum=30, default=0, suffix=" px")
        for r in (self.shift, self.feather):
            r.edited.connect(lambda v, k=r.key: self._edge(k, v))
            r.pressed.connect(self._pressed)
            r.released.connect(self._released)
            mb.addWidget(r)
        self.invert = QCheckBox("Invert (keep the background instead)")
        self.invert.setObjectName("invertToggle")
        self.invert.toggled.connect(self._invert)
        mb.addWidget(self.invert)
        actions = QHBoxLayout()
        self.extract = _button("Extract to new layer")
        self.extract.setToolTip("Put the cut-out on its own layer; the original shows everything again")
        self.extract.clicked.connect(self._extract)
        self.remove_mask = _button("Remove cut-out")
        self.remove_mask.clicked.connect(self._remove_mask)
        actions.addWidget(self.extract)
        actions.addWidget(self.remove_mask)
        mb.addLayout(actions)
        col.addWidget(self.mask_box)

        # ---- brush ----
        self.brush_btn = _button("Brush…")
        self.brush_btn.setToolTip("Paint the mask: erase or restore by hand")
        self.brush_btn.clicked.connect(self._start_brush)
        col.addWidget(self.brush_btn)
        self.brush_box = QWidget()
        bb = QVBoxLayout(self.brush_box)
        bb.setContentsMargins(0, 0, 0, 0)
        bb.setSpacing(4)
        hint = QLabel("Drag on the canvas to paint. Alt = the opposite mode · [ ] = size · Enter = done")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        bb.addWidget(hint)
        modes = QHBoxLayout()
        self.erase_btn, self.restore_btn = _button("Erase"), _button("Restore")
        for b, restore in ((self.erase_btn, False), (self.restore_btn, True)):
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, r=restore: self._set_mode(r))
            modes.addWidget(b)
        bb.addLayout(modes)
        self.size_row = SliderRow("size", "Size", minimum=2, maximum=800, default=80, suffix=" px")
        self.size_row.edited.connect(lambda v: self._brush_setting("size", float(v)))
        self.hardness_row = SliderRow("hardness", "Hardness", minimum=0, maximum=100, default=60, suffix="%")
        self.hardness_row.edited.connect(lambda v: self._brush_setting("hardness", v / 100.0))
        bb.addWidget(self.size_row)
        bb.addWidget(self.hardness_row)
        done_row = QHBoxLayout()
        self.done_btn, self.cancel_btn = _button("Done", primary=True), _button("Cancel")
        self.done_btn.clicked.connect(lambda: self.brush.finish(apply=True))
        self.cancel_btn.clicked.connect(lambda: self.brush.finish(apply=False))
        done_row.addWidget(self.done_btn)
        done_row.addWidget(self.cancel_btn)
        bb.addLayout(done_row)
        col.addWidget(self.brush_box)

        controller.busy_changed.connect(lambda _b: self.refresh())
        brush.changed.connect(self.refresh)

    # ------------------------------------------------------------ document → widgets
    def _layer(self) -> ImageLayer | None:
        doc = self.editor.doc
        if self._layer_id is None or not doc.has_layer(self._layer_id):
            return None
        layer = doc.layer(self._layer_id)
        return layer if isinstance(layer, ImageLayer) else None

    def set_layer(self, layer_id: str | None) -> None:
        self._layer_id = layer_id
        self.refresh()

    def refresh(self) -> None:
        layer = self._layer()
        brushing = self.brush.active and self.brush.layer_id == self._layer_id
        busy = self.controller.busy
        self.remove_bg.setEnabled(layer is not None and not busy and not brushing)
        spec = self.controller.current_spec()
        self.model_btn.setText(f"Model: {spec.name}" if spec else "Model…")
        has_mask = layer is not None and layer.mask is not None
        self.mask_box.setVisible(has_mask and not brushing)
        self.brush_btn.setVisible(not brushing)
        self.brush_btn.setText("Brush…" if has_mask else "Cut out by hand…")
        self.brush_box.setVisible(brushing)
        self.erase_btn.setChecked(not self.brush.restore)
        self.restore_btn.setChecked(self.brush.restore)
        self.size_row.set_value(self.brush.size)
        self.hardness_row.set_value(self.brush.hardness * 100)
        if has_mask and not self._dragging:
            self._updating = True
            try:
                self.shift.set_value(layer.mask.shift)
                self.feather.set_value(layer.mask.feather)
                self.invert.setChecked(layer.mask.invert)
            finally:
                self._updating = False

    # ------------------------------------------------------------ actions
    def _remove_bg(self) -> None:
        if self._layer() is not None:
            self.controller.remove_background(self._layer_id, self.refine.isChecked())

    def _choose_model(self) -> None:
        self.controller.choose_model()
        self.refresh()

    def _pressed(self) -> None:
        self._dragging = True
        self._serial += 1

    def _released(self) -> None:
        self._dragging = False
        self.refresh()

    def _set_mask(self, new, text: str, merge: bool = False) -> None:
        layer = self._layer()
        if layer is None or self._updating or new == layer.mask:
            return
        key = f"mask-drag-{self._serial}" if (merge and self._dragging) else None
        self.editor.push(edits.SetMask(layer.id, layer.mask, new, text=text, merge_key=key))

    def _edge(self, key: str, v: int) -> None:
        layer = self._layer()
        if layer is not None and layer.mask is not None:
            new = copy.deepcopy(layer.mask)
            setattr(new, key, float(v))
            self._set_mask(new, "Edge shift" if key == "shift" else "Feather", merge=True)

    def _invert(self, on: bool) -> None:
        layer = self._layer()
        if layer is not None and layer.mask is not None:
            new = copy.deepcopy(layer.mask)
            new.invert = on
            self._set_mask(new, "Invert cut-out")

    def _remove_mask(self) -> None:
        self._set_mask(None, "Remove cut-out")

    def _extract(self) -> None:
        layer = self._layer()
        if layer is not None and layer.mask is not None:
            batch = edits.extract_to_layer(self.editor.doc, layer.id)
            self.editor.push(batch)
            self.editor.select(batch.parts[0].layer.id)  # select the new cut-out layer

    def _start_brush(self) -> None:
        if self._layer() is not None:
            self.brush.start(self._layer_id)

    def _set_mode(self, restore: bool) -> None:
        self.brush.restore = restore
        self.refresh()

    def _brush_setting(self, name: str, value: float) -> None:
        setattr(self.brush, name, value)
