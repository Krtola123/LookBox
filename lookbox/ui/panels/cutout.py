"""Adjust tab → Cut-out: remove background (M6), pick an object from the render's
ID pass and lasso (M8), mask brush, edge controls, extract to a new layer.

Pick, Lasso and Brush are tools of one mask session on the canvas (canvas/mask_brush.py):
switch between them freely; Done (or Extract) makes the whole session one undo step."""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

from lookbox.commands import edits
from lookbox.core.io import render_sets as RS
from lookbox.core.io.images import ImageError
from lookbox.core.model import ImageLayer
from lookbox.ui.canvas.mask_brush import MaskBrush
from lookbox.ui.cutout import CutoutController
from lookbox.ui.editor import Editor
from lookbox.ui.panels.fill_row import FillRow
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
        self.keep_bg = QCheckBox("Keep the background as its own layer")
        self.keep_bg.setObjectName("invertToggle")
        self.keep_bg.setToolTip("Puts what was removed on a layer below, so you can blur, adjust or hide it separately")
        self.keep_bg.setChecked(controller.keep_background)
        self.keep_bg.toggled.connect(lambda on: setattr(controller, "keep_background", on))
        col.addWidget(self.keep_bg)

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

        # ---- select by hand: entry buttons ----
        self.start_row = QWidget()
        sr = QHBoxLayout(self.start_row)
        sr.setContentsMargins(0, 0, 0, 0)
        self.pick_btn = _button("Pick object…")
        self.pick_btn.setToolTip("Click an object in the render: selected exactly, from its ID pass")
        self.pick_btn.clicked.connect(lambda: self._start("pick"))
        self.lasso_btn = _button("Lasso…")
        self.lasso_btn.setToolTip("Draw around what to keep")
        self.lasso_btn.clicked.connect(lambda: self._start("lasso"))
        self.brush_btn = _button("Brush…")
        self.brush_btn.setToolTip("Paint the mask: erase or restore by hand")
        self.brush_btn.clicked.connect(lambda: self._start("brush"))
        for b in (self.pick_btn, self.lasso_btn, self.brush_btn):
            sr.addWidget(b)
        col.addWidget(self.start_row)
        self.passes_label = QLabel()
        self.passes_label.setObjectName("hintLabel")
        self.passes_label.setWordWrap(True)
        self.attach_btn = _button("Attach ID pass…")
        self.attach_btn.setToolTip("Add the render's Object ID or Material ID pass (same size as the render)")
        self.attach_btn.clicked.connect(self._attach_pass)
        pass_row = QHBoxLayout()
        pass_row.addWidget(self.passes_label, 1)
        pass_row.addWidget(self.attach_btn)
        col.addLayout(pass_row)

        # ---- the session (on the canvas) ----
        self.brush_box = QWidget()
        bb = QVBoxLayout(self.brush_box)
        bb.setContentsMargins(0, 0, 0, 0)
        bb.setSpacing(4)
        tools = QHBoxLayout()
        self.tool_btns: dict[str, QPushButton] = {}
        for tool, label in (("pick", "Pick"), ("lasso", "Lasso"), ("brush", "Brush")):
            b = _button(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, t=tool: self._set_tool(t))
            tools.addWidget(b)
            self.tool_btns[tool] = b
        bb.addLayout(tools)
        self.hint = QLabel()
        self.hint.setObjectName("hintLabel")
        self.hint.setWordWrap(True)
        bb.addWidget(self.hint)
        # pick options
        self.pick_box = QWidget()
        pb = QVBoxLayout(self.pick_box)
        pb.setContentsMargins(0, 0, 0, 0)
        self.pick_pass = QComboBox()
        self.pick_pass.currentIndexChanged.connect(self._pick_pass_changed)
        pp = QHBoxLayout()
        lab = QLabel("Pick by")
        lab.setObjectName("sliderLabel")
        pp.addWidget(lab)
        pp.addStretch(1)
        pp.addWidget(self.pick_pass)
        pb.addLayout(pp)
        self.tolerance = SliderRow("tolerance", "Tolerance", minimum=0, maximum=60, default=0)
        self.tolerance.edited.connect(lambda v: setattr(self.brush, "tolerance", v))
        pb.addWidget(self.tolerance)
        bb.addWidget(self.pick_box)
        # brush options
        self.paint_box = QWidget()
        bb2 = QVBoxLayout(self.paint_box)
        bb2.setContentsMargins(0, 0, 0, 0)
        modes = QHBoxLayout()
        self.erase_btn, self.restore_btn = _button("Erase"), _button("Restore")
        for b, restore in ((self.erase_btn, False), (self.restore_btn, True)):
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, r=restore: self._set_mode(r))
            modes.addWidget(b)
        bb2.addLayout(modes)
        self.size_row = SliderRow("size", "Size", minimum=2, maximum=800, default=80, suffix=" px")
        self.size_row.edited.connect(lambda v: self._brush_setting("size", float(v)))
        self.hardness_row = SliderRow("hardness", "Hardness", minimum=0, maximum=100, default=60, suffix="%")
        self.hardness_row.edited.connect(lambda v: self._brush_setting("hardness", v / 100.0))
        bb2.addWidget(self.size_row)
        bb2.addWidget(self.hardness_row)
        bb.addWidget(self.paint_box)
        done_row = QHBoxLayout()
        self.done_btn, self.cancel_btn = _button("Done", primary=True), _button("Cancel")
        self.session_extract = _button("Extract")
        self.session_extract.setToolTip("Done, and put the selection on its own layer")
        self.done_btn.clicked.connect(lambda: self.brush.finish(apply=True))
        self.session_extract.clicked.connect(self._done_and_extract)
        self.cancel_btn.clicked.connect(lambda: self.brush.finish(apply=False))
        done_row.addWidget(self.done_btn)
        done_row.addWidget(self.session_extract)
        done_row.addWidget(self.cancel_btn)
        bb.addLayout(done_row)
        self.fill_row = FillRow(brush, controller)
        bb.addWidget(self.fill_row)
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
        kinds = [k for k in RS.ID_KINDS if layer is not None and k in layer.passes]
        self.start_row.setVisible(layer is not None and not brushing)
        self.pick_btn.setVisible(bool(kinds))
        self.passes_label.setVisible(layer is not None and not brushing)
        self.attach_btn.setVisible(layer is not None and not brushing)
        self.passes_label.setText("ID passes: " + ", ".join(RS.PASS_LABELS[k] for k in kinds) if kinds
                                  else "No ID pass (name it like render_objectid.png to attach it on import)")
        self.attach_btn.setText("Attach…" if kinds else "Attach ID pass…")
        self.brush_box.setVisible(brushing)
        tool = self.brush.tool
        for t, b in self.tool_btns.items():
            b.setChecked(t == tool)
        self.tool_btns["pick"].setVisible(bool(kinds))
        self.hint.setText({
            "pick": "Click an object to select it. Shift+click adds, Alt+click removes. Red = hidden.",
            "lasso": "Drag around it, or click corner by corner and press Enter. Shift adds, Alt removes.",
            "brush": "Drag to paint. Alt = the opposite mode · [ ] = size.",
        }[tool] + " Enter = done, Esc = cancel.")
        self.pick_box.setVisible(tool == "pick")
        self.paint_box.setVisible(tool == "brush")
        self.pick_pass.blockSignals(True)
        self.pick_pass.clear()
        for k in kinds:
            self.pick_pass.addItem(RS.PASS_LABELS[k], k)
        if self.brush.pick_kind in kinds:
            self.pick_pass.setCurrentIndex(kinds.index(self.brush.pick_kind))
        self.pick_pass.setVisible(len(kinds) > 1)
        self.pick_pass.blockSignals(False)
        self.tolerance.set_value(self.brush.tolerance)
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

    def _start(self, tool: str) -> None:
        if self._layer() is not None:
            self.brush.tool = tool
            self.brush.start(self._layer_id)

    def _set_tool(self, tool: str) -> None:
        self.brush.set_tool(tool)

    def _pick_pass_changed(self, _i: int) -> None:
        kind = self.pick_pass.currentData()
        if kind:
            self.brush.pick_kind = kind

    def _done_and_extract(self) -> None:
        layer_id = self.brush.layer_id
        if layer_id is None or not self.editor.doc.has_layer(layer_id):
            return
        batch = edits.extract_session(self.editor.doc, layer_id, self.brush.take_edit())
        self.brush.end()
        if batch is not None:  # one undo step: selection + extract
            self.editor.push(batch)
            self.editor.select(batch.parts[0].layer.id)

    def _attach_pass(self) -> None:
        layer = self._layer()
        if layer is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Attach ID pass", "",
                                              "Images (" + " ".join(f"*{e}" for e in RS.SUPPORTED_EXTS) + ")")
        if not path:
            return
        kind = RS.guess_kind(path)
        if kind not in RS.ID_KINDS:
            labels = [RS.PASS_LABELS[k] for k in RS.ID_KINDS]
            choice, ok = QInputDialog.getItem(self, "Attach ID pass", "What kind of pass is it?", labels, 0, False)
            if not ok:
                return
            kind = RS.ID_KINDS[labels.index(choice)]
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            info = self.editor.store.add_file(path)
        except (ImageError, MemoryError, OSError) as exc:
            QMessageBox.warning(self, "Couldn't attach the pass", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        src = self.editor.doc.assets[layer.source]
        if (info.width, info.height) != (src.width, src.height):
            QMessageBox.warning(self, "Couldn't attach the pass",
                                f"The pass is {info.width} × {info.height}; the render is {src.width} × {src.height}. "
                                "A pass must be the same size as its render.")
            return
        self.editor.push(edits.SetPass(layer.id, kind, info, text=f"Attach {RS.PASS_LABELS[kind]} pass"))
        self.refresh()

    def _set_mode(self, restore: bool) -> None:
        self.brush.restore = restore
        self.refresh()

    def _brush_setting(self, name: str, value: float) -> None:
        setattr(self.brush, name, value)
