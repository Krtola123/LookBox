"""Adjust panel (ARCHITECTURE §7, §12) — Canva's layout: Filters, Cut-out, White
balance, Light, Color, Color edit, Texture, Reset.

Two targets (M10): **This layer** (an image) or **Whole design** (the global grade:
adjust + filter on the flattened composite). Every change goes through
SetAdjustments / SetLut with the layer id, or None for the whole design. One
slider drag = one undo step (merge key per drag). While a slider drags the
canvas renders half-res; the full-res preview lands on release.
"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QToolButton, QVBoxLayout, QWidget)

from lookbox.commands import edits
from lookbox.core.model import Adjustments, ColorBand, ImageLayer
from lookbox.core.render.pipeline import layer_source, render
from lookbox.ui.editor import Editor
from lookbox.ui.panels.color_edit import ColorEditSection, same_hue
from lookbox.ui.panels.filters import FiltersSection
from lookbox.ui.widgets.slider_row import SliderRow

GROUPS = [
    ("White balance", [("temperature", "Temperature"), ("tint", "Tint")]),
    ("Light", [("brightness", "Brightness"), ("contrast", "Contrast"), ("highlights", "Highlights"),
               ("shadows", "Shadows"), ("whites", "Whites"), ("blacks", "Blacks")]),
    ("Color", [("vibrance", "Vibrance"), ("saturation", "Saturation")]),
    ("Texture", [("sharpness", "Sharpness"), ("clarity", "Clarity"), ("vignette", "Vignette")]),
]
LABELS = {k: label for _, rows in GROUPS for k, label in rows}
DEBOUNCE_MS = 30


def _title(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("sectionTitle")
    return lab


class AdjustPanel(QWidget):
    interactive = Signal(str, bool)  # layer id, slider being dragged
    compare = Signal(object)  # layer id to show "before", or None
    global_interactive = Signal(bool)  # a whole-design slider is being dragged
    global_compare = Signal(bool)  # show the design without its whole-design grade

    def __init__(self, editor: Editor, cutout: QWidget | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.editor = editor
        self.cutout = cutout  # the Cut-out section (M6), shown at the top of the body
        self._layer_id: str | None = None
        self._whole = False  # editing the whole-design grade instead of a layer
        self._dragging = False
        self._drag_serial = 0
        self._pending: dict[str, int] = {}
        self._pending_bands: dict[tuple[float, str], int] = {}
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(DEBOUNCE_MS)
        self._flush_timer.timeout.connect(self._flush)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.setContentsMargins(16, 10, 12, 4)
        header.addWidget(_title("Adjust"))
        header.addStretch(1)
        self.target_btns: dict[bool, QToolButton] = {}
        for whole, label, tip in ((False, "This layer", "Adjust the selected image"),
                                  (True, "Whole design", "Grade everything together: applied to the finished image")):
            b = QToolButton()
            b.setObjectName("targetButton")
            b.setText(label)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, w=whole: self.set_whole(w))
            header.addWidget(b)
            self.target_btns[whole] = b
        header.addSpacing(6)
        self.compare_btn = QToolButton()
        self.compare_btn.setObjectName("compareButton")
        self.compare_btn.setText("Before / After")
        self.compare_btn.setCheckable(True)
        self.compare_btn.setToolTip("Show this layer without adjustments")
        self.compare_btn.toggled.connect(self._on_compare)
        header.addWidget(self.compare_btn)
        outer.addLayout(header)

        self.hint = QLabel("Select an image to adjust it.")
        self.hint.setObjectName("emptyHint")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.hint, 1)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("adjustBody")
        col = QVBoxLayout(body)
        col.setContentsMargins(16, 4, 16, 16)
        col.setSpacing(6)
        self.filters = FiltersSection(editor)
        self.filters.pressed.connect(self._pressed)
        self.filters.released.connect(self._released)
        col.addWidget(self.filters)
        if cutout is not None:
            col.addWidget(cutout)
        self.rows: dict[str, SliderRow] = {}
        for group, rows in GROUPS:
            head = QHBoxLayout()
            head.addWidget(_title(group))
            head.addStretch(1)
            if group == "Color":
                self.invert = QCheckBox("Invert")
                self.invert.setObjectName("invertToggle")
                self.invert.toggled.connect(self._on_invert)
                head.addWidget(self.invert)
            col.addSpacing(6)
            col.addLayout(head)
            for key, label in rows:
                row = SliderRow(key, label)
                row.edited.connect(lambda v, k=key: self._edited(k, v))
                row.pressed.connect(self._pressed)
                row.released.connect(self._released)
                col.addWidget(row)
                self.rows[key] = row
            if group == "Color":
                col.addSpacing(6)
                col.addWidget(_title("Color edit"))
                self.color_edit = ColorEditSection()
                self.color_edit.band_edited.connect(self._band_edited)
                self.color_edit.pressed.connect(self._pressed)
                self.color_edit.released.connect(self._released)
                col.addWidget(self.color_edit)
        col.addSpacing(12)
        self.reset_btn = QPushButton("Reset adjustments")
        self.reset_btn.setObjectName("resetButton")
        self.reset_btn.clicked.connect(self._reset_all)
        col.addWidget(self.reset_btn)
        col.addStretch(1)
        self.scroll.setWidget(body)
        outer.addWidget(self.scroll, 1)

        editor.selection_changed.connect(self._on_selection)
        editor.changed.connect(self.refresh)
        editor.document_replaced.connect(self._on_selection)
        self._on_selection()

    # ------------------------------------------------------------ document → widgets
    def _layer(self) -> ImageLayer | None:
        """The layer this panel is editing, even mid-way through a selection change,
        so a pending edit still lands on the layer it was made for."""
        doc = self.editor.doc
        if self._layer_id is None or not doc.has_layer(self._layer_id):
            return None
        layer = doc.layer(self._layer_id)
        return layer if isinstance(layer, ImageLayer) else None

    def _target_adjust(self) -> Adjustments | None:
        """The Adjustments being edited: the whole design's, or the layer's (None = nothing)."""
        if self._whole:
            return self.editor.doc.global_adjust
        layer = self._layer()
        return layer.adjust if layer is not None else None

    def _target_id(self) -> str | None:
        return None if self._whole else self._layer_id

    def set_whole(self, whole: bool) -> None:
        if whole != self._whole:
            self._flush()
            self.compare_btn.setChecked(False)  # ends a before/after for the old target first
            self._whole = whole
        self._on_selection()

    def _composite_thumb(self):
        """A small render of the whole design (for filter thumbnails and colour swatches)."""
        rev = (self.editor.generation, self.editor.revision)
        if getattr(self, "_thumb_rev", None) != rev:
            doc = self.editor.doc
            s = min(1.0, 256.0 / max(doc.canvas.w, doc.canvas.h))
            self._thumb, self._thumb_rev = render(doc, self.editor.store, s, preview=True), rev
        return self._thumb

    def _on_selection(self) -> None:
        self._flush()
        if self.compare_btn.isChecked():
            self.compare_btn.setChecked(False)  # emits compare(None) / global_compare(False)
        for whole, b in self.target_btns.items():
            b.setChecked(whole == self._whole)
        layer = self.editor.selected_layer()
        if layer is not None and not isinstance(layer, ImageLayer):
            self.hint.setText("Adjustments are for images.\nUse the Style tab to style this layer,\n"
                              "or Whole design to grade everything.")
            layer = None
        else:
            self.hint.setText("Select an image to adjust it,\nor choose Whole design to grade everything.")
        self._layer_id = layer.id if layer is not None else None
        active = self._whole or layer is not None
        self.hint.setVisible(not active)
        self.scroll.setVisible(active)
        self.compare_btn.setEnabled(active)
        self.compare_btn.setToolTip("Show the design without its whole-design grade" if self._whole
                                    else "Show this layer without adjustments")
        if self._whole:
            thumb = self._composite_thumb()
            self.color_edit.set_source(thumb, self.editor.doc.global_adjust.color_edit)
            self.filters.set_target(True, None, thumb)
        elif layer is not None:
            src = layer_source(layer, self.editor.store)
            self.color_edit.set_source(src, layer.adjust.color_edit)
            self.filters.set_target(True, layer.id, src)
        else:
            self.filters.set_target(False, None, None)
        if self.cutout is not None:
            self.cutout.setVisible(not self._whole)
            self.cutout.set_layer(None if self._whole else self._layer_id)
        self.refresh()

    def refresh(self) -> None:
        a = self._target_adjust()
        if a is None:
            return
        self.filters.refresh()
        if self.cutout is not None and not self._whole:
            self.cutout.refresh()
        for key, row in self.rows.items():
            if not (self._dragging and key in self._pending):  # don't fight the hand on the slider
                row.set_value(getattr(a, key))
        self.invert.blockSignals(True)
        self.invert.setChecked(a.invert)
        self.invert.blockSignals(False)
        if not self._dragging:
            self.color_edit.set_bands(a.color_edit)
        self.reset_btn.setEnabled(not a.is_identity())

    # ------------------------------------------------------------ widgets → document
    def _pressed(self) -> None:
        self._flush()
        self._dragging = True
        self._drag_serial += 1
        if self._whole:
            self.global_interactive.emit(True)
        elif self._layer_id:
            self.interactive.emit(self._layer_id, True)

    def _released(self) -> None:
        self._flush()
        self._dragging = False
        if self._whole:
            self.global_interactive.emit(False)
        elif self._layer_id:
            self.interactive.emit(self._layer_id, False)
        self.refresh()

    def _edited(self, key: str, v: int) -> None:
        self._pending[key] = v
        self._flush_timer.start()

    def _band_edited(self, hue: float, field: str, v: int) -> None:
        self._pending_bands[(hue, field)] = v
        self._flush_timer.start()

    def _flush(self) -> None:
        self._flush_timer.stop()
        current = self._target_adjust()
        if current is None or not (self._pending or self._pending_bands):
            self._pending.clear()
            self._pending_bands.clear()
            return
        new = copy.deepcopy(current)
        for key, v in self._pending.items():
            setattr(new, key, float(v))
        for (hue, field), v in self._pending_bands.items():
            band = next((b for b in new.color_edit if same_hue(b.hue, hue)), None)
            if band is None:
                band = ColorBand(hue=hue)
                new.color_edit.append(band)
            setattr(band, field, float(v))
        names = [LABELS[k] for k in self._pending] + (["Color edit"] if self._pending_bands else [])
        self._pending.clear()
        self._pending_bands.clear()
        if new == current:
            return
        # One drag = one undo step; outside a drag, repeated edits to one control merge too.
        tid = self._target_id()
        key = f"adj-drag-{self._drag_serial}" if self._dragging else f"adj-{tid or 'design'}-{'+'.join(names)}"
        where = "Whole design: " if tid is None else "Adjust "
        self.editor.push(edits.SetAdjustments(tid, current, new, text=f"{where}{', '.join(names)}", merge_key=key))

    def _on_invert(self, on: bool) -> None:
        current = self._target_adjust()
        if current is not None and current.invert != on:
            self._flush()
            new = copy.deepcopy(current)
            new.invert = on
            self.editor.push(edits.SetAdjustments(self._target_id(), current, new, text="Invert colours"))

    def _reset_all(self) -> None:
        current = self._target_adjust()
        if current is not None and not current.is_identity():
            self._flush()
            self.editor.push(edits.SetAdjustments(self._target_id(), current, Adjustments(),
                                                  text="Reset adjustments"))

    def _on_compare(self, on: bool) -> None:
        if self._whole:
            self.global_compare.emit(on)
        else:
            self.compare.emit(self._layer_id if on else None)
