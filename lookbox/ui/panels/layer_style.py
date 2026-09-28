"""Layer tab: opacity, blend mode, gradient fade (every layer) and the fill
editor (fill layers). Every change is a SetLayerField edit; one slider drag =
one undo step."""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget)

from lookbox.commands import edits
from lookbox.core.model import BLEND_MODES, FILL_KINDS, FillLayer, GradientFade, GradientStop, Layer
from lookbox.ui.editor import Editor
from lookbox.ui.widgets.slider_row import SliderRow

BLEND_LABELS = {"normal": "Normal", "multiply": "Multiply", "screen": "Screen", "overlay": "Overlay",
                "add": "Add (glow)", "soft_light": "Soft light"}
FADE_KINDS = (None, "linear", "radial")
FADE_LABELS = ("None", "Linear", "Radial")
FILL_LABELS = {"solid": "Solid colour", "linear": "Linear gradient", "radial": "Radial gradient"}


def _title(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("sectionTitle")
    return lab


def _row(label: str, widget: QWidget) -> QHBoxLayout:
    h = QHBoxLayout()
    lab = QLabel(label)
    lab.setObjectName("sliderLabel")
    h.addWidget(lab)
    h.addStretch(1)
    h.addWidget(widget)
    return h


class _ColourButton(QToolButton):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("colourButton")
        self.setFixedSize(44, 26)
        self.rgba = (0.0, 0.0, 0.0, 1.0)

    def set_rgba(self, rgba) -> None:
        self.rgba = tuple(rgba)
        c = QColor.fromRgbF(*self.rgba)
        css = f"rgba({c.red()}, {c.green()}, {c.blue()}, {c.alpha()})"
        self.setStyleSheet(f"QToolButton#colourButton {{ background: {css}; }}")
        self.setToolTip(f"{c.name().upper()}  ·  {round(c.alphaF() * 100)}% opaque")


class LayerPanel(QWidget):
    def __init__(self, editor: Editor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.editor = editor
        self._layer_id: str | None = None
        self._dragging = False
        self._serial = 0
        self._updating = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.hint = QLabel("Select a layer to style it.")
        self.hint.setObjectName("emptyHint")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.hint, 1)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("adjustBody")
        col = QVBoxLayout(body)
        col.setContentsMargins(16, 10, 16, 16)
        col.setSpacing(6)

        # ---- fill (fill layers only) ----
        self.fill_box = QWidget()
        fl = QVBoxLayout(self.fill_box)
        fl.setContentsMargins(0, 0, 0, 8)
        fl.addWidget(_title("Fill"))
        self.fill_kind = QComboBox()
        for k in FILL_KINDS:
            self.fill_kind.addItem(FILL_LABELS[k], k)
        self.fill_kind.currentIndexChanged.connect(lambda _i: self._fill_change(kind=self.fill_kind.currentData()))
        fl.addLayout(_row("Type", self.fill_kind))
        colours = QHBoxLayout()
        self.c0, self.c1 = _ColourButton(), _ColourButton()
        self.c0.clicked.connect(lambda: self._pick_colour(0))
        self.c1.clicked.connect(lambda: self._pick_colour(1))
        self.swap = QPushButton("Swap")
        self.swap.clicked.connect(self._swap_colours)
        colours.addWidget(self.c0)
        colours.addWidget(self.c1)
        colours.addStretch(1)
        colours.addWidget(self.swap)
        fl.addLayout(colours)
        self.angle = self._slider("angle_deg", "Angle", -180, 180, 90, "°", fl, self._fill_slider)
        self.cx = self._slider("cx", "Centre X", 0, 100, 50, "%", fl, self._fill_slider)
        self.cy = self._slider("cy", "Centre Y", 0, 100, 45, "%", fl, self._fill_slider)
        self.radius = self._slider("radius", "Size", 5, 200, 80, "%", fl, self._fill_slider)
        self.fit_btn = QPushButton("Fit to canvas")
        self.fit_btn.clicked.connect(self._fit_to_canvas)
        fl.addWidget(self.fit_btn)
        col.addWidget(self.fill_box)

        # ---- layer ----
        col.addWidget(_title("Layer"))
        self.opacity = self._slider("opacity", "Opacity", 0, 100, 100, "%", col, self._opacity)
        self.blend = QComboBox()
        for m in BLEND_MODES:
            self.blend.addItem(BLEND_LABELS[m], m)
        self.blend.currentIndexChanged.connect(self._blend)
        col.addLayout(_row("Blend mode", self.blend))

        # ---- gradient fade ----
        col.addSpacing(8)
        col.addWidget(_title("Gradient fade"))
        self.fade_kind = QComboBox()
        self.fade_kind.addItems(FADE_LABELS)
        self.fade_kind.currentIndexChanged.connect(self._fade_kind)
        col.addLayout(_row("Type", self.fade_kind))
        self.fade_angle = self._slider("angle_deg", "Direction", -180, 180, 90, "°", col, self._fade_slider)
        self.fade_start = self._slider("start", "Start", 0, 100, 50, "%", col, self._fade_slider)
        self.fade_end = self._slider("end", "End", 0, 100, 100, "%", col, self._fade_slider)
        self.fade_invert = QCheckBox("Invert")
        self.fade_invert.setObjectName("invertToggle")
        self.fade_invert.toggled.connect(lambda on: self._fade_change(invert=on))
        col.addWidget(self.fade_invert)
        col.addStretch(1)
        self.scroll.setWidget(body)
        outer.addWidget(self.scroll, 1)

        editor.selection_changed.connect(self.refresh)
        editor.changed.connect(self.refresh)
        editor.document_replaced.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------ building
    def _slider(self, key, label, lo, hi, default, suffix, layout, handler) -> SliderRow:
        row = SliderRow(key, label, minimum=lo, maximum=hi, default=default, suffix=suffix)
        row.edited.connect(lambda v, k=key: handler(k, v))
        row.pressed.connect(self._pressed)
        row.released.connect(self._released)
        layout.addWidget(row)
        return row

    # ------------------------------------------------------------ document → widgets
    def _layer(self) -> Layer | None:
        doc = self.editor.doc
        return doc.layer(self._layer_id) if self._layer_id and doc.has_layer(self._layer_id) else None

    def refresh(self) -> None:
        if self._updating:
            return  # a widget we're setting fired a signal; never re-enter
        layer = self.editor.selected_layer()
        self._layer_id = layer.id if layer is not None else None
        self.hint.setVisible(layer is None)
        self.scroll.setVisible(layer is not None)
        if layer is None or self._dragging:
            return
        self._updating = True
        try:
            is_fill = isinstance(layer, FillLayer)
            self.fill_box.setVisible(is_fill)
            if is_fill:
                f = layer.fill
                self.fill_kind.setCurrentIndex(FILL_KINDS.index(f.kind))
                stops = sorted(f.stops, key=lambda s: s.pos)
                self.c0.set_rgba(stops[0].color)
                self.c1.set_rgba(stops[-1].color)
                self.c1.setVisible(f.kind != "solid")
                self.swap.setVisible(f.kind != "solid")
                self.angle.set_value(f.angle_deg)
                self.angle.setVisible(f.kind == "linear")
                for row, v in ((self.cx, f.cx), (self.cy, f.cy), (self.radius, f.radius)):
                    row.set_value(v * 100)
                    row.setVisible(f.kind == "radial")
            self.opacity.set_value(layer.opacity * 100)
            self.blend.setCurrentIndex(BLEND_MODES.index(layer.blend_mode))
            fade = layer.fade
            self.fade_kind.setCurrentIndex(FADE_KINDS.index(fade.kind if fade else None))
            for row in (self.fade_angle, self.fade_start, self.fade_end):
                row.setVisible(fade is not None)
            self.fade_invert.setVisible(fade is not None)
            if fade is not None:
                self.fade_angle.setVisible(fade.kind == "linear")
                self.fade_angle.set_value(fade.angle_deg)
                self.fade_start.set_value(fade.start * 100)
                self.fade_end.set_value(fade.end * 100)
                self.fade_invert.setChecked(fade.invert)
        finally:
            self._updating = False

    # ------------------------------------------------------------ widgets → document
    def _pressed(self) -> None:
        self._dragging = True
        self._serial += 1

    def _released(self) -> None:
        self._dragging = False
        self.refresh()

    def _push(self, name: str, new, text: str) -> None:
        layer = self._layer()
        if layer is None or self._updating or getattr(layer, name) == new:
            return
        key = f"style-drag-{self._serial}" if self._dragging else f"style-{layer.id}-{name}"
        self.editor.push(edits.SetLayerField(layer.id, name, getattr(layer, name), new, text=text, merge_key=key))

    def _opacity(self, _key: str, v: int) -> None:
        self._push("opacity", v / 100.0, "Opacity")

    def _blend(self, _i: int) -> None:
        self._push("blend_mode", self.blend.currentData(), "Blend mode")

    def _fade_kind(self, i: int) -> None:
        layer = self._layer()
        if layer is None or self._updating:
            return
        kind = FADE_KINDS[i]
        if kind is None:
            self._push("fade", None, "Remove fade")
        else:
            base = copy.deepcopy(layer.fade) if layer.fade else GradientFade()
            base.kind = kind
            self._push("fade", base, "Gradient fade")
        self.refresh()

    def _fade_change(self, **changes) -> None:
        layer = self._layer()
        if self._updating or layer is None or layer.fade is None:
            return
        new = copy.deepcopy(layer.fade)
        for k, v in changes.items():
            setattr(new, k, v)
        self._push("fade", new, "Gradient fade")

    def _fade_slider(self, key: str, v: int) -> None:
        self._fade_change(**{key: float(v) if key == "angle_deg" else v / 100.0})

    def _fill_change(self, **changes) -> None:
        layer = self._layer()
        if self._updating or not isinstance(layer, FillLayer):
            return
        new = copy.deepcopy(layer.fill)
        for k, v in changes.items():
            setattr(new, k, v)
        self._push("fill", new, "Fill")
        if "kind" in changes:
            self.refresh()

    def _fill_slider(self, key: str, v: int) -> None:
        self._fill_change(**{key: float(v) if key == "angle_deg" else v / 100.0})

    def _pick_colour(self, which: int) -> None:
        layer = self._layer()
        if not isinstance(layer, FillLayer):
            return
        stops = sorted(copy.deepcopy(layer.fill.stops), key=lambda s: s.pos)
        idx = 0 if which == 0 else len(stops) - 1
        c = QColorDialog.getColor(QColor.fromRgbF(*stops[idx].color), self, "Fill colour",
                                  QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            stops[idx] = GradientStop(pos=stops[idx].pos, color=(c.redF(), c.greenF(), c.blueF(), c.alphaF()))
            self._fill_change(stops=stops)

    def _swap_colours(self) -> None:
        layer = self._layer()
        if isinstance(layer, FillLayer):
            stops = [GradientStop(pos=1.0 - s.pos, color=s.color) for s in layer.fill.stops]
            self._fill_change(stops=sorted(stops, key=lambda s: s.pos))

    def _fit_to_canvas(self) -> None:
        layer = self._layer()
        if not isinstance(layer, FillLayer):
            return
        doc = self.editor.doc
        from lookbox.core.model import Transform  # local: only this action needs it

        stack = self.editor.stack
        stack.beginMacro("Fit fill to canvas")
        try:
            self.editor.push(edits.SetLayerField(layer.id, "width", layer.width, doc.canvas.w))
            self.editor.push(edits.SetLayerField(layer.id, "height", layer.height, doc.canvas.h))
            self.editor.push(edits.SetTransform(layer.id, layer.transform,
                                                Transform(x=doc.canvas.w / 2.0, y=doc.canvas.h / 2.0)))
        finally:
            stack.endMacro()
