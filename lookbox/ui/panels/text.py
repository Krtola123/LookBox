"""Style tab → Text (text layers only): content, font, size, weight, italic, colour,
alignment, letter spacing, line height, wrapping.

Typing updates the canvas live (the real render, effects included). One focus
session in the text box = one undo step; one slider drag = one undo step. Every
change goes through edits.change_text, which re-places the box so text grows
from its left edge / centre / right edge instead of from the middle.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFontComboBox, QHBoxLayout, QLabel,
                               QPlainTextEdit, QToolButton, QVBoxLayout, QWidget)

from lookbox.commands import edits
from lookbox.core.model import TEXT_ALIGNS, TextLayer
from lookbox.core.render import text as text_render
from lookbox.ui.widgets.colour_button import ColourButton

WEIGHTS = ((300, "Light"), (400, "Regular"), (500, "Medium"), (600, "Semibold"), (700, "Bold"),
           (800, "Extra bold"), (900, "Black"))
ALIGN_LABELS = {"left": "Left", "center": "Centre", "right": "Right"}


class _TextBox(QPlainTextEdit):
    focused = Signal()

    def focusInEvent(self, e) -> None:
        self.focused.emit()
        super().focusInEvent(e)


def _labelled(label: str, widget: QWidget) -> QHBoxLayout:
    h = QHBoxLayout()
    lab = QLabel(label)
    lab.setObjectName("sliderLabel")
    h.addWidget(lab)
    h.addStretch(1)
    h.addWidget(widget)
    return h


class TextSection(QWidget):
    def __init__(self, panel) -> None:
        super().__init__()
        self.panel = panel
        self._typing = 0  # typing session counter → merge key
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 8)
        col.setSpacing(6)
        title = QLabel("Text")
        title.setObjectName("sectionTitle")
        col.addWidget(title)

        self.box = _TextBox()
        self.box.setObjectName("textContent")
        self.box.setPlaceholderText("Type here…")
        self.box.setFixedHeight(84)
        self.box.focused.connect(self._new_typing_session)
        self.box.textChanged.connect(self._typed)
        col.addWidget(self.box)

        self.font_box = QFontComboBox()
        self.font_box.setMaximumWidth(190)
        self.font_box.currentFontChanged.connect(lambda f: self._set("Font", font_family=f.family()))
        col.addLayout(_labelled("Font", self.font_box))

        style = QHBoxLayout()
        self.weight = QComboBox()
        for w, name in WEIGHTS:
            self.weight.addItem(name, w)
        self.weight.currentIndexChanged.connect(lambda _i: self._set("Font weight", weight=self.weight.currentData()))
        self.italic = QToolButton()
        self.italic.setText("I")
        self.italic.setCheckable(True)
        f = QFont()
        f.setItalic(True)
        self.italic.setFont(f)
        self.italic.setToolTip("Italic")
        self.italic.toggled.connect(lambda on: self._set("Italic", italic=on))
        self.colour = ColourButton()
        self.colour.clicked.connect(self._pick_colour)
        style.addWidget(self.weight, 1)
        style.addWidget(self.italic)
        style.addWidget(self.colour)
        col.addLayout(style)

        align = QHBoxLayout()
        self.align_group = QButtonGroup(self)
        self.align_group.setExclusive(True)
        self.align_btns: dict[str, QToolButton] = {}
        for a in TEXT_ALIGNS:
            b = QToolButton()
            b.setText(ALIGN_LABELS[a])
            b.setCheckable(True)
            b.setObjectName("alignButton")
            b.clicked.connect(lambda _=False, a=a: self._set("Align text", align=a))
            self.align_group.addButton(b)
            self.align_btns[a] = b
            align.addWidget(b)
        col.addLayout(align)

        self.size_row = panel._slider("font_size", "Size", 4, 1000, 96, " px", col, self._slider)
        self.spacing_row = panel._slider("letter_spacing", "Letter spacing", -50, 200, 0, " px", col, self._slider)
        self.line_row = panel._slider("line_height", "Line height", 50, 300, 120, "%", col, self._slider)
        self.wrap = QCheckBox("Wrap at a fixed width")
        self.wrap.setObjectName("effectToggle")
        self.wrap.toggled.connect(self._toggle_wrap)
        col.addWidget(self.wrap)
        self.width_row = panel._slider("box_width", "Width", 1, 8000, 600, " px", col, self._slider)

    # ------------------------------------------------------------ document → widgets
    def refresh(self, layer: TextLayer) -> None:
        """Called by the panel with its re-entrancy guard up (widget signals are ignored)."""
        if self.box.toPlainText() != layer.text:  # never while typing: that would move the cursor
            self.box.setPlainText(layer.text)
        if self.font_box.currentFont().family() != layer.font_family:
            self.font_box.setCurrentFont(QFont(layer.font_family))
        idx = min(range(len(WEIGHTS)), key=lambda i: abs(WEIGHTS[i][0] - layer.weight))
        self.weight.setCurrentIndex(idx)
        self.italic.setChecked(layer.italic)
        self.colour.set_rgba(layer.color)
        self.align_btns.get(layer.align, self.align_btns["left"]).setChecked(True)
        self.size_row.set_value(layer.font_size)
        self.spacing_row.set_value(layer.letter_spacing)
        self.line_row.set_value(layer.line_height * 100)
        self.wrap.setChecked(layer.box_width is not None)
        self.width_row.setVisible(layer.box_width is not None)
        if layer.box_width is not None:
            self.width_row.set_value(layer.box_width)

    def start_editing(self) -> None:
        """Double-click on the canvas / new text: put the cursor in the text box, all selected."""
        self.box.setFocus()
        self.box.selectAll()

    # ------------------------------------------------------------ widgets → document
    def _layer(self) -> TextLayer | None:
        layer = self.panel._layer()
        return layer if isinstance(layer, TextLayer) else None

    def _set(self, label: str, merge_key: str | None = None, **fields) -> None:
        layer = self._layer()
        if layer is None or self.panel._updating:
            return
        if merge_key is None and self.panel._dragging:
            merge_key = f"style-drag-{self.panel._serial}"
        edit = edits.change_text(self.panel.editor.doc, layer.id, label=label, merge_key=merge_key, **fields)
        if edit is not None:
            self.panel.editor.push(edit)

    def _new_typing_session(self) -> None:
        self._typing += 1

    def _typed(self) -> None:
        self._set("Type", merge_key=f"typing-{self._typing}", text=self.box.toPlainText())

    def _slider(self, key: str, v: int) -> None:
        value = v / 100.0 if key == "line_height" else float(v)
        self._set({"font_size": "Font size", "letter_spacing": "Letter spacing", "line_height": "Line height",
                   "box_width": "Text width"}[key], **{key: value})

    def _toggle_wrap(self, on: bool) -> None:
        layer = self._layer()
        if layer is None or self.panel._updating:
            return
        width = math.ceil(text_render.content_width(layer)) if on else None  # ceil: the widest line must still fit
        self._set("Wrap text" if on else "Don't wrap", box_width=width)
        self.panel.refresh()

    def _pick_colour(self) -> None:
        if self._layer() is None:
            return
        rgba = self.colour.pick("Text colour")
        if rgba is not None:
            self._set("Text colour", color=rgba)
