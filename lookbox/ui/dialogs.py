"""Small dialogs: New design / Resize page, Export pages."""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QColorDialog, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QHBoxLayout, QPushButton, QSpinBox, QWidget)

PRESETS = [
    ("Full HD — 1920 × 1080", 1920, 1080),
    ("4K UHD — 3840 × 2160", 3840, 2160),
    ("Square — 1080 × 1080", 1080, 1080),
    ("Square 4K — 2160 × 2160", 2160, 2160),
    ("Portrait post — 1080 × 1350", 1080, 1350),
    ("Story — 1080 × 1920", 1080, 1920),
    ("ArtStation cover — 1920 × 1080", 1920, 1080),
    ("Custom", 0, 0),
]
BACKGROUNDS = ["Transparent", "White", "Black", "Custom colour…"]


class NewDocumentDialog(QDialog):
    """New design, or (with `resize=(w, h)`) Resize page: size only, starting from it."""

    def __init__(self, parent: QWidget | None = None, resize: tuple[int, int] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resize page" if resize else "New design")
        self._custom = QColor("#202124")

        self.preset = QComboBox()
        for label, _, _ in PRESETS:
            self.preset.addItem(label)
        self.w = QSpinBox()
        self.h = QSpinBox()
        for s in (self.w, self.h):
            s.setRange(16, 16384)
            s.setSuffix(" px")
        self.w.setValue(1920)
        self.h.setValue(1080)

        size_row = QWidget()
        sl = QHBoxLayout(size_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.w)
        sl.addWidget(self.h)

        self.bg = QComboBox()
        self.bg.addItems(BACKGROUNDS)
        self.pick = QPushButton("Pick…")
        self.pick.setVisible(False)
        bg_row = QWidget()
        bl = QHBoxLayout(bg_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.bg, 1)
        bl.addWidget(self.pick)

        form = QFormLayout(self)
        form.addRow("Size", self.preset)
        form.addRow("", size_row)
        if not resize:
            form.addRow("Background", bg_row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Resize" if resize else "Create")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.preset.currentIndexChanged.connect(self._on_preset)
        self.w.valueChanged.connect(self._on_manual)
        self.h.valueChanged.connect(self._on_manual)
        self.bg.currentIndexChanged.connect(lambda i: self.pick.setVisible(i == 3))
        self.pick.clicked.connect(self._pick)
        self._applying = False
        if resize:
            self.w.setValue(resize[0])
            self.h.setValue(resize[1])

    def _on_preset(self, i: int) -> None:
        _, w, h = PRESETS[i]
        if w:
            self._applying = True
            self.w.setValue(w)
            self.h.setValue(h)
            self._applying = False

    def _on_manual(self) -> None:
        if not self._applying:
            self.preset.setCurrentIndex(len(PRESETS) - 1)

    def _pick(self) -> None:
        c = QColorDialog.getColor(self._custom, self, "Background colour")
        if c.isValid():
            self._custom = c

    def result_values(self) -> tuple[int, int, tuple[float, float, float, float] | None]:
        i = self.bg.currentIndex()
        bg = {0: None, 1: (1.0, 1.0, 1.0, 1.0), 2: (0.0, 0.0, 0.0, 1.0)}.get(i)
        if i == 3:
            bg = (self._custom.redF(), self._custom.greenF(), self._custom.blueF(), 1.0)
        return self.w.value(), self.h.value(), bg


class ExportPagesDialog(QDialog):
    """Which pages to export (projects with more than one page)."""

    def __init__(self, pages: list[tuple[str, str]], active: int, parent: QWidget | None = None) -> None:
        from PySide6.QtWidgets import QCheckBox, QLabel, QVBoxLayout

        super().__init__(parent)
        self.setWindowTitle("Export pages")
        col = QVBoxLayout(self)
        col.addWidget(QLabel("Export these pages (one PNG each):"))
        self.boxes = []
        for i, (label, _pid) in enumerate(pages):
            b = QCheckBox(label)
            b.setChecked(True)
            col.addWidget(b)
            self.boxes.append(b)
        row = QHBoxLayout()
        for text, fn in (("All", lambda: self._set(True)), ("None", lambda: self._set(False)),
                         ("Only this page", lambda: self._only(active))):
            btn = QPushButton(text)
            btn.clicked.connect(fn)
            row.addWidget(btn)
        col.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        col.addWidget(buttons)
        self._ids = [pid for _, pid in pages]

    def _set(self, on: bool) -> None:
        for b in self.boxes:
            b.setChecked(on)

    def _only(self, i: int) -> None:
        for j, b in enumerate(self.boxes):
            b.setChecked(j == i)

    def chosen(self) -> list[str]:
        return [pid for pid, b in zip(self._ids, self.boxes) if b.isChecked()]
