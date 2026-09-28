"""Adjust panel 'Color edit' (§7 #9): swatches of the layer's dominant colours;
pick one to shift that colour's hue, saturation and lightness."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from lookbox.core.model import ColorBand
from lookbox.core.render.adjust import suggest_swatch_hues
from lookbox.core.render.levels import thumbnail_array
from lookbox.ui.widgets.slider_row import SliderRow

MAX_SWATCHES = 5
BAND_FIELDS = (("hue_shift", "Hue"), ("saturation", "Saturation"), ("lightness", "Lightness"))


def same_hue(a: float, b: float) -> bool:
    return abs(((a - b) + 180.0) % 360.0 - 180.0) < 0.5


class ColorEditSection(QWidget):
    band_edited = Signal(float, str, int)  # band hue, field, value
    pressed = Signal()
    released = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.swatch_row = QHBoxLayout()
        self.swatch_row.setSpacing(8)
        self.swatch_row.addStretch(1)
        lay.addLayout(self.swatch_row)
        self.none_label = QLabel("No distinct colours found in this image.")
        self.none_label.setObjectName("hintLabel")
        lay.addWidget(self.none_label)

        self.rows: dict[str, SliderRow] = {}
        for key, label in BAND_FIELDS:
            row = SliderRow(key, label)
            row.edited.connect(lambda v, k=key: self._edited(k, v))
            row.pressed.connect(self.pressed)
            row.released.connect(self.released)
            lay.addWidget(row)
            self.rows[key] = row
        self._buttons: list[tuple[float, QToolButton]] = []
        self.selected: float | None = None
        self._set_rows_visible(False)

    # ---- driven by the panel ----
    def set_source(self, pixels, bands: list[ColorBand]) -> None:
        """New layer selected: rebuild swatches from its pixels plus any edited bands."""
        hues = [b.hue for b in bands]
        if pixels is not None:
            for h in suggest_swatch_hues(thumbnail_array(pixels, 160), k=4):
                if all(abs(((h - x) + 180) % 360 - 180) > 20 for x in hues):
                    hues.append(h)
        hues = hues[:MAX_SWATCHES]
        for _, b in self._buttons:
            self.swatch_row.removeWidget(b)
            b.deleteLater()
        self._buttons = []
        for i, h in enumerate(hues):
            b = QToolButton()
            b.setObjectName("swatch")
            b.setCheckable(True)
            b.setFixedSize(30, 30)
            c = QColor.fromHsvF((h % 360.0) / 360.0, 0.75, 0.85)
            b.setStyleSheet(f"QToolButton#swatch {{ background: {c.name()}; }}")
            b.setToolTip("Edit this colour")
            b.clicked.connect(lambda _=False, hue=h: self._select(hue))
            self.swatch_row.insertWidget(i, b)
            self._buttons.append((h, b))
        self.none_label.setVisible(not hues)
        keep = self.selected if self.selected is not None and any(same_hue(self.selected, h) for h in hues) else None
        self._select(keep)
        self.set_bands(bands)

    def set_bands(self, bands: list[ColorBand]) -> None:
        """Show the selected band's values (from the document)."""
        band = next((b for b in bands if self.selected is not None and same_hue(b.hue, self.selected)), None)
        for key, row in self.rows.items():
            row.set_value(getattr(band, key) if band is not None else 0)

    # ---- internals ----
    def _select(self, hue: float | None) -> None:
        self.selected = hue
        for h, b in self._buttons:
            b.setChecked(hue is not None and same_hue(h, hue))
        self._set_rows_visible(hue is not None)

    def _set_rows_visible(self, on: bool) -> None:
        for row in self.rows.values():
            row.setVisible(on)

    def _edited(self, key: str, v: int) -> None:
        if self.selected is not None:
            self.band_edited.emit(self.selected, key, v)
