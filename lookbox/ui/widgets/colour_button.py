"""A small swatch button showing an RGBA colour; `pick()` opens the colour dialog."""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QToolButton, QWidget


class ColourButton(QToolButton):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("colourButton")
        self.setFixedSize(44, 26)
        self.rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def set_rgba(self, rgba) -> None:
        self.rgba = tuple(float(c) for c in rgba)
        c = QColor.fromRgbF(*self.rgba)
        css = f"rgba({c.red()}, {c.green()}, {c.blue()}, {c.alpha()})"
        self.setStyleSheet(f"QToolButton#colourButton {{ background: {css}; }}")
        self.setToolTip(f"{c.name().upper()}  ·  {round(c.alphaF() * 100)}% opaque")

    def pick(self, title: str = "Colour") -> tuple[float, float, float, float] | None:
        """Ask for a colour (with alpha). None if the user cancels."""
        c = QColorDialog.getColor(QColor.fromRgbF(*self.rgba), self, title,
                                  QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if not c.isValid():
            return None
        return (c.redF(), c.greenF(), c.blueF(), c.alphaF())
