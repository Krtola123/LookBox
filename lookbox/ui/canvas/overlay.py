"""Canvas overlay painting: the dimmed area outside the design, hover outline,
selection box and handles. Pure drawing, no state."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF

from lookbox.core.model import Transform
from lookbox.ui.canvas import handles as H

ACCENT = QColor("#8b3dff")
HOVER = QColor("#4aa3ff")
OUTSIDE_DIM = QColor(17, 18, 20, 200)


def dim_outside(painter: QPainter, visible: QRectF, canvas: QRectF) -> None:
    outside = QPainterPath()
    outside.addRect(visible)
    inside = QPainterPath()
    inside.addRect(canvas)
    painter.fillPath(outside.subtracted(inside), OUTSIDE_DIM)


def quad(painter: QPainter, t: Transform, w: int, h: int, color: QColor, width: float) -> None:
    pen = QPen(color, width)
    pen.setCosmetic(True)  # constant screen width at any zoom
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in H.quad(t, w, h)]))


def selection(painter: QPainter, t: Transform, w: int, h: int, zoom: float, locked: bool) -> None:
    quad(painter, t, w, h, ACCENT, 2.0)
    if locked:
        return
    painter.setPen(QPen(ACCENT, 1.5 / zoom))
    painter.setBrush(QColor("white"))
    r = 5.5 / zoom
    for name, p in H.handle_positions(t, w, h, zoom).items():
        if name == H.ROTATE:
            painter.drawEllipse(QPointF(p[0], p[1]), r * 1.4, r * 1.4)
            painter.drawArc(QRectF(p[0] - r * 0.7, p[1] - r * 0.7, r * 1.4, r * 1.4), 30 * 16, 270 * 16)
        elif name in H.CORNERS:
            painter.drawEllipse(QPointF(p[0], p[1]), r, r)
        else:
            painter.drawRoundedRect(QRectF(p[0] - r, p[1] - r, 2 * r, 2 * r), r * 0.4, r * 0.4)
