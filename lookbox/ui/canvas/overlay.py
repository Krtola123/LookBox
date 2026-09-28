"""Canvas overlay painting: the dimmed area outside the design, hover outline,
selection box and handles. Pure drawing, no state."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from lookbox.core.model import Transform
from lookbox.ui.canvas import handles as H

ACCENT = QColor("#8b3dff")
GUIDE = QColor("#ff4fa3")
HOVER = QColor("#4aa3ff")
OUTSIDE_DIM = QColor(17, 18, 20, 200)


def checker() -> QPixmap:
    """The transparency checkerboard tile."""
    pm = QPixmap(16, 16)
    pm.fill(QColor("#e6e6e6"))
    p = QPainter(pm)
    p.fillRect(0, 0, 8, 8, QColor("#c8c8c8"))
    p.fillRect(8, 8, 8, 8, QColor("#c8c8c8"))
    p.end()
    return pm


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


def selection(painter: QPainter, t: Transform, w: int, h: int, zoom: float, locked: bool,
              corners_only: bool = False) -> None:
    quad(painter, t, w, h, ACCENT, 2.0)
    if locked:
        return
    painter.setPen(QPen(ACCENT, 1.5 / zoom))
    painter.setBrush(QColor("white"))
    r = 5.5 / zoom
    for name, p in H.handle_positions(t, w, h, zoom, corners_only).items():
        if name == H.ROTATE:
            painter.drawEllipse(QPointF(p[0], p[1]), r * 1.4, r * 1.4)
            painter.drawArc(QRectF(p[0] - r * 0.7, p[1] - r * 0.7, r * 1.4, r * 1.4), 30 * 16, 270 * 16)
        elif name in H.CORNERS:
            painter.drawEllipse(QPointF(p[0], p[1]), r, r)
        else:
            painter.drawRoundedRect(QRectF(p[0] - r, p[1] - r, 2 * r, 2 * r), r * 0.4, r * 0.4)


def guides(painter: QPainter, lines: list[tuple[str, float]], canvas: QRectF) -> None:
    """Snap guides across the whole canvas: ("v", x) vertical, ("h", y) horizontal."""
    pen = QPen(GUIDE, 1.0)
    pen.setCosmetic(True)
    painter.setPen(pen)
    for kind, v in lines:
        if kind == "v":
            painter.drawLine(QPointF(v, canvas.top()), QPointF(v, canvas.bottom()))
        else:
            painter.drawLine(QPointF(canvas.left(), v), QPointF(canvas.right(), v))


def brush_cursor(painter: QPainter, x: float, y: float, radius: float) -> None:
    """Brush outline, readable on light and dark images (dark ring + white ring)."""
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for colour, width in ((QColor(0, 0, 0, 160), 3.0), (QColor(255, 255, 255, 230), 1.2)):
        pen = QPen(colour, width)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(x, y), radius, radius)


def handle_cursor(t: Transform, w: int, h: int, zoom: float, name: str) -> Qt.CursorShape:
    """Resize cursor pointing the way the handle actually moves (the layer may be rotated)."""
    if name == H.ROTATE:
        return Qt.CursorShape.CrossCursor
    p = H.handle_positions(t, w, h, zoom)[name]
    ang = math.degrees(math.atan2(p[1] - t.y, p[0] - t.x)) % 180.0
    if ang < 22.5 or ang >= 157.5:
        return Qt.CursorShape.SizeHorCursor
    if ang < 67.5:
        return Qt.CursorShape.SizeFDiagCursor
    if ang < 112.5:
        return Qt.CursorShape.SizeVerCursor
    return Qt.CursorShape.SizeBDiagCursor
