"""Line icons drawn in code (no image files to ship or license): 24-unit grid, round
caps, one stroke weight, so the rail reads as one family. Each QIcon has a muted
"off" look, a brighter hover look and a white "on" (checked) look."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap

MUTED = QColor("#a9abb3")
HOVER = QColor("#e6e7ea")
ON = QColor("#ffffff")
STROKE = 1.7


def _select(p: QPainter, c: QColor) -> None:
    path = QPainterPath(QPointF(6.5, 3.5))
    for x, y in ((6.5, 18.5), (10.6, 14.9), (13.4, 20.6), (15.9, 19.4), (13.2, 13.8), (18.6, 13.5)):
        path.lineTo(x, y)
    path.closeSubpath()
    p.drawPath(path)


def _image(p: QPainter, c: QColor) -> None:
    p.drawRoundedRect(QRectF(3.5, 5, 17, 14), 2.5, 2.5)
    path = QPainterPath(QPointF(4, 17.5))
    for x, y in ((9, 12), (12.5, 15.5), (15, 13), (20, 17.5)):
        path.lineTo(x, y)
    p.drawPath(path)
    p.drawEllipse(QPointF(15.5, 9.3), 1.5, 1.5)


def _text(p: QPainter, c: QColor) -> None:
    p.drawLine(QPointF(5.5, 6), QPointF(18.5, 6))
    p.drawLine(QPointF(12, 6), QPointF(12, 19))
    p.drawLine(QPointF(9.5, 19), QPointF(14.5, 19))
    p.drawLine(QPointF(5.5, 6), QPointF(5.5, 8))
    p.drawLine(QPointF(18.5, 6), QPointF(18.5, 8))


def _backdrop(p: QPainter, c: QColor) -> None:
    r = QRectF(3.5, 4.5, 17, 15)
    g = QLinearGradient(r.topLeft(), r.bottomRight())
    top, bottom = QColor(c), QColor(c)
    top.setAlphaF(0.55)
    bottom.setAlphaF(0.0)
    g.setColorAt(0, top)
    g.setColorAt(1, bottom)
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(g)
    p.drawRoundedRect(r, 2.5, 2.5)
    p.restore()
    p.drawRoundedRect(r, 2.5, 2.5)


_DRAW = {"select": _select, "image": _image, "text": _text, "backdrop": _backdrop}


def _pixmap(name: str, colour: QColor, size: int, ratio: float = 2.0) -> QPixmap:
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.scale(size * ratio / 24.0, size * ratio / 24.0)
    pen = QPen(colour, STROKE)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    _DRAW[name](p, colour)
    p.end()
    pm.setDevicePixelRatio(ratio)
    return pm


def icon(name: str, size: int = 24) -> QIcon:
    ic = QIcon()
    ic.addPixmap(_pixmap(name, MUTED, size), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(_pixmap(name, HOVER, size), QIcon.Mode.Active, QIcon.State.Off)
    ic.addPixmap(_pixmap(name, ON, size), QIcon.Mode.Normal, QIcon.State.On)
    ic.addPixmap(_pixmap(name, ON, size), QIcon.Mode.Active, QIcon.State.On)
    return ic
