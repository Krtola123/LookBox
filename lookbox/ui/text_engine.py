"""The Qt text engine (§3: text is drawn by Qt). Plugged into core/render/text.py
at startup; it only measures strings and draws laid-out lines as coverage. Line
breaking, alignment and colour are the core's job (tested without Qt).

Font sizes are fractional pixels: fonts are resolved against 72-dpi images, where
one point is one pixel, so `setPointSizeF(font_size)` gives exactly font_size px.
Hinting is off so text scales linearly (a resize doesn't reflow it).

Runs on render threads too: Qt allows painting QImages off the GUI thread. Painting
is lock-free (typing must not wait behind a big render); the metrics cache is locked
and per-thread, so no QFontMetricsF is shared between threads.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter

from lookbox.core.model import TextLayer
from lookbox.core.render.text import FontMetrics, Layout

_DPM_72DPI = 2835  # dots per metre ≈ 72 dpi: 1 pt = 1 px
_WEIGHTS = (100, 200, 300, 400, 500, 600, 700, 800, 900)


def _image(w: int, h: int) -> QImage:
    img = QImage(max(1, w), max(1, h), QImage.Format.Format_ARGB32_Premultiplied)
    img.setDotsPerMeterX(_DPM_72DPI)
    img.setDotsPerMeterY(_DPM_72DPI)
    img.fill(Qt.GlobalColor.transparent)
    return img


def make_font(layer: TextLayer) -> QFont:
    """The QFont for a layer (also used by the panel's font preview)."""
    f = QFont(layer.font_family)
    f.setPointSizeF(max(1.0, float(layer.font_size)))
    weight = min(_WEIGHTS, key=lambda w: abs(w - int(layer.weight)))
    f.setWeight(QFont.Weight(weight))
    f.setItalic(bool(layer.italic))
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, float(layer.letter_spacing))
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    # Greyscale antialiasing only: we read coverage from alpha, ClearType would fringe it.
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)
    return f


def _font_key(layer: TextLayer) -> tuple:
    return (layer.font_family, float(layer.font_size), int(layer.weight), bool(layer.italic),
            float(layer.letter_spacing))


class QtTextEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()  # per-thread 1×1 72-dpi device for measuring
        self._metrics: dict[tuple, QFontMetricsF] = {}

    def _fm(self, layer: TextLayer) -> QFontMetricsF:
        """Call with the lock held."""
        key = (threading.get_ident(),) + _font_key(layer)
        fm = self._metrics.get(key)
        if fm is None:
            device = getattr(self._local, "device", None)
            if device is None:
                device = self._local.device = _image(1, 1)
            if len(self._metrics) > 128:
                self._metrics.clear()
            fm = self._metrics[key] = QFontMetricsF(make_font(layer), device)
        return fm

    def metrics(self, layer: TextLayer) -> FontMetrics:
        with self._lock:
            fm = self._fm(layer)
            return FontMetrics(ascent=float(fm.ascent()), descent=float(fm.descent()))

    def advance(self, layer: TextLayer, s: str) -> float:
        with self._lock:
            return float(self._fm(layer).horizontalAdvance(s))

    def coverage(self, layer: TextLayer, layout: Layout, w: int, h: int) -> np.ndarray:
        img = _image(w, h)
        if img.isNull():  # Qt couldn't allocate it
            raise MemoryError(f"Not enough memory to draw {w} × {h} px of text.")
        p = QPainter(img)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            p.scale(w / layout.width, h / layout.height)
            p.setFont(QFont(make_font(layer), img))
            p.setPen(QColor(255, 255, 255))
            for line in layout.lines:
                if line.text:
                    p.drawText(QPointF(line.x, line.baseline), line.text)
        finally:
            p.end()
        # BGRA bytes, rows padded to bytesPerLine; alpha (byte 3) is the coverage.
        raw = np.frombuffer(img.constBits(), np.uint8, count=img.bytesPerLine() * h)
        alpha = raw.reshape(h, img.bytesPerLine())[:, : w * 4].reshape(h, w, 4)[:, :, 3]
        return alpha.astype(np.float32) * np.float32(1.0 / 255.0)
