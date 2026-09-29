"""Text layout and rasterising (ARCHITECTURE §5, §6.1 "text").

Real fonts need a font engine, which for this app means Qt, and the render core
must stay Qt-free. So the split is:

- this module (Qt-free, tested): line breaking, alignment, line spacing, the box
  size, colouring, and where the box sits when the text changes;
- a `TextEngine` plugged in at startup (ui/text_engine.py): it only *measures*
  strings and *draws* them as coverage. Tests plug in a fake one.

Everything is in canvas px at scale 1. Rendering at a level scales the finished
layout, so every level is the same picture at a different resolution.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from lookbox.core.model import TEXT_FIELDS, TextLayer, Transform

MARGIN_EM = 0.12  # room round the text for italic overhang and accents; part of the box
ITALIC_EXTRA_EM = 0.12  # more on the sides for slanted glyphs
EMPTY_WIDTH_EM = 0.5  # an empty text box stays wide enough to click


@dataclass(frozen=True)
class FontMetrics:
    ascent: float  # px above the baseline
    descent: float  # px below the baseline (positive)


@dataclass(frozen=True)
class Line:
    text: str
    x: float  # left edge of the line's advance box, layout px
    baseline: float  # layout px from the top
    advance: float


@dataclass(frozen=True)
class Layout:
    lines: tuple[Line, ...]
    width: int  # the layer box, whole px
    height: int


class TextEngine(Protocol):
    def metrics(self, layer: TextLayer) -> FontMetrics: ...

    def advance(self, layer: TextLayer, s: str) -> float:
        """Width of `s` in px with kerning, WITHOUT letter spacing (the core adds that:
        Qt's own letter spacing isn't reflected in its measurements on every platform).
        When drawing with letter spacing, character i goes at
        advance(text[:i]) + i × letter_spacing."""

    def coverage(self, layer: TextLayer, layout: Layout, w: int, h: int) -> np.ndarray:
        """Draw `layout` scaled to w × h pixels: float32 (h, w) coverage 0–1."""


_engine: TextEngine | None = None
_lock = threading.Lock()
_cache: dict[tuple, Layout] = {}
_CACHE_MAX = 512


def set_engine(engine: TextEngine | None) -> None:
    global _engine
    with _lock:
        _engine = engine
        _cache.clear()


def engine() -> TextEngine:
    if _engine is None:
        raise RuntimeError("No text engine registered (the app registers the Qt one at startup).")
    return _engine


def layout_key(layer: TextLayer) -> tuple:
    return tuple(getattr(layer, f) for f in TEXT_FIELDS)


# ------------------------------------------------------------------ line breaking


def _break_word(word: str, fits) -> list[str]:
    """Split a word too long for the line into pieces that fit (at least 1 char each)."""
    out, cur = [], ""
    for ch in word:
        if cur and not fits(cur + ch):
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def wrap(paragraph: str, width: float | None, measure) -> list[str]:
    """Greedy word wrap. `measure(s)` → px. No width = the paragraph is one line.
    Spaces the line breaks on are dropped; spaces the user typed elsewhere are kept."""
    if width is None or paragraph == "":
        return [paragraph]
    fits = lambda s: measure(s) <= width + 1e-6  # noqa: E731
    lines: list[str] = []
    cur = ""
    for word in paragraph.split(" "):
        cand = word if cur == "" else f"{cur} {word}"
        if fits(cand):
            cur = cand
            continue
        if cur != "":
            lines.append(cur)
        if fits(word):
            cur = word
        else:
            pieces = _break_word(word, fits)
            lines.extend(pieces[:-1])
            cur = pieces[-1] if pieces else ""
    lines.append(cur)
    return lines


def compute_layout(layer: TextLayer, eng: TextEngine) -> Layout:
    m = eng.metrics(layer)
    size = max(1.0, float(layer.font_size))
    spacing = float(layer.letter_spacing)
    measure = lambda s: eng.advance(layer, s) + spacing * len(s)  # noqa: E731
    box = None if layer.box_width is None else max(1.0, float(layer.box_width))
    rows: list[str] = []
    for para in layer.text.split("\n"):
        rows.extend(wrap(para, box, measure))
    adv = [measure(r) if r else 0.0 for r in rows]
    content_w = box if box is not None else max(max(adv, default=0.0), EMPTY_WIDTH_EM * size)
    mx = MARGIN_EM * size + (ITALIC_EXTRA_EM * size if layer.italic else 0.0)
    my = MARGIN_EM * size
    step = (m.ascent + m.descent) * max(0.1, float(layer.line_height))
    lines = []
    for i, (r, a) in enumerate(zip(rows, adv)):
        free = content_w - a
        off = {"center": free / 2.0, "right": free}.get(layer.align, 0.0)
        lines.append(Line(r, mx + off, my + m.ascent + i * step, a))
    height = 2 * my + (len(rows) - 1) * step + m.ascent + m.descent
    return Layout(tuple(lines), max(1, math.ceil(content_w + 2 * mx)), max(1, math.ceil(height)))


def layout(layer: TextLayer) -> Layout:
    """Cached layout (thread-safe: the UI and render threads both ask)."""
    key = layout_key(layer)
    with _lock:
        hit = _cache.get(key)
        eng = _engine
    if hit is not None:
        return hit
    if eng is None:
        engine()  # raises the clear error
    lay = compute_layout(layer, eng)
    with _lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = lay
    return lay


def content_width(layer: TextLayer) -> float:
    """Width of the text itself (the box minus its side margins): where to start
    when the user switches on wrapping."""
    lay = layout(layer)
    return max(1.0, max((ln.advance for ln in lay.lines), default=1.0))


def layer_box(layer: TextLayer) -> tuple[int, int]:
    lay = layout(layer)
    return lay.width, lay.height


def render_text(layer: TextLayer, w: int, h: int) -> np.ndarray:
    """Straight float32 RGBA, w × h: the text in its colour on transparent."""
    cov = engine().coverage(layer, layout(layer), w, h)
    out = np.empty((h, w, 4), np.float32)
    out[:, :, :3] = np.asarray(layer.color[:3], np.float32)
    out[:, :, 3] = np.clip(cov, 0.0, 1.0) * np.float32(layer.color[3])
    return out


# ------------------------------------------------------------------ placement


def anchor_x(align: str) -> float:
    return {"center": 0.5, "right": 1.0}.get(align, 0.0)


def anchored(t: Transform, old_box: tuple[int, int], new_box: tuple[int, int], align: str) -> Transform:
    """Where to put a text box that changed size so it grows like text should: the top
    edge stays put, and so does the left edge (left-aligned), centre, or right edge.
    Works with rotation and flips (it's the same point in the layer's own frame)."""
    from lookbox.core.render.transform import layer_matrix

    ax = anchor_x(align)
    (ow, oh), (nw, nh) = old_box, new_box
    p = layer_matrix(t, ow, oh) @ np.array([ax * ow, 0.0, 1.0])
    at_origin = layer_matrix(replace(t, x=0.0, y=0.0), nw, nh) @ np.array([ax * nw, 0.0, 1.0])
    return replace(t, x=float(p[0] - at_origin[0]), y=float(p[1] - at_origin[1]))


def baked_resize(layer: TextLayer, t: Transform) -> tuple[dict, Transform]:
    """A canvas resize of text becomes a font-size change (so text is re-rendered sharp,
    never upscaled). Returns (new text fields, transform with scale folded in)."""
    s = t.scale_y
    fields = {"font_size": round(max(1.0, layer.font_size * s), 2),
              "letter_spacing": round(layer.letter_spacing * s, 2)}
    if layer.box_width is not None:
        fields["box_width"] = round(max(1.0, layer.box_width * t.scale_x), 2)
    # Uniform drags leave scale 1; an old non-uniform scale keeps its proportions.
    return fields, replace(t, scale_x=t.scale_x / s if layer.box_width is None else 1.0, scale_y=1.0)
