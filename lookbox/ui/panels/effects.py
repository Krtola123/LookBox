"""Style tab → Effects (ARCHITECTURE §8): shadow (with a one-click contact
shadow), glow, outline, layer blur. Sizes are canvas pixels. Every change is a
SetLayerField('effects') edit through the owning panel, so undo and drag-merging
behave like the rest of the tab."""

from __future__ import annotations

import copy

from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from lookbox.core.model import OUTLINE_POSITIONS, DropShadow, Effects, Glow, Outline
from lookbox.ui.widgets.colour_button import ColourButton

PRESETS = {
    "Soft drop": DropShadow(angle_deg=90, distance=24, blur=40, spread=0, squash=0.0, opacity=0.45),
    "Contact shadow": DropShadow(angle_deg=90, distance=4, blur=28, spread=0, squash=0.85, opacity=0.75),
}
PART_CLASS = {"shadow": DropShadow, "glow": Glow, "outline": Outline}
PART_LABEL = {"shadow": "Shadow", "glow": "Glow", "outline": "Outline"}
PERCENT_KEYS = {"opacity", "squash"}


def _title(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("sectionTitle")
    return lab


class EffectsSection(QWidget):
    def __init__(self, panel) -> None:
        super().__init__()
        self.panel = panel
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        col.addWidget(_title("Effects"))
        self.toggles: dict[str, QCheckBox] = {}
        self.boxes: dict[str, QWidget] = {}
        self.rows: dict[tuple[str, str], object] = {}
        self.colours: dict[str, ColourButton] = {}

        # ---- shadow ----
        box = self._part(col, "shadow")
        presets = QHBoxLayout()
        for name, preset in PRESETS.items():
            b = QPushButton(name)
            b.setObjectName("presetButton")
            b.clicked.connect(lambda _=False, p=preset, n=name: self._set_part("shadow", copy.deepcopy(p), n))
            presets.addWidget(b)
        box.layout().addLayout(presets)
        self._row(box, "shadow", "angle_deg", "Direction", -180, 180, 90, "°")
        self._row(box, "shadow", "distance", "Distance", 0, 400, 20, " px")
        self._row(box, "shadow", "blur", "Blur", 0, 400, 30, " px")
        self._row(box, "shadow", "spread", "Spread", 0, 200, 0, " px")
        self._row(box, "shadow", "squash", "Floor squash", 0, 95, 0, "%")
        self._row(box, "shadow", "opacity", "Opacity", 0, 100, 55, "%")
        self._colour(box, "shadow")

        # ---- glow ----
        box = self._part(col, "glow")
        self._row(box, "glow", "blur", "Blur", 0, 400, 40, " px")
        self._row(box, "glow", "spread", "Spread", 0, 200, 0, " px")
        self._row(box, "glow", "opacity", "Opacity", 0, 100, 60, "%")
        self._colour(box, "glow")

        # ---- outline ----
        box = self._part(col, "outline")
        self._row(box, "outline", "width", "Width", 1, 100, 6, " px")
        self._row(box, "outline", "opacity", "Opacity", 0, 100, 100, "%")
        pos_row = QHBoxLayout()
        lab = QLabel("Position")
        lab.setObjectName("sliderLabel")
        self.position = QComboBox()
        for p in OUTLINE_POSITIONS:
            self.position.addItem(p.capitalize(), p)
        self.position.currentIndexChanged.connect(
            lambda _i: self._set("outline", "position", self.position.currentData()))
        pos_row.addWidget(lab)
        pos_row.addStretch(1)
        pos_row.addWidget(self.position)
        box.layout().addLayout(pos_row)
        self._colour(box, "outline")

        # ---- layer blur ----
        self.blur = panel._slider("blur", "Layer blur", 0, 200, 0, " px", col, self._blur)

    # ------------------------------------------------------------ building
    def _part(self, col: QVBoxLayout, part: str) -> QWidget:
        toggle = QCheckBox(PART_LABEL[part])
        toggle.setObjectName("effectToggle")
        toggle.toggled.connect(lambda on, p=part: self._toggle(p, on))
        col.addWidget(toggle)
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 0, 0, 6)
        lay.setSpacing(4)
        col.addWidget(box)
        self.toggles[part], self.boxes[part] = toggle, box
        return box

    def _row(self, box: QWidget, part: str, key: str, label: str, lo: int, hi: int, default: int,
             suffix: str) -> None:
        row = self.panel._slider(key, label, lo, hi, default, suffix, box.layout(),
                                 lambda k, v, p=part: self._set(p, k, v))
        self.rows[(part, key)] = row

    def _colour(self, box: QWidget, part: str) -> None:
        h = QHBoxLayout()
        lab = QLabel("Colour")
        lab.setObjectName("sliderLabel")
        btn = ColourButton()
        btn.clicked.connect(lambda _=False, p=part: self._pick(p))
        h.addWidget(lab)
        h.addStretch(1)
        h.addWidget(btn)
        box.layout().addLayout(h)
        self.colours[part] = btn

    # ------------------------------------------------------------ document → widgets
    def refresh(self, fx: Effects) -> None:
        for part, toggle in self.toggles.items():
            obj = getattr(fx, part)
            toggle.setChecked(obj is not None)
            self.boxes[part].setVisible(obj is not None)
            if obj is None:
                continue
            for (p, key), row in self.rows.items():
                if p == part:
                    v = getattr(obj, key)
                    row.set_value(v * 100 if key in PERCENT_KEYS else v)
            self.colours[part].set_rgba(obj.color)
        if fx.outline is not None:
            self.position.setCurrentIndex(OUTLINE_POSITIONS.index(fx.outline.position))
        self.blur.set_value(fx.blur)

    # ------------------------------------------------------------ widgets → document
    def _current(self) -> Effects | None:
        layer = self.panel._layer()
        return None if layer is None or self.panel._updating else copy.deepcopy(layer.effects)

    def _toggle(self, part: str, on: bool) -> None:
        fx = self._current()
        if fx is None or (getattr(fx, part) is not None) == on:
            return
        setattr(fx, part, PART_CLASS[part]() if on else None)
        self.panel._push("effects", fx, f"{'Add' if on else 'Remove'} {PART_LABEL[part].lower()}")
        self.panel.refresh()

    def _set_part(self, part: str, obj, text: str) -> None:
        fx = self._current()
        if fx is not None:
            setattr(fx, part, obj)
            self.panel._push("effects", fx, text)
            self.panel.refresh()

    def _set(self, part: str, key: str, v) -> None:
        fx = self._current()
        if fx is None or getattr(fx, part) is None:
            return
        value = v / 100.0 if key in PERCENT_KEYS else (v if isinstance(v, str) else float(v))
        setattr(getattr(fx, part), key, value)
        self.panel._push("effects", fx, PART_LABEL[part])

    def _pick(self, part: str) -> None:
        fx = self._current()
        if fx is None or getattr(fx, part) is None:
            return
        rgba = self.colours[part].pick(f"{PART_LABEL[part]} colour")
        if rgba is not None:
            getattr(fx, part).color = rgba
            self.panel._push("effects", fx, f"{PART_LABEL[part]} colour")
            self.panel.refresh()

    def _blur(self, _key: str, v: int) -> None:
        fx = self._current()
        if fx is not None:
            fx.blur = float(v)
            self.panel._push("effects", fx, "Layer blur")
