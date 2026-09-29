"""Adjust tab → Filters (ARCHITECTURE §7a, M10): LUT looks as thumbnails, a strength
slider, and "Import .cube…". Works on the selected layer or the whole design; the
owning panel says which (`set_target`).

The library is the bundled looks (lookbox/luts) plus every .cube imported before
(copied to %LOCALAPPDATA%\\RRIPP\\luts, so they're there in every design). Applying
one copies its bytes into the document, so a saved design never depends on files
outside it.
"""

from __future__ import annotations

import glob
import hashlib
import os
import shutil

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QToolButton, QVBoxLayout, QWidget)

from lookbox.branding import data_root
from lookbox.commands import edits
from lookbox.core.model import LutRef
from lookbox.core.render import lut as L
from lookbox.core.render.levels import thumbnail_array
from lookbox.ui.pixmaps import to_qimage
from lookbox.ui.widgets.slider_row import SliderRow

BUNDLED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "luts")
THUMB = 60
COLUMNS = 3


def user_lut_dir() -> str:
    return os.path.join(data_root(), "luts")


class _Entry:
    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, "rb") as fh:
            self.data = fh.read()
        self.id = hashlib.sha256(self.data).hexdigest()  # = the asset id once applied
        self.lut = L.parse_cube(self.data)
        stem = os.path.splitext(os.path.basename(path))[0]
        self.name = self.lut.title or stem.replace("_", " ").capitalize()


def library() -> list[_Entry]:
    """Bundled looks, then imported ones. Unreadable files are skipped."""
    out, seen = [], set()
    for folder in (BUNDLED_DIR, user_lut_dir()):
        for path in sorted(glob.glob(os.path.join(folder, "*.cube"))):
            try:
                e = _Entry(path)
            except (OSError, L.LutError):
                continue
            if e.id not in seen:
                seen.add(e.id)
                out.append(e)
    return out


class FiltersSection(QWidget):
    pressed = Signal()  # strength slider drag starts / ends (the panel switches previews)
    released = Signal()

    def __init__(self, editor) -> None:
        super().__init__()
        self.editor = editor
        self._target: str | None = None  # layer id, or None = the whole design
        self._enabled = False
        self._source: np.ndarray | None = None  # thumbnail-size straight RGBA the looks are shown on
        self._dragging, self._serial = False, 0
        self._entries = library()

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 8)
        col.setSpacing(6)
        head = QHBoxLayout()
        title = QLabel("Filters")
        title.setObjectName("sectionTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.import_btn = QPushButton("Import .cube…")
        self.import_btn.setToolTip("Add a LUT (.cube) from Resolve, Photoshop, a LUT pack…")
        self.import_btn.clicked.connect(self._import)
        head.addWidget(self.import_btn)
        col.addLayout(head)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(4)
        col.addWidget(self.grid_host)
        self.strength = SliderRow("strength", "Strength", minimum=0, maximum=100, default=100, suffix="%")
        self.strength.edited.connect(self._strength)
        self.strength.pressed.connect(self._pressed)
        self.strength.released.connect(self._released)
        col.addWidget(self.strength)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str | None, QToolButton] = {}
        self._build_grid()

    # ------------------------------------------------------------ owner API
    def set_target(self, enabled: bool, layer_id: str | None, source: np.ndarray | None) -> None:
        """What the filters apply to: a layer (its id) or the whole design (None), and
        the picture the thumbnails show (straight RGBA, any size)."""
        self._enabled, self._target = enabled, layer_id
        self._source = thumbnail_array(source, THUMB) if source is not None else None
        self._build_grid()
        self.refresh()

    def refresh(self) -> None:
        current = self._current()
        key = current.asset if current is not None else None
        if key is not None and key not in self.buttons and self.editor.store.has(key):
            self._build_grid()  # a LUT from the project that isn't in the library: show it
        btn = self.buttons.get(key)
        if btn is not None:
            btn.setChecked(True)
        self.strength.setVisible(current is not None)
        if current is not None and not self._dragging:
            self.strength.set_value(current.strength * 100)
        self.setEnabled(self._enabled)

    # ------------------------------------------------------------ grid
    def _current(self) -> LutRef | None:
        doc = self.editor.doc
        if not self._enabled:
            return None
        if self._target is None:
            return doc.global_lut
        return doc.layer(self._target).lut if doc.has_layer(self._target) else None

    def _icon(self, lut: L.Lut | None) -> QIcon:
        if self._source is None:
            pm = QPixmap(THUMB, THUMB)
            pm.fill(Qt.GlobalColor.darkGray)
            return QIcon(pm)
        px = self._source.copy()
        if lut is not None:
            px[..., :3] = L.apply(px[..., :3], lut)
        return QIcon(QPixmap.fromImage(to_qimage(px)))

    def _add_button(self, i: int, key: str | None, name: str, lut: L.Lut | None) -> None:
        b = QToolButton()
        b.setObjectName("filterButton")
        b.setCheckable(True)
        b.setText(name.replace("&", "&&"))  # "&" would be taken as a keyboard shortcut marker
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        b.setIcon(self._icon(lut))
        b.setIconSize(QSize(THUMB, THUMB))
        b.setToolTip(name)
        b.clicked.connect(lambda _=False, k=key: self._choose(k))
        self.group.addButton(b)
        self.grid.addWidget(b, i // COLUMNS, i % COLUMNS)
        self.buttons[key] = b

    def _build_grid(self) -> None:
        for b in list(self.buttons.values()):
            self.group.removeButton(b)
            self.grid.removeWidget(b)
            b.setParent(None)
            b.deleteLater()
        self.buttons.clear()
        self._add_button(0, None, "None", None)
        i = 1
        for e in self._entries:
            self._add_button(i, e.id, e.name, e.lut)
            i += 1
        current = self._current()
        if current is not None and current.asset not in self.buttons and self.editor.store.has(current.asset):
            self._add_button(i, current.asset, current.name or "From design", self.editor.store.lut(current.asset))

    # ------------------------------------------------------------ actions
    def _push(self, new: LutRef | None, asset=None, text: str = "Filter", merge_key: str | None = None) -> None:
        old = self._current()
        if not self._enabled or new == old:
            return
        self.editor.push(edits.SetLut(self._target, old, new, asset=asset, text=text, merge_key=merge_key))

    def _choose(self, key: str | None) -> None:
        if key is None:
            self._push(None, text="No filter")
            return
        current = self._current()
        strength = current.strength if current is not None else 1.0
        entry = next((e for e in self._entries if e.id == key), None)
        if entry is not None:
            info = self.editor.store.add_bytes(entry.data, ".cube", entry.name)
            self._push(LutRef(asset=info.id, strength=strength, name=entry.name), asset=info,
                       text=f"Filter: {entry.name}")
        elif current is not None:  # the project's own LUT, already in the document
            self._push(LutRef(asset=key, strength=strength, name=current.name), text="Filter")

    def _strength(self, v: int) -> None:
        current = self._current()
        if current is None:
            return
        key = f"lut-drag-{self._serial}" if self._dragging else None
        self._push(LutRef(asset=current.asset, strength=v / 100.0, name=current.name), text="Filter strength",
                   merge_key=key)

    def _pressed(self) -> None:
        self._dragging = True
        self._serial += 1
        self.pressed.emit()

    def _released(self) -> None:
        self._dragging = False
        self.released.emit()
        self.refresh()

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import LUT", "", "LUT (*.cube)")
        if not path:
            return
        try:
            entry = _Entry(path)
        except (OSError, L.LutError) as exc:
            QMessageBox.warning(self, "Couldn't use this LUT", str(exc))
            return
        os.makedirs(user_lut_dir(), exist_ok=True)
        dest = os.path.join(user_lut_dir(), os.path.basename(path))
        if os.path.exists(dest) and _Entry(dest).id != entry.id:
            stem, ext = os.path.splitext(dest)
            dest = f"{stem}-{entry.id[:6]}{ext}"
        if not os.path.exists(dest):
            shutil.copyfile(path, dest)  # kept for every design from now on
        self._entries = library()
        self._build_grid()
        if self._enabled:
            self._choose(entry.id)
        self.refresh()
