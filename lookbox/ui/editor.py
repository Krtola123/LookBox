"""Editor state: the open document, its assets, undo history and selection.

The UI talks to this object; it never mutates the Document itself (§4.2).
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoStack

from lookbox.commands import edits
from lookbox.commands.qt import EditCommand
from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.io.images import ImageError
from lookbox.core.model import Document, ImageLayer, Size, Transform


class Editor(QObject):
    changed = Signal()  # document content changed (edit, undo, redo)
    selection_changed = Signal()
    document_replaced = Signal()  # new / open
    path_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.doc = Document()
        self.store = AssetStore()
        self.stack = QUndoStack(self)
        self.path: str | None = None
        self.selected: str | None = None

    # ---- edits ----
    def push(self, edit: edits.Edit) -> None:
        self.stack.push(EditCommand(self.doc, edit, self._after_change))

    def _after_change(self) -> None:
        if self.selected is not None and not self.doc.has_layer(self.selected):
            self.selected = None
            self.selection_changed.emit()
        self.changed.emit()

    # ---- selection (not part of undo, like Canva) ----
    def select(self, layer_id: str | None) -> None:
        if layer_id != self.selected:
            self.selected = layer_id
            self.selection_changed.emit()

    def selected_layer(self) -> ImageLayer | None:
        if self.selected is None or not self.doc.has_layer(self.selected):
            return None
        layer = self.doc.layer(self.selected)
        return layer if isinstance(layer, ImageLayer) else None

    # ---- import ----
    def import_files(self, paths: list[str], at: tuple[float, float] | None = None) -> list[str]:
        """Add each image as a new layer on top. Returns error messages (empty = all fine)."""
        errors: list[str] = []
        infos = []
        for path in paths:
            try:
                infos.append(self.store.add_file(path))
            except ImageError as exc:
                errors.append(str(exc))
        if not infos:
            return errors

        cw, ch = self.doc.canvas.w, self.doc.canvas.h
        bx, by = at if at is not None else (cw / 2.0, ch / 2.0)
        added: list[str] = []
        self.stack.beginMacro("Import image" if len(infos) == 1 else f"Import {len(infos)} images")
        try:
            for i, info in enumerate(infos):
                # Fit inside the canvas only if it's bigger; same-size renders land 1:1.
                s = min(1.0, cw / info.width, ch / info.height)
                layer = ImageLayer(
                    name=os.path.splitext(info.name)[0] or "Image",
                    source=info.id,
                    transform=Transform(x=bx + 20 * i, y=by + 20 * i, scale_x=s, scale_y=s),
                )
                self.push(edits.AddLayer(layer, asset=info, text="Import image"))
                added.append(layer.id)
        finally:
            self.stack.endMacro()
        self.select(added[-1])
        return errors

    # ---- documents ----
    def _replace(self, doc: Document, store: AssetStore, path: str | None) -> None:
        self.doc, self.store, self.path, self.selected = doc, store, path, None
        self.stack.clear()
        self.stack.setClean()
        self.document_replaced.emit()
        self.selection_changed.emit()
        self.path_changed.emit()

    def new_document(self, w: int, h: int, background: tuple[float, float, float, float] | None) -> None:
        self._replace(Document(canvas=Size(w=w, h=h), background=background), AssetStore(), None)

    def open(self, path: str) -> None:
        """Raises serialize.ProjectError with a user-facing message."""
        doc, store = serialize.load(path)
        self._replace(doc, store, path)

    def save(self, path: str) -> None:
        """Raises serialize.ProjectError / OSError."""
        serialize.save(path, self.doc, self.store)
        self.path = path
        self.stack.setClean()
        self.path_changed.emit()

    def is_dirty(self) -> bool:
        return not self.stack.isClean()

    def title(self) -> str:
        name = os.path.basename(self.path) if self.path else "Untitled"
        return f"{name}{'*' if self.is_dirty() else ''} — LookBox"
