"""Editor state: the open document, its assets, undo history and selection.

The UI talks to this object; it never mutates the Document itself (§4.2).
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoStack

from lookbox.branding import APP_NAME
from lookbox.commands import edits
from lookbox.commands.qt import EditCommand
from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.model import Document, ImageLayer, Layer, LayerMask, Size, Transform


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
        self.generation = 0  # bumps whenever a different document is opened
        self.revision = 0  # bumps on every edit, undo and redo

    # ---- edits ----
    def push(self, edit: edits.Edit) -> None:
        self.stack.push(EditCommand(self.doc, edit, self._after_change))

    def _after_change(self) -> None:
        self.revision += 1
        if self.selected is not None and not self.doc.has_layer(self.selected):
            self.selected = None
            self.selection_changed.emit()
        self.changed.emit()

    # ---- selection (not part of undo, like Canva) ----
    def select(self, layer_id: str | None) -> None:
        if layer_id != self.selected:
            self.selected = layer_id
            self.selection_changed.emit()

    def selected_layer(self) -> Layer | None:
        if self.selected is None or not self.doc.has_layer(self.selected):
            return None
        return self.doc.layer(self.selected)

    # ---- import (files are decoded off-thread by ui/jobs.ImportJob) ----
    def add_imported(self, store: AssetStore, items: list,
                     at: tuple[float, float] | None = None) -> bool:
        """Add decoded images (render_sets.ImportItem) as layers on top, with their passes
        and alpha-pass cut-out. False if the document changed meanwhile."""
        if store is not self.store or not items:
            return False
        cw, ch = self.doc.canvas.w, self.doc.canvas.h
        bx, by = at if at is not None else (cw / 2.0, ch / 2.0)
        added: list[str] = []
        self.stack.beginMacro("Import image" if len(items) == 1 else f"Import {len(items)} images")
        try:
            for i, item in enumerate(items):
                info = item.info
                # Fit inside the canvas only if it's bigger; same-size renders land 1:1.
                s = min(1.0, cw / info.width, ch / info.height)
                layer = ImageLayer(
                    name=os.path.splitext(info.name)[0] or "Image",
                    source=info.id,
                    transform=Transform(x=bx + 20 * i, y=by + 20 * i, scale_x=s, scale_y=s),
                    passes={k: p.id for k, p in item.passes.items()},
                    mask=LayerMask(asset=item.mask.id) if item.mask is not None else None,
                )
                self.push(edits.AddLayer(layer, asset=info, text="Import image", extra_assets=item.extra_assets()))
                added.append(layer.id)
        finally:
            self.stack.endMacro()
        self.select(added[-1])
        return True

    # ---- documents ----
    def _replace(self, doc: Document, store: AssetStore, path: str | None) -> None:
        self.doc, self.store, self.path, self.selected = doc, store, path, None
        self.generation += 1
        self.stack.clear()
        self.stack.setClean()
        self.document_replaced.emit()
        self.selection_changed.emit()
        self.path_changed.emit()

    def new_document(self, w: int, h: int, background: tuple[float, float, float, float] | None) -> None:
        self._replace(Document(canvas=Size(w=w, h=h), background=background), AssetStore(), None)

    def adopt(self, doc: Document, store: AssetStore, path: str) -> None:
        """Take over a document loaded off-thread (ui/jobs.OpenJob)."""
        self._replace(doc, store, path)

    def save(self, path: str) -> None:
        """Blocking save, for the 'save before closing?' prompt. Raises ProjectError / OSError."""
        serialize.save(path, self.doc, self.store)
        self.finish_save(path, self.save_token())

    def save_token(self) -> tuple[int, int]:
        """Identifies the exact state being saved, so a background save only marks *that*
        state clean. (Not the undo index: undo + a new edit can land on the same index.)"""
        return self.generation, self.revision

    def finish_save(self, path: str, token: tuple[int, int]) -> None:
        generation, revision = token
        if generation != self.generation:
            return  # a different document is open now
        self.path = path
        if self.revision == revision:
            self.stack.setClean()  # anything changed during the save keeps the doc dirty
        self.path_changed.emit()

    def is_dirty(self) -> bool:
        return not self.stack.isClean()

    def title(self) -> str:
        name = os.path.basename(self.path) if self.path else "Untitled"
        return f"{name}{'*' if self.is_dirty() else ''} — {APP_NAME}"
