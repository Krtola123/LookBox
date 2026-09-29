"""Editor state: the open project (pages), the page being edited, its assets, undo
history, selection and the layer clipboard.

The UI talks to this object; it never mutates a Document itself (§4.2). `doc` is the
active page, so everything that edits "the design" edits that page. One undo history
covers all pages; undoing a change on another page shows that page.
"""

from __future__ import annotations

import copy
import os

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoStack

from lookbox.branding import APP_NAME
from lookbox.commands import edits
from lookbox.commands.qt import EditCommand
from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.model import Document, ImageLayer, Layer, LayerMask, Project, Size, Transform


class Editor(QObject):
    changed = Signal()  # content changed (edit, undo, redo), any page
    selection_changed = Signal()
    document_replaced = Signal()  # new / open, and switching page (then page_switch is True)
    path_changed = Signal()
    pages_changed = Signal()  # pages added, removed, moved or resized
    about_to_switch = Signal()  # the user is switching page (finish what's open on this one)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.project = Project()
        self.active = self.project.pages[0].id
        self.page_switch = False  # True while document_replaced is emitted for a page switch
        self._last_index = 0  # where the active page was: if it's deleted, its neighbour takes over
        self._clipboard: tuple | None = None  # (layer, assets) copied with Ctrl+C
        self.store = AssetStore()
        self.stack = QUndoStack(self)
        self.path: str | None = None
        self.selected: str | None = None
        self.generation = 0  # bumps whenever a different document is opened
        self.revision = 0  # bumps on every edit, undo and redo

    @property
    def doc(self) -> Document:
        """The page being edited."""
        return self.project.page(self.active)

    # ---- edits ----
    def push(self, edit: edits.Edit) -> None:
        """An edit to the active page."""
        self.stack.push(EditCommand(self.doc, edit, self._after_change))

    def push_pages(self, edit: edits.Edit) -> None:
        """An edit to the page list (add, delete, move)."""
        self.stack.push(EditCommand(self.project, edit, self._after_change))

    def _after_change(self, target=None) -> None:
        self.revision += 1
        if target is self.project:
            if not self.project.has_page(self.active):  # the active page was deleted (or un-added)
                self._set_active(self.project.pages[min(self._last_index, len(self.project.pages) - 1)].id)
            self.pages_changed.emit()
        elif isinstance(target, Document) and target is not self.doc and any(p is target for p in self.project.pages):
            self._set_active(target.id)  # undo/redo on another page: show it
        if self.selected is not None and not self.doc.has_layer(self.selected):
            self.selected = None
            self.selection_changed.emit()
        self._last_index = self.project.page_index(self.active)
        self.changed.emit()

    # ---- pages ----
    def set_active(self, page_id: str) -> None:
        """Edit another page (clicked on the canvas or in the pages bar)."""
        if page_id != self.active and self.project.has_page(page_id):
            self.about_to_switch.emit()
            self._set_active(page_id)

    def _set_active(self, page_id: str) -> None:
        self.active, self.selected = page_id, None
        self._last_index = self.project.page_index(page_id)
        self.page_switch = True
        try:
            self.document_replaced.emit()
        finally:
            self.page_switch = False
        self.selection_changed.emit()

    def page_number(self, page_id: str | None = None) -> int:
        return self.project.page_index(page_id or self.active) + 1

    # ---- clipboard (copy a layer to another page, or the same one) ----
    def copy_selected(self) -> bool:
        layer = self.selected_layer()
        if layer is None:
            return False
        self._clipboard = (copy.deepcopy(layer), edits.layer_assets(self.doc, layer), self.active)
        return True

    def paste(self) -> bool:
        if self._clipboard is None:
            return False
        layer, assets, from_page = self._clipboard
        edit = edits.paste_layer(self.doc, layer, assets, offset=20.0 if from_page == self.active else 0.0)
        self.push(edit)
        self.select(edit.layer.id)
        return True

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
    def _replace(self, project: Project, store: AssetStore, path: str | None) -> None:
        self.project, self.store, self.path, self.selected = project, store, path, None
        self.active, self._last_index = project.pages[0].id, 0
        self.generation += 1
        self.stack.clear()
        self.stack.setClean()
        self.document_replaced.emit()
        self.selection_changed.emit()
        self.path_changed.emit()

    def new_document(self, w: int, h: int, background: tuple[float, float, float, float] | None) -> None:
        self._replace(Project(pages=[Document(canvas=Size(w=w, h=h), background=background)]), AssetStore(), None)

    def adopt(self, project: Project, store: AssetStore, path: str) -> None:
        """Take over a project loaded off-thread (ui/jobs.OpenJob)."""
        self._replace(project, store, path)

    def save(self, path: str) -> None:
        """Blocking save, for the 'save before closing?' prompt. Raises ProjectError / OSError."""
        serialize.save(path, self.project, self.store)
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
