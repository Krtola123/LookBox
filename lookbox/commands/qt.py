"""QUndoStack adapter for the plain edits in `edits.py` (ARCHITECTURE §3, §4.2)."""

from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QUndoCommand

from lookbox.commands.edits import Edit
from lookbox.core.model import Document, Project

_merge_ids: dict[str, int] = {}


def _merge_id(key: str | None) -> int:
    if key is None:
        return -1
    return _merge_ids.setdefault(key, len(_merge_ids) + 1)


class EditCommand(QUndoCommand):
    """Wraps an Edit and what it applies to: a page (Document) or, for page-list edits,
    the Project. `on_change(target)` runs after every redo/undo so the UI can resync
    (and show the page that changed)."""

    def __init__(self, target: Document | Project, edit: Edit, on_change: Callable[[object], None]) -> None:
        super().__init__(edit.text)
        self.target = target
        self.edit = edit
        self.on_change = on_change

    def redo(self) -> None:  # also called once by QUndoStack.push
        self.edit.apply(self.target)
        self.on_change(self.target)

    def undo(self) -> None:
        self.edit.revert(self.target)
        self.on_change(self.target)

    def id(self) -> int:
        return _merge_id(self.edit.merge_key)

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, EditCommand):
            return False
        return self.edit.merge(other.edit)
