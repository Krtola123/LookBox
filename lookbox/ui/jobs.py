"""Background jobs (§4.5: nothing blocks the UI thread)."""

from __future__ import annotations

import copy

from PySide6.QtCore import QThread, Signal

from lookbox.core.assets import AssetStore
from lookbox.core.io.images import save_png
from lookbox.core.model import Document
from lookbox.core.render.pipeline import render


class ExportJob(QThread):
    """Renders at full resolution and writes a PNG. Emits finished_with("") on
    success or a user-facing error message."""

    finished_with = Signal(str)

    def __init__(self, doc: Document, store: AssetStore, path: str, bits: int = 8, scale: float = 1.0,
                 parent=None) -> None:
        super().__init__(parent)
        self.doc = copy.deepcopy(doc)  # snapshot: edits during export can't affect it
        self.store = store  # immutable assets, thread-safe access
        self.path, self.bits, self.scale = path, bits, scale

    def run(self) -> None:
        try:
            save_png(self.path, render(self.doc, self.store, self.scale), self.bits)
        except MemoryError:
            self.finished_with.emit("Not enough memory to export at this size.")
        except Exception as exc:  # surfaced to the user, never swallowed (§16.5)
            self.finished_with.emit(f"Export failed: {exc}")
        else:
            self.finished_with.emit("")
