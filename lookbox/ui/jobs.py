"""Background jobs (§4.5: nothing blocks the UI thread).

Each job snapshots what it needs, runs in its own QThread, and reports back
with one signal. MainWindow connects those signals to its own slots, so the
handlers run on the UI thread.
"""

from __future__ import annotations

import copy
import threading

from PySide6.QtCore import QThread, Signal

from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.io.images import ImageError, atomic_write, save_png
from lookbox.core.model import Document
from lookbox.core.render.pipeline import Cancelled, render


class ImportJob(QThread):
    """Decodes image files into `store`. Emits (store, infos, errors, at)."""

    done = Signal(object, list, list, object)

    def __init__(self, store: AssetStore, paths: list[str], at, parent=None) -> None:
        super().__init__(parent)
        self.store, self.paths, self.at = store, list(paths), at

    def run(self) -> None:
        infos, errors = [], []
        for path in self.paths:
            try:
                infos.append(self.store.add_file(path))
            except ImageError as exc:
                errors.append(str(exc))
            except MemoryError:
                errors.append(f"{path}: not enough memory to open this image.")
        self.done.emit(self.store, infos, errors, self.at)


class OpenJob(QThread):
    """Loads a .lookbox. Emits (path, doc, store, error)."""

    done = Signal(str, object, object, str)

    def __init__(self, path: str, parent=None) -> None:
        super().__init__(parent)
        self.path = path

    def run(self) -> None:
        try:
            doc, store = serialize.load(self.path)
        except serialize.ProjectError as exc:
            self.done.emit(self.path, None, None, str(exc))
        except MemoryError:
            self.done.emit(self.path, None, None, "Not enough memory to open this project.")
        else:
            self.done.emit(self.path, doc, store, "")


class SaveJob(QThread):
    """Writes a .lookbox from a snapshot. Emits (path, token, error)."""

    done = Signal(str, object, str)

    def __init__(self, doc: Document, store: AssetStore, path: str, token, parent=None) -> None:
        super().__init__(parent)
        self.doc = copy.deepcopy(doc)  # edits during the save can't leak into the file
        self.store, self.path, self.token = store, path, token

    def run(self) -> None:
        try:
            atomic_write(self.path, serialize.to_bytes(self.doc, self.store))
        except (serialize.ProjectError, OSError) as exc:
            self.done.emit(self.path, self.token, str(exc))
        else:
            self.done.emit(self.path, self.token, "")


class ExportJob(QThread):
    """Full-resolution float render → PNG. Cancellable; reports progress 0–100."""

    progress = Signal(int)
    done = Signal(str, str)  # path, error ("" = success, "cancelled" = user cancelled)

    def __init__(self, doc: Document, store: AssetStore, path: str, bits: int = 8, scale: float = 1.0,
                 parent=None) -> None:
        super().__init__(parent)
        self.doc = copy.deepcopy(doc)  # snapshot: edits during export can't affect it
        self.store = store  # immutable assets, thread-safe access
        self.path, self.bits, self.scale = path, bits, scale
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            px = render(self.doc, self.store, self.scale, self.cancel_event,
                        progress=lambda f: self.progress.emit(int(f * 90)))
            if self.cancel_event.is_set():
                raise Cancelled()
            save_png(self.path, px, self.bits)
            self.progress.emit(100)
        except Cancelled:
            self.done.emit(self.path, "cancelled")
        except MemoryError:
            self.done.emit(self.path, "Not enough memory to export at this size.")
        except Exception as exc:  # surfaced to the user, never swallowed (§16.5)
            self.done.emit(self.path, f"Export failed: {exc}")
        else:
            self.done.emit(self.path, "")
