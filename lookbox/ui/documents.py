"""File-level actions: new, open, save, import, export — each heavy step in a
background job (ui/jobs.py), with results handled back on the UI thread."""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QProgressDialog

from lookbox.core import serialize
from lookbox.core.io.images import SUPPORTED_EXTS
from lookbox.ui.dialogs import NewDocumentDialog
from lookbox.ui.editor import Editor
from lookbox.ui.jobs import ExportJob, ImportJob, OpenJob, SaveJob

IMAGE_FILTER = "Images (" + " ".join(f"*{e}" for e in SUPPORTED_EXTS) + ")"
PROJECT_FILTER = "RRIPP project (" + " ".join(f"*{e}" for e in serialize.OPEN_EXTENSIONS) + ")"


class DocumentActions(QObject):
    status = Signal(str, int)  # message, timeout ms (0 = until replaced)
    busy_changed = Signal()

    def __init__(self, window: QMainWindow, editor: Editor) -> None:
        super().__init__(window)
        self.window, self.editor = window, editor
        self.last_dir = os.path.expanduser("~")
        self._jobs: set = set()  # keep running jobs referenced
        self.export_job: ExportJob | None = None
        self.save_job: SaveJob | None = None
        self.open_job: OpenJob | None = None
        self.before_discard = None  # callable run before asking to save (set by MainWindow)

    # ------------------------------------------------------------ helpers
    def _error(self, title: str, message: str) -> None:
        QMessageBox.warning(self.window, title, message)

    def _start(self, job) -> None:
        self._jobs.add(job)
        job.start()
        self.busy_changed.emit()

    def _finish(self, job) -> None:
        """Called from each job's done-slot (UI thread): reap the thread safely."""
        if job is None:
            return
        job.wait()  # run() has returned; this only waits for the thread to exit
        self._jobs.discard(job)
        job.deleteLater()

    def wait_all(self) -> None:
        if self.export_job is not None:
            self.export_job.cancel()
        for job in list(self._jobs):
            job.wait()

    def confirm_discard(self) -> bool:
        """True if it's OK to throw away the current document."""
        if self.before_discard is not None:
            self.before_discard()  # e.g. apply an open brush session so the prompt covers it
        if not self.editor.is_dirty():
            return True
        r = QMessageBox.question(
            self.window, "Unsaved changes", "Save changes to this design first?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Save:
            return self.save_project(blocking=True)
        return r == QMessageBox.StandardButton.Discard

    # ------------------------------------------------------------ new / open
    def new_document(self) -> None:
        if not self.confirm_discard():
            return
        dlg = NewDocumentDialog(self.window)
        if dlg.exec():
            w, h, bg = dlg.result_values()
            self.editor.new_document(w, h, bg)

    def open_project(self) -> None:
        if not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self.window, "Open project", self.last_dir, PROJECT_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        if self.open_job is not None:
            return
        self.open_job = OpenJob(path)
        self.open_job.done.connect(self._on_opened)
        self.status.emit(f"Opening {os.path.basename(path)}…", 0)
        self._start(self.open_job)

    @Slot(str, object, object, str)
    def _on_opened(self, path: str, doc, store, error: str) -> None:
        self._finish(self.open_job)
        self.open_job = None
        self.busy_changed.emit()
        if error:
            self.status.emit("", 1)
            self._error("Couldn't open project", error)
            return
        self.editor.adopt(doc, store, path)
        self.last_dir = os.path.dirname(path)
        self.status.emit(f"Opened {os.path.basename(path)}", 3000)

    # ------------------------------------------------------------ save
    def save_project(self, blocking: bool = False) -> bool:
        if self.editor.path is None:
            return self.save_project_as(blocking)
        return self._save_to(self.editor.path, blocking)

    def save_project_as(self, blocking: bool = False) -> bool:
        start = self.editor.path or os.path.join(self.last_dir, "Untitled" + serialize.EXTENSION)
        path, _ = QFileDialog.getSaveFileName(self.window, "Save project", start, PROJECT_FILTER)
        if not path:
            return False
        if not serialize.is_project(path):
            path += serialize.EXTENSION
        return self._save_to(path, blocking)

    def _save_to(self, path: str, blocking: bool) -> bool:
        self.last_dir = os.path.dirname(path)
        if blocking:  # the "save before closing?" path must finish before we continue
            if self.save_job is not None:
                self.save_job.wait()
            self.window.setCursor(Qt.CursorShape.WaitCursor)
            try:
                self.editor.save(path)
            except (serialize.ProjectError, OSError) as exc:
                self._error("Couldn't save", str(exc))
                return False
            finally:
                self.window.unsetCursor()
            self.status.emit(f"Saved {os.path.basename(path)}", 3000)
            return True
        if self.save_job is not None:
            self.status.emit("Still saving the previous version…", 3000)
            return False
        self.save_job = SaveJob(self.editor.doc, self.editor.store, path, self.editor.save_token())
        self.save_job.done.connect(self._on_saved)
        self.status.emit(f"Saving {os.path.basename(path)}…", 0)
        self._start(self.save_job)
        return True

    @Slot(str, object, str)
    def _on_saved(self, path: str, token, error: str) -> None:
        self._finish(self.save_job)
        self.save_job = None
        self.busy_changed.emit()
        if error:
            self.status.emit("", 1)
            self._error("Couldn't save", error)
            return
        self.editor.finish_save(path, token)
        self.status.emit(f"Saved {os.path.basename(path)}", 3000)

    # ------------------------------------------------------------ import
    def import_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self.window, "Import images", self.last_dir, IMAGE_FILTER)
        if paths:
            self.last_dir = os.path.dirname(paths[0])
            self.import_paths(paths, None)

    def import_paths(self, paths: list[str], at) -> None:
        job = ImportJob(self.editor.store, paths, at)
        job.done.connect(self._on_imported)
        n = len(paths)
        self.status.emit(f"Importing {n} image{'s' if n != 1 else ''}…", 0)
        self._start(job)

    @Slot(object, list, list, list, object)
    def _on_imported(self, store, items: list, errors: list, notes: list, at) -> None:
        self._finish(self.sender())
        added = self.editor.add_imported(store, items, at)
        if not added and items:
            self.status.emit("Import skipped: a different design was opened.", 4000)
        else:
            self.status.emit("Render passes found: " + "; ".join(notes) if notes else "", 8000)
        if errors:
            self._error("Some images couldn't be imported", "\n".join(errors))

    def files_dropped(self, paths: list[str], at) -> None:
        projects = [p for p in paths if serialize.is_project(p)]
        if projects:
            if self.confirm_discard():
                self.open_path(projects[0])
            return
        self.import_paths(paths, at)

    # ------------------------------------------------------------ export
    def export_png(self) -> None:
        if self.export_job is not None:
            return
        base = os.path.splitext(os.path.basename(self.editor.path or "Untitled"))[0]
        path, _ = QFileDialog.getSaveFileName(self.window, "Export PNG",
                                              os.path.join(self.last_dir, base + ".png"), "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        job = ExportJob(self.editor.doc, self.editor.store, path)
        progress = QProgressDialog("Exporting…", "Cancel", 0, 100, self.window)
        progress.setWindowTitle("Export")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(400)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.canceled.connect(job.cancel)
        job.progress.connect(progress.setValue)
        job.done.connect(self._on_exported)
        self.export_job, self._export_progress = job, progress
        self._start(job)

    @Slot(str, str)
    def _on_exported(self, path: str, error: str) -> None:
        self._finish(self.export_job)
        self._export_progress.close()
        self._export_progress.deleteLater()
        self.export_job = None
        self.busy_changed.emit()
        if error == "cancelled":
            self.status.emit("Export cancelled.", 3000)
        elif error:
            self._error("Export failed", error)
        else:
            self.last_dir = os.path.dirname(path)
            self.status.emit(f"Exported {os.path.basename(path)}", 4000)
