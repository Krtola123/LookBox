"""Main window: top bar, tool rail, canvas, context panel (§12)."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QProgressDialog, QSizePolicy, QTabWidget, QToolBar, QToolButton,
                               QWidget)

from lookbox.commands import edits
from lookbox.core import serialize
from lookbox.core.io.images import SUPPORTED_EXTS
from lookbox.ui.canvas.view import CanvasView
from lookbox.ui.dialogs import NewDocumentDialog
from lookbox.ui.editor import Editor
from lookbox.ui.jobs import ExportJob
from lookbox.ui.panels.layers import LayersPanel
from lookbox.ui.pixmaps import PixmapCache

IMAGE_FILTER = "Images (" + " ".join(f"*{e}" for e in SUPPORTED_EXTS) + ")"
PROJECT_FILTER = f"LookBox project (*{serialize.EXTENSION})"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.editor = Editor(self)
        self.pixmaps = PixmapCache()
        self._export_job: ExportJob | None = None
        self._last_dir = os.path.expanduser("~")

        self.canvas = CanvasView(self.editor, self.pixmaps)
        self.setCentralWidget(self.canvas)
        self._build_actions()
        self._build_top_bar()
        self._build_rail()
        self._build_panel()
        self._build_status()

        self.editor.stack.cleanChanged.connect(self._update_title)
        self.editor.path_changed.connect(self._update_title)
        self.editor.document_replaced.connect(self._update_status)
        self.editor.selection_changed.connect(self._update_actions)
        self.editor.changed.connect(self._update_actions)
        self.canvas.zoom_changed.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self.canvas.files_dropped.connect(self._on_files_dropped)

        self.resize(1400, 860)
        self.editor.new_document(1920, 1080, None)
        self._update_title()
        self._update_actions()

    # ------------------------------------------------------------ building
    def _action(self, text: str, slot, shortcut=None, tip: str | None = None) -> QAction:
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut is not None:
            seqs = shortcut if isinstance(shortcut, list) else [shortcut]
            a.setShortcuts([QKeySequence(s) for s in seqs])
        if tip:
            a.setToolTip(tip)
        self.addAction(a)  # shortcuts work wherever focus is
        return a

    def _build_actions(self) -> None:
        S = QKeySequence.StandardKey
        self.act_new = self._action("New", self.new_document, S.New)
        self.act_open = self._action("Open", self.open_project, S.Open)
        self.act_save = self._action("Save", self.save_project, S.Save)
        self.act_save_as = self._action("Save as", self.save_project_as, "Ctrl+Shift+S")
        self.act_import = self._action("Import image", self.import_images, "Ctrl+I")
        self.act_export = self._action("Export", self.export_png, "Ctrl+E", "Export PNG (Ctrl+E)")
        stack = self.editor.stack
        self.act_undo = self._action("Undo", stack.undo, "Ctrl+Z")
        self.act_redo = self._action("Redo", stack.redo, ["Ctrl+Y", "Ctrl+Shift+Z"])
        stack.canUndoChanged.connect(self.act_undo.setEnabled)
        stack.canRedoChanged.connect(self.act_redo.setEnabled)
        stack.undoTextChanged.connect(lambda t: self.act_undo.setToolTip(f"Undo {t} (Ctrl+Z)".replace("  ", " ")))
        stack.redoTextChanged.connect(lambda t: self.act_redo.setToolTip(f"Redo {t} (Ctrl+Y)".replace("  ", " ")))
        self.act_undo.setEnabled(False)
        self.act_redo.setEnabled(False)
        self.act_duplicate = self._action("Duplicate", self.duplicate_layer, "Ctrl+D")
        self.act_delete = self._action("Delete", self.delete_layer)
        self.act_fit = self._action("Fit", self.canvas.fit, "Ctrl+0", "Fit to screen (Ctrl+0)")
        self.act_100 = self._action("100%", lambda: self.canvas.set_zoom(1.0), "Ctrl+1", "Actual size (Ctrl+1)")
        self._action("Zoom in", lambda: self.canvas.set_zoom(self.canvas.zoom() * 1.25), ["Ctrl+=", "Ctrl++"])
        self._action("Zoom out", lambda: self.canvas.set_zoom(self.canvas.zoom() / 1.25), "Ctrl+-")

    def _build_top_bar(self) -> None:
        bar = QToolBar("Top bar")
        bar.setObjectName("topBar")
        bar.setMovable(False)
        for a in (self.act_new, self.act_open, self.act_save):
            bar.addAction(a)
        bar.addSeparator()
        bar.addAction(self.act_undo)
        bar.addAction(self.act_redo)
        bar.addSeparator()
        bar.addAction(self.act_duplicate)
        bar.addAction(self.act_delete)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        bar.addAction(self.act_fit)
        bar.addAction(self.act_100)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("zoomLabel")
        self.zoom_label.setMinimumWidth(52)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar.addWidget(self.zoom_label)
        bar.addSeparator()
        export_btn = QToolButton()
        export_btn.setDefaultAction(self.act_export)
        export_btn.setObjectName("primaryButton")
        bar.addWidget(export_btn)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)

    def _build_rail(self) -> None:
        rail = QToolBar("Tools")
        rail.setObjectName("toolRail")
        rail.setMovable(False)
        rail.setOrientation(Qt.Orientation.Vertical)
        rail.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        select = QAction("Select", self)
        select.setCheckable(True)
        select.setChecked(True)
        select.setToolTip("Select, move, resize, rotate")
        rail.addAction(select)
        rail.addAction(self.act_import)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, rail)

    def _build_panel(self) -> None:
        tabs = QTabWidget()
        tabs.setObjectName("contextPanel")
        tabs.addTab(LayersPanel(self.editor, self.pixmaps), "Layers")
        tabs.setMinimumWidth(290)
        tabs.setMaximumWidth(360)
        dock = QDockWidget("", self)
        dock.setObjectName("panelDock")
        dock.setTitleBarWidget(QWidget())
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setWidget(tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_status(self) -> None:
        self.size_label = QLabel()
        self.statusBar().addPermanentWidget(self.size_label)

    # ------------------------------------------------------------ state
    def _update_title(self) -> None:
        self.setWindowTitle(self.editor.title())

    def _update_status(self) -> None:
        c = self.editor.doc.canvas
        self.size_label.setText(f"{c.w} × {c.h} px")
        self._update_title()

    def _update_actions(self) -> None:
        layer = self.editor.selected_layer()
        self.act_duplicate.setEnabled(layer is not None)
        self.act_delete.setEnabled(layer is not None)
        self.act_export.setEnabled(self._export_job is None)

    def _error(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _confirm_discard(self) -> bool:
        """True if it's OK to throw away the current document."""
        if not self.editor.is_dirty():
            return True
        r = QMessageBox.question(
            self, "Unsaved changes", "Save changes to this design first?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Save:
            return self.save_project()
        return r == QMessageBox.StandardButton.Discard

    # ------------------------------------------------------------ file actions
    def new_document(self) -> None:
        if not self._confirm_discard():
            return
        dlg = NewDocumentDialog(self)
        if dlg.exec():
            w, h, bg = dlg.result_values()
            self.editor.new_document(w, h, bg)

    def open_path(self, path: str) -> None:
        try:
            self.editor.open(path)
        except serialize.ProjectError as exc:
            self._error("Couldn't open project", str(exc))
            return
        self._last_dir = os.path.dirname(path)

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", self._last_dir, PROJECT_FILTER)
        if path:
            self.open_path(path)

    def save_project(self) -> bool:
        if self.editor.path is None:
            return self.save_project_as()
        return self._save_to(self.editor.path)

    def save_project_as(self) -> bool:
        start = self.editor.path or os.path.join(self._last_dir, "Untitled" + serialize.EXTENSION)
        path, _ = QFileDialog.getSaveFileName(self, "Save project", start, PROJECT_FILTER)
        if not path:
            return False
        if not path.lower().endswith(serialize.EXTENSION):
            path += serialize.EXTENSION
        return self._save_to(path)

    def _save_to(self, path: str) -> bool:
        try:
            self.editor.save(path)
        except (serialize.ProjectError, OSError) as exc:
            self._error("Couldn't save", str(exc))
            return False
        self._last_dir = os.path.dirname(path)
        self.statusBar().showMessage(f"Saved {os.path.basename(path)}", 3000)
        return True

    def import_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Import images", self._last_dir, IMAGE_FILTER)
        if paths:
            self._last_dir = os.path.dirname(paths[0])
            self.import_paths(paths, None)

    def import_paths(self, paths: list[str], at) -> None:
        errors = self.editor.import_files(paths, at)
        if errors:
            self._error("Some images couldn't be imported", "\n".join(errors))

    def _on_files_dropped(self, paths: list[str], at) -> None:
        projects = [p for p in paths if p.lower().endswith(serialize.EXTENSION)]
        if projects:
            if self._confirm_discard():
                self.open_path(projects[0])
            return
        self.import_paths(paths, at)

    def export_png(self) -> None:
        if self._export_job is not None:
            return
        base = os.path.splitext(os.path.basename(self.editor.path or "Untitled"))[0]
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", os.path.join(self._last_dir, base + ".png"),
                                              "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        progress = QProgressDialog(self)
        progress.setWindowTitle("Export")
        progress.setLabelText("Exporting…")
        progress.setCancelButton(None)  # M1 render can't be interrupted mid-way
        progress.setRange(0, 0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(300)
        job = ExportJob(self.editor.doc, self.editor.store, path, parent=self)
        self._export_job = job
        self._export_progress = progress
        self._export_path = path
        self._update_actions()
        # Bound slot on this QObject → queued onto the UI thread, never run on the worker.
        job.finished_with.connect(self._on_export_done)
        job.start()

    @Slot(str)
    def _on_export_done(self, error: str) -> None:
        self._export_progress.close()
        self._export_job.wait()
        self._export_job.deleteLater()
        self._export_job = None
        self._update_actions()
        if error:
            self._error("Export failed", error)
        else:
            self.statusBar().showMessage(f"Exported {os.path.basename(self._export_path)}", 4000)

    # ------------------------------------------------------------ layer actions
    def duplicate_layer(self) -> None:
        layer = self.editor.selected_layer()
        if layer is None:
            return
        edit = edits.duplicate(self.editor.doc, layer.id)
        self.editor.push(edit)
        self.editor.select(edit.layer.id)

    def delete_layer(self) -> None:
        layer = self.editor.selected_layer()
        if layer is not None:
            self.editor.push(edits.RemoveLayer(layer.id))

    # ------------------------------------------------------------ lifecycle
    def showEvent(self, e) -> None:
        super().showEvent(e)
        if not getattr(self, "_fitted", False):
            self._fitted = True
            QTimer.singleShot(0, self.canvas.fit)  # viewport has its real size only now

    def closeEvent(self, e) -> None:
        if self._export_job is not None:
            self._export_job.wait()
        if self._confirm_discard():
            e.accept()
        else:
            e.ignore()
