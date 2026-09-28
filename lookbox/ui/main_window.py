"""Main window: top bar, tool rail, canvas, context panel (§12)."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QDockWidget, QLabel, QMainWindow, QSizePolicy, QTabWidget, QToolBar,
                               QToolButton, QWidget)

from lookbox.commands import edits
from lookbox.core.model import FillLayer, Transform
from lookbox.ui.canvas.view import CanvasView
from lookbox.ui.documents import DocumentActions
from lookbox.ui.editor import Editor
from lookbox.ui.panels.adjust import AdjustPanel
from lookbox.ui.panels.layer_style import LayerPanel
from lookbox.ui.panels.layers import LayersPanel
from lookbox.ui.pixmaps import ThumbCache
from lookbox.ui.render_service import RenderService


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.editor = Editor(self)
        self.thumbs = ThumbCache()
        self.renderer = RenderService(self)
        self.docs = DocumentActions(self, self.editor)

        self.canvas = CanvasView(self.editor, self.renderer)
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
        self.canvas.files_dropped.connect(self.docs.files_dropped)
        self.canvas.frame_stats.connect(lambda text: self.statusBar().showMessage(text, 8000))
        self.docs.status.connect(self._show_status)
        self.docs.busy_changed.connect(self._update_actions)
        self.renderer.failed.connect(lambda msg: self.statusBar().showMessage(msg, 8000))
        self.editor.document_replaced.connect(self.thumbs.clear)

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
        d = self.docs
        self.act_new = self._action("New", d.new_document, S.New)
        self.act_open = self._action("Open", d.open_project, S.Open)
        self.act_save = self._action("Save", lambda: d.save_project(), S.Save)
        self.act_save_as = self._action("Save as", lambda: d.save_project_as(), "Ctrl+Shift+S")
        self.act_import = self._action("Import image", d.import_images, "Ctrl+I")
        self.act_export = self._action("Export", d.export_png, "Ctrl+E", "Export PNG (Ctrl+E)")
        stack = self.editor.stack
        self.act_undo = self._action("Undo", stack.undo, "Ctrl+Z")
        self.act_redo = self._action("Redo", stack.redo, ["Ctrl+Y", "Ctrl+Shift+Z"])
        stack.canUndoChanged.connect(self.act_undo.setEnabled)
        stack.canRedoChanged.connect(self.act_redo.setEnabled)
        stack.undoTextChanged.connect(lambda t: self.act_undo.setToolTip(f"Undo {t} (Ctrl+Z)".replace("  ", " ")))
        stack.redoTextChanged.connect(lambda t: self.act_redo.setToolTip(f"Redo {t} (Ctrl+Y)".replace("  ", " ")))
        self.act_undo.setEnabled(False)
        self.act_redo.setEnabled(False)
        self.act_fill = self._action("Backdrop", self.add_backdrop, "Ctrl+B",
                                     "Add a gradient backdrop behind everything (Ctrl+B)")
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
        spacer.setObjectName("barSpacer")
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
        rail.addAction(self.act_fill)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, rail)

    def _build_panel(self) -> None:
        tabs = QTabWidget()
        tabs.setObjectName("contextPanel")
        self.adjust_panel = AdjustPanel(self.editor)
        self.adjust_panel.interactive.connect(self.canvas.layers.set_interactive)
        self.adjust_panel.compare.connect(self.canvas.layers.set_bypass)
        self.layer_panel = LayerPanel(self.editor)
        tabs.addTab(self.adjust_panel, "Adjust")
        self.layer_panel.interactive.connect(self.canvas.layers.set_interactive)
        tabs.addTab(self.layer_panel, "Style")
        tabs.addTab(LayersPanel(self.editor, self.thumbs), "Layers")
        self.tabs = tabs
        self.editor.selection_changed.connect(self._follow_selection)
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
        self.act_export.setEnabled(self.docs.export_job is None)

    def _show_status(self, text: str, timeout: int) -> None:
        if text:
            self.statusBar().showMessage(text, timeout)
        else:
            self.statusBar().clearMessage()

    # ------------------------------------------------------------ entry points used by app.py
    def open_path(self, path: str) -> None:
        self.docs.open_path(path)

    def import_paths(self, paths: list[str], at) -> None:
        self.docs.import_paths(paths, at)

    def _follow_selection(self) -> None:
        """Selecting a backdrop while on Adjust (images only) jumps to the Style tab."""
        if isinstance(self.editor.selected_layer(), FillLayer) and self.tabs.currentWidget() is self.adjust_panel:
            self.tabs.setCurrentWidget(self.layer_panel)

    # ------------------------------------------------------------ layer actions
    def add_backdrop(self) -> None:
        c = self.editor.doc.canvas
        layer = FillLayer(name="Backdrop", width=c.w, height=c.h, transform=Transform(x=c.w / 2.0, y=c.h / 2.0))
        self.editor.push(edits.AddLayer(layer, index=0, text="Add backdrop"))
        self.editor.select(layer.id)

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
        if not self.docs.confirm_discard():
            e.ignore()
            return
        self.docs.wait_all()  # an in-flight save must land before we exit
        self.renderer.shutdown()
        e.accept()
