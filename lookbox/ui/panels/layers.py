"""Layers list (Canva's Position → Layers): thumbnails, drag to reorder,
eye and lock toggles, double-click to rename. Top of the list = top layer.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QListWidgetItem, QToolButton, QVBoxLayout, QWidget)

from lookbox.commands import edits
from lookbox.core.model import ImageLayer
from lookbox.core.render.pipeline import layer_source
from lookbox.ui.editor import Editor
from lookbox.ui.pixmaps import ThumbCache

THUMB = 44
ID_ROLE = Qt.ItemDataRole.UserRole


class _List(QListWidget):
    reordered = Signal()

    def dropEvent(self, e) -> None:
        super().dropEvent(e)
        self.reordered.emit()


class _Row(QWidget):
    def __init__(self, panel: "LayersPanel", layer: ImageLayer) -> None:
        super().__init__()
        self.setObjectName("layerRow")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)

        thumb = QLabel()
        thumb.setFixedSize(THUMB, THUMB)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setObjectName("layerThumb")
        px = layer_source(layer, panel.editor.store)
        thumb.setPixmap(panel.thumbs.get((layer.source, layer.crop), px, THUMB - 4))
        lay.addWidget(thumb)

        name = QLabel(layer.name)
        name.setObjectName("layerName")
        name.setEnabled(layer.visible)
        lay.addWidget(name, 1)

        eye = QToolButton()
        eye.setObjectName("rowToggle")
        eye.setText("👁" if layer.visible else "—")
        eye.setToolTip("Hide layer" if layer.visible else "Show layer")
        eye.clicked.connect(lambda: panel.later(edits.SetLayerProps(
            layer.id, text="Show layer" if not layer.visible else "Hide layer", visible=not layer.visible)))
        lay.addWidget(eye)

        lock = QToolButton()
        lock.setObjectName("rowToggle")
        lock.setText("🔒" if layer.locked else "🔓")
        lock.setToolTip("Unlock layer" if layer.locked else "Lock layer")
        lock.clicked.connect(lambda: panel.later(edits.SetLayerProps(
            layer.id, text="Unlock layer" if layer.locked else "Lock layer", locked=not layer.locked)))
        lay.addWidget(lock)


class LayersPanel(QWidget):
    def __init__(self, editor: Editor, thumbs: ThumbCache, parent=None) -> None:
        super().__init__(parent)
        self.editor = editor
        self.thumbs = thumbs
        self._syncing = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.list = _List()
        self.list.setObjectName("layerList")
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setSpacing(3)
        lay.addWidget(self.list)

        self.empty = QLabel("No layers yet.\nImport an image or drop one on the canvas.")
        self.empty.setObjectName("emptyHint")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        lay.addWidget(self.empty)

        self.list.currentItemChanged.connect(self._on_current)
        self.list.itemDoubleClicked.connect(self._rename)
        # Rebuild after Qt finishes its own drop handling, not inside it.
        self.list.reordered.connect(lambda: QTimer.singleShot(0, self._on_reordered))
        self._signature: tuple = ()
        editor.changed.connect(self._maybe_rebuild)
        editor.document_replaced.connect(self.rebuild)
        editor.selection_changed.connect(self._sync_selection)
        self.rebuild()

    def later(self, edit: edits.Edit) -> None:
        """Push after the current event returns: the push rebuilds rows, and a
        button must not be deleted while its own click is being handled."""
        QTimer.singleShot(0, lambda: self.editor.push(edit))

    def _ids_bottom_to_top(self) -> list[str]:
        return [self.list.item(i).data(ID_ROLE) for i in reversed(range(self.list.count()))]

    def _row_signature(self) -> tuple:
        """What the rows display. Transform/adjust edits don't change it, so a slider
        drag or a canvas drag doesn't rebuild the list 30 times a second."""
        return tuple((layer.id, layer.name, layer.visible, layer.locked, layer.source,
                      getattr(layer, "crop", None)) for layer in self.editor.doc.layers)

    def _maybe_rebuild(self) -> None:
        if self._row_signature() != self._signature:
            self.rebuild()

    def rebuild(self) -> None:
        self._signature = self._row_signature()
        self._syncing = True
        try:
            self.list.clear()
            for layer in reversed(self.editor.doc.layers):
                if not isinstance(layer, ImageLayer):
                    continue
                item = QListWidgetItem()
                item.setData(ID_ROLE, layer.id)
                item.setSizeHint(QSize(0, THUMB + 10))
                flags = item.flags() | Qt.ItemFlag.ItemIsDragEnabled
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsDropEnabled)
                self.list.addItem(item)
                self.list.setItemWidget(item, _Row(self, layer))
            self.empty.setVisible(self.list.count() == 0)
            self.list.setVisible(self.list.count() > 0)
        finally:
            self._syncing = False
        self._sync_selection()

    def _sync_selection(self) -> None:
        self._syncing = True
        try:
            target = self.editor.selected
            for i in range(self.list.count()):
                if self.list.item(i).data(ID_ROLE) == target:
                    self.list.setCurrentRow(i)
                    break
            else:
                self.list.setCurrentRow(-1)
                self.list.clearSelection()
        finally:
            self._syncing = False

    def _on_current(self, cur: QListWidgetItem | None, _prev) -> None:
        if not self._syncing:
            self.editor.select(cur.data(ID_ROLE) if cur is not None else None)

    def _on_reordered(self) -> None:
        old = [layer.id for layer in self.editor.doc.layers]
        edit = edits.reorder_from(old, self._ids_bottom_to_top())
        if edit is not None:
            self.editor.push(edit)
        # Always rebuild: Qt drops the moved row's widget during a drag-and-drop.
        self.rebuild()

    def _rename(self, item: QListWidgetItem) -> None:
        lid = item.data(ID_ROLE)
        layer = self.editor.doc.layer(lid)
        name, ok = QInputDialog.getText(self, "Rename layer", "Name:", text=layer.name)
        name = name.strip()
        if ok and name and name != layer.name:
            self.editor.push(edits.SetLayerProps(lid, text="Rename layer", name=name))

    def keyPressEvent(self, e) -> None:
        layer = self.editor.selected_layer()
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and layer is not None:
            self.editor.push(edits.RemoveLayer(layer.id))
            return
        super().keyPressEvent(e)
