"""Drag image or project files from Explorer onto the canvas."""

from __future__ import annotations


class FileDropMixin:
    """Mixed into CanvasView (needs `files_dropped` and `mapToScene`)."""

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if not paths:
            super().dropEvent(e)
            return
        p = self.mapToScene(e.position().toPoint())
        e.acceptProposedAction()
        self.files_dropped.emit(paths, (p.x(), p.y()))
