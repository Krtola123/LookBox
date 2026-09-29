"""Layers tab → Pages (M14): which page you're on, and add / duplicate / move /
resize / delete it. The same actions work from the canvas ("+ Add page", click a
page) and the keyboard (Page Up / Page Down)."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QToolButton, QVBoxLayout, QWidget

from lookbox.commands import edits


class PagesBar(QWidget):
    def __init__(self, editor, window) -> None:
        super().__init__()
        self.editor, self.window = editor, window
        self.setObjectName("pagesBar")
        col = QVBoxLayout(self)
        col.setContentsMargins(12, 10, 12, 6)
        col.setSpacing(6)
        head = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("sectionTitle")
        head.addWidget(self.title)
        head.addStretch(1)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Page name")
        self.name.setMaximumWidth(150)
        self.name.editingFinished.connect(self._rename)
        head.addWidget(self.name)
        col.addLayout(head)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.buttons = {}
        for key, label, tip, fn in (
                ("add", "Add", "Add a page after this one (same size)", window.add_page),
                ("dup", "Duplicate", "Copy this page: same layers, images shared (no extra memory)",
                 window.duplicate_page),
                ("up", "↑", "Move this page up", lambda: window.move_page(-1)),
                ("down", "↓", "Move this page down", lambda: window.move_page(+1)),
                ("resize", "Resize…", "Change this page's size (layers stay centred)", window.resize_page),
                ("del", "Delete", "Delete this page", window.delete_page)):
            b = QToolButton()
            b.setObjectName("pageButton")
            b.setText(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
            self.buttons[key] = b
        row.addStretch(1)
        col.addLayout(row)
        for sig in (editor.changed, editor.pages_changed, editor.document_replaced):
            sig.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        pages = self.editor.project.pages
        n, i = len(pages), self.editor.page_number()
        self.title.setText(f"Page {i} of {n}")
        if not self.name.hasFocus():
            self.name.setText(self.editor.doc.name)
        self.buttons["del"].setEnabled(n > 1)
        self.buttons["up"].setEnabled(i > 1)
        self.buttons["down"].setEnabled(i < n)

    def _rename(self) -> None:
        name = self.name.text().strip()
        if name != self.editor.doc.name:
            self.editor.push(edits.SetPageName(self.editor.doc.name, name))
