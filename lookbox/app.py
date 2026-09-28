"""Entry point: `python -m lookbox [file ...]`."""

from __future__ import annotations

import os
import sys

import lookbox  # noqa: F401  (EXR env var before cv2)
from PySide6.QtWidgets import QApplication, QMessageBox

from lookbox.core import serialize
from lookbox.crash import ErrorReporter
from lookbox.ui.main_window import MainWindow


def _resource(name: str) -> str:
    # Works from source and from a PyInstaller build (data kept at lookbox/ui/).
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", name)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setApplicationName("LookBox")
    app.setOrganizationName("LookBox")
    app.setStyle("Fusion")
    with open(_resource("theme.qss"), encoding="utf-8") as fh:
        app.setStyleSheet(fh.read())

    # Errors become a dialog + a log entry instead of a silent exit.
    ErrorReporter(show=lambda title, msg: QMessageBox.critical(None, title, msg)).install()

    win = MainWindow()
    win.show()

    files = [a for a in argv[1:] if os.path.isfile(a)]
    projects = [f for f in files if f.lower().endswith(serialize.EXTENSION)]
    if projects:
        win.open_path(projects[0])
    elif files:
        win.import_paths(files, None)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
