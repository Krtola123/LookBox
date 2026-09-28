"""Entry point: `python -m lookbox [file ...]`, or LookBox.exe.

`--selftest [report.txt]` checks a build without showing a window (lookbox/selftest.py).
"""

from __future__ import annotations

import os
import sys

import lookbox  # noqa: F401  (EXR env var before cv2)


def resource(name: str) -> str:
    """A file shipped next to the UI code (theme, icon). Works from source and from the
    PyInstaller bundle, which keeps the package layout under _internal/lookbox/ui/."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", name)


def _harden_frozen() -> None:
    """A windowed exe has no console: sys.stdout/stderr are None, and anything that
    prints (tracebacks included) would crash. Send them nowhere instead; errors still
    reach the log file (crash.py)."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _windows_taskbar_identity() -> None:
    """Own taskbar button + icon instead of Python's (harmless when not on Windows)."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LookBox.LookBox")
        except (AttributeError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    selftest = "--selftest" in argv
    if selftest:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window, works on build machines
    _harden_frozen()
    _windows_taskbar_identity()

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    from lookbox.core import serialize
    from lookbox.core.render import text as text_render
    from lookbox.crash import ErrorReporter
    from lookbox.ui.text_engine import QtTextEngine

    app = QApplication(argv)
    app.setApplicationName("LookBox")
    app.setOrganizationName("LookBox")
    app.setApplicationVersion(lookbox.__version__)
    app.setWindowIcon(QIcon(resource("icon.png")))
    app.setStyle("Fusion")
    with open(resource("theme.qss"), encoding="utf-8") as fh:
        app.setStyleSheet(fh.read())
    text_render.set_engine(QtTextEngine())  # needs the QApplication (fonts); before any document loads

    if selftest:
        from lookbox import selftest as st

        rest = argv[argv.index("--selftest") + 1:]
        return st.run(rest[0] if rest else None)

    # Errors become a dialog + a log entry instead of a silent exit.
    ErrorReporter(show=lambda title, msg: QMessageBox.critical(None, title, msg)).install()

    from lookbox.ui.main_window import MainWindow

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
