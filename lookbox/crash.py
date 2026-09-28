"""Crash reporting: every error ends up in a log file with its full traceback.

- Python errors anywhere (UI slots, worker threads) are written to
  <data dir>/LookBox/logs/lookbox.log and, when a UI is up, shown in a dialog
  instead of taking the app down.
- Hard native crashes (inside Qt / onnxruntime) can't be caught, but
  faulthandler writes their stack trace to the same log before the process dies.

The formatting/writing part is Qt-free and tested; the dialog is optional.
"""

from __future__ import annotations

import datetime
import faulthandler
import os
import sys
import threading
import time
import traceback
from typing import Callable

MAX_LOG_BYTES = 2_000_000


def logs_dir() -> str:
    override = os.environ.get("LOOKBOX_LOG_DIR")
    if override:
        return override
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    else:
        base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return os.path.join(base, "LookBox", "logs")


def log_path() -> str:
    return os.path.join(logs_dir(), "lookbox.log")


def write_report(exc_type, exc, tb, where: str = "") -> str:
    """Append a timestamped traceback to the log; returns the text written."""
    os.makedirs(logs_dir(), exist_ok=True)
    path = log_path()
    try:
        if os.path.getsize(path) > MAX_LOG_BYTES:  # keep it small: start over
            os.replace(path, path + ".old")
    except OSError:
        pass
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    body = "".join(traceback.format_exception(exc_type, exc, tb))
    text = f"\n===== {stamp} {where}\n{body}"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text)
    return text


class ErrorReporter:
    """Installs sys/threading excepthooks. `show(title, message)` is called on the UI
    thread for errors there; repeats of the same error within a few seconds are
    logged but not shown again (a failing paint handler would otherwise spam)."""

    def __init__(self, show: Callable[[str, str], None] | None = None, quiet_seconds: float = 5.0) -> None:
        self.show = show
        self.quiet = quiet_seconds
        self._last: dict[str, float] = {}
        self._fault_file = None

    def install(self) -> None:
        sys.excepthook = self._hook
        threading.excepthook = lambda args: self._report(args.exc_type, args.exc_value, args.exc_traceback,
                                                           where=f"thread {args.thread.name if args.thread else ''}",
                                                           ui=False)
        try:
            os.makedirs(logs_dir(), exist_ok=True)
            self._fault_file = open(log_path(), "a", encoding="utf-8")  # kept open for faulthandler
            faulthandler.enable(file=self._fault_file, all_threads=True)
        except OSError:
            pass  # logging is best-effort; the app must still start

    def _hook(self, exc_type, exc, tb) -> None:
        self._report(exc_type, exc, tb, where="", ui=threading.current_thread() is threading.main_thread())

    def _report(self, exc_type, exc, tb, where: str, ui: bool) -> None:
        try:
            write_report(exc_type, exc, tb, where)
        except OSError:
            traceback.print_exception(exc_type, exc, tb)
        traceback.print_exception(exc_type, exc, tb)  # still visible in the run.bat console
        if not ui or self.show is None:
            return
        key = f"{exc_type.__name__}: {exc}"
        now = time.monotonic()
        if now - self._last.get(key, -1e9) < self.quiet:
            return
        self._last[key] = now
        self.show("Something went wrong",
                  f"{key}\n\nLookBox kept running. The full details were saved to:\n{log_path()}\n\n"
                  "Please send that file (or the last entry in it) so this can be fixed.")
