"""The app's name and where it keeps its files. One place, so a rename is one edit.

The Python package stays `lookbox` (internal only; renaming it would churn every
import for nothing the user sees). Everything user-facing comes from here.
"""

from __future__ import annotations

import os
import sys

APP_NAME = "RRIPP"
FULL_NAME = "Reshi Renders' Image Post Processing App"
APP_USER_MODEL_ID = "ReshiRenders.RRIPP"  # Windows taskbar identity
DATA_DIR = "RRIPP"
LEGACY_DATA_DIR = "LookBox"  # the app's name before 1.1: its folder is moved over once
LOG_NAME = "rripp.log"


def _base() -> str:
    if sys.platform == "win32":
        return os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    return os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")


def data_root() -> str:
    """%LOCALAPPDATA%\\RRIPP (models, logs). A LookBox folder from before the rename is
    moved here the first time, so downloaded AI models (up to ~1 GB) aren't fetched again."""
    base = _base()
    root = os.path.join(base, DATA_DIR)
    legacy = os.path.join(base, LEGACY_DATA_DIR)
    if not os.path.exists(root) and os.path.isdir(legacy):
        try:
            os.replace(legacy, root)
        except OSError:
            return legacy  # in use or not allowed: keep using the old folder rather than fail
    return root
