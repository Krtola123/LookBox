"""App settings (QSettings, HKCU\\Software\\RRIPP on Windows). Settings saved under the
app's old name (LookBox) are copied over the first time, so choices survive the rename."""

from __future__ import annotations

from PySide6.QtCore import QSettings

from lookbox import branding

_migrated = False


def app_settings() -> QSettings:
    global _migrated
    s = QSettings(branding.APP_NAME, branding.APP_NAME)
    if not _migrated:
        _migrated = True
        if not s.allKeys():
            old = QSettings(branding.LEGACY_DATA_DIR, branding.LEGACY_DATA_DIR)
            for key in old.allKeys():
                s.setValue(key, old.value(key))
    return s
