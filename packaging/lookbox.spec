# PyInstaller build for RRIPP (ARCHITECTURE §15, M12): one folder, windowed.
#   pyinstaller packaging/lookbox.spec --noconfirm      → dist/RRIPP/RRIPP.exe
# The build is checked by running `dist/RRIPP/RRIPP.exe --selftest report.txt`.

import os

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 (SPECPATH is set by PyInstaller)

datas = [
    (os.path.join(ROOT, "lookbox", "ui", "theme.qss"), "lookbox/ui"),
    (os.path.join(ROOT, "lookbox", "ui", "icon.png"), "lookbox/ui"),
    (os.path.join(ROOT, "lookbox", "ui", "icon.ico"), "lookbox/ui"),
    (os.path.join(ROOT, "lookbox", "ui", "check.png"), "lookbox/ui"),
    (os.path.join(ROOT, "lookbox", "models", "models.json"), "lookbox/models"),
    (os.path.join(ROOT, "lookbox", "luts"), "lookbox/luts"),  # bundled filter looks (.cube)
]

# onnxruntime-directml ships DirectML.dll next to its Python extension; make sure it comes along.
binaries = collect_dynamic_libs("onnxruntime")

# Qt modules LookBox never imports. PyInstaller only bundles what's imported, but some
# packages pull these in optionally; excluding them keeps the folder ~40% smaller.
excludes = [
    "tkinter", "pytest", "PIL",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets", "PySide6.QtWebChannel", "PySide6.QtMultimedia", "PySide6.Qt3DCore",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf", "PySide6.QtSql",
    "PySide6.QtNetwork", "PySide6.QtBluetooth", "PySide6.QtPositioning", "PySide6.QtSensors",
    "PySide6.QtSerialPort", "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets", "PySide6.QtSvg", "PySide6.QtSvgWidgets",
]

a = Analysis(  # noqa: F821
    [os.path.join(ROOT, "lookbox", "__main__.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=["onnxruntime", "onnxruntime.capi._pybind_state"],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RRIPP",
    icon=os.path.join(ROOT, "lookbox", "ui", "icon.ico"),
    console=False,  # a windowed app; errors go to %LOCALAPPDATA%\RRIPP\logs\rripp.log
    upx=False,  # UPX-packed DLLs trip antivirus and save little
    version=os.path.join(SPECPATH, "version_info.txt"),  # noqa: F821
)
coll = COLLECT(exe, a.binaries, a.datas, name="RRIPP", upx=False)  # noqa: F821
