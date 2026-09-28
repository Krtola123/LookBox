"""Write packaging/version_info.txt (the exe's Properties → Details) from lookbox.__version__."""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
with open(os.path.join(ROOT, "lookbox", "__init__.py"), encoding="utf-8") as fh:
    version = re.search(r'__version__ = "([^"]+)"', fh.read()).group(1)
nums = tuple(int(x) for x in (version.split(".") + ["0", "0", "0"])[:4])
text = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={nums}, prodvers={nums}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('ProductName', 'LookBox'),
      StringStruct('FileDescription', 'LookBox'),
      StringStruct('FileVersion', '{version}'),
      StringStruct('ProductVersion', '{version}'),
      StringStruct('OriginalFilename', 'LookBox.exe'),
      StringStruct('InternalName', 'LookBox')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
with open(os.path.join(ROOT, "packaging", "version_info.txt"), "w", encoding="utf-8") as fh:
    fh.write(text)
print(f"version_info.txt for {version}")
