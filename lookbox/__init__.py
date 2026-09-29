"""RRIPP (Reshi Renders' Image Post Processing App): an offline, Canva-simple compositor
for finishing renders. Formerly LookBox; the package keeps that name internally.

See ARCHITECTURE.md for the rules this package follows.
"""

import os

# OpenCV only enables its EXR codec if this is set before cv2 is imported (§3).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

__version__ = "1.2.0"
