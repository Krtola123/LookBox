"""LookBox — an offline, Canva-simple compositor for finishing renders.

See ARCHITECTURE.md for the rules this package follows.
"""

import os

# OpenCV only enables its EXR codec if this is set before cv2 is imported (§3).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

__version__ = "1.0.0"
