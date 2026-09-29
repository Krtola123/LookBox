"""Background jobs for the AI features (§4.5: nothing blocks the UI thread).

Each job reports back with one signal; the controller connects them to its own
slots so handlers run on the UI thread.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QThread, Signal

from lookbox.ai import birefnet
from lookbox.ai.registry import DownloadCancelled, DownloadError, ModelSpec, download, model_path
from lookbox.ai.runtime import ModelManager
from lookbox.core.masks.ops import encode_mask_png


class DownloadJob(QThread):
    progress = Signal(int)  # MB done (connect straight to a QProgressDialog's setValue: queued, thread-safe)
    done = Signal(str)  # "" = ok, "cancelled", or a user-facing error

    def __init__(self, spec: ModelSpec, parent=None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            download(self.spec, progress=lambda d, _t: self.progress.emit(d // 1_000_000),
                     cancel=self.cancel_event)
        except DownloadCancelled:
            self.done.emit("cancelled")
        except DownloadError as exc:
            self.done.emit(str(exc))
        except OSError as exc:
            self.done.emit(f"Couldn't save the model: {exc}.")
        else:
            self.done.emit("")


class RemoveBackgroundJob(QThread):
    """Runs the model on a layer's full source image; returns the mask as PNG bytes."""

    done = Signal(bytes, str, str)  # mask png, error ("" = ok), notice (GPU→CPU etc.)

    def __init__(self, manager: ModelManager, spec: ModelSpec, rgba: np.ndarray, refine: bool,
                 parent=None) -> None:
        super().__init__(parent)
        self.manager, self.spec, self.rgba, self.refine = manager, spec, rgba, refine

    def run(self) -> None:
        path = model_path(self.spec)
        try:
            size = self.manager.input_size(path, self.spec.input_size)
            mask = birefnet.remove_background(lambda x: self.manager.run(path, x), self.rgba, self.spec,
                                              size=size, refine=self.refine)
            png = encode_mask_png(mask)
        except MemoryError:
            self.done.emit(b"", "Not enough memory to run background removal on this image.", "")
        except Exception as exc:  # surfaced to the user, never swallowed (§16.5)
            self.done.emit(b"", f"Background removal failed: {exc}", "")
        else:
            where = "GPU" if self.manager.provider == "DmlExecutionProvider" else "CPU"
            self.done.emit(png, "", self.manager.notice or f"Ran on the {where}.")


class FillJob(QThread):
    """Fills the selection (§9a) and returns the fill image (same size as the source) as a
    16-bit PNG. `method`: "ai" (LaMa via `manager`), "quick" (OpenCV) or "plate" (`plate`)."""

    done = Signal(bytes, str, str)  # fill png, error ("" = ok), notice

    def __init__(self, method: str, rgba: np.ndarray, mask: np.ndarray, manager: ModelManager | None = None,
                 spec: ModelSpec | None = None, plate: np.ndarray | None = None, parent=None) -> None:
        super().__init__(parent)
        self.method, self.rgba, self.mask = method, rgba, mask
        self.manager, self.spec, self.plate = manager, spec, plate

    def run(self) -> None:
        from lookbox.ai import lama
        from lookbox.core.io.images import encode_png
        from lookbox.core.masks import fill as F

        rgb = np.ascontiguousarray(self.rgba[:, :, :3], np.float32)
        notice = ""
        try:
            if self.method == "ai":
                path = model_path(self.spec)
                filled = lama.fill(lambda feeds: self.manager.run(path, feeds), rgb, self.mask)
                where = "GPU" if self.manager.provider == "DmlExecutionProvider" else "CPU"
                notice = self.manager.notice or f"Ran on the {where}."
            elif self.method == "plate":
                filled = F.plate_fill(rgb, self.plate, self.mask)
            else:
                filled = F.quick_fill(rgb, self.mask)
            patch = F.make_patch(filled, self.mask)
            png = encode_png(patch, bits=16) if patch is not None else b""
        except MemoryError:
            self.done.emit(b"", "Not enough memory to fill this area.", "")
        except Exception as exc:  # surfaced to the user, never swallowed (§16.5)
            self.done.emit(b"", f"Fill failed: {exc}", "")
        else:
            self.done.emit(png, "", notice)
