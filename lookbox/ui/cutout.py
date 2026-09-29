"""Cut-out flows: choose + download a model, run background removal (M6) and fills
(M13: AI, quick or from a clean render; Fill and Grab), apply the result as edits
(§4.2). One AI job at a time."""

from __future__ import annotations

import importlib.util

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QMainWindow, QMessageBox, QProgressDialog

from lookbox.ai.registry import ModelSpec, is_ready, load_registry
from lookbox.ai.runtime import ModelManager
from lookbox.commands import edits
from lookbox.core.io.images import ImageError
from lookbox.core.model import ImageLayer, LayerMask
from lookbox.ui.ai_jobs import DownloadJob, FillJob, RemoveBackgroundJob
from lookbox.ui.editor import Editor
from lookbox.ui.settings import app_settings

SETTING_MODEL = "ai/background_model"
SETTING_KEEP_BG = "ai/keep_background"


class CutoutController(QObject):
    status = Signal(str, int)
    busy_changed = Signal(bool)

    def __init__(self, window: QMainWindow, editor: Editor) -> None:
        super().__init__(window)
        self.window, self.editor = window, editor
        self.manager = ModelManager()
        self.settings = app_settings()
        models = load_registry()
        self.registry = {k: v for k, v in models.items() if v.task == "background"}
        self.lama = models.get("lama")
        self._job = None
        self._progress: QProgressDialog | None = None
        self._pending = None  # what to run once a model download finishes (a callable)
        self._run_ctx: tuple | None = None

    @property
    def busy(self) -> bool:
        return self._job is not None

    # ------------------------------------------------------------ model choice
    def current_spec(self) -> ModelSpec | None:
        return self.registry.get(self.settings.value(SETTING_MODEL, "", type=str))

    def choose_model(self) -> ModelSpec | None:
        full, lite = self.registry["birefnet"], self.registry["birefnet-lite"]
        box = QMessageBox(self.window)
        box.setWindowTitle("Background removal model")
        box.setText("Background removal runs on your computer, with an AI model that downloads once.")
        box.setInformativeText(
            f"• Best quality: {full.size_mb} MB. Recommended for GPUs with 8 GB or more (e.g. RTX 4060).\n"
            f"• Fast: {lite.size_mb} MB. For 4 GB GPUs (e.g. RX 570) or quicker results.\n\n"
            "You can switch later with “Model…” in the Cut-out section.")
        b_full = box.addButton(f"Best quality ({full.size_mb} MB)", QMessageBox.ButtonRole.AcceptRole)
        b_lite = box.addButton(f"Fast ({lite.size_mb} MB)", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(b_full)
        box.exec()
        chosen = {b_full: full, b_lite: lite}.get(box.clickedButton())
        if chosen is not None:
            self.settings.setValue(SETTING_MODEL, chosen.id)
            self.manager.unload()
        return chosen

    # ------------------------------------------------------------ remove background
    @property
    def keep_background(self) -> bool:
        return self.settings.value(SETTING_KEEP_BG, True, type=bool)

    @keep_background.setter
    def keep_background(self, on: bool) -> None:
        self.settings.setValue(SETTING_KEEP_BG, bool(on))

    def remove_background(self, layer_id: str, refine: bool) -> None:
        if self.busy:
            return
        if importlib.util.find_spec("onnxruntime") is None:
            QMessageBox.warning(self.window, "Background removal",
                                "The AI runtime isn't installed. Close RRIPP and start it again with run.bat "
                                "(it installs onnxruntime-directml automatically).")
            return
        spec = self.current_spec() or self.choose_model()
        if spec is None:
            return
        keep = self.keep_background
        if not is_ready(spec):
            self._pending = lambda: self._start_run(spec, layer_id, refine, keep)
            self._start_download(spec, "Background removal")
            return
        self._start_run(spec, layer_id, refine, keep)

    def _start_download(self, spec: ModelSpec, title: str) -> None:
        job = DownloadJob(spec)
        dlg = QProgressDialog(f"Downloading the {spec.name.lower()} model ({spec.size_mb} MB)…", "Cancel",
                              0, max(1, spec.size_mb), self.window)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0)
        dlg.canceled.connect(job.cancel)
        job.progress.connect(dlg.setValue)  # a Qt slot on a UI-thread object: delivered queued
        job.done.connect(self._on_downloaded)
        self._begin(job, dlg)

    @Slot(str)
    def _on_downloaded(self, error: str) -> None:
        self._end()
        pending, self._pending = self._pending, None
        if error == "cancelled":
            self.status.emit("Model download cancelled.", 4000)
            return
        if error:
            QMessageBox.warning(self.window, "Model download failed", error)
            return
        if pending is not None:
            pending()

    def _start_run(self, spec: ModelSpec, layer_id: str, refine: bool, keep_bg: bool) -> None:
        doc = self.editor.doc
        if not doc.has_layer(layer_id) or not isinstance(doc.layer(layer_id), ImageLayer):
            return
        layer = doc.layer(layer_id)
        rgba = self.editor.store.pixels(layer.source)  # read-only; the job doesn't modify it
        self._run_ctx = (self.editor.generation, layer_id, layer.source, self.editor.store, keep_bg)
        job = RemoveBackgroundJob(self.manager, spec, rgba, refine)
        dlg = QProgressDialog("Removing the background…\n(the first run on a GPU can take a little longer)",
                              "Cancel", 0, 0, self.window)
        dlg.setWindowTitle("Background removal")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(300)
        dlg.canceled.connect(lambda: setattr(self, "_run_ctx", None))  # result will be discarded
        job.done.connect(self._on_removed)
        self._begin(job, dlg)

    @Slot(bytes, str, str)
    def _on_removed(self, png: bytes, error: str, notice: str) -> None:
        ctx, self._run_ctx = self._run_ctx, None  # read before cleanup touches the dialog
        self._end()
        if error:
            QMessageBox.warning(self.window, "Background removal", error)
            return
        if ctx is None:
            self.status.emit("Background removal cancelled.", 4000)
            return
        generation, layer_id, source, store, keep_bg = ctx
        doc = self.editor.doc
        if (generation != self.editor.generation or store is not self.editor.store or not doc.has_layer(layer_id)
                or getattr(doc.layer(layer_id), "source", None) != source):
            self.status.emit("Background removal finished, but that layer changed meanwhile; nothing applied.", 6000)
            return
        try:
            info = store.add_bytes(bytes(png), ".png", "mask")
        except ImageError as exc:
            QMessageBox.warning(self.window, "Background removal", str(exc))
            return
        self.editor.push(edits.remove_background(doc, layer_id, LayerMask(asset=info.id), info, keep_bg))
        kept = " The background is on its own layer below." if keep_bg else ""
        self.status.emit(f"Background removed.{kept} {notice}", 8000)

    # ------------------------------------------------------------ fill + grab (M13)
    def fill(self, layer_id: str, mask, method: str, set_mask=None, grab: bool = False) -> None:
        """Fill the selection `mask` (source-size, 1 = selected) on an image layer: a fill
        layer above it; with `grab`, the selection also goes on its own layer on top
        (`set_mask`: the session's SetMask, not applied yet). method: "ai" | "quick" | "plate"."""
        from lookbox.core.masks import fill as F

        doc = self.editor.doc
        if self.busy or not doc.has_layer(layer_id) or not isinstance(doc.layer(layer_id), ImageLayer):
            return
        layer = doc.layer(layer_id)
        rgba = self.editor.store.pixels(layer.source)
        selected = float((mask > F.HOLE_THRESHOLD).mean())
        if selected == 0.0 or selected > 0.9:
            QMessageBox.information(self.window, "Fill", "Select what to remove first (Pick, Lasso or Brush): "
                                    "the selected area is what gets filled.")
            return
        ctx = (self.editor.generation, layer_id, layer.source, self.editor.store, set_mask, grab)
        if grab and F.on_transparency(rgba[:, :, 3], mask):
            self._apply_fill(ctx, None, "It's on a transparent background, so there's nothing to fill behind it.")
            return
        if method == "plate":
            plate = self._ask_plate(rgba.shape)
            if plate is None:
                return
            self._start_fill(ctx, "plate", rgba, mask, plate=plate)
            return
        if method == "ai":
            if importlib.util.find_spec("onnxruntime") is None or self.lama is None:
                QMessageBox.warning(self.window, "AI fill", "The AI runtime isn't installed.")
                return
            if not is_ready(self.lama):
                ok = QMessageBox.question(
                    self.window, "AI fill",
                    f"AI fill runs on your computer with a model that downloads once ({self.lama.size_mb} MB). "
                    "Download it now?") == QMessageBox.StandardButton.Yes
                if ok:
                    self._pending = lambda: self._start_fill(ctx, "ai", rgba, mask)
                    self._start_download(self.lama, "AI fill")
                return
        self._start_fill(ctx, method, rgba, mask)

    def _ask_plate(self, shape):
        from PySide6.QtWidgets import QFileDialog

        from lookbox.core.io import images

        path, _ = QFileDialog.getOpenFileName(
            self.window, "Fill from a render: pick the same shot rendered without the object", "",
            "Images (" + " ".join(f"*{e}" for e in images.SUPPORTED_EXTS) + ")")
        if not path:
            return None
        try:
            plate = images.decode(images.read_file(path), path[path.rfind("."):])
        except (ImageError, OSError) as exc:
            QMessageBox.warning(self.window, "Fill from a render", str(exc))
            return None
        if plate.shape[:2] != shape[:2]:
            QMessageBox.warning(self.window, "Fill from a render",
                                f"That render is {plate.shape[1]} × {plate.shape[0]}; this image is "
                                f"{shape[1]} × {shape[0]}. Render the clean plate at the same size and camera.")
            return None
        return plate

    def _start_fill(self, ctx, method: str, rgba, mask, plate=None) -> None:
        self._fill_ctx = ctx
        job = FillJob(method, rgba, mask, manager=self.manager, spec=self.lama, plate=plate)
        label = {"ai": "Filling with AI…\n(the first run on a GPU can take a little longer)",
                 "quick": "Filling…", "plate": "Filling from the render…"}[method]
        dlg = QProgressDialog(label, "Cancel", 0, 0, self.window)
        dlg.setWindowTitle("Fill")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(300)
        dlg.canceled.connect(lambda: setattr(self, "_fill_ctx", None))  # result will be discarded
        job.done.connect(self._on_filled)
        self._begin(job, dlg)

    @Slot(bytes, str, str)
    def _on_filled(self, png: bytes, error: str, notice: str) -> None:
        ctx, self._fill_ctx = getattr(self, "_fill_ctx", None), None
        self._end()
        if error:
            QMessageBox.warning(self.window, "Fill", error)
            return
        if ctx is None:
            self.status.emit("Fill cancelled.", 4000)
            return
        store = ctx[3]
        info = store.add_bytes(bytes(png), ".png", "fill") if png else None
        self._apply_fill(ctx, info, notice)

    def _apply_fill(self, ctx, info, notice: str) -> None:
        generation, layer_id, source, store, set_mask, grab = ctx
        doc = self.editor.doc
        if (generation != self.editor.generation or store is not self.editor.store or not doc.has_layer(layer_id)
                or getattr(doc.layer(layer_id), "source", None) != source):
            self.status.emit("The fill finished, but that layer changed meanwhile; nothing applied.", 6000)
            return
        if grab:
            batch = edits.grab(doc, layer_id, set_mask, info)
            if batch is None:
                return
            self.editor.push(batch)
            self.editor.select(batch.parts[-2 if info is None else 1].layer.id)
            self.status.emit(f"Grabbed: it's on its own layer, the hole behind it is filled. {notice}", 8000)
        elif info is not None:
            self.editor.push(edits.fill_layer(doc, layer_id, info))
            self.status.emit(f"Filled, on its own layer above (hide it to see the original). {notice}", 8000)

    # ------------------------------------------------------------ job plumbing
    def _begin(self, job, dlg: QProgressDialog) -> None:
        self._job, self._progress = job, dlg
        self.busy_changed.emit(True)
        job.start()

    def _end(self) -> None:
        job, self._job = self._job, None
        if job is not None:
            job.wait()
            job.deleteLater()
        if self._progress is not None:
            # QProgressDialog emits `canceled` when closed; block it, or a finished job
            # would look cancelled and its result would be thrown away.
            self._progress.blockSignals(True)
            self._progress.close()
            self._progress.deleteLater()
            self._progress = None
        self.busy_changed.emit(False)

    def shutdown(self) -> None:
        if self._job is not None:
            if isinstance(self._job, DownloadJob):
                self._job.cancel()
            self._job.wait()
