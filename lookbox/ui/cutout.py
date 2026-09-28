"""Cut-out flows (M6): choose + download the model, run background removal,
apply the result as a layer mask. Every document change is an edit (§4.2)."""

from __future__ import annotations

import importlib.util

from PySide6.QtCore import QObject, QSettings, Qt, Signal, Slot
from PySide6.QtWidgets import QMainWindow, QMessageBox, QProgressDialog

from lookbox.ai.registry import ModelSpec, is_ready, load_registry
from lookbox.ai.runtime import ModelManager
from lookbox.commands import edits
from lookbox.core.io.images import ImageError
from lookbox.core.model import ImageLayer, LayerMask
from lookbox.ui.ai_jobs import DownloadJob, RemoveBackgroundJob
from lookbox.ui.editor import Editor

SETTING_MODEL = "ai/background_model"


class CutoutController(QObject):
    status = Signal(str, int)
    busy_changed = Signal(bool)

    def __init__(self, window: QMainWindow, editor: Editor) -> None:
        super().__init__(window)
        self.window, self.editor = window, editor
        self.manager = ModelManager()
        self.settings = QSettings("LookBox", "LookBox")
        self.registry = {k: v for k, v in load_registry().items() if v.task == "background"}
        self._job = None
        self._progress: QProgressDialog | None = None
        self._pending: tuple[str, bool] | None = None  # (layer id, refine) waiting on a download
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
    def remove_background(self, layer_id: str, refine: bool) -> None:
        if self.busy:
            return
        if importlib.util.find_spec("onnxruntime") is None:
            QMessageBox.warning(self.window, "Background removal",
                                "The AI runtime isn't installed. Close LookBox and start it again with run.bat "
                                "(it installs onnxruntime-directml automatically).")
            return
        spec = self.current_spec() or self.choose_model()
        if spec is None:
            return
        if not is_ready(spec):
            self._pending = (layer_id, refine)
            self._start_download(spec)
            return
        self._start_run(spec, layer_id, refine)

    def _start_download(self, spec: ModelSpec) -> None:
        job = DownloadJob(spec)
        dlg = QProgressDialog(f"Downloading the {spec.name.lower()} model ({spec.size_mb} MB)…", "Cancel",
                              0, max(1, spec.size_mb), self.window)
        dlg.setWindowTitle("Background removal")
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
        spec = self.current_spec()
        if pending is not None and spec is not None:
            self._start_run(spec, *pending)

    def _start_run(self, spec: ModelSpec, layer_id: str, refine: bool) -> None:
        doc = self.editor.doc
        if not doc.has_layer(layer_id) or not isinstance(doc.layer(layer_id), ImageLayer):
            return
        layer = doc.layer(layer_id)
        rgba = self.editor.store.pixels(layer.source)  # read-only; the job doesn't modify it
        self._run_ctx = (self.editor.generation, layer_id, layer.source, self.editor.store)
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
        generation, layer_id, source, store = ctx
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
        old = doc.layer(layer_id).mask
        self.editor.push(edits.SetMask(layer_id, old, LayerMask(asset=info.id), asset=info,
                                       text="Remove background"))
        self.status.emit(f"Background removed. {notice}", 6000)

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
