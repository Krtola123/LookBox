"""Background rendering of per-layer previews (ARCHITECTURE §6.2–6.4).

Layers render in a thread pool (one job per layer × level), results go into a
byte-budgeted LRU cache as display-ready BGRA arrays, and `ready` fires on the
UI thread. The UI never waits: it shows the best level already cached and
swaps in the sharper one when it lands.
"""

from __future__ import annotations

import copy

from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal, Slot

from lookbox.core.assets import AssetStore
from lookbox.core.model import ImageLayer
from lookbox.core.render.cache import LRUCache
from lookbox.core.render.levels import MIN_LEVEL, to_display_bgra
from lookbox.core.render.pipeline import render_key, render_layer

CACHE_BUDGET_BYTES = 4 * 1024**3  # §6.2 default; becomes a user setting later


class _Emitter(QObject):
    """Lives on the UI thread; workers emit through it so slots run there (queued)."""
    done = Signal(str, float)
    failed = Signal(str, float, str)


class _Job(QRunnable):
    def __init__(self, layer: ImageLayer, store: AssetStore, level: float, key: str,
                 cache: LRUCache, emitter: _Emitter) -> None:
        super().__init__()
        self.layer, self.store, self.level, self.key = layer, store, level, key
        self.cache, self.emitter = cache, emitter

    def run(self) -> None:
        try:
            bgra = to_display_bgra(render_layer(self.layer, self.store, self.level))
            self.cache.put((self.key, self.level), bgra)
        except Exception as exc:  # reported to the UI, never swallowed (§16.5)
            self.emitter.failed.emit(self.key, self.level, str(exc))
        else:
            self.emitter.done.emit(self.key, self.level)


class RenderService(QObject):
    ready = Signal(str, float)  # render_key, level
    failed = Signal(str)  # user-facing message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.cache = LRUCache(CACHE_BUDGET_BYTES)
        self.pool = QThreadPool(self)
        # Leave headroom for the UI thread and the OS.
        self.pool.setMaxThreadCount(max(1, QThread.idealThreadCount() - 2))
        self._inflight: set[tuple[str, float]] = set()
        self._failed: set[tuple[str, float]] = set()  # never retried in a loop
        self._emitter = _Emitter()
        self._emitter.done.connect(self._on_done)
        self._emitter.failed.connect(self._on_failed)

    def request(self, layer: ImageLayer, store: AssetStore, level: float, key: str | None = None):
        """Return the cached BGRA array for (layer, level), or None and start rendering it."""
        key = key or render_key(layer)
        arr = self.cache.get((key, level))
        if arr is not None:
            return arr
        if (key, level) not in self._inflight and (key, level) not in self._failed:
            self._inflight.add((key, level))
            # Snapshot the layer: the document may change while the job runs.
            self.pool.start(_Job(copy.deepcopy(layer), store, level, key, self.cache, self._emitter))
        return None

    def best_cached(self, key: str) -> tuple[float, object] | None:
        """Sharpest level already cached for `key`, as (level, array)."""
        level = 1.0
        while level >= MIN_LEVEL:
            if (key, level) in self.cache:
                arr = self.cache.get((key, level))
                if arr is not None:
                    return level, arr
            level /= 2.0
        return None

    def clear(self) -> None:
        self.pool.clear()  # drop queued jobs; running ones finish harmlessly
        self._inflight.clear()
        self._failed.clear()
        self.cache.clear()

    def shutdown(self) -> None:
        self.pool.clear()
        self.pool.waitForDone()

    @Slot(str, float)
    def _on_done(self, key: str, level: float) -> None:
        self._inflight.discard((key, level))
        self.ready.emit(key, level)

    @Slot(str, float, str)
    def _on_failed(self, key: str, level: float, message: str) -> None:
        self._inflight.discard((key, level))
        self._failed.add((key, level))
        self.failed.emit(f"Couldn't render a layer preview: {message}")
