"""Content-addressed asset storage (ARCHITECTURE §5).

An asset is an imported file, kept byte-for-byte (so 16-bit PNG and EXR are
never degraded) and identified by the sha256 of those bytes. Decoded pixels are
cached. Assets are immutable: nothing ever writes into a decoded array.
"""

from __future__ import annotations

import hashlib
import os
import threading

import numpy as np

from lookbox.core.io import images
from lookbox.core.model import AssetInfo


class AssetStore:
    def __init__(self) -> None:
        self._bytes: dict[str, bytes] = {}
        self._ext: dict[str, str] = {}
        self._pixels: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()  # export runs in a worker thread

    # ---- adding ----
    def add_bytes(self, data: bytes, ext: str, name: str = "") -> AssetInfo:
        """Store encoded bytes; decodes once to validate and learn the size."""
        ext = ext.lower()
        asset_id = hashlib.sha256(data).hexdigest()
        with self._lock:
            known = asset_id in self._bytes
        if known:
            px = self.pixels(asset_id)
        else:
            px = images.decode(data, ext)
            px.setflags(write=False)
            with self._lock:
                self._bytes[asset_id] = data
                self._ext[asset_id] = ext
                self._pixels[asset_id] = px
        return AssetInfo(id=asset_id, ext=ext, width=px.shape[1], height=px.shape[0], name=name)

    def add_file(self, path: str) -> AssetInfo:
        data = images.read_file(path)
        return self.add_bytes(data, os.path.splitext(path)[1], os.path.basename(path))

    # ---- reading ----
    def has(self, asset_id: str) -> bool:
        with self._lock:
            return asset_id in self._bytes

    def raw(self, asset_id: str) -> tuple[bytes, str]:
        with self._lock:
            return self._bytes[asset_id], self._ext[asset_id]

    def pixels(self, asset_id: str) -> np.ndarray:
        """Read-only float32 RGBA array."""
        with self._lock:
            cached = self._pixels.get(asset_id)
            if cached is not None:
                return cached
            data, ext = self._bytes[asset_id], self._ext[asset_id]
        px = images.decode(data, ext)
        px.setflags(write=False)
        with self._lock:
            self._pixels.setdefault(asset_id, px)
            return self._pixels[asset_id]

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._bytes)
