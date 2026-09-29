"""Model registry and verified downloads (ARCHITECTURE §10.2, §10.3). Qt-free.

Models are listed in lookbox/models/models.json (url, size, sha256) — never in
code. They download on first use into the user's data folder, streamed and
hashed; a file that doesn't match is rejected. A small ".verified" marker next
to a good file avoids re-hashing ~1 GB at every launch.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.request
from dataclasses import dataclass
from typing import Callable

from lookbox.branding import data_root

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "models.json")
CHUNK = 1 << 20


class DownloadError(Exception):
    """Raised with a message that can be shown to the user as-is."""


class DownloadCancelled(Exception):
    pass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    task: str
    url: str
    source: str
    filename: str
    sha256: str
    size: int
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @property
    def size_mb(self) -> int:
        return round(self.size / 1e6)


def load_registry(path: str = REGISTRY_PATH) -> dict[str, ModelSpec]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = {}
    for mid, m in data["models"].items():
        out[mid] = ModelSpec(id=mid, name=m["name"], task=m["task"], url=m["url"], source=m.get("source", ""),
                             filename=m["filename"], sha256=m["sha256"].lower(), size=int(m["size"]),
                             input_size=int(m.get("input_size", 1024)), mean=tuple(m["mean"]), std=tuple(m["std"]))
    return out


def models_dir() -> str:
    """%LOCALAPPDATA%\\RRIPP\\models on Windows; ~/.local/share/RRIPP/models elsewhere.
    LOOKBOX_MODELS_DIR overrides (tests, portable installs)."""
    override = os.environ.get("LOOKBOX_MODELS_DIR")
    if override:
        return override
    return os.path.join(data_root(), "models")


def model_path(spec: ModelSpec) -> str:
    return os.path.join(models_dir(), spec.filename)


def _marker(spec: ModelSpec) -> str:
    return model_path(spec) + ".verified"


def is_ready(spec: ModelSpec) -> bool:
    """Downloaded, the right size, and verified once against its sha256."""
    path = model_path(spec)
    try:
        if os.path.getsize(path) != spec.size:
            return False
        with open(_marker(spec), encoding="utf-8") as fh:
            return fh.read().strip() == spec.sha256
    except OSError:
        return False


def download(spec: ModelSpec, progress: Callable[[int, int], None] | None = None,
             cancel: threading.Event | None = None, opener=urllib.request.urlopen) -> str:
    """Stream the model to disk, hashing as it goes. Only a verified file ever takes
    the final name; a failed or cancelled download leaves nothing behind."""
    os.makedirs(models_dir(), exist_ok=True)
    final = model_path(spec)
    part = final + ".part"
    digest = hashlib.sha256()
    done = 0
    try:
        try:
            resp = opener(spec.url, timeout=60)
        except OSError as exc:
            raise DownloadError(f"Couldn't reach the model server: {exc}.") from exc
        with resp, open(part, "wb") as out:
            while True:
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled()
                try:
                    chunk = resp.read(CHUNK)
                except OSError as exc:
                    raise DownloadError(f"The download was interrupted: {exc}.") from exc
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if done > spec.size:
                    raise DownloadError("The server sent more data than expected; download rejected.")
                if progress is not None:
                    progress(done, spec.size)
        if done != spec.size:
            raise DownloadError(f"Download incomplete ({done:,} of {spec.size:,} bytes). Please try again.")
        if digest.hexdigest() != spec.sha256:
            raise DownloadError("The downloaded model failed its integrity check and was deleted.")
        os.replace(part, final)
        with open(_marker(spec), "w", encoding="utf-8") as fh:
            fh.write(spec.sha256)
        return final
    finally:
        if os.path.exists(part):
            os.remove(part)
