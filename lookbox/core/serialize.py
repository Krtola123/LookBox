""".lookbox project files (ARCHITECTURE §5).

A .lookbox file is a zip:
  document.json          {"format_version": N, "document": {...}}
  assets/<sha256><ext>   original imported file bytes, unmodified
"""

from __future__ import annotations

import copy
import io
import json
import zipfile

from lookbox.core.assets import AssetStore
from lookbox.core.io.images import atomic_write
from lookbox.core.model import Document, document_from_dict, document_to_dict

FORMAT_VERSION = 1
EXTENSION = ".lookbox"


class ProjectError(Exception):
    """Raised with a message that can be shown to the user as-is."""


def migrate(data: dict) -> dict:
    """Bring an older document.json payload up to FORMAT_VERSION.

    Add one step per version bump: `if version == 1: ...; version = 2`.
    """
    version = data.get("format_version")
    if not isinstance(version, int):
        raise ProjectError("This file has no format version; it isn't a LookBox project.")
    if version > FORMAT_VERSION:
        raise ProjectError(
            f"This project was saved by a newer LookBox (format {version}); update the app to open it."
        )
    return data


def to_bytes(doc: Document, store: AssetStore) -> bytes:
    used = doc.referenced_assets()
    missing = [a for a in used if not store.has(a) or a not in doc.assets]
    if missing:
        raise ProjectError(f"{len(missing)} image(s) used by this project are missing from memory.")

    pruned = copy.copy(doc)
    pruned.assets = {k: v for k, v in doc.assets.items() if k in used}
    payload = {"format_version": FORMAT_VERSION, "document": document_to_dict(pruned)}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("document.json", json.dumps(payload, indent=1), zipfile.ZIP_DEFLATED)
        for asset_id in sorted(used):
            data, ext = store.raw(asset_id)
            # Images are already compressed; storing avoids wasted CPU.
            zf.writestr(f"assets/{asset_id}{ext}", data, zipfile.ZIP_STORED)
    return buf.getvalue()


def save(path: str, doc: Document, store: AssetStore) -> None:
    atomic_write(path, to_bytes(doc, store))


def from_bytes(data: bytes) -> tuple[Document, AssetStore]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ProjectError("This file isn't a valid LookBox project.") from exc
    with zf:
        try:
            payload = json.loads(zf.read("document.json"))
        except KeyError as exc:
            raise ProjectError("This project is missing document.json.") from exc
        except json.JSONDecodeError as exc:
            raise ProjectError("This project's document.json is corrupted.") from exc
        payload = migrate(payload)
        doc = document_from_dict(payload["document"])

        store = AssetStore()
        names = set(zf.namelist())
        for asset_id, info in doc.assets.items():
            name = f"assets/{asset_id}{info.ext}"
            if name not in names:
                raise ProjectError(f"The image '{info.name or asset_id[:8]}' is missing from this project.")
            stored = store.add_bytes(zf.read(name), info.ext, info.name)
            if stored.id != asset_id:
                raise ProjectError(f"The image '{info.name or asset_id[:8]}' in this project is corrupted.")

    missing = doc.referenced_assets() - set(doc.assets)
    if missing:
        raise ProjectError(f"{len(missing)} layer image(s) are missing from this project.")
    return doc, store


def load(path: str) -> tuple[Document, AssetStore]:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise ProjectError(f"Could not open the project: {exc.strerror}.") from exc
    return from_bytes(data)
