""".rripp project files (ARCHITECTURE §5). Files from before the rename (.lookbox) are
the same format and open as they are.

A project file is a zip:
  document.json          {"format_version": 2, "project": {"pages": [{...}, ...]}}
                         (version 1, before pages: {"format_version": 1, "document": {...}})
  assets/<sha256><ext>   original imported file bytes, unmodified, once for all pages
"""

from __future__ import annotations

import copy
import io
import json
import zipfile

from lookbox.core.assets import AssetStore
from lookbox.core.io.images import atomic_write
from lookbox.core.model import Document, Project, project_from_dict, project_to_dict

FORMAT_VERSION = 2
EXTENSION = ".rripp"
OPEN_EXTENSIONS = (".rripp", ".lookbox")  # .lookbox: the app's name before 1.1, same format


def is_project(path: str) -> bool:
    return path.lower().endswith(OPEN_EXTENSIONS)


class ProjectError(Exception):
    """Raised with a message that can be shown to the user as-is."""


def migrate(data: dict) -> dict:
    """Bring an older document.json payload up to FORMAT_VERSION.

    Add one step per version bump: `if version == 1: ...; version = 2`.
    """
    version = data.get("format_version")
    if not isinstance(version, int):
        raise ProjectError("This file has no format version; it isn't an RRIPP project.")
    if version > FORMAT_VERSION:
        raise ProjectError(
            f"This project was saved by a newer RRIPP (format {version}); update the app to open it."
        )
    if version == 1:  # one design → a project with one page (M14)
        data = {"format_version": 2, "project": {"pages": [data["document"]]}}
    return data


def _as_project(doc_or_project) -> Project:
    return doc_or_project if isinstance(doc_or_project, Project) else Project(pages=[doc_or_project])


def to_bytes(doc_or_project, store: AssetStore) -> bytes:
    project = _as_project(doc_or_project)
    used = project.referenced_assets()
    missing = [a for a in used if not store.has(a) or not any(a in p.assets for p in project.pages)]
    if missing:
        raise ProjectError(f"{len(missing)} image(s) used by this project are missing from memory.")

    pruned = Project(pages=[copy.copy(p) for p in project.pages])
    for page in pruned.pages:
        mine = page.referenced_assets()
        page.assets = {k: v for k, v in page.assets.items() if k in mine}
    payload = {"format_version": FORMAT_VERSION, "project": project_to_dict(pruned)}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("document.json", json.dumps(payload, indent=1), zipfile.ZIP_DEFLATED)
        for asset_id in sorted(used):
            data, ext = store.raw(asset_id)
            # Images are already compressed; storing avoids wasted CPU.
            zf.writestr(f"assets/{asset_id}{ext}", data, zipfile.ZIP_STORED)
    return buf.getvalue()


def save(path: str, doc_or_project, store: AssetStore) -> None:
    atomic_write(path, to_bytes(doc_or_project, store))


def from_bytes(data: bytes) -> tuple[Document, AssetStore]:
    """The first page (single-page callers and older code)."""
    project, store = project_from_bytes(data)
    return project.pages[0], store


def project_from_bytes(data: bytes) -> tuple[Project, AssetStore]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ProjectError("This file isn't a valid RRIPP project.") from exc
    with zf:
        try:
            payload = json.loads(zf.read("document.json"))
        except KeyError as exc:
            raise ProjectError("This project is missing document.json.") from exc
        except json.JSONDecodeError as exc:
            raise ProjectError("This project's document.json is corrupted.") from exc
        payload = migrate(payload)
        project = project_from_dict(payload["project"])

        store = AssetStore()
        names = set(zf.namelist())
        for page in project.pages:
            for asset_id, info in page.assets.items():
                if store.has(asset_id):
                    continue  # shared by several pages: stored once
                name = f"assets/{asset_id}{info.ext}"
                if name not in names:
                    raise ProjectError(f"The image '{info.name or asset_id[:8]}' is missing from this project.")
                stored = store.add_bytes(zf.read(name), info.ext, info.name)
                if stored.id != asset_id:
                    raise ProjectError(f"The image '{info.name or asset_id[:8]}' in this project is corrupted.")

    for page in project.pages:
        missing = page.referenced_assets() - set(page.assets)
        if missing:
            raise ProjectError(f"{len(missing)} layer image(s) are missing from this project.")
    return project, store


def _read(path: str) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as exc:
        raise ProjectError(f"Could not open the project: {exc.strerror}.") from exc


def load(path: str) -> tuple[Document, AssetStore]:
    """The first page of a project (see load_project for all of them)."""
    return from_bytes(_read(path))


def load_project(path: str) -> tuple[Project, AssetStore]:
    return project_from_bytes(_read(path))
