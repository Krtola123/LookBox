"""M14: projects with pages: page edits and undo, resizing, shared assets,
copy/paste between pages, file format v2 (and v1 files opening as one page)."""

import copy
import io
import json
import zipfile

import numpy as np
import pytest

from lookbox.commands import edits
from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.io import images
from lookbox.core.model import (Document, ImageLayer, Project, Size, TextLayer, Transform,
                                document_to_dict)
from lookbox.core.render.pipeline import render
from tests.helpers import make_rgba


def _project(store):
    info = store.add_bytes(images.encode_png(make_rgba(20, 10, seed=1, alpha=False)), ".png", "r.png")
    p1 = Document(canvas=Size(w=40, h=30))
    edits.AddLayer(ImageLayer(source=info.id, transform=Transform(x=20, y=15)), asset=info).apply(p1)
    return Project(pages=[p1]), info


def test_add_duplicate_move_remove_pages_undo():
    store = AssetStore()
    project, info = _project(store)
    before = copy.deepcopy(project)
    dup = edits.duplicate_page(project, project.pages[0].id)
    dup.apply(project)
    assert len(project.pages) == 2 and project.pages[1].id != project.pages[0].id
    assert project.pages[1].layers[0].id != project.pages[0].layers[0].id  # fresh layer ids
    assert project.pages[1].layers[0].source == info.id  # the same image, not a copy
    add = edits.AddPage(Document(canvas=Size(w=10, h=20)), index=0)
    add.apply(project)
    mv = edits.MovePage(project.pages[0].id, 2)
    mv.apply(project)
    assert project.pages[2].canvas == Size(w=10, h=20)
    rm = edits.RemovePage(project.pages[0].id)
    rm.apply(project)
    for e in (rm, mv, add, dup):
        e.revert(project)
    assert project == before
    with pytest.raises(ValueError):
        edits.RemovePage(project.pages[0].id).apply(project)  # never zero pages


def test_resize_page_keeps_the_layout_centred():
    doc = Document(canvas=Size(w=100, h=100))
    edits.AddLayer(TextLayer(transform=Transform(x=50, y=50))).apply(doc)
    e = edits.SetCanvas(doc.canvas, Size(w=100, h=160))
    e.apply(doc)
    assert doc.canvas == Size(w=100, h=160) and doc.layers[0].transform.y == 80
    e.revert(doc)
    assert doc.canvas == Size(w=100, h=100) and doc.layers[0].transform.y == 50


def test_copy_paste_between_pages_brings_the_assets():
    store = AssetStore()
    project, info = _project(store)
    other = Document(canvas=Size(w=40, h=30))
    project.pages.append(other)
    layer = project.pages[0].layers[0]
    paste = edits.paste_layer(other, layer, edits.layer_assets(project.pages[0], layer), offset=5)
    paste.apply(other)
    assert other.layers[0].source == info.id and info.id in other.assets
    assert other.layers[0].id != layer.id and other.layers[0].transform.x == layer.transform.x + 5
    paste.revert(other)
    assert other.layers == [] and other.assets == {}


def test_project_round_trip_shares_assets_and_renders_each_page(tmp_path):
    store = AssetStore()
    project, info = _project(store)
    edits.duplicate_page(project, project.pages[0].id).apply(project)
    project.pages[1].name = "Square"
    project.pages[1].canvas = Size(w=30, h=30)
    path = str(tmp_path / "p.rripp")
    serialize.save(path, project, store)
    with zipfile.ZipFile(path) as zf:
        assert sum(n.startswith("assets/") for n in zf.namelist()) == 1  # one copy for both pages
        assert json.loads(zf.read("document.json"))["format_version"] == 2
    p2, store2 = serialize.load_project(path)
    assert p2 == project
    for a, b in zip(project.pages, p2.pages):
        assert np.array_equal(render(a, store), render(b, store2))
    first, _ = serialize.load(path)
    assert first == project.pages[0]


def test_version_1_files_open_as_one_page(tmp_path):
    store = AssetStore()
    project, info = _project(store)
    doc = project.pages[0]
    old = {k: v for k, v in document_to_dict(doc).items() if k not in ("id", "name")}  # as 1.2 wrote it
    payload = {"format_version": 1, "document": old}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("document.json", json.dumps(payload))
        data, ext = store.raw(info.id)
        zf.writestr(f"assets/{info.id}{ext}", data)
    p2, _ = serialize.project_from_bytes(buf.getvalue())
    assert len(p2.pages) == 1 and p2.pages[0].layers == doc.layers and p2.pages[0].id
