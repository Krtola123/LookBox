import copy
import io
import json
import zipfile

import numpy as np
import pytest

from lookbox.commands import edits
from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.model import Document, ImageLayer, Size, Transform
from lookbox.core.render.pipeline import render
from tests.helpers import tmp_png


def _doc(tmp_path):
    store = AssetStore()
    doc = Document(canvas=Size(w=64, h=48))
    for i in range(3):
        info = store.add_file(tmp_png(tmp_path, f"i{i}.png", 20 + i, 16, seed=i))
        layer = ImageLayer(name=info.name, source=info.id,
                           transform=Transform(x=10 + 15 * i, y=20, rotation_deg=10 * i))
        edits.AddLayer(layer, asset=info).apply(doc)
    return doc, store


def test_roundtrip_document_and_pixels(tmp_path):
    doc, store = _doc(tmp_path)
    path = str(tmp_path / "p.lookbox")
    serialize.save(path, doc, store)
    doc2, store2 = serialize.load(path)
    assert doc2 == doc
    assert np.array_equal(render(doc2, store2), render(doc, store))


def test_unreferenced_assets_pruned_on_save_without_touching_doc(tmp_path):
    doc, store = _doc(tmp_path)
    removed_source = doc.layers[0].source
    edits.RemoveLayer(doc.layers[0].id).apply(doc)
    before = copy.deepcopy(doc)
    data = serialize.to_bytes(doc, store)
    assert doc == before  # saving must not mutate the document
    doc2, _ = serialize.from_bytes(data)
    assert removed_source not in doc2.assets


def test_newer_format_is_refused(tmp_path):
    doc, store = _doc(tmp_path)
    data = serialize.to_bytes(doc, store)
    zin = zipfile.ZipFile(io.BytesIO(data))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zout:
        for name in zin.namelist():
            blob = zin.read(name)
            if name == "document.json":
                payload = json.loads(blob)
                payload["format_version"] = 99
                blob = json.dumps(payload).encode()
            zout.writestr(name, blob)
    with pytest.raises(serialize.ProjectError, match="newer"):
        serialize.from_bytes(buf.getvalue())


def test_corrupted_asset_is_detected(tmp_path):
    doc, store = _doc(tmp_path)
    data = serialize.to_bytes(doc, store)
    zin = zipfile.ZipFile(io.BytesIO(data))
    buf = io.BytesIO()
    victim = f"assets/{doc.layers[0].source}.png"
    other = f"assets/{doc.layers[1].source}.png"
    with zipfile.ZipFile(buf, "w") as zout:
        for name in zin.namelist():
            zout.writestr(name, zin.read(other if name == victim else name))
    with pytest.raises(serialize.ProjectError, match="corrupted"):
        serialize.from_bytes(buf.getvalue())


def test_not_a_zip():
    with pytest.raises(serialize.ProjectError):
        serialize.from_bytes(b"hello")


def test_failed_save_leaves_existing_file_intact(tmp_path):
    doc, store = _doc(tmp_path)
    path = str(tmp_path / "p.lookbox")
    serialize.save(path, doc, store)
    good = open(path, "rb").read()
    broken = copy.deepcopy(doc)
    broken.assets.clear()  # makes to_bytes fail before anything is written
    with pytest.raises(serialize.ProjectError):
        serialize.save(path, broken, store)
    assert open(path, "rb").read() == good


def test_adjustments_roundtrip_and_old_files_load(tmp_path):
    from lookbox.core.model import Adjustments, ColorBand, document_from_dict, document_to_dict
    doc, store = _doc(tmp_path)
    edits.SetAdjustments(doc.layers[0].id, doc.layers[0].adjust, Adjustments(
        brightness=12, invert=True, color_edit=[ColorBand(hue=33.5, hue_shift=-40)])).apply(doc)
    doc2, store2 = serialize.from_bytes(serialize.to_bytes(doc, store))
    assert doc2 == doc
    assert np.array_equal(render(doc2, store2), render(doc, store))
    # An M1/M2 file has no "adjust" key at all: it must load with neutral adjustments.
    d = document_to_dict(doc)
    for layer in d["layers"]:
        del layer["adjust"]
    assert all(layer.adjust == Adjustments() for layer in document_from_dict(d).layers)
