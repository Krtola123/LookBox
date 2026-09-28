import copy
from dataclasses import replace

import pytest

from lookbox.commands import edits
from lookbox.core.model import AssetInfo, Document, ImageLayer, Size, Transform


def _doc():
    doc = Document(canvas=Size(w=100, h=100))
    for i in range(3):
        info = AssetInfo(id=f"a{i}", ext=".png", width=10, height=10)
        doc.assets[info.id] = info
        doc.layers.append(ImageLayer(id=f"L{i}", name=f"L{i}", source=info.id,
                                     transform=Transform(x=10 * i, y=5)))
    return doc


def _check_do_undo(edit, doc):
    before = copy.deepcopy(doc)
    edit.apply(doc)
    after = copy.deepcopy(doc)
    assert after != before, "edit did nothing"
    edit.revert(doc)
    assert doc == before
    edit.apply(doc)  # redo must reproduce the same state
    assert doc == after


def test_add_layer_with_new_asset():
    doc = _doc()
    info = AssetInfo(id="new", ext=".png", width=1, height=1)
    _check_do_undo(edits.AddLayer(ImageLayer(id="N", source="new"), asset=info), doc)
    assert doc.layers[-1].id == "N" and "new" in doc.assets


def test_add_layer_does_not_remove_shared_asset_on_undo():
    doc = _doc()
    e = edits.AddLayer(ImageLayer(id="N", source="a0"), asset=doc.assets["a0"])
    e.apply(doc)
    e.revert(doc)
    assert "a0" in doc.assets


def test_remove_layer():
    _check_do_undo(edits.RemoveLayer("L1"), _doc())


def test_set_transform_and_merge():
    doc = _doc()
    old = doc.layer("L0").transform
    _check_do_undo(edits.SetTransform("L0", old, replace(old, x=50, rotation_deg=12)), doc)

    doc = _doc()
    t0 = replace(doc.layer("L0").transform)
    a = edits.SetTransform("L0", t0, replace(t0, x=t0.x + 1), merge_key="nudge")
    a.apply(doc)
    b = edits.SetTransform("L0", doc.layer("L0").transform, replace(t0, x=t0.x + 2), merge_key="nudge")
    b.apply(doc)
    assert a.merge(b)
    a.revert(doc)
    assert doc.layer("L0").transform == t0
    assert not a.merge(edits.SetTransform("L1", t0, t0))


def test_edit_does_not_alias_caller_objects():
    doc = _doc()
    t = replace(doc.layer("L0").transform, x=77)
    edits.SetTransform("L0", doc.layer("L0").transform, t).apply(doc)
    t.x = 999
    assert doc.layer("L0").transform.x == 77


@pytest.mark.parametrize("src,dst", [(0, 2), (2, 0), (1, 1), (0, 5)])
def test_reorder(src, dst):
    doc = _doc()
    edit = edits.ReorderLayer(f"L{src}", dst)
    before = copy.deepcopy(doc)
    edit.apply(doc)
    edit.revert(doc)
    assert doc == before


def test_set_layer_props():
    _check_do_undo(edits.SetLayerProps("L2", visible=False, locked=True, name="x"), _doc())
    with pytest.raises(ValueError):
        edits.SetLayerProps("L2", source="hack")


def test_reorder_from_list_orders():
    old = ["a", "b", "c", "d"]
    for new in (["b", "c", "a", "d"], ["d", "a", "b", "c"], ["a", "c", "b", "d"], ["a", "b", "d", "c"]):
        doc = Document()
        doc.layers = [ImageLayer(id=x) for x in old]
        e = edits.reorder_from(old, new)
        e.apply(doc)
        assert [layer.id for layer in doc.layers] == new
    assert edits.reorder_from(old, old) is None
    assert edits.reorder_from(old, ["b", "a", "d", "c"]) is None  # two moves


def test_duplicate():
    doc = _doc()
    e = edits.duplicate(doc, "L0")
    _check_do_undo(e, doc)
    assert doc.layers[1].source == "a0" and doc.layers[1].id != "L0"
