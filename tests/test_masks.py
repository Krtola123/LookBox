"""M6 core: layer masks, edge controls, refine edge, brush, and the mask edits."""

import copy

import cv2
import numpy as np
import pytest

from lookbox.commands import edits
from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.io import images
from lookbox.core.masks import ops as M
from lookbox.core.model import (Document, FillLayer, ImageLayer, LayerMask, Size, Transform, document_from_dict,
                                document_to_dict)
from lookbox.core.render.pipeline import render, render_key, render_layer, unpremultiply


def _disc_mask(w=120, h=80, r=25):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    d = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    return np.clip(r - d + 0.5, 0, 1).astype(np.float32)


def _doc_with_mask(mask, **mask_kw):
    store = AssetStore()
    img = np.ones((mask.shape[0], mask.shape[1], 4), np.float32)
    img[:, :, 0] = 0.8
    info = store.add_bytes(images.encode_png(img), ".png", "photo.png")
    minfo = store.add_bytes(M.encode_mask_png(mask), ".png", "mask")
    layer = ImageLayer(id="L", source=info.id, transform=Transform(x=mask.shape[1] / 2, y=mask.shape[0] / 2),
                       mask=LayerMask(asset=minfo.id, **mask_kw))
    doc = Document(canvas=Size(w=mask.shape[1], h=mask.shape[0]), assets={info.id: info, minfo.id: minfo},
                   layers=[layer])
    return doc, store


def test_mask_png_roundtrip_keeps_16bit_precision():
    m = np.linspace(0, 1, 300 * 7, dtype=np.float32).reshape(7, 300)
    store = AssetStore()
    info = store.add_bytes(M.encode_mask_png(m), ".png", "mask")
    back = M.mask_from_pixels(store.pixels(info.id))
    assert np.abs(back - m).max() <= 0.5 / 65535 + 1e-7


def test_mask_multiplies_alpha_and_changes_render_key():
    mask = _disc_mask()
    doc, store = _doc_with_mask(mask)
    out = render(doc, store)
    assert np.allclose(out[:, :, 3], mask, atol=1e-4)
    k = render_key(doc.layers[0])
    doc.layers[0].mask.feather = 3
    assert render_key(doc.layers[0]) != k
    doc.layers[0].mask.invert = True
    doc.layers[0].mask.feather = 0
    assert np.allclose(render(doc, store)[:, :, 3], 1 - mask, atol=1e-4)


def test_grow_shrink_and_feather():
    mask = _disc_mask(r=20)
    area = mask.sum()
    grown, shrunk = M.grow_shrink(mask, 5), M.grow_shrink(mask, -5)
    assert grown.sum() > area * 1.4 and shrunk.sum() < area * 0.7
    soft = M.feather(mask, 10)
    assert soft.sum() == pytest.approx(area, rel=0.03)  # feathering moves the edge, not the area
    assert (soft > 0.02).sum() > (mask > 0.02).sum()  # …but spreads it
    assert np.array_equal(M.grow_shrink(mask, 0.3), mask)


@pytest.mark.parametrize("kw", [dict(shift=6), dict(shift=-6), dict(feather=10), dict(feather=6, shift=4, invert=True)])
def test_mask_edge_controls_preview_matches_export(kw):
    mask = _disc_mask(w=240, h=160, r=50)
    doc, store = _doc_with_mask(mask, **kw)
    layer = doc.layers[0]
    full = unpremultiply(render_layer(layer, store, 1.0))[:, :, 3]
    half = unpremultiply(render_layer(layer, store, 0.5))[:, :, 3]
    ref = cv2.resize(full, (half.shape[1], half.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(ref - half).mean() < 0.01 and (ref - half).sum() / ref.sum() < 0.03


def test_mask_follows_layer_crop():
    mask = np.zeros((40, 60), np.float32)
    mask[:, 30:] = 1.0  # right half visible
    doc, store = _doc_with_mask(mask)
    doc.layers[0].crop = (20, 0, 20, 40)  # straddles the mask edge at x = 30
    alpha = render_layer(doc.layers[0], store)[:, :, 3]
    assert alpha.shape == (40, 20) and alpha[:, :10].max() == 0 and alpha[:, 10:].min() == 1


def test_refine_edge_snaps_the_mask_to_the_photos_real_edge():
    """What the guided filter is for: a model's upsampled mask is soft and slightly off;
    refined, it gets a sharp step exactly where the photo's edge is (hair, fur, rims)."""
    h, w = 256, 512
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[:, 256:] = 1.0  # the real edge is at x = 256
    truth = (np.arange(w) >= 256).astype(np.float32)[None, :].repeat(h, 0)
    coarse = cv2.GaussianBlur(np.roll(truth, 3, axis=1), (0, 0), 3)  # soft AND 3 px off
    refined = M.refine_edge(coarse, rgb)
    step = lambda m: m[128, 256] - m[128, 255]  # noqa: E731
    assert step(refined) > 2.5 * step(coarse)  # a real step appears exactly at the photo's edge
    assert refined.min() >= 0 and refined.max() <= 1
    # Deliberately NOT asserted: lower overall error. On clean, noisy edges it doesn't
    # have it (measured), which is why refining is optional and off by default.


def test_brush_restore_and_erase_with_soft_edges():
    m = np.zeros((50, 50), np.float32)
    rect = M.stamp(m, 25, 25, 10, 0.5, restore=True)
    assert m[25, 25] == 1.0 and m[25, 44] == 0.0 and 0 < m[25, 33] < 1  # soft falloff
    assert rect == (15, 15, 36, 36)
    before = m.copy()
    M.stamp(m, 25, 25, 10, 0.5, restore=True)
    assert np.array_equal(m, before)  # dabbing twice doesn't build up
    M.stamp(m, 25, 25, 4, 1.0, restore=False)
    assert m[25, 25] == 0.0 and m[25, 32] == before[25, 32]
    assert M.stamp(m, -100, -100, 5, 0.5, restore=True) is None  # off the mask: no-op


def test_brush_stroke_has_no_gaps_at_speed():
    m = np.zeros((20, 400), np.float32)
    M.stroke(m, (10, 10), (390, 10), 6, 0.8, restore=True)
    assert m[10, 10:390].min() > 0.99


def test_set_mask_edit_adds_and_removes_its_asset():
    doc = Document()
    edits.AddLayer(ImageLayer(id="L", source="s")).apply(doc)
    doc.assets["s"] = None
    from lookbox.core.model import AssetInfo
    info = AssetInfo(id="m1", ext=".png", width=4, height=4)
    e = edits.SetMask("L", None, LayerMask(asset="m1"), asset=info)
    before = copy.deepcopy(doc)
    e.apply(doc)
    assert doc.layer("L").mask.asset == "m1" and "m1" in doc.assets
    e.revert(doc)
    assert doc == before
    with pytest.raises(ValueError):
        d2 = Document(layers=[FillLayer(id="F")])
        edits.SetMask("F", None, LayerMask(asset="m1")).apply(d2)


def test_edge_drag_merges_but_new_mask_images_do_not():
    a = edits.SetMask("L", LayerMask(asset="m"), LayerMask(asset="m", shift=2), merge_key="d")
    b = edits.SetMask("L", LayerMask(asset="m", shift=2), LayerMask(asset="m", shift=5), merge_key="d")
    assert a.merge(b) and a.new.shift == 5 and a.old.shift == 0
    from lookbox.core.model import AssetInfo
    c = edits.SetMask("L", None, LayerMask(asset="n"), asset=AssetInfo(id="n", ext=".png", width=1, height=1))
    assert not a.merge(c)


def test_extract_to_new_layer_is_one_undo_step():
    mask = _disc_mask()
    doc, store = _doc_with_mask(mask)
    before = copy.deepcopy(doc)
    e = edits.extract_to_layer(doc, "L")
    e.apply(doc)
    assert len(doc.layers) == 2 and doc.layers[0].mask is None and doc.layers[1].mask is not None
    assert doc.layers[1].source == doc.layers[0].source  # no pixels copied
    out = render(doc, store)
    assert out[:, :, 3].min() == pytest.approx(1.0)  # original shows everything again
    e.revert(doc)
    assert doc == before
    with pytest.raises(ValueError):
        edits.extract_to_layer(before, "L") if False else edits.extract_to_layer(
            Document(layers=[ImageLayer(id="X")]), "X")


def test_batch_rolls_back_on_failure():
    doc = Document(layers=[ImageLayer(id="A")])
    before = copy.deepcopy(doc)
    bad = edits.Batch([edits.SetLayerProps("A", name="renamed"), edits.RemoveLayer("missing")], text="x")
    with pytest.raises(KeyError):
        bad.apply(doc)
    assert doc == before


def test_mask_survives_save_and_old_files_load():
    doc, store = _doc_with_mask(_disc_mask(), shift=2.5, feather=4)
    assert doc.layers[0].mask.asset in doc.referenced_assets()
    doc2, store2 = serialize.from_bytes(serialize.to_bytes(doc, store))
    assert doc2 == doc and np.array_equal(render(doc2, store2), render(doc, store))
    d = document_to_dict(doc)
    del d["layers"][0]["mask"]
    assert document_from_dict(d).layers[0].mask is None


def test_remove_background_keeps_the_background_as_its_own_layer():
    from lookbox.core.model import AssetInfo, DropShadow, Effects
    mask = _disc_mask()
    doc, store = _doc_with_mask(np.ones_like(mask))
    layer = doc.layers[0]
    layer.mask = None
    layer.effects = Effects(shadow=DropShadow())
    minfo = store.add_bytes(M.encode_mask_png(mask), ".png", "mask")
    before = copy.deepcopy(doc)
    e = edits.remove_background(doc, "L", LayerMask(asset=minfo.id), minfo, keep_background=True)
    e.apply(doc)
    assert [ly.name for ly in doc.layers] == [f"{layer.name} background", layer.name]  # background below
    bg, subject = doc.layers
    assert bg.source == subject.source and bg.mask.asset == subject.mask.asset  # no pixels copied
    assert bg.mask.invert and not subject.mask.invert and not bg.effects.active()
    no_fx = copy.deepcopy(subject)
    no_fx.effects = Effects()  # compare the cut itself, not the subject's shadow
    subject_alpha = render_layer(no_fx, store)[:, :, 3]
    bg_alpha = render_layer(bg, store)[:, :, 3]
    assert np.allclose(subject_alpha + bg_alpha, 1.0, atol=1e-4)  # together they're the whole photo
    e.revert(doc)
    assert doc == before  # one undo step
    only = edits.remove_background(doc, "L", LayerMask(asset=minfo.id), minfo, keep_background=False)
    only.apply(doc)
    assert len(doc.layers) == 1 and doc.layers[0].mask.asset == minfo.id
