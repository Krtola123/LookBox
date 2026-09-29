"""M10: .cube LUT filters (parse, apply, strength), bundled looks, LUT assets,
layer filters and the whole-design grade (global adjust + LUT) in the pipeline."""

import glob
import os

import numpy as np
import pytest

from lookbox.commands import edits
from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.io import images
from lookbox.core.model import (Adjustments, Document, FillLayer, GradientStop, ImageLayer, LutRef, Size,
                                Transform, Fill, document_from_dict, document_to_dict)
from lookbox.core.render import lut as L
from lookbox.core.render.pipeline import render, render_key, render_layer
from tests.helpers import make_rgba

LUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lookbox", "luts")


def _cube(fn, n=9, title="t") -> bytes:
    return L.to_cube(fn(L.identity_table(n)).astype(np.float32), title).encode()


# ------------------------------------------------------------------ parsing


def test_parse_3d_with_comments_title_and_domain():
    text = "# made by hand\nTITLE \"Swap\"\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\nLUT_3D_SIZE 2\n\n" + \
        "\n".join(f"{b} {g} {r}  # swapped" for b in (0, 1) for g in (0, 1) for r in (0, 1))
    lut = L.parse_cube(text)
    assert lut.dims == 3 and lut.size == 2 and lut.title == "Swap"
    x = np.array([[0.2, 0.5, 0.9]], np.float32)
    assert np.allclose(L.apply(x, lut), [[0.9, 0.5, 0.2]], atol=1e-6)  # red↔blue, exact (linear map)


def test_parse_1d_and_input_range():
    rows = "\n".join(f"{v:.4f} {v:.4f} {v:.4f}" for v in np.linspace(1, 0, 16))
    lut = L.parse_cube("LUT_1D_SIZE 16\nLUT_1D_INPUT_RANGE 0 1\n" + rows)
    assert lut.dims == 1
    x = np.array([[0.0, 0.25, 1.0]], np.float32)
    assert np.allclose(L.apply(x, lut), 1 - x, atol=1e-4)


@pytest.mark.parametrize("text,msg", [
    ("0 0 0\n", "no LUT_3D_SIZE"),
    ("LUT_3D_SIZE 2\n0 0 0\n", "should have 8"),
    ("LUT_3D_SIZE 1\n0 0 0\n", "isn't supported"),
    ("LUT_3D_SIZE 2\n" + "0 0\n" * 8, "3 numbers"),
    ("LUT_3D_SIZE x\n", "can't be read"),
    ("DOMAIN_MIN 1 1 1\nDOMAIN_MAX 0 0 0\nLUT_3D_SIZE 2\n" + "0 0 0\n" * 8, "DOMAIN_MAX"),
])
def test_bad_files_say_what_is_wrong(text, msg):
    with pytest.raises(L.LutError, match=msg):
        L.parse_cube(text)


# ------------------------------------------------------------------ applying


def test_identity_is_exact_and_linear_maps_are_exact_between_grid_points():
    x = np.random.default_rng(0).random((64, 80, 3), dtype=np.float32)
    assert np.allclose(L.apply(x, L.parse_cube(_cube(lambda t: t))), x, atol=1e-6)
    m = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.0, 0.3, 0.7]], np.float32)
    lin = L.parse_cube(_cube(lambda t: t @ m.T, n=5))
    assert np.allclose(L.apply(x, lin), x @ m.T, atol=1e-5)  # tetrahedral is exact for linear maps


def test_strength_blends_and_domain_clamps():
    inv = L.parse_cube(_cube(lambda t: 1 - t))
    x = np.full((2, 2, 3), 0.2, np.float32)
    assert np.allclose(L.apply(x, inv, 0.0), x)
    assert np.allclose(L.apply(x, inv, 0.25), 0.2 + 0.25 * (0.8 - 0.2), atol=1e-6)
    assert np.allclose(L.apply(np.array([[1.5, -0.2, 0.5]], np.float32), inv), [[0.0, 1.0, 0.5]], atol=1e-6)


def test_big_images_go_through_in_chunks():
    inv = L.parse_cube(_cube(lambda t: 1 - t))
    x = np.random.default_rng(1).random((900, 700, 3), dtype=np.float32)  # > one chunk
    assert np.allclose(L.apply(x, inv), 1 - x, atol=1e-6)


def test_bundled_looks_load_and_keep_greys_in_order():
    paths = sorted(glob.glob(os.path.join(LUT_DIR, "*.cube")))
    assert len(paths) >= 6
    ramp = np.repeat(np.linspace(0, 1, 64, dtype=np.float32)[:, None], 3, axis=1)
    for p in paths:
        lut = L.parse_cube(open(p, "rb").read())
        assert lut.title, p
        y = L.apply(ramp, lut).mean(axis=1)
        assert np.all(np.diff(y) >= -1e-4), p  # a look never makes a brighter grey darker


# ------------------------------------------------------------------ assets + model


def test_lut_assets_store_save_and_load(tmp_path):
    store = AssetStore()
    info = store.add_bytes(_cube(lambda t: 1 - t), ".cube", "Invert.cube")
    assert info.width == 9 and store.lut(info.id).size == 9
    with pytest.raises(L.LutError):
        store.add_bytes(b"not a lut", ".cube")
    src = store.add_bytes(images.encode_png(make_rgba(8, 6, seed=1, alpha=False)), ".png", "a.png")
    doc = Document(canvas=Size(w=8, h=6))
    layer = ImageLayer(source=src.id, transform=Transform(x=4, y=3))
    edits.AddLayer(layer, asset=src).apply(doc)
    edits.SetLut(layer.id, None, LutRef(asset=info.id, strength=0.5, name="Invert"), asset=info).apply(doc)
    edits.SetAdjustments(None, doc.global_adjust, Adjustments(contrast=20)).apply(doc)
    edits.SetLut(None, None, LutRef(asset=info.id, strength=0.3)).apply(doc)
    assert doc.referenced_assets() == {src.id, info.id}
    assert document_from_dict(document_to_dict(doc)) == doc
    path = str(tmp_path / "g.rripp")
    serialize.save(path, doc, store)
    doc2, store2 = serialize.load(path)
    assert doc2 == doc and np.array_equal(render(doc2, store2), render(doc, store))


def test_set_lut_undo_and_merge():
    store = AssetStore()
    info = store.add_bytes(_cube(lambda t: 1 - t), ".cube", "Invert.cube")
    doc = Document()
    doc.layers.append(FillLayer())
    lid = doc.layers[0].id
    e1 = edits.SetLut(lid, None, LutRef(asset=info.id), asset=info)
    e1.apply(doc)
    e2 = edits.SetLut(lid, doc.layers[0].lut, LutRef(asset=info.id, strength=0.4), merge_key="k")
    e2.apply(doc)
    e3 = edits.SetLut(lid, doc.layers[0].lut, LutRef(asset=info.id, strength=0.2), merge_key="k")
    e3.apply(doc)
    assert e2.merge(e3) and not e1.merge(e2)  # a new .cube never merges
    e2.revert(doc)
    assert doc.layers[0].lut.strength == 1.0
    e1.revert(doc)
    assert doc.layers[0].lut is None and info.id not in doc.assets


# ------------------------------------------------------------------ pipeline


def _two_layer_doc(store):
    doc = Document(canvas=Size(w=40, h=30), background=(0.1, 0.2, 0.3, 1.0))
    fill = Fill(kind="linear", angle_deg=0, stops=[GradientStop(pos=0, color=(0.9, 0.2, 0.1, 1)),
                                                   GradientStop(pos=1, color=(0.1, 0.4, 0.9, 1))])
    edits.AddLayer(FillLayer(fill=fill, width=20, height=30, transform=Transform(x=10, y=15))).apply(doc)
    src = store.add_bytes(images.encode_png(make_rgba(16, 12, seed=3)), ".png", "b.png")
    edits.AddLayer(ImageLayer(source=src.id, transform=Transform(x=28, y=14), opacity=0.8), asset=src).apply(doc)
    return doc


def test_layer_filter_is_applied_and_changes_the_render_key():
    store = AssetStore()
    info = store.add_bytes(_cube(lambda t: 1 - t), ".cube")
    layer = FillLayer(width=4, height=4, fill=Fill(kind="solid", stops=[GradientStop(pos=0, color=(0.2, 0.3, 0.4, 1))]))
    before = render_key(layer)
    layer.lut = LutRef(asset=info.id, strength=1.0)
    assert render_key(layer) != before
    px = render_layer(layer, store)
    assert np.allclose(px[0, 0, :3], (0.8, 0.7, 0.6), atol=1e-5)


def test_global_grade_acts_on_the_flattened_image():
    store = AssetStore()
    doc = _two_layer_doc(store)
    plain = render(doc, store)
    info = store.add_bytes(_cube(lambda t: 1 - t), ".cube")
    doc.global_lut = LutRef(asset=info.id, strength=1.0)
    graded = render(doc, store)
    assert np.allclose(graded[..., :3], 1 - plain[..., :3], atol=1e-5)  # the composite, not each layer
    assert np.array_equal(graded[..., 3], plain[..., 3])
    doc.global_lut = None
    doc.global_adjust = Adjustments(brightness=40)
    assert render(doc, store)[..., :3].mean() > plain[..., :3].mean() + 0.02


def test_preview_render_matches_the_export_downscaled():
    import cv2

    store = AssetStore()
    doc = _two_layer_doc(store)
    info = store.add_bytes(open(os.path.join(LUT_DIR, "warm_film.cube"), "rb").read(), ".cube")
    doc.global_lut = LutRef(asset=info.id, strength=0.8)
    doc.global_adjust = Adjustments(contrast=30, vignette=40)
    full = render(doc, store, 1.0)
    prev = render(doc, store, 0.5, preview=True)
    ref = cv2.resize(full, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(prev - ref).mean() < 0.02
