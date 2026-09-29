"""M13: filling a selection: regions, quick fill, clean-plate fill, AI fill plumbing
(with a stand-in model), the fill layer lining up exactly, Grab as one undo step.
With RRIPP_TEST_LAMA=1 (set in the Windows build) the real LaMa model is downloaded,
verified and run."""

import copy
import os

import cv2
import numpy as np
import pytest

from lookbox.ai import lama
from lookbox.commands import edits
from lookbox.core.assets import AssetStore
from lookbox.core.io import images
from lookbox.core.masks import fill as F
from lookbox.core.masks.ops import encode_mask_png
from lookbox.core.model import (Adjustments, Document, ImageLayer, LayerMask, Size, Transform)
from lookbox.core.render.pipeline import render


def _gradient(h=120, w=160):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([xx / w, yy / h, 0.5 + 0 * xx], axis=-1).astype(np.float32)
    return rgb


def _blob(h=120, w=160, c=(80, 60), r=12):
    m = np.zeros((h, w), np.float32)
    cv2.circle(m, c, r, 1.0, -1)
    return m


def test_regions_merge_near_blobs_and_keep_far_ones_apart():
    m = _blob(r=6, c=(30, 30)) + _blob(r=6, c=(50, 30)) + _blob(r=6, c=(140, 100))
    boxes = F.regions(np.clip(m, 0, 1), min_context=8, context_ratio=0.5)
    assert len(boxes) == 2
    big = boxes[0]
    assert big[0] <= 24 - 8 and big[2] >= 56 + 8  # the two near blobs share one box
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= 160 and 0 <= y0 < y1 <= 120


def test_quick_fill_repairs_a_smooth_gradient_and_leaves_the_rest_alone():
    rgb = _gradient()
    damaged = rgb.copy()
    hole = _blob()
    damaged[hole > 0] = (1.0, 0.0, 1.0)  # an object in the way
    out = F.quick_fill(damaged, hole)
    assert np.abs(out[hole > 0] - rgb[hole > 0]).mean() < 0.03
    far = F.coverage(hole) == 0
    assert np.array_equal(out[far], damaged[far])


def test_plate_fill_is_exact_and_checks_size():
    rgb = _gradient()
    plate = np.random.default_rng(0).random((120, 160, 4)).astype(np.float32)
    hole = _blob()
    out = F.plate_fill(rgb, plate, hole)
    assert np.allclose(out[hole > 0], plate[..., :3][hole > 0])
    with pytest.raises(ValueError, match="same size"):
        F.plate_fill(rgb, plate[:10], hole)


def test_lama_wrapper_feeds_512_and_reads_0_255_output():
    seen = {}

    def fake_run(feeds):
        seen.update({k: v.shape for k, v in feeds.items()})
        assert feeds["mask"].max() == 1.0 and set(np.unique(feeds["mask"])) <= {0.0, 1.0}
        img = feeds["image"][0].transpose(1, 2, 0)
        filled = np.where(feeds["mask"][0, 0][..., None] > 0, 0.25, img)  # "AI": grey in the hole
        return (filled.transpose(2, 0, 1)[None] * 255.0).astype(np.float32)

    rgb = _gradient()
    hole = _blob()
    out = lama.fill(fake_run, rgb, hole)
    assert seen == {"image": (1, 3, 512, 512), "mask": (1, 1, 512, 512)}
    assert np.abs(out[hole > 0] - 0.25).max() < 0.02
    far = F.coverage(hole) == 0
    assert np.array_equal(out[far], rgb[far])


def _doc_with_image(store, t=Transform(x=100, y=70), crop=None, adjust=None):
    rgba = np.concatenate([_gradient(), np.ones((120, 160, 1), np.float32)], axis=-1)
    info = store.add_bytes(images.encode_png(rgba, bits=16), ".png", "g.png")
    doc = Document(canvas=Size(w=220, h=160))
    layer = ImageLayer(source=info.id, transform=t, crop=crop, adjust=adjust or Adjustments())
    edits.AddLayer(layer, asset=info).apply(doc)
    return doc, layer, rgba


@pytest.mark.parametrize("t", [Transform(x=100, y=70), Transform(x=110, y=80, scale_x=0.7, scale_y=0.7,
                                                                   rotation_deg=25, flip_h=True)])
def test_fill_layer_lines_up_with_the_original(t):
    store = AssetStore()
    doc, layer, rgba = _doc_with_image(store, t=t, adjust=Adjustments(vignette=50))
    hole = _blob()
    red = rgba[..., :3].copy()
    red[...] = (1.0, 0.0, 0.0)
    before = render(doc, store)
    patch = F.make_patch(F.blend_in(rgba[..., :3], red, hole), hole)
    info = store.add_bytes(images.encode_png(patch, bits=16), ".png", "fill")
    e = edits.fill_layer(doc, layer.id, info)
    e.apply(doc)
    after = render(doc, store)
    changed = np.abs(after - before).max(axis=2) > 0.05
    # The change sits where the blob is on the canvas: project the blob through the same transform.
    from lookbox.core.render.transform import warp_to_canvas
    blob = np.zeros((120, 160, 4), np.float32)
    blob[..., 3] = F.coverage(hole)  # the selection plus the fill's soft edge
    placed, (x0, y0) = warp_to_canvas(blob, t, 220, 160)
    expected = np.zeros((160, 220), bool)
    expected[y0:y0 + placed.shape[0], x0:x0 + placed.shape[1]] = placed[..., 3] > 0.15
    inter = (changed & expected).sum()
    assert inter / changed.sum() > 0.95 and inter / expected.sum() > 0.8  # in place, nowhere else
    assert doc.layers[1].adjust == layer.adjust and doc.layers[1].name.endswith("fill")
    e.revert(doc)
    assert len(doc.layers) == 1 and info.id not in doc.assets


def test_grab_is_one_step_and_uses_a_mask_on_transparent_renders():
    store = AssetStore()
    doc, layer, rgba = _doc_with_image(store)
    hole = _blob()
    minfo = store.add_bytes(encode_mask_png(hole), ".png", "mask")
    set_mask = edits.SetMask(layer.id, None, LayerMask(asset=minfo.id), asset=minfo)
    patch = F.make_patch(rgba[..., :3], hole)
    pinfo = store.add_bytes(images.encode_png(patch, bits=16), ".png", "fill")
    before = copy.deepcopy(doc)
    batch = edits.grab(doc, layer.id, set_mask, pinfo)
    batch.apply(doc)
    names = [lay.name for lay in doc.layers]
    assert names == [layer.name, f"{layer.name} fill", f"{layer.name} grab"]
    assert doc.layers[2].mask == LayerMask(asset=minfo.id) and doc.layers[0].mask is None
    batch.revert(doc)
    assert doc == before
    # No patch (object on transparency): the original hides the object instead.
    b2 = edits.grab(doc, layer.id, set_mask, None)
    b2.apply(doc)
    assert doc.layers[0].mask.invert and not doc.layers[1].mask.invert


def test_on_transparency():
    alpha = np.zeros((60, 60), np.float32)
    alpha[20:40, 20:40] = 1.0
    obj = np.zeros((60, 60), np.float32)
    obj[22:38, 22:38] = 1.0
    assert F.on_transparency(alpha, np.pad(obj[2:-2, 2:-2], 2) * 0 + (alpha > 0)) is True
    assert F.on_transparency(np.ones((60, 60), np.float32), obj) is False


@pytest.mark.skipif(os.environ.get("RRIPP_TEST_LAMA") != "1", reason="set RRIPP_TEST_LAMA=1 to download + run LaMa")
def test_real_lama_fills_a_gradient():
    from lookbox.ai.registry import download, is_ready, load_registry, model_path
    from lookbox.ai.runtime import ModelManager

    spec = load_registry()["lama"]
    if not is_ready(spec):
        download(spec)  # verifies size + sha256
    mgr = ModelManager()
    rgb = _gradient(240, 320)
    damaged = rgb.copy()
    hole = _blob(240, 320, c=(160, 120), r=30)
    damaged[hole > 0] = (1.0, 0.0, 1.0)
    out = lama.fill(lambda feeds: mgr.run(model_path(spec), feeds), damaged, hole)
    err = np.abs(out[hole > 0] - rgb[hole > 0]).mean()
    assert err < 0.08, f"mean error in the hole {err:.3f} ({mgr.provider})"
