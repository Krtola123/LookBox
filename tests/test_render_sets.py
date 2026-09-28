"""M8: render sets (pass detection + attaching), ID pick with soft edges, lasso,
combining selections."""

import copy
import os

import cv2
import numpy as np
import pytest

from lookbox.commands import edits
from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.io import render_sets as RS
from lookbox.core.io.images import srgb_oetf
from lookbox.core.masks import idpick as P
from lookbox.core.masks import ops as M
from lookbox.core.masks.lasso import lasso_mask
from lookbox.core.model import Document, ImageLayer, Size
from tests.helpers import make_rgba, write_png

# ------------------------------------------------------------------ names


@pytest.mark.parametrize("beauty,other,kind", [
    ("shot.png", "shot_objectid.png", "object_id"),
    ("shot.png", "shot_Object ID.png", "object_id"),
    ("shot.png", "shot-ObjectID.exr", "object_id"),
    ("shot.png", "ObjectID_shot.png", "object_id"),
    ("shot.png", "shot_MaterialID.png", "material_id"),
    ("shot.png", "shot_matid.png", "material_id"),
    ("shot.png", "shot_Alpha Mask.png", "alpha"),
    ("shot_0001.png", "shot_objectid_0001.png", "object_id"),
    ("shot_0001.png", "shot_0001_objectid.png", "object_id"),
    ("shot.png", "shot_normal.png", None),
    ("shot.png", "other_objectid.png", None),
    ("shot.png", "shot.exr", None),
    ("shot_0001.png", "shot_objectid_0002.png", None),
])
def test_pass_names(beauty, other, kind):
    assert RS.pass_kind(beauty, other) == kind


def test_group_attaches_passes_among_the_files_and_from_the_folder(tmp_path):
    for n in ("a.png", "a_objectid.png", "a_matid.png", "b.png", "b_mask.png", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    p = lambda n: str(tmp_path / n)  # noqa: E731
    # Passes picked alongside their beauty don't become layers of their own.
    groups = RS.group([p("a.png"), p("a_objectid.png"), p("b.png")], scan_folders=False)
    assert groups == [(p("a.png"), {"object_id": p("a_objectid.png")}), (p("b.png"), {})]
    # Importing only the beauty finds its passes next to it.
    groups = RS.group([p("a.png")])
    assert groups == [(p("a.png"), {"object_id": p("a_objectid.png"), "material_id": p("a_matid.png")})]
    assert RS.group([p("b.png")]) == [(p("b.png"), {"alpha": p("b_mask.png")})]


def test_guess_kind_for_attaching_by_hand():
    assert RS.guess_kind("C:/x/whatever_MaterialID.png") == "material_id"
    assert RS.guess_kind("IDmap.png") == "object_id"
    assert RS.guess_kind("render.png") is None


# ------------------------------------------------------------------ attaching


def _store_with(tmp_path, arrays: dict):
    store = AssetStore()
    infos = {k: store.add_file(write_png(os.path.join(str(tmp_path), f"{k}.png"), v)) for k, v in arrays.items()}
    return store, infos


def test_prepare_drops_wrong_sizes_and_turns_alpha_pass_into_a_mask(tmp_path):
    beauty = make_rgba(40, 30, seed=1, alpha=False)
    grey = np.zeros((30, 40, 4), np.float32)
    grey[:, :20] = 1.0
    grey[:, :, 3] = 1.0
    store, infos = _store_with(tmp_path, {"beauty": beauty, "id": make_rgba(40, 30, seed=2),
                                          "small": make_rgba(20, 10, seed=3), "alpha": grey})
    keep, mask, warnings = RS.prepare(store, infos["beauty"], {"object_id": infos["id"],
                                                               "material_id": infos["small"],
                                                               "alpha": infos["alpha"]})
    assert keep == {"object_id": infos["id"]}
    assert len(warnings) == 1 and "Material ID" in warnings[0]
    m = M.mask_from_pixels(store.pixels(mask.id))
    assert m[:, :20].min() > 0.99 and m[:, 20:].max() < 0.01


def test_alpha_pass_is_not_applied_to_a_render_that_already_has_alpha(tmp_path):
    store, infos = _store_with(tmp_path, {"beauty": make_rgba(8, 8, seed=1), "alpha": make_rgba(8, 8, seed=2)})
    _, mask, _ = RS.prepare(store, infos["beauty"], {"alpha": infos["alpha"]})
    assert mask is None


def test_add_layer_with_passes_and_set_pass_undo_and_save(tmp_path):
    store, infos = _store_with(tmp_path, {"beauty": make_rgba(16, 12, seed=1), "id": make_rgba(16, 12, seed=2),
                                          "mat": make_rgba(16, 12, seed=3), "odd": make_rgba(5, 5, seed=4)})
    doc = Document(canvas=Size(w=16, h=12))
    layer = ImageLayer(source=infos["beauty"].id, passes={"object_id": infos["id"].id})
    add = edits.AddLayer(layer, asset=infos["beauty"], extra_assets=(infos["id"],))
    add.apply(doc)
    assert set(doc.assets) == {infos["beauty"].id, infos["id"].id}
    sp = edits.SetPass(layer.id, "material_id", infos["mat"])
    sp.apply(doc)
    assert doc.layers[0].passes == {"object_id": infos["id"].id, "material_id": infos["mat"].id}
    path = str(tmp_path / "p.lookbox")
    serialize.save(path, doc, store)
    doc2, store2 = serialize.load(path)
    assert doc2 == doc and store2.has(infos["mat"].id)
    sp.revert(doc)
    assert doc.layers[0].passes == {"object_id": infos["id"].id} and infos["mat"].id not in doc.assets
    with pytest.raises(ValueError, match="same size"):
        edits.SetPass(layer.id, "material_id", infos["odd"]).apply(doc)
    add.revert(doc)
    assert doc.layers == [] and doc.assets == {}


# ------------------------------------------------------------------ ID pick

SS, W, H = 8, 160, 120
COLS = {1: (1.0, 0.0, 0.0, 1.0), 2: (0.0, 0.5, 1.0, 1.0), 3: (0.2, 0.9, 0.2, 1.0)}


def _id_scene(background=(0.0, 0.0, 0.0, 0.0), linear=False):
    """An anti-aliased ID pass (rendered 8× and area-averaged, stored 8-bit) + exact coverage.
    `linear`: the renderer mixes in linear light and writes sRGB (colours are sRGB values)."""
    lab = np.zeros((H * SS, W * SS), np.uint8)
    cv2.circle(lab, (60 * SS, 60 * SS), 35 * SS, 1, -1)
    cv2.fillPoly(lab, [np.array([[80, 20], [150, 40], [120, 110], [70, 90]]) * SS], 2)
    cv2.rectangle(lab, (10 * SS, 90 * SS), (60 * SS, 115 * SS), 3, -1)
    big = np.zeros((H * SS, W * SS, 4), np.float32)
    big[:] = background
    for k, c in COLS.items():
        big[lab == k] = c
    if linear:
        s = big[:, :, :3]
        big[:, :, :3] = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    big[:, :, :3] *= big[:, :, 3:4]
    small = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    a = small[:, :, 3:4]
    np.divide(small[:, :, :3], a, out=small[:, :, :3], where=a > 0)
    if linear:
        small[:, :, :3] = srgb_oetf(small[:, :, :3])
    small = np.round(small * 255) / 255  # as decoded from an 8-bit PNG
    truth = {k: cv2.resize((lab == k).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) for k in COLS}
    return small.astype(np.float32), truth


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("background", [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 1.0)])
def test_id_pick_edges_match_the_renderers_antialiasing(background, linear):
    px, truth = _id_scene(background, linear)
    ids = P.IdPass(px)
    assert ids.linear == linear  # measured from the pass's own edges
    for k in COLS:
        ys, xs = np.nonzero(cv2.erode((truth[k] > 0.999).astype(np.uint8), np.ones((5, 5), np.uint8)))
        col = ids.sample(xs[0] + 0.5, ys[0] + 0.5)  # a click inside: the pass's own colour
        soft, hard = ids.mask(col), ids.mask(col, soft=False)
        edge = (truth[k] > 0.02) & (truth[k] < 0.98)
        assert np.abs(soft - truth[k])[edge].mean() < 0.03, k  # vs ~0.5 for a plain colour match
        assert np.abs(hard - truth[k])[edge].mean() > 0.1
        assert np.abs(soft - truth[k]).mean() < 0.002
        inside, outside = truth[k] > 0.999, truth[k] < 0.001
        assert soft[inside].min() == 1.0 and soft[outside].max() < 0.2


def test_sample_prefers_the_object_over_an_edge_blend():
    px, truth = _id_scene()
    ids = P.IdPass(px)
    ys, xs = np.nonzero((truth[1] > 0.3) & (truth[1] < 0.7))
    for y, x in zip(ys[:20], xs[:20]):
        c = ids.sample(x + 0.5, y + 0.5)  # an edge blend: you get a real ID, never the blend
        assert any(np.allclose(c, v, atol=1 / 255) for v in list(COLS.values()) + [(0, 0, 0, 0)])
    assert np.allclose(ids.sample(60.5, 60.5), COLS[1], atol=1 / 255)
    assert ids.sample(-1, 5) is None and ids.sample(W, 5) is None


def test_tolerance_widens_the_match():
    px = np.zeros((10, 10, 4), np.float32)
    px[..., 3] = 1.0
    px[:, :5, 0] = 0.50
    px[:, 5:, 0] = 0.53
    ids = P.IdPass(px)
    col = ids.sample(0.5, 0.5)
    assert ids.mask(col, soft=False)[:, 5:].max() == 0.0
    assert ids.mask(col, tolerance=0.05, soft=False).min() == 1.0
    assert ids.mask(col)[:, 5:].max() == 0.0  # two flat regions: nothing to blend


# ------------------------------------------------------------------ lasso + combine


def test_lasso_square_is_exact_and_triangle_has_the_right_area():
    sq = lasso_mask([(10, 10), (20, 10), (20, 20), (10, 20)], 40, 30)
    assert sq[10:20, 10:20].min() > 0.99 and sq.sum() == pytest.approx(100, abs=3)
    assert sq[:10].max() == 0.0 and sq[:, 21:].max() == 0.0
    tri = lasso_mask([(0.0, 0.0), (30.0, 0.0), (0.0, 30.0)], 40, 40)
    assert tri.sum() == pytest.approx(450, rel=0.03)
    half = tri[5, 24]  # the diagonal crosses pixel (24, 5) exactly through its centre
    assert 0.3 < half < 0.7


def test_lasso_clips_and_ignores_degenerate_input():
    m = lasso_mask([(-50, -50), (10, -50), (10, 10), (-50, 10)], 20, 20)
    assert m[:10, :10].min() > 0.99 and m[11:].max() == 0.0
    assert lasso_mask([(1, 1), (5, 5)], 10, 10).max() == 0.0
    assert lasso_mask([(100, 100), (110, 100), (110, 110)], 10, 10).max() == 0.0


def test_combine_modes():
    a = np.array([[0.0, 0.5, 1.0]], np.float32)
    b = np.array([[1.0, 0.25, 0.0]], np.float32)
    assert np.allclose(M.combine(a, b, "replace"), b)
    assert np.allclose(M.combine(a, b, "add"), [[1.0, 0.5, 1.0]])
    assert np.allclose(M.combine(a, b, "subtract"), [[0.0, 0.5, 1.0]])
    with pytest.raises(ValueError):
        M.combine(a, b, "xor")


def test_import_files_brings_the_render_set_along(tmp_path):
    beauty = make_rgba(24, 16, seed=1, alpha=False)
    alpha = np.zeros((16, 24, 4), np.float32)
    alpha[:, :12] = 1.0
    alpha[:, :, 3] = 1.0
    write_png(str(tmp_path / "shot.png"), beauty)
    write_png(str(tmp_path / "shot_Object ID.png"), make_rgba(24, 16, seed=2))
    write_png(str(tmp_path / "shot_MaterialID.png"), make_rgba(10, 10, seed=3))  # wrong size
    write_png(str(tmp_path / "shot_alpha.png"), alpha)
    (tmp_path / "shot_objectid_broken.png").write_bytes(b"not an image")
    store = AssetStore()
    items, errors, notes = RS.import_files(store, [str(tmp_path / "shot.png")])
    assert len(items) == 1
    item = items[0]
    assert set(item.passes) == {"object_id"} and item.mask is not None
    assert len(errors) == 1 and "Material ID" in errors[0]
    assert notes and "Object ID" in notes[0]
    # As the editor builds it: the layer renders cut out by the alpha pass.
    from lookbox.core.model import LayerMask, Transform
    from lookbox.core.render.pipeline import render_layer
    layer = ImageLayer(source=item.info.id, passes={k: p.id for k, p in item.passes.items()},
                       mask=LayerMask(asset=item.mask.id), transform=Transform(x=12, y=8))
    doc = Document(canvas=Size(w=24, h=16))
    edits.AddLayer(layer, asset=item.info, extra_assets=item.extra_assets()).apply(doc)
    assert doc.referenced_assets() == set(doc.assets)
    out = render_layer(layer, store)
    assert out[:, :12, 3].min() > 0.99 and out[:, 12:, 3].max() < 0.01


def test_importing_passes_with_their_render_doesnt_make_extra_layers(tmp_path):
    for n, seed in (("a.png", 1), ("a_objectid.png", 2)):
        write_png(str(tmp_path / n), make_rgba(8, 8, seed=seed))
    items, errors, _ = RS.import_files(AssetStore(), [str(tmp_path / "a_objectid.png"), str(tmp_path / "a.png")])
    assert len(items) == 1 and items[0].info.name == "a.png" and not errors


@pytest.mark.parametrize("beauty,other,kind", [
    ("cam2_0001.png", "cam2_objectid_0001.png", "object_id"),
    ("shot_v02_0001.png", "shot_v02_Object ID_0001.png", "object_id"),
    ("shot_v02.png", "shot_v02_matid.png", "material_id"),
])
def test_pass_names_with_digits_in_the_name(beauty, other, kind):
    assert RS.pass_kind(beauty, other) == kind


def test_nothing_selected_disappears(tmp_path):
    for n in ("a.png", "a_id.png", "a_objectid.png"):
        (tmp_path / n).write_bytes(b"x")
    p = lambda n: str(tmp_path / n)  # noqa: E731
    groups = RS.group([p("a.png"), p("a_id.png"), p("a_objectid.png")], scan_folders=False)
    names = [os.path.basename(b) for b, _ in groups]
    assert names == ["a.png", "a_objectid.png"]  # one Object ID attached, the other becomes a layer
    assert RS.guess_kind("objects_render.png") is None and RS.guess_kind("x_matid_0003.exr") == "material_id"


def test_extract_session_is_one_undo_step(tmp_path):
    store, infos = _store_with(tmp_path, {"beauty": make_rgba(8, 6, seed=1, alpha=False)})
    doc = Document(canvas=Size(w=8, h=6))
    layer = ImageLayer(source=infos["beauty"].id)
    edits.AddLayer(layer, asset=infos["beauty"]).apply(doc)
    m = np.zeros((6, 8), np.float32)
    m[:, :4] = 1.0
    info = store.add_bytes(M.encode_mask_png(m), ".png", "mask")
    from lookbox.core.model import LayerMask
    set_mask = edits.SetMask(layer.id, None, LayerMask(asset=info.id), asset=info)
    batch = edits.extract_session(doc, layer.id, set_mask)
    before = copy.deepcopy(doc)
    batch.apply(doc)
    assert [lay.mask for lay in doc.layers] == [None, LayerMask(asset=info.id)] and info.id in doc.assets
    batch.revert(doc)
    assert doc == before
    assert edits.extract_session(doc, layer.id, None) is None  # no selection, nothing to extract
