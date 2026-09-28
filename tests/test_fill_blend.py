"""M4: fill layers, gradient fade, blend modes, and the banding acceptance test."""

import math

import numpy as np
import pytest

from lookbox.commands import edits
from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.io import images
from lookbox.core.model import (BLEND_MODES, Document, Fill, FillLayer, GradientFade, GradientStop,
                                ImageLayer, Size, Transform, document_from_dict, document_to_dict)
from lookbox.core.render import blend
from lookbox.core.render import fill as F
from lookbox.core.render.pipeline import premultiply, render, render_key, render_layer, unpremultiply


def _two(c0, c1, **kw):
    return Fill(stops=[GradientStop(pos=0.0, color=c0), GradientStop(pos=1.0, color=c1)], **kw)


BLACK, WHITE = (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)


# ------------------------------------------------------------------ fills


def test_linear_gradient_directions_and_corners():
    out = F.render_fill(_two(BLACK, WHITE, kind="linear", angle_deg=0), 200, 10)
    assert out[5, 0, 0] < 0.01 and out[5, -1, 0] > 0.99  # left→right
    assert np.all(np.diff(out[5, :, 0]) >= 0)
    down = F.render_fill(_two(BLACK, WHITE, kind="linear", angle_deg=90), 10, 200)
    assert down[0, 5, 0] < 0.01 and down[-1, 5, 0] > 0.99  # top→bottom
    diag = F.render_fill(_two(BLACK, WHITE, kind="linear", angle_deg=45), 100, 100)
    assert diag[0, 0, 0] < 0.02 and diag[-1, -1, 0] > 0.98  # corners land on 0 and 1


def test_radial_gradient_centre_and_radius():
    out = F.render_fill(_two(WHITE, BLACK, kind="radial", cx=0.5, cy=0.5, radius=0.5), 101, 101)
    assert out[50, 50, 0] > 0.98 and out[0, 0, 0] < 0.01
    ring = F.radial_t(0.5, 0.5, 101, 101)
    assert ring[50, 50] < 0.02 and abs(ring[0, 0] - 1.0) < 0.02  # half-diagonal units


def test_solid_and_premultiplied_interpolation():
    solid = F.render_fill(Fill(kind="solid", stops=[GradientStop(pos=0, color=(0.2, 0.4, 0.6, 0.5))]), 4, 3)
    assert np.allclose(solid, (0.2, 0.4, 0.6, 0.5))
    # Red → fully transparent: the middle must stay red, not go dark (premultiplied interpolation).
    out = F.render_fill(_two((1, 0, 0, 1), (0, 0, 0, 0), kind="linear", angle_deg=0), 101, 1)
    assert np.allclose(out[0, 50, :3], (1, 0, 0), atol=1e-4) and 0.4 < out[0, 50, 3] < 0.6


def test_fill_is_resolution_independent():
    f = _two((0.9, 0.3, 0.1, 1), (0.1, 0.2, 0.8, 1), kind="radial", cx=0.3, cy=0.6, radius=0.7)
    layer = FillLayer(fill=f, width=400, height=240)
    import cv2
    full = unpremultiply(render_layer(layer, AssetStore(), 1.0))
    quarter = unpremultiply(render_layer(layer, AssetStore(), 0.25))
    ref = cv2.resize(full, (100, 60), interpolation=cv2.INTER_AREA)
    assert quarter.shape == (60, 100, 4) and np.abs(quarter - ref).max() < 0.01


# ------------------------------------------------------------------ fade


def test_linear_fade_opaque_before_start_clear_after_end():
    m = F.fade_mask(GradientFade(kind="linear", angle_deg=90, start=0.5, end=1.0), 10, 100)
    assert np.allclose(m[:45], 1.0) and m[-1].max() < 0.02 and np.all(np.diff(m[:, 5]) <= 1e-6)
    inv = F.fade_mask(GradientFade(kind="linear", angle_deg=90, start=0.5, end=1.0, invert=True), 10, 100)
    assert np.allclose(inv, 1.0 - m)
    hard = F.fade_mask(GradientFade(kind="linear", angle_deg=0, start=0.5, end=0.5), 100, 4)
    assert hard[0, 10] == 1.0 and hard[0, 90] == 0.0


def test_radial_fade_and_fade_on_image_layer():
    m = F.fade_mask(GradientFade(kind="radial", start=0.2, end=0.9), 101, 101)
    assert m[50, 50] == 1.0 and m[0, 0] < 0.05
    store = AssetStore()
    info = store.add_bytes(images.encode_png(np.ones((20, 30, 4), np.float32)), ".png")
    layer = ImageLayer(source=info.id, fade=GradientFade(angle_deg=0, start=0.0, end=1.0))
    out = render_layer(layer, store)
    assert out[10, 0, 3] > 0.95 and out[10, -1, 3] < 0.05
    assert render_key(layer) != render_key(ImageLayer(source=info.id))  # fade invalidates the cache


# ------------------------------------------------------------------ blend modes (W3C)


def _composite(mode, cb, ab, cs, as_):
    dst = premultiply(np.array([[[*cb, ab]]], np.float32))
    src = premultiply(np.array([[[*cs, as_]]], np.float32))
    blend.composite(dst, src, mode)
    return unpremultiply(dst)[0, 0]


@pytest.mark.parametrize("mode,expected", [
    ("multiply", 0.6 * 0.3),
    ("screen", 0.6 + 0.3 - 0.6 * 0.3),
    ("overlay", 1 - 2 * (1 - 0.6) * (1 - 0.3)),  # backdrop 0.6 > 0.5 → screen branch
    ("soft_light", 0.6 - (1 - 2 * 0.3) * 0.6 * (1 - 0.6)),  # source 0.3 ≤ 0.5 branch
    ("add", 0.9),
])
def test_blend_formulas_opaque(mode, expected):
    out = _composite(mode, (0.6, 0.6, 0.6), 1.0, (0.3, 0.3, 0.3), 1.0)
    assert out[0] == pytest.approx(expected, abs=1e-5) and out[3] == pytest.approx(1.0)


def test_soft_light_upper_branch_uses_w3c_d():
    cb, cs = 0.2, 0.8
    d = ((16 * cb - 12) * cb + 4) * cb
    assert _composite("soft_light", (cb,) * 3, 1, (cs,) * 3, 1)[0] == pytest.approx(cb + (2 * cs - 1) * (d - cb), abs=1e-5)


@pytest.mark.parametrize("mode", [m for m in BLEND_MODES if m != "add"])
def test_blend_over_empty_is_just_the_layer(mode):
    out = _composite(mode, (0, 0, 0), 0.0, (0.2, 0.5, 0.9), 0.7)
    assert np.allclose(out, (0.2, 0.5, 0.9, 0.7), atol=1e-5)


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_transparent_layer_changes_nothing(mode):
    out = _composite(mode, (0.4, 0.5, 0.6), 0.8, (0.9, 0.1, 0.3), 0.0)
    assert np.allclose(out, (0.4, 0.5, 0.6, 0.8), atol=1e-5)


def test_neutral_colours():
    assert np.allclose(_composite("multiply", (0.3, 0.6, 0.9), 1, (1, 1, 1), 1)[:3], (0.3, 0.6, 0.9), atol=1e-5)
    assert np.allclose(_composite("screen", (0.3, 0.6, 0.9), 1, (0, 0, 0), 1)[:3], (0.3, 0.6, 0.9), atol=1e-5)
    assert np.allclose(_composite("soft_light", (0.3, 0.6, 0.9), 1, (0.5,) * 3, 1)[:3], (0.3, 0.6, 0.9), atol=1e-5)


def test_unknown_blend_mode_errors():
    with pytest.raises(ValueError):
        blend.composite(np.zeros((1, 1, 4), np.float32), np.zeros((1, 1, 4), np.float32), "dissolve")


def test_blend_mode_in_document_render():
    store = AssetStore()
    doc = Document(canvas=Size(w=8, h=8), background=(0.5, 0.5, 0.5, 1.0))
    doc.layers = [FillLayer(fill=Fill(kind="solid", stops=[GradientStop(pos=0, color=(0.5, 0.5, 0.5, 1))]),
                            width=8, height=8, transform=Transform(x=4, y=4), blend_mode="multiply")]
    assert render(doc, store)[4, 4, 0] == pytest.approx(0.25, abs=1e-5)


# ------------------------------------------------------------------ M4 acceptance: no banding


def test_backdrop_gradient_has_no_visible_banding_after_8bit_export(tmp_path):
    """A subtle dark backdrop across a wide canvas is the worst case for 8-bit:
    ~10 shades over 1920 px → bands ~190 px wide without dithering."""
    doc = Document(canvas=Size(w=1920, h=64))
    doc.layers = [FillLayer(fill=_two((0.10, 0.10, 0.11, 1), (0.14, 0.14, 0.155, 1), kind="linear", angle_deg=0),
                            width=1920, height=64, transform=Transform(x=960, y=32))]
    px = render(doc, AssetStore())
    path = str(tmp_path / "backdrop.png")
    images.save_png(path, px)
    with open(path, "rb") as fh:
        out = images.decode(fh.read(), ".png")

    ideal = np.convolve(px[:, :, 0].mean(axis=0), np.ones(31) / 31, mode="valid")
    got = np.convolve(out[:, :, 0].mean(axis=0), np.ones(31) / 31, mode="valid")
    plain = np.rint(px[:, :, 0] * 255) / 255
    stairs = np.convolve(plain.mean(axis=0), np.ones(31) / 31, mode="valid")
    err_dither = np.abs(got - ideal).max() * 255
    err_plain = np.abs(stairs - ideal).max() * 255
    assert err_plain > 0.3  # the test image really would band without dithering
    assert err_dither < 0.1, f"banding error {err_dither:.3f} LSB"


# ------------------------------------------------------------------ model / edits / files


def test_fill_layer_roundtrip_and_edits(tmp_path):
    doc = Document(canvas=Size(w=100, h=50))
    fl = FillLayer(width=100, height=50, transform=Transform(x=50, y=25),
                   fade=GradientFade(kind="radial", start=0.3, end=0.8), blend_mode="screen")
    edits.AddLayer(fl, index=0).apply(doc)
    d = document_to_dict(doc)
    assert document_from_dict(d) == doc
    store = AssetStore()
    doc2, store2 = serialize.from_bytes(serialize.to_bytes(doc, store))
    assert doc2 == doc and np.array_equal(render(doc2, store2), render(doc, store))

    new_fill = _two(WHITE, BLACK, kind="linear", angle_deg=30)
    e = edits.SetLayerField(fl.id, "fill", fl.fill, new_fill, merge_key="d1")
    before = document_to_dict(doc)
    e.apply(doc)
    assert doc.layer(fl.id).fill == new_fill
    e2 = edits.SetLayerField(fl.id, "fill", new_fill, _two(BLACK, BLACK), merge_key="d1")
    e2.apply(doc)
    assert e.merge(e2)
    e.revert(doc)
    assert document_to_dict(doc) == before
    with pytest.raises(ValueError):
        edits.SetLayerField(fl.id, "source", None, "x")


def test_fill_field_on_image_layer_is_refused():
    doc = Document()
    edits.AddLayer(ImageLayer(id="img", source="a")).apply(doc)
    with pytest.raises(ValueError):
        edits.SetLayerField("img", "fill", None, Fill()).apply(doc)


def test_fill_layers_reference_no_assets():
    doc = Document(layers=[FillLayer()])
    assert doc.referenced_assets() == set()
    assert math.isfinite(render(Document(canvas=Size(w=4, h=4), layers=[FillLayer(width=4, height=4,
                                         transform=Transform(x=2, y=2))]), AssetStore()).sum())
