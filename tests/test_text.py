"""M7: text layers. Layout, rendering through the normal pipeline (so effects,
fade and blend apply), export sharpness, placement when text changes, edits,
hit testing and save/load. The Qt engine is swapped for a deterministic fake."""

import math

import cv2
import numpy as np
import pytest

from lookbox.commands import edits
from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.model import (Document, Effects, FillLayer, Outline, Size, TextLayer, Transform,
                                document_from_dict, document_to_dict)
from lookbox.core.render import levels
from lookbox.core.render import text as T
from lookbox.core.render.pipeline import render, render_key, render_layer_full, unpremultiply
from lookbox.core.render.transform import layer_matrix
from lookbox.ui.canvas import handles as H
from tests.fake_text import FakeTextEngine

T.set_engine(FakeTextEngine())


def _text(**kw) -> TextLayer:
    kw.setdefault("font_size", 20.0)
    kw.setdefault("color", (1.0, 0.5, 0.25, 1.0))
    return TextLayer(**kw)


def _margins(layer):
    m = T.MARGIN_EM * layer.font_size
    return m + (T.ITALIC_EXTRA_EM * layer.font_size if layer.italic else 0.0), m


# ------------------------------------------------------------------ layout


def test_wrap_is_greedy_and_breaks_overlong_words():
    measure = len  # 1 px per char
    assert T.wrap("aa bb cc", 5, measure) == ["aa bb", "cc"]
    assert T.wrap("aa bb cc", None, measure) == ["aa bb cc"]
    assert T.wrap("abcdefgh", 3, measure) == ["abc", "def", "gh"]
    assert T.wrap("hi abcdefgh", 3, measure) == ["hi", "abc", "def", "gh"]
    assert T.wrap("", 3, measure) == [""]
    assert T.wrap("a  b", 10, measure) == ["a  b"]  # typed double spaces survive


def test_box_size_lines_and_alignment():
    layer = _text(text="ab\nabcd", align="left")
    lay = T.layout(layer)
    mx, my = _margins(layer)
    step = 20.0 * 1.2  # (ascent + descent) × line_height
    assert [ln.text for ln in lay.lines] == ["ab", "abcd"]
    assert lay.width == math.ceil(4 * 10 + 2 * mx)
    assert lay.height == math.ceil(2 * my + step + 20)
    assert lay.lines[1].baseline - lay.lines[0].baseline == pytest.approx(step)
    assert lay.lines[0].x == pytest.approx(mx)
    centre = T.layout(_text(text="ab\nabcd", align="center"))
    assert centre.lines[0].x == pytest.approx(mx + 10)  # (40 - 20) / 2
    right = T.layout(_text(text="ab\nabcd", align="right"))
    assert right.lines[0].x == pytest.approx(mx + 20)


def test_fixed_width_wraps_and_fixes_the_box():
    layer = _text(text="aa bb cc dd", box_width=55.0)  # 5 chars = 50 px fit, 8 don't
    lay = T.layout(layer)
    assert [ln.text for ln in lay.lines] == ["aa bb", "cc dd"]
    assert lay.width == math.ceil(55 + 2 * _margins(layer)[0])


def test_empty_text_still_has_a_clickable_box():
    lay = T.layout(_text(text=""))
    assert lay.width >= 10 and lay.height >= 20


def test_letter_spacing_and_italic_widen_the_box():
    base = T.layout(_text(text="abcd")).width
    assert T.layout(_text(text="abcd", letter_spacing=5)).width == base + 20
    assert T.layout(_text(text="abcd", italic=True)).width > base


def test_no_engine_is_a_clear_error():
    T.set_engine(None)
    try:
        with pytest.raises(RuntimeError, match="text engine"):
            T.layout(_text(text="new"))
    finally:
        T.set_engine(FakeTextEngine())


# ------------------------------------------------------------------ rendering


def test_text_renders_in_its_colour_where_the_glyphs_are():
    layer = _text(text="a b")
    px = unpremultiply(render_layer_full(layer, AssetStore(), 1.0).pixels)
    lay = T.layout(layer)
    line = lay.lines[0]
    ink_y = int(line.baseline - 5)
    a_x = int(line.x + 4)  # inside "a"
    gap_x = int(line.x + 15)  # the space
    assert px[ink_y, a_x, 3] == pytest.approx(1.0)
    assert np.allclose(px[ink_y, a_x, :3], (1.0, 0.5, 0.25))
    assert px[ink_y, gap_x, 3] == 0.0
    assert px[1, 1, 3] == 0.0  # margin is empty


def test_levels_are_the_same_picture():
    layer = _text(text="Hello there", font_size=40.0)
    full = render_layer_full(layer, AssetStore(), 1.0).pixels
    half = render_layer_full(layer, AssetStore(), 0.5).pixels
    ref = cv2.resize(full, (half.shape[1], half.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(half - ref).mean() < 0.02
    double = render_layer_full(layer, AssetStore(), 2.0).pixels
    assert double.shape[:2] == (round(full.shape[0] * 2), round(full.shape[1] * 2))


def test_effects_apply_to_text():
    plain = _text(text="ab")
    outlined = _text(text="ab", effects=Effects(outline=Outline(width=4.0)))
    r0 = render_layer_full(plain, AssetStore(), 1.0)
    r1 = render_layer_full(outlined, AssetStore(), 1.0)
    assert r1.pad > 0
    assert r1.pixels[:, :, 3].sum() > r0.pixels[:, :, 3].sum() * 1.2


def test_max_level_text_is_finer_but_capped_with_effects():
    assert levels.max_level(FillLayer()) == 1.0
    assert levels.max_level(_text()) == levels.TEXT_MAX_LEVEL
    assert levels.max_level(_text(effects=Effects(blur=3))) == levels.TEXT_FX_MAX_LEVEL
    assert levels.choose_level(3.0, levels.TEXT_MAX_LEVEL) == 4.0
    assert levels.choose_level(3.0) == 1.0


def _alpha_centroid(a):
    ys, xs = np.nonzero(a > 0.5)
    return xs.mean(), ys.mean()


@pytest.mark.parametrize("rot", [0.0, 30.0])
def test_export_at_2x_renders_text_sharp_and_in_the_same_place(rot):
    layer = _text(text="Hi you", font_size=30.0, transform=Transform(x=100, y=60, rotation_deg=rot))
    doc = Document(canvas=Size(w=200, h=120), layers=[layer])
    one = render(doc, AssetStore(), 1.0)
    two = render(doc, AssetStore(), 2.0)
    assert levels.export_level(layer, 2.0) == 2.0
    c1, c2 = _alpha_centroid(one[:, :, 3]), _alpha_centroid(two[:, :, 3])
    assert c2[0] == pytest.approx(2 * c1[0] + 0.5, abs=0.6) and c2[1] == pytest.approx(2 * c1[1] + 0.5, abs=0.6)
    if rot == 0.0:
        # Sharp: edges are hard at 2× (a level-1 render upscaled 2× would have 2-px ramps).
        a = two[:, :, 3]
        partial = ((a > 0.05) & (a < 0.95)).sum()
        solid = (a >= 0.95).sum()
        assert partial < 0.25 * solid


def test_render_key_follows_text_but_not_position():
    a = _text(text="one")
    assert render_key(a) != render_key(_text(text="two"))
    moved = _text(text="one", transform=Transform(x=50))
    moved.id = a.id
    assert render_key(a) == render_key(moved)


# ------------------------------------------------------------------ placement


def _canvas_pt(t, box, u, v):
    return (layer_matrix(t, *box) @ np.array([u, v, 1.0]))[:2]


@pytest.mark.parametrize("align,ax", [("left", 0.0), ("center", 0.5), ("right", 1.0)])
@pytest.mark.parametrize("rot,flip", [(0.0, False), (90.0, False), (33.0, True)])
def test_anchored_keeps_the_top_left_centre_or_right_point(align, ax, rot, flip):
    t = Transform(x=120, y=80, rotation_deg=rot, flip_h=flip, scale_x=1.5, scale_y=1.5)
    old, new = (100, 40), (160, 90)
    t2 = T.anchored(t, old, new, align)
    assert np.allclose(_canvas_pt(t, old, ax * old[0], 0), _canvas_pt(t2, new, ax * new[0], 0))


def test_baked_resize_turns_scale_into_font_size():
    layer = _text(text="ab", font_size=20.0, letter_spacing=2.0, box_width=100.0)
    t = Transform(x=10, y=20, scale_x=2.0, scale_y=2.0, rotation_deg=15)
    fields, t2 = T.baked_resize(layer, t)
    assert fields == {"font_size": 40.0, "letter_spacing": 4.0, "box_width": 200.0}
    assert (t2.scale_x, t2.scale_y, t2.x, t2.y, t2.rotation_deg) == (1.0, 1.0, 10, 20, 15)
    # The box scales with it, so the text lands where the drag left it.
    ow, oh = T.layer_box(layer)
    nw, nh = T.layer_box(_text(text="ab", **fields))
    assert abs(nw - 2 * ow) <= 2 and abs(nh - 2 * oh) <= 2


# ------------------------------------------------------------------ edits


def test_set_text_apply_revert_merge():
    layer = _text(text="a")
    doc = Document(layers=[layer])
    t0 = layer.transform
    e1 = edits.SetText(layer.id, {"text": "a"}, {"text": "ab"}, t0, Transform(x=1), merge_key="typing")
    e1.apply(doc)
    e2 = edits.SetText(layer.id, {"text": "ab", "font_size": 20.0}, {"text": "abc", "font_size": 30.0},
                       Transform(x=1), Transform(x=2), merge_key="typing")
    e2.apply(doc)
    assert e1.merge(e2)
    assert doc.layers[0].text == "abc" and doc.layers[0].font_size == 30.0
    e1.revert(doc)
    assert doc.layers[0].text == "a" and doc.layers[0].font_size == 20.0 and doc.layers[0].transform == t0
    e1.apply(doc)
    assert doc.layers[0].text == "abc" and doc.layers[0].transform == Transform(x=2)


def test_set_text_rejects_wrong_layers_and_fields():
    doc = Document(layers=[FillLayer()])
    with pytest.raises(ValueError):
        edits.SetText(doc.layers[0].id, {"text": "a"}, {"text": "b"}).apply(doc)
    with pytest.raises(ValueError):
        edits.SetText("x", {"opacity": 1}, {"opacity": 0.5})
    with pytest.raises(ValueError):
        edits.SetText("x", {"align": "left"}, {"align": "justify"})


def test_style_edits_work_on_text():
    layer = _text()
    doc = Document(layers=[layer])
    e = edits.SetLayerField(layer.id, "effects", layer.effects, Effects(blur=2.0))
    e.apply(doc)
    assert doc.layers[0].effects.blur == 2.0
    with pytest.raises(ValueError):  # text has no mask
        edits.SetMask(layer.id, None, object()).apply(doc)


# ------------------------------------------------------------------ canvas + files


def test_layer_size_and_hit_test_use_the_text_box():
    layer = _text(text="abc", transform=Transform(x=100, y=50))
    doc = Document(canvas=Size(w=200, h=100), layers=[layer])
    w, h = H.layer_size(doc, layer)
    assert (w, h) == T.layer_box(layer)
    lay = T.layout(layer)
    gap_y = 50 - h / 2 + 1  # inside the margin: no ink, still a hit (text is thin)
    assert H.layer_at(doc, AssetStore(), 100, gap_y) == layer.id
    assert H.layer_at(doc, AssetStore(), 100 + w, 50) is None
    assert lay.lines


def test_text_layer_roundtrips_and_old_style_dicts_load(tmp_path):
    layer = _text(text="Line 1\nLine 2", font_family="Arial", italic=True, align="right",
                  box_width=300.0, effects=Effects(outline=Outline(width=3)))
    doc = Document(canvas=Size(w=64, h=48), layers=[layer])
    assert document_from_dict(document_to_dict(doc)) == doc
    path = str(tmp_path / "t.lookbox")
    serialize.save(path, doc, AssetStore())
    doc2, _ = serialize.load(path)
    assert doc2 == doc and isinstance(doc2.layers[0], TextLayer)
    bare = document_from_dict({"canvas": {"w": 10, "h": 10}, "layers": [{"kind": "text", "text": "x"}]})
    assert bare.layers[0].text == "x" and bare.layers[0].font_size == TextLayer().font_size


def test_resize_text_edit_bakes_and_undoes():
    layer = _text(text="ab", font_size=20.0)
    doc = Document(layers=[layer])
    t0 = layer.transform
    t1 = Transform(x=5, y=6, scale_x=1.5, scale_y=1.5)
    e = edits.resize_text(doc, layer.id, t0, t1)
    e.apply(doc)
    assert doc.layers[0].font_size == 30.0 and doc.layers[0].transform == Transform(x=5, y=6)
    e.revert(doc)
    assert doc.layers[0].font_size == 20.0 and doc.layers[0].transform == t0


def test_change_text_grows_from_the_left_edge_and_skips_no_ops():
    layer = _text(text="ab", transform=Transform(x=100, y=50))
    doc = Document(layers=[layer])
    assert edits.change_text(doc, layer.id, text="ab") is None
    old_box = T.layer_box(layer)
    left_top = _canvas_pt(layer.transform, old_box, 0, 0)
    e = edits.change_text(doc, layer.id, text="abcdef\nsecond line")
    e.apply(doc)
    new = doc.layers[0]
    assert np.allclose(_canvas_pt(new.transform, T.layer_box(new), 0, 0), left_top)
    assert T.content_width(new) == pytest.approx(11 * 10)


def test_huge_text_caps_its_level_by_pixel_count():
    big = _text(text="HEADLINE " * 6, font_size=600.0)
    w, h = T.layer_box(big)
    top = levels.max_level(big)
    assert top >= 1.0 and (top == 1.0 or w * h * top * top <= levels.TEXT_MAX_PIXELS)
    assert top < levels.TEXT_MAX_LEVEL


def test_baked_resize_keeps_an_old_non_uniform_scale():
    layer = _text(text="ab", font_size=20.0)
    fields, t2 = T.baked_resize(layer, Transform(scale_x=3.0, scale_y=2.0))
    assert fields["font_size"] == 40.0 and (t2.scale_x, t2.scale_y) == (1.5, 1.0)
