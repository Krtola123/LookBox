"""M5: layer effects. The promises being tested:
- sizes are canvas pixels and the shadow direction is canvas space, whatever
  the layer's scale / rotation / flip;
- nothing is clipped at the layer box;
- a preview level looks like the downscaled export;
- moving never re-renders; rotating/scaling re-renders only when effects are on.
"""

import math

import cv2
import numpy as np
import pytest

from lookbox.core.assets import AssetStore
from lookbox.core.model import (Document, DropShadow, Effects, Fill, FillLayer, Glow, GradientStop, Outline,
                                Size, Transform, document_from_dict, document_to_dict)
from lookbox.core.render import effects as E
from lookbox.core.render.pipeline import render, render_key, render_layer_full, unpremultiply

WHITE_FILL = Fill(kind="solid", stops=[GradientStop(pos=0, color=(1, 1, 1, 1))])


def _square(size=40, **fx) -> FillLayer:
    return FillLayer(fill=WHITE_FILL, width=size, height=size, transform=Transform(x=100, y=100),
                     effects=Effects(**fx))


def _canvas(layer, bg=None, w=200, h=200):
    doc = Document(canvas=Size(w=w, h=h), background=bg, layers=[layer])
    return render(doc, AssetStore())


def _shadow_only(layer):
    """Export alpha that isn't the layer itself (the layer is an opaque white square)."""
    out = _canvas(layer)
    a = out[:, :, 3].copy()
    white = (out[:, :, :3].min(axis=2) > 0.99) & (a > 0.99)
    a[white] = 0
    return a


def _centroid(a):
    ys, xs = np.nonzero(a > 0.02)
    wts = a[ys, xs]
    return (xs * wts).sum() / wts.sum(), (ys * wts).sum() / wts.sum()


def test_no_effects_means_no_padding():
    r = render_layer_full(_square(), AssetStore())
    assert r.pad == 0 and r.pixels.shape == (40, 40, 4)


@pytest.mark.parametrize("rot,flip", [(0, False), (90, False), (-135, False), (37, True)])
def test_shadow_falls_down_in_canvas_space_whatever_the_rotation(rot, flip):
    sq = _square(shadow=DropShadow(angle_deg=90, distance=30, blur=6, opacity=1.0))
    sq.transform.rotation_deg = rot
    sq.transform.flip_h = flip
    cx, cy = _centroid(_shadow_only(sq))
    assert abs(cx - 100) < 3 and cy > 110, (rot, cx, cy)  # below the layer, not sideways


def test_shadow_sizes_are_canvas_pixels_whatever_the_layer_scale():
    big = _square(size=40, shadow=DropShadow(angle_deg=0, distance=50, blur=0, opacity=1.0))
    small = _square(size=80, shadow=DropShadow(angle_deg=0, distance=50, blur=0, opacity=1.0))
    small.transform.scale_x = small.transform.scale_y = 0.5  # same 40 px on canvas
    for layer in (big, small):
        cx, _ = _centroid(_shadow_only(layer))
        # Shadow square (40 px, centred 50 px right) minus the part hidden under the layer.
        assert 135 < cx < 150, cx
    a1, a2 = _shadow_only(big), _shadow_only(small)
    assert np.abs(a1 - a2).max() < 0.05


@pytest.mark.parametrize("fx", [
    dict(shadow=DropShadow(angle_deg=45, distance=20, blur=16, spread=6, opacity=1.0)),
    dict(glow=Glow(blur=12, spread=5, opacity=1.0)),
    dict(outline=Outline(width=7, color=(1, 0, 0, 1))),
    dict(blur=9),
])
def test_blur_spread_and_width_are_canvas_pixels_whatever_the_layer_scale(fx):
    """Same 40 px square on canvas, built two ways: effects must come out identical."""
    a = _square(size=40, **fx)
    b = _square(size=160, **fx)
    b.transform.scale_x = b.transform.scale_y = 0.25
    ea, eb = _canvas(a), _canvas(b)
    pa, pb = ea[:, :, :3] * ea[:, :, 3:], eb[:, :, :3] * eb[:, :, 3:]  # premultiplied: no noise from ~0 alpha
    assert np.abs(pa - pb).mean() < 0.002, np.abs(pa - pb).mean()
    # One sample per pixel vs a 4×-supersampled ring differ slightly on anti-aliased curved
    # corners; a wrong size would be off by whole pixels everywhere, not ~0.15 at a corner.
    worst = 0.16 if "outline" in fx else 0.1
    assert np.abs(ea[:, :, 3] - eb[:, :, 3]).max() < worst, np.abs(ea[:, :, 3] - eb[:, :, 3]).max()


def test_long_shadow_and_big_blur_are_not_clipped():
    sq = _square(shadow=DropShadow(angle_deg=90, distance=60, blur=30, opacity=1.0))
    r = render_layer_full(sq, AssetStore())
    assert r.pad >= 60 + 30 * E.SIGMA_PER_BLUR * E.REACH_SIGMAS
    a = r.pixels[:, :, 3]
    assert a[0, :].max() < 1e-3 and a[-1, :].max() < 1e-3 and a[:, 0].max() < 1e-3  # faded out before the edge


def test_floor_squash_flattens_the_shadow_onto_the_bottom_edge():
    # Shadow straight down, no offset: plain → a blurred copy of the square behind it;
    # squashed 0.8 → a thin band hugging the bottom edge (a contact shadow).
    base = dict(angle_deg=90, distance=0, blur=4, spread=0, opacity=1.0)
    plain = render_layer_full(_square(shadow=DropShadow(**base)), AssetStore())
    flat = render_layer_full(_square(shadow=DropShadow(**base, squash=0.8)), AssetStore())

    def shadow_rows(r):  # rows where the shadow shows outside the square (left margin column)
        col = r.pixels[:, r.pad - 3, 3]
        return np.nonzero(col > 0.05)[0] - r.pad  # relative to the square's top

    p_rows, f_rows = shadow_rows(plain), shadow_rows(flat)
    assert p_rows.min() < 5 and p_rows.max() > 35  # plain spans the full height
    assert f_rows.min() > 25 and f_rows.max() <= 44  # squashed: only near the bottom edge (y = 40)


def test_glow_is_symmetric_and_outline_has_its_width():
    glow = _shadow_only(_square(glow=Glow(blur=10, spread=4, opacity=1.0, color=(1, 0, 0, 1))))
    cx, cy = _centroid(glow)
    assert abs(cx - 99.5) < 0.05 and abs(cy - 99.5) < 0.05  # pixel indices: the square spans 80–119
    ol = _canvas(_square(outline=Outline(width=6, color=(1, 0, 0, 1))))
    row = ol[100, :, :]
    red = (row[:, 0] > 0.5) & (row[:, 1] < 0.5)
    assert 11 <= red.sum() <= 13  # 6 px each side of the 40 px square
    centred = _canvas(_square(outline=Outline(width=6, color=(1, 0, 0, 1), position="center")))
    red_c = (centred[100, :, 0] > 0.5) & (centred[100, :, 1] < 0.5)
    assert 10 <= red_c.sum() <= 14 and red_c[100 - 20 + 1]  # straddles the edge: inside part visible


def test_layer_blur_spreads_past_the_box():
    out = _canvas(_square(blur=10))
    assert out[100, 100 - 20 - 5, 3] > 0.02  # soft edge outside the square
    assert out[100, 100, 3] > 0.95


def test_outline_ignores_transparent_hole():
    # Fill with a transparent stop: the outline follows the opaque part only.
    f = Fill(kind="linear", angle_deg=0, stops=[GradientStop(pos=0, color=(1, 1, 1, 1)),
                                                GradientStop(pos=0.49, color=(1, 1, 1, 1)),
                                                GradientStop(pos=0.51, color=(1, 1, 1, 0)),
                                                GradientStop(pos=1, color=(1, 1, 1, 0))])
    layer = FillLayer(fill=f, width=40, height=40, transform=Transform(x=100, y=100),
                      effects=Effects(outline=Outline(width=4, color=(1, 0, 0, 1))))
    out = _canvas(layer)
    assert out[100, 100 + 2, 0] > 0.5 and out[100, 100 + 2, 1] < 0.5  # ring right of the opaque half


CASES = [
    Effects(shadow=DropShadow(angle_deg=60, distance=24, blur=20, spread=4, opacity=0.8)),
    Effects(shadow=DropShadow(angle_deg=90, distance=0, blur=12, squash=0.7, opacity=0.9)),
    Effects(glow=Glow(blur=16, spread=6)),
    Effects(outline=Outline(width=8)),
    Effects(blur=12),
]


@pytest.mark.parametrize("fx", CASES)
def test_preview_level_matches_export(fx):
    layer = FillLayer(fill=Fill(kind="radial", cx=0.4, cy=0.4), width=160, height=120,
                      transform=Transform(x=150, y=150, rotation_deg=20), effects=fx)
    layer.fill.stops = [GradientStop(pos=0, color=(0.9, 0.5, 0.2, 1)), GradientStop(pos=1, color=(0.2, 0.3, 0.8, 1))]
    full = render_layer_full(layer, AssetStore(), 1.0)
    half = render_layer_full(layer, AssetStore(), 0.5)
    # Compare the half level against the full render downscaled, aligned on the layer box.
    f = unpremultiply(full.pixels)
    hh = unpremultiply(half.pixels)
    box = f[full.pad:full.pad + 120, full.pad:full.pad + 160]
    ref = cv2.resize(np.ascontiguousarray(box), (80, 60), interpolation=cv2.INTER_AREA)
    got = hh[half.pad:half.pad + 60, half.pad:half.pad + 80]
    err = np.abs(ref[:, :, 3] - got[:, :, 3]).mean() + np.abs(ref[:, :, :3] - got[:, :, :3]).mean()
    assert err < 0.03, err
    # And the padding scales with the level (±2 px of rounding margin).
    assert abs(half.pad * 2 - full.pad) <= 4


def test_render_key_moving_never_rerenders_rotating_does_only_with_effects():
    plain = _square()
    fx = _square(shadow=DropShadow())
    for layer, rotation_matters in ((plain, False), (fx, True)):
        k0 = render_key(layer)
        layer.transform.x += 37
        assert render_key(layer) == k0
        layer.transform.rotation_deg += 10
        assert (render_key(layer) != k0) == rotation_matters


def test_effects_roundtrip_and_old_files():
    layer = _square(shadow=DropShadow(color=(0.1, 0.2, 0.3, 0.9), squash=0.5), glow=Glow(),
                    outline=Outline(position="center"), blur=3.5)
    doc = Document(layers=[layer])
    d = document_to_dict(doc)
    assert document_from_dict(d) == doc
    del d["layers"][0]["effects"]
    assert document_from_dict(d).layers[0].effects == Effects()  # M1–M4 files load clean


def test_canvas_vec_to_local_inverts_the_transform():
    t = Transform(scale_x=2.0, scale_y=0.5, rotation_deg=30, flip_v=True)
    lx, ly = E.canvas_vec_to_local(t, 10.0, 5.0)
    r = math.radians(30)
    sx, sy = lx * 2.0, ly * 0.5 * -1
    back = (math.cos(r) * sx - math.sin(r) * sy, math.sin(r) * sx + math.cos(r) * sy)
    assert np.allclose(back, (10.0, 5.0))
