"""Behaviour tests for the Adjust panel (§7). These hold whatever the tuning;
the golden tests (test_adjust_golden.py) freeze the exact look."""

import numpy as np
import pytest

from lookbox.core.assets import AssetStore
from lookbox.core.io import images
from lookbox.core.model import ADJUST_SLIDERS, Adjustments, ColorBand, ImageLayer, Transform
from lookbox.core.render import adjust as A
from lookbox.core.render.pipeline import render_key, render_layer, unpremultiply
from tests import adjust_images as IM


def _apply(img, **kw):
    bands = kw.pop("color_edit", [])
    return A.apply(img[:, :, :3], img[:, :, 3], Adjustments(color_edit=bands, **kw))


def _lum(x):
    return A.luma(x[:, :, :3])


def test_identity_is_exact():
    for make in IM.IMAGES.values():
        img = make()
        assert np.array_equal(_apply(img), np.clip(img[:, :, :3], 0, 1))
    assert Adjustments().is_identity()
    assert not Adjustments(invert=True).is_identity()
    assert not Adjustments(color_edit=[ColorBand(hue=0, saturation=10)]).is_identity()
    assert Adjustments(color_edit=[ColorBand(hue=0)]).is_identity()


@pytest.mark.parametrize("name", ADJUST_SLIDERS)
def test_extremes_are_finite_and_in_range(name):
    for make in IM.IMAGES.values():
        for v in (-100, 100):
            out = _apply(make(), **{name: v})
            assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_everything_at_once_is_finite():
    img = IM.photo()
    kw = {k: (100 if i % 2 else -100) for i, k in enumerate(ADJUST_SLIDERS)}
    out = _apply(img, invert=True, color_edit=[ColorBand(hue=30, hue_shift=100, saturation=-100, lightness=100)], **kw)
    assert np.isfinite(out).all()


def test_directions():
    img = IM.photo()
    base = img[:, :, :3]
    lum = _lum(img)
    assert _apply(img, brightness=50).mean() > base.mean()
    assert _apply(img, brightness=-50).mean() < base.mean()
    assert A.luma(_apply(img, contrast=80)).std() > lum.std()
    assert A.luma(_apply(img, contrast=-80)).std() < lum.std()
    warm = _apply(img, temperature=80)
    assert (warm[:, :, 0] - warm[:, :, 2]).mean() > (base[:, :, 0] - base[:, :, 2]).mean()
    assert abs(A.luma(warm).mean() - lum.mean()) < 0.02  # white balance keeps brightness
    assert _apply(img, tint=80)[:, :, 1].mean() < base[:, :, 1].mean()  # + = magenta
    grey = _apply(img, saturation=-100)
    assert np.allclose(grey[:, :, 0], grey[:, :, 1], atol=1e-5) and np.allclose(grey[:, :, 1], grey[:, :, 2], atol=1e-5)
    assert _apply(img, whites=80).mean() > base.mean()
    assert _apply(img, blacks=-80).mean() < base.mean()


def test_highlights_and_shadows_target_their_range():
    img = IM.ramp()
    bright = _lum(img) > 0.8
    dark = _lum(img) < 0.2
    hi = _apply(img, highlights=-100)
    assert A.luma(hi)[bright].mean() < _lum(img)[bright].mean() - 0.05
    assert abs(A.luma(hi)[dark].mean() - _lum(img)[dark].mean()) < 0.01
    sh = _apply(img, shadows=100)
    assert A.luma(sh)[dark].mean() > _lum(img)[dark].mean() + 0.03
    assert abs(A.luma(sh)[bright].mean() - _lum(img)[bright].mean()) < 0.01


def test_vibrance_boosts_muted_more_than_saturated():
    muted = np.full((8, 8, 4), (0.55, 0.5, 0.45, 1.0), np.float32)
    vivid = np.full((8, 8, 4), (0.9, 0.2, 0.1, 1.0), np.float32)
    chroma = lambda x: (x.max(axis=2) - x.min(axis=2)).mean()  # noqa: E731
    gain_muted = chroma(_apply(muted, vibrance=100)) / chroma(muted[:, :, :3])
    gain_vivid = chroma(_apply(vivid, vibrance=100)) / chroma(vivid[:, :, :3])
    assert gain_muted > gain_vivid > 1.0
    grey = np.full((4, 4, 4), 0.5, np.float32)
    assert np.allclose(_apply(grey, vibrance=100), 0.5)


def test_clarity_and_sharpness_add_local_contrast():
    img = IM.photo()
    grad = lambda x: np.abs(np.diff(A.luma(x), axis=1)).mean()  # noqa: E731
    assert grad(_apply(img, sharpness=100)) > grad(img[:, :, :3]) * 1.3
    assert grad(_apply(img, sharpness=-100)) < grad(img[:, :, :3])
    import cv2
    local_std = lambda x: np.abs(A.luma(x) - cv2.GaussianBlur(A.luma(x), (0, 0), 3)).mean()  # noqa: E731
    assert local_std(_apply(img, clarity=100)) > local_std(img[:, :, :3])


def test_vignette_darkens_corners_not_centre():
    img = np.full((64, 96, 4), 0.6, np.float32)
    img[:, :, 3] = 1
    out = _apply(img, vignette=-100)
    assert out[0, 0].mean() < 0.3 and abs(out[32, 48].mean() - 0.6) < 1e-5
    assert _apply(img, vignette=100)[0, 0].mean() > 0.6


def test_invert_twice_is_identity():
    img = IM.photo()
    once = _apply(img, invert=True)
    again = A.apply(once, img[:, :, 3], Adjustments(invert=True))
    assert np.allclose(again, img[:, :, :3], atol=1e-6)


def test_color_edit_targets_its_hue_only():
    img = np.zeros((10, 30, 4), np.float32)
    img[:, :10] = (0.9, 0.1, 0.1, 1)  # red
    img[:, 10:20] = (0.1, 0.2, 0.9, 1)  # blue
    img[:, 20:] = (0.5, 0.5, 0.5, 1)  # grey
    out = _apply(img, color_edit=[ColorBand(hue=0.0, hue_shift=100, saturation=-50)])
    assert not np.allclose(out[:, :10], img[:, :10, :3], atol=0.02)
    assert np.allclose(out[:, 10:20], img[:, 10:20, :3], atol=1e-5)
    assert np.allclose(out[:, 20:], img[:, 20:, :3], atol=1e-5)


def test_swatch_suggestions_find_dominant_hues():
    img = np.zeros((40, 40, 4), np.float32)
    img[:, :, 3] = 1
    img[:, :25] = (0.9, 0.15, 0.1, 1)  # red-orange (hue ≈ 4°), biggest
    img[:, 25:] = (0.1, 0.3, 0.9, 1)  # blue (hue ≈ 227°)
    hues = A.suggest_swatch_hues(img)
    assert len(hues) == 2
    assert abs(((hues[0] - 4) + 180) % 360 - 180) < 8 and abs(hues[1] - 227) < 8
    assert A.suggest_swatch_hues(np.full((20, 20, 4), 0.5, np.float32)) == []  # grey: no swatches


@pytest.mark.parametrize("name", ("highlights", "shadows", "clarity", "sharpness"))
def test_transparent_background_colour_never_leaks_into_edges(name):
    """Renders often store black (or junk) in fully transparent pixels. Alpha-aware
    blurs must give the same object whatever that hidden colour is."""
    a = IM.render(bg_rgb=0.0)
    b = IM.render(bg_rgb=1.0)
    oa, ob = _apply(a, **{name: 100}), _apply(b, **{name: 100})
    visible = a[:, :, 3] > 0.5
    assert np.abs(oa[visible] - ob[visible]).max() < 1e-4


def _layer(img):
    store = AssetStore()
    info = store.add_bytes(images.encode_png(img, bits=16), ".png")
    return ImageLayer(source=info.id, transform=Transform()), store


CASES = [{k: v} for k in ADJUST_SLIDERS for v in (-100, 100)] + [
    {"invert": True},
    {"color_edit": [ColorBand(hue=20.0, hue_shift=60, saturation=40, lightness=-30)]},
]


# Max relative error of the *effect* (adjusted − unadjusted) between a half-res
# preview level and the downscaled full-res export. Sharpness is inherently
# resolution-bound (a 1 px edge doesn't survive downscaling the same way; judge
# it at 100% zoom), so it gets a looser bound that still fails if its radius
# stops scaling with the level (measured: ~0.44 correct vs ≥0.67 broken).
PREVIEW_TOL = {"sharpness": 0.5, "clarity": 0.2}


@pytest.mark.parametrize("case", CASES)
def test_preview_level_matches_export(case):
    """§15 M3: 'previews match exports', for every control."""
    import cv2
    name = next(iter(case))
    tol = PREVIEW_TOL.get(name, 0.05)
    for make in (IM.photo, IM.render):
        layer, store = _layer(make())

        def r(level, adj):
            layer.adjust = adj
            return unpremultiply(render_layer(layer, store, level))

        f0, f1 = r(1.0, Adjustments()), r(1.0, Adjustments(**case))
        h0, h1 = r(0.5, Adjustments()), r(0.5, Adjustments(**case))
        size = (h0.shape[1], h0.shape[0])
        d_ref = cv2.resize(f1, size, interpolation=cv2.INTER_AREA) - cv2.resize(f0, size, interpolation=cv2.INTER_AREA)
        d_half = h1 - h0
        vis = h0[:, :, 3] > 0.9
        err = np.abs(d_half[vis][:, :3] - d_ref[vis][:, :3]).mean()
        effect = np.abs(d_ref[vis][:, :3]).mean()
        if effect < 0.005:  # control barely touches this image: compare absolutely
            assert err < 0.005, (case, make.__name__, err)
        else:
            assert err / effect < tol, (case, make.__name__, err / effect)


def test_adjust_changes_render_key_and_keeps_alpha():
    layer, store = _layer(IM.render())
    k0 = render_key(layer)
    layer.adjust = Adjustments(brightness=30)
    assert render_key(layer) != k0
    out = render_layer(layer, store)
    src = store.pixels(layer.source)
    assert np.allclose(out[:, :, 3], src[:, :, 3])
