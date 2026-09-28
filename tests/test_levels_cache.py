import threading

import numpy as np
import pytest

from lookbox.core.assets import AssetStore
from lookbox.core.io import images
from lookbox.core.model import Document, ImageLayer, Size, Transform
from lookbox.core.render import levels
from lookbox.core.render.cache import LRUCache
from lookbox.core.render.pipeline import (Cancelled, premultiply, render, render_key, render_layer,
                                          unpremultiply)
from lookbox.core.render.transform import level_matrix
from tests.helpers import make_rgba


def _layer_store(img):
    store = AssetStore()
    info = store.add_bytes(images.encode_png(img), ".png")
    return ImageLayer(source=info.id, transform=Transform(x=img.shape[1] / 2, y=img.shape[0] / 2)), store, info


def test_render_key_ignores_placement_only():
    a = ImageLayer(source="s")
    b = ImageLayer(source="s", name="other", visible=False, locked=True, opacity=0.3,
                   transform=Transform(x=5, rotation_deg=40))
    assert render_key(a) == render_key(b)  # moving/hiding never invalidates the cache
    assert render_key(a) != render_key(ImageLayer(source="s", crop=(0, 0, 4, 4)))
    assert render_key(a) != render_key(ImageLayer(source="t"))


def test_choose_level_is_power_of_two_just_above():
    assert levels.choose_level(1.7) == 1.0
    assert levels.choose_level(1.0) == 1.0
    assert levels.choose_level(0.5) == 0.5
    assert levels.choose_level(0.3) == 0.5
    assert levels.choose_level(0.2) == 0.25
    assert levels.choose_level(1e-6) == levels.MIN_LEVEL
    assert levels.on_screen_scale(Transform(scale_x=0.5, scale_y=2.0), 0.25) == 0.5


def test_render_layer_level_is_area_average():
    img = make_rgba(64, 32, seed=1, alpha=False)
    layer, store, _ = _layer_store(img)
    full = render_layer(layer, store, 1.0)
    assert np.allclose(full, premultiply(img))
    quarter = render_layer(layer, store, 0.25)
    assert quarter.shape == (8, 16, 4)
    assert np.allclose(quarter[0, 0], full[:4, :4].mean(axis=(0, 1)), atol=1e-5)


def test_level_matrix_places_level_like_full_res():
    img = make_rgba(64, 48, seed=2, alpha=False)
    layer, store, _ = _layer_store(img)
    t = Transform(x=50, y=40, scale_x=0.8, scale_y=0.8, rotation_deg=17)
    full = render_layer(layer, store, 1.0)
    half = render_layer(layer, store, 0.5)
    # Warp the half-res level with its own matrix; compare against full-res placement.
    from lookbox.core.render import transform as T

    def place(src, m):
        import cv2
        out = cv2.warpAffine(src, T._to_cv(m)[:2], (100, 80), flags=cv2.INTER_LINEAR)
        return unpremultiply(out)

    a = place(full, T.layer_matrix(t, 64, 48))
    b = place(half, level_matrix(t, 64, 48, 32, 24))
    inner = np.s_[30:50, 40:60]  # well inside the layer, away from edges
    assert np.abs(a[inner] - b[inner]).mean() < 0.12  # same content, just softer
    assert np.allclose(a[inner].mean(axis=(0, 1)), b[inner].mean(axis=(0, 1)), atol=0.03)


def test_display_bgra_order_and_premult_invariant():
    exact = np.array([[[1.0, 128 / 255, 0.0, 1.0]]], np.float32)  # on the 8-bit grid: no dither
    assert levels.to_display_bgra(premultiply(exact), dither=False).tolist() == [[[0, 128, 255, 255]]]  # BGRA
    rnd = premultiply(make_rgba(50, 50, seed=3))
    q = levels.to_display_bgra(rnd)
    assert np.all(q[:, :, :3] <= q[:, :, 3:4])
    assert np.array_equal(q, levels.to_display_bgra(rnd))  # deterministic


def test_display_dither_removes_banding():
    ramp = np.tile(np.linspace(0.10, 0.14, 1024, dtype=np.float32)[None, :, None], (64, 1, 4))
    ramp[:, :, 3] = 1.0
    q = levels.to_display_bgra(ramp).astype(np.float32) / 255.0
    col = q[:, :, 2].mean(axis=0)  # red channel sits at index 2 (BGRA)
    smooth = np.convolve(col, np.ones(31) / 31, mode="valid")
    ideal = np.convolve(ramp[0, :, 0], np.ones(31) / 31, mode="valid")
    assert np.abs(smooth - ideal).max() < 0.25 / 255  # tracks the ramp; no stair steps
    plain = levels.to_display_bgra(ramp, dither=False).astype(np.float32) / 255.0
    stairs = np.convolve(plain[:, :, 2].mean(axis=0), np.ones(31) / 31, mode="valid")
    assert np.abs(stairs - ideal).max() > 0.3 / 255  # sanity: without dither it *does* band


def test_needs_dither():
    from lookbox.core.model import Adjustments, FillLayer, GradientFade, ImageLayer
    assert not levels.needs_dither(ImageLayer(), 1.0)
    assert levels.needs_dither(ImageLayer(), 0.5)
    assert levels.needs_dither(ImageLayer(adjust=Adjustments(brightness=1)), 1.0)
    assert levels.needs_dither(ImageLayer(fade=GradientFade()), 1.0)
    assert levels.needs_dither(FillLayer(), 1.0)


def test_lru_evicts_oldest_by_bytes():
    c = LRUCache(budget_bytes=300)
    for k in "abc":
        c.put(k, np.zeros(100, np.uint8))
    assert c.get("a") is not None  # touch a → b is now oldest
    c.put("d", np.zeros(100, np.uint8))
    assert "b" not in c and "a" in c and "d" in c
    assert c.bytes_used == 300
    c.put("huge", np.zeros(1000, np.uint8))
    assert "huge" not in c and c.bytes_used == 300
    c.put("a", np.zeros(50, np.uint8))  # replacing an entry updates the byte count
    assert c.bytes_used == 250


def test_render_cancel_and_progress():
    img = make_rgba(8, 8, seed=4)
    layer, store, info = _layer_store(img)
    doc = Document(canvas=Size(w=8, h=8), assets={info.id: info}, layers=[layer, ImageLayer(
        source=info.id, transform=Transform(x=4, y=4))])
    seen = []
    render(doc, store, progress=seen.append)
    assert seen == [0.5, 1.0]
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        render(doc, store, cancel=ev)


def test_opacity_applies_to_whole_layer_result():
    img = np.ones((4, 4, 4), np.float32)
    layer, store, info = _layer_store(img)
    layer.opacity = 0.25
    doc = Document(canvas=Size(w=4, h=4), assets={info.id: info}, layers=[layer])
    out = render(doc, store)
    assert np.allclose(out[1, 1], (1, 1, 1, 0.25), atol=1 / 255)


def test_thumbnail_fits_and_keeps_average():
    big = np.zeros((4000, 6000, 4), np.float32)
    big[:, :, 3] = 1
    big[:, 3000:, 0] = 1  # right half red
    th = levels.thumbnail_array(big, 44)
    assert th.shape == (29, 44, 4)
    assert abs(th[:, :, 0].mean() - 0.5) < 0.03
    small = np.ones((10, 20, 4), np.float32)
    assert levels.thumbnail_array(small, 44).shape == (10, 20, 4)  # never upscales
