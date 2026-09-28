import numpy as np

from lookbox.core.model import Transform
from lookbox.core.render.pipeline import premultiply, unpremultiply
from lookbox.core.render.transform import canvas_to_local, corners, warp_to_canvas
from tests.helpers import make_rgba


def _place(img, t, cw, ch, scale=1.0):
    """Warp and paste into a full canvas (straight alpha out)."""
    out = np.zeros((round(ch * scale), round(cw * scale), 4), np.float32)
    placed = warp_to_canvas(premultiply(img), t, out.shape[1], out.shape[0], scale)
    if placed is not None:
        patch, (x0, y0) = placed
        out[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]] = patch
    return unpremultiply(out)


def test_corners_identity_and_rotation():
    t = Transform(x=50, y=40, scale_x=1, scale_y=1)
    assert np.allclose(corners(t, 20, 10), [[40, 35], [60, 35], [60, 45], [40, 45]])
    t.rotation_deg = 90  # clockwise: top-left goes to top-right
    assert np.allclose(corners(t, 20, 10)[0], [55, 30])


def test_canvas_to_local_is_inverse():
    t = Transform(x=13.5, y=7.25, scale_x=1.7, scale_y=0.6, rotation_deg=33, flip_h=True)
    for lx, ly in [(0, 0), (12, 3), (19.5, 9.9)]:
        c = corners(t, 20, 10)  # just to exercise; real check below
        m_pt = np.array([lx, ly])
        # map forward using the corner basis
        u = (c[1] - c[0]) / 20
        v = (c[3] - c[0]) / 10
        canvas_pt = c[0] + u * m_pt[0] + v * m_pt[1]
        back = canvas_to_local(t, 20, 10, *canvas_pt)
        assert np.allclose(back, (lx, ly), atol=1e-9)


def test_identity_placement_is_pixel_exact():
    img = make_rgba(8, 6, seed=1, alpha=False)
    out = _place(img, Transform(x=4 + 10, y=3 + 5), 30, 20)
    assert np.allclose(out[5:11, 10:18], img, atol=1e-6)
    assert np.all(out[:5] == 0) and np.all(out[:, :10] == 0)


def test_flip_h_mirrors_exactly():
    img = make_rgba(8, 6, seed=2, alpha=False)
    out = _place(img, Transform(x=4, y=3, flip_h=True), 8, 6)
    assert np.allclose(out, img[:, ::-1], atol=1e-6)


def test_rotate_90_matches_rot90():
    img = make_rgba(8, 6, seed=3, alpha=False)
    # Rotated layer is 6 wide × 8 tall; centre it on a 6×8 canvas.
    out = _place(img, Transform(x=3, y=4, rotation_deg=90), 6, 8)
    assert np.allclose(out, np.rot90(img, k=-1), atol=1e-5)


def test_downscale_uses_area_average():
    img = np.zeros((4, 4, 4), np.float32)
    img[:, :, 3] = 1
    img[::2, ::2, :3] = 1  # checkerboard-ish dots, mean 0.25
    out = _place(img, Transform(x=1, y=1, scale_x=0.5, scale_y=0.5), 2, 2)
    assert np.allclose(out[:, :, 0], 0.25, atol=1e-6)


def test_off_canvas_returns_none():
    img = make_rgba(4, 4)
    assert warp_to_canvas(premultiply(img), Transform(x=-100, y=-100), 10, 10) is None


def test_out_scale_renders_scaled_canvas():
    img = make_rgba(8, 8, seed=5, alpha=False)
    full = _place(img, Transform(x=8, y=8), 16, 16)
    half = _place(img, Transform(x=8, y=8), 16, 16, scale=0.5)
    assert half.shape == (8, 8, 4)
    assert np.allclose(half[2:6, 2:6, :3].mean(), full[4:12, 4:12, :3].mean(), atol=1e-5)
