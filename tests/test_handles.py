import math

import numpy as np

from lookbox.core.assets import AssetStore
from lookbox.core.io import images
from lookbox.core.model import Document, ImageLayer, Size, Transform
from lookbox.core.render.transform import corners
from lookbox.ui.canvas import handles as H


def test_handles_sit_on_corners_and_edges():
    t = Transform(x=100, y=50)
    pos = H.handle_positions(t, 40, 20, zoom=1.0)
    assert np.allclose(pos["nw"], [80, 40]) and np.allclose(pos["se"], [120, 60])
    assert np.allclose(pos["e"], [120, 50])
    assert np.allclose(pos[H.ROTATE], [100, 60 + H.ROTATE_OFFSET_PX])
    # Rotate handle distance is in screen px: at 2× zoom it's half as far in canvas px.
    assert np.allclose(H.handle_positions(t, 40, 20, zoom=2.0)[H.ROTATE], [100, 60 + H.ROTATE_OFFSET_PX / 2])


def test_rotate_handle_stays_below_when_flipped():
    t = Transform(x=100, y=50, flip_v=True)
    assert H.handle_positions(t, 40, 20, 1.0)[H.ROTATE][1] > 60


def test_handle_hit_radius_is_screen_space():
    t = Transform(x=100, y=50)
    assert H.handle_at(t, 40, 20, 1.0, 81, 41) == "nw"
    assert H.handle_at(t, 40, 20, 1.0, 100, 50) is None
    assert H.handle_at(t, 40, 20, 4.0, 83, 43) is None  # 3 canvas px = 12 screen px at 4×


def test_corner_scale_is_uniform_and_anchors_opposite_corner():
    t0 = Transform(x=100, y=50, rotation_deg=30)
    anchor = corners(t0, 40, 20)[0]  # nw stays put when dragging se
    se = H.handle_positions(t0, 40, 20, 1.0)["se"]
    target = anchor + (se - anchor) * 2.0
    t1 = H.drag_scale(t0, 40, 20, "se", tuple(target))
    assert math.isclose(t1.scale_x, 2.0, rel_tol=1e-9) and math.isclose(t1.scale_y, 2.0, rel_tol=1e-9)
    assert np.allclose(corners(t1, 40, 20)[0], anchor)
    assert np.allclose(corners(t1, 40, 20)[2], target)


def test_edge_scale_changes_one_axis():
    t0 = Transform(x=100, y=50, rotation_deg=90, flip_h=True)
    e = H.handle_positions(t0, 40, 20, 1.0)["e"]
    w_pt = H.handle_positions(t0, 40, 20, 1.0)["w"]
    target = w_pt + (e - w_pt) * 1.5
    t1 = H.drag_scale(t0, 40, 20, "e", tuple(target))
    assert math.isclose(t1.scale_x, 1.5) and t1.scale_y == 1.0 and t1.flip_h
    assert np.allclose(H.handle_positions(t1, 40, 20, 1.0)["w"], w_pt)


def test_scale_from_centre_and_never_flips():
    t0 = Transform(x=100, y=50)
    t1 = H.drag_scale(t0, 40, 20, "e", (140, 50), from_centre=True)
    assert math.isclose(t1.scale_x, 2.0) and t1.x == 100
    t2 = H.drag_scale(t0, 40, 20, "e", (0, 50))  # dragged past the anchor
    assert t2.scale_x == H.MIN_SIZE_PX / 40 and not t2.flip_h


def test_free_corner_scale():
    t0 = Transform(x=100, y=50)
    t1 = H.drag_scale(t0, 40, 20, "se", (160, 60), free=True)
    assert math.isclose(t1.scale_x, 2.0) and math.isclose(t1.scale_y, 1.0)


def test_rotate_and_snaps():
    t0 = Transform(x=0, y=0)
    t1 = H.drag_rotate(t0, (10, 0), (0, 10))
    assert math.isclose(t1.rotation_deg, 90.0)
    assert H.drag_rotate(t0, (10, 0), (10, 0.2)).rotation_deg == 0.0  # 1.1° soft-snaps to 0
    assert H.drag_rotate(t0, (10, 0), (10, 1.5)).rotation_deg > 8.0  # 8.5° doesn't
    assert H.drag_rotate(t0, (10, 0), (10, 3.0), snap_15=True).rotation_deg == 15.0
    assert math.isclose(H.drag_rotate(Transform(rotation_deg=170), (10, 0), (0, 10)).rotation_deg, -100.0)


def test_move_constrained():
    t = H.drag_move(Transform(x=5, y=5), (0, 0), (10, 3), constrain=True)
    assert (t.x, t.y) == (15, 5)


def test_layer_at_is_alpha_aware_and_skips_locked():
    img = np.zeros((10, 10, 4), np.float32)
    img[:, :5] = (1, 0, 0, 1)  # left half opaque, right half transparent
    store = AssetStore()
    info = store.add_bytes(images.encode_png(img), ".png")
    solid = store.add_bytes(images.encode_png(np.ones((10, 10, 4), np.float32)), ".png")
    doc = Document(canvas=Size(w=10, h=10), assets={info.id: info, solid.id: solid})
    doc.layers = [
        ImageLayer(id="bottom", source=solid.id, transform=Transform(x=5, y=5)),
        ImageLayer(id="top", source=info.id, transform=Transform(x=5, y=5)),
    ]
    assert H.layer_at(doc, store, 2, 5) == "top"
    assert H.layer_at(doc, store, 8, 5) == "bottom"  # click passes through transparency
    doc.layers[1].locked = True
    assert H.layer_at(doc, store, 2, 5) == "bottom"
    assert H.layer_at(doc, store, 50, 50) is None


def test_fill_layers_size_and_hit_by_box():
    from lookbox.core.model import Fill, FillLayer, GradientStop
    doc = Document(canvas=Size(w=100, h=100))
    backdrop = FillLayer(id="bg", width=100, height=100, transform=Transform(x=50, y=50))
    doc.layers = [backdrop]
    assert H.layer_size(doc, backdrop) == (100, 100)
    assert H.layer_at(doc, AssetStore(), 10, 90) == "bg"  # anywhere inside the box
    assert H.layer_at(doc, AssetStore(), 150, 50) is None
    backdrop.locked = True
    assert H.layer_at(doc, AssetStore(), 10, 90) is None  # locked backdrops don't steal clicks
    clear = FillLayer(id="clear", width=100, height=100, transform=Transform(x=50, y=50),
                      fill=Fill(kind="solid", stops=[GradientStop(pos=0, color=(0, 0, 0, 0))]))
    doc.layers = [clear]
    assert H.layer_at(doc, AssetStore(), 10, 90) is None  # fully transparent fill isn't clickable


def _snap_doc():
    from lookbox.core.model import FillLayer
    doc = Document(canvas=Size(w=1000, h=800))
    other = FillLayer(id="other", width=100, height=100, transform=Transform(x=700, y=300))
    doc.layers = [other]
    return doc


def test_snap_move_to_canvas_edges_and_centre():
    doc = _snap_doc()
    targets = H.snap_targets(doc, exclude="me")
    t, guides = H.snap_move(Transform(x=55, y=400), 100, 100, targets, zoom=1.0)  # left edge at 5 → 0
    assert (t.x, t.y) == (50, 400) and ("v", 0.0) in guides and ("h", 400.0) in guides  # centre on centre
    t, guides = H.snap_move(Transform(x=503, y=733), 100, 100, targets, zoom=1.0)
    assert t.x == 500 and ("v", 500.0) in guides  # centre → canvas centre


def test_snap_threshold_is_in_screen_pixels():
    targets = H.snap_targets(_snap_doc(), exclude=None)
    t, _ = H.snap_move(Transform(x=56, y=123), 100, 100, targets, zoom=1.0)
    assert t.x == 50  # 6 px away at zoom 1: within the 8 screen-px threshold → snaps
    t, g = H.snap_move(Transform(x=60, y=123), 100, 100, targets, zoom=1.0)
    assert t.x == 60 and not [x for x in g if x[0] == "v"]  # 10 px away: no snap
    t, _ = H.snap_move(Transform(x=60, y=123), 100, 100, targets, zoom=0.5)
    assert t.x == 50  # zoomed out: 10 canvas px = 5 screen px → snaps


def test_snap_to_other_layers_and_rotated_bbox():
    doc = _snap_doc()
    targets = H.snap_targets(doc, exclude="me")
    t, guides = H.snap_move(Transform(x=804, y=100), 100, 100, targets, zoom=1.0)  # left edge at 754
    assert t.x == 800 and ("v", 750.0) in guides  # my left edge onto the other layer's right edge
    rot = Transform(x=74, y=400, rotation_deg=45)  # 100 px square rotated: half-extent ≈ 70.7
    t, _ = H.snap_move(rot, 100, 100, targets, zoom=1.0)
    assert abs(H.bbox(t, 100, 100)[0]) < 1e-6  # rotated bounds snap to the canvas edge


def test_snap_point_for_resize():
    targets = H.snap_targets(_snap_doc(), exclude=None)
    x, y, g = H.snap_point(Transform(), "se", 996, 403, targets, zoom=1.0)
    assert (x, y) == (1000, 400) and len(g) == 2
    x, y, g = H.snap_point(Transform(), "e", 996, 403, targets, zoom=1.0)
    assert (x, y) == (1000, 403)  # an edge handle only moves x
    x, y, g = H.snap_point(Transform(rotation_deg=30), "se", 996, 403, targets, zoom=1.0)
    assert (x, y, g) == (996, 403, [])  # off-axis rotation: no resize snapping


def test_canvas_to_source_for_brushing_handles_scale_rotation_and_crop():
    doc = Document(canvas=Size(w=400, h=400))
    from lookbox.core.model import AssetInfo
    doc.assets["a"] = AssetInfo(id="a", ext=".png", width=200, height=100)
    layer = ImageLayer(id="L", source="a", transform=Transform(x=200, y=200, scale_x=2, scale_y=2))
    assert np.allclose(H.canvas_to_source(doc, layer, 200, 200), (100, 50))  # centre → centre
    assert np.allclose(H.canvas_to_source(doc, layer, 0, 100), (0, 0))  # top-left corner
    layer.crop = (50, 20, 100, 60)  # 100×60 window starting at (50, 20)
    assert np.allclose(H.canvas_to_source(doc, layer, 200, 200), (100, 50))  # crop centre = (50+50, 20+30)
    layer.transform.rotation_deg = 90
    x, y = H.canvas_to_source(doc, layer, 200 + 2 * 10, 200)  # 10 source px "down" after rotating 90°
    assert np.allclose((x, y), (100, 40))
