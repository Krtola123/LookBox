import numpy as np

from lookbox.core.assets import AssetStore
from lookbox.core.io import images
from lookbox.core.model import Document, ImageLayer, Size, Transform
from lookbox.core.render.pipeline import render


def _store_with(img):
    store = AssetStore()
    info = store.add_bytes(images.encode_png(img), ".png", "t.png")
    return store, info


def _solid(w, h, rgba):
    img = np.zeros((h, w, 4), np.float32)
    img[:] = rgba
    return img


def test_empty_doc_is_transparent_or_background():
    doc = Document(canvas=Size(w=4, h=3))
    store = AssetStore()
    assert np.all(render(doc, store) == 0)
    doc = Document(canvas=Size(w=4, h=3), background=(1.0, 0.0, 0.0, 1.0))
    assert np.allclose(render(doc, store)[1, 1], (1, 0, 0, 1))


def test_over_and_opacity():
    red = _solid(4, 4, (1, 0, 0, 1))
    store, info = _store_with(red)
    blue_store_info = store.add_bytes(images.encode_png(_solid(4, 4, (0, 0, 1, 1))), ".png")
    doc = Document(canvas=Size(w=4, h=4), assets={info.id: info, blue_store_info.id: blue_store_info})
    doc.layers = [
        ImageLayer(source=info.id, transform=Transform(x=2, y=2)),
        ImageLayer(source=blue_store_info.id, transform=Transform(x=2, y=2), opacity=0.5),
    ]
    out = render(doc, store)
    assert np.allclose(out[0, 0], (0.5, 0, 0.5, 1), atol=1e-6)


def test_half_transparent_over_transparent_keeps_colour():
    store, info = _store_with(_solid(2, 2, (0.2, 0.4, 0.6, 0.5)))
    doc = Document(canvas=Size(w=2, h=2), assets={info.id: info})
    doc.layers = [ImageLayer(source=info.id, transform=Transform(x=1, y=1))]
    out = render(doc, store)
    assert np.allclose(out[0, 0], (0.2, 0.4, 0.6, 0.5), atol=2 / 255)


def test_hidden_layer_skipped_and_crop_applied():
    img = _solid(4, 2, (0, 1, 0, 1))
    img[:, 2:] = (1, 1, 1, 1)
    store, info = _store_with(img)
    doc = Document(canvas=Size(w=2, h=2), assets={info.id: info})
    doc.layers = [ImageLayer(source=info.id, crop=(2, 0, 2, 2), transform=Transform(x=1, y=1))]
    assert np.allclose(render(doc, store), 1.0)
    doc.layers[0].visible = False
    assert np.all(render(doc, store) == 0)
