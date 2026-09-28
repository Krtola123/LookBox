"""M1 acceptance (ARCHITECTURE §15): import 3 images, arrange them, undo 20
steps, save, reopen, export; everything is identical.

Uses the Qt-free core + edits. The UI drives exactly these edits.
"""

import copy
import os
from dataclasses import replace

import numpy as np

from lookbox.commands import edits
from lookbox.core import serialize
from lookbox.core.assets import AssetStore
from lookbox.core.io import images
from lookbox.core.model import Document, ImageLayer, Size, Transform
from lookbox.core.render.pipeline import render
from tests.helpers import tmp_png


class History:
    """Minimal stand-in for QUndoStack."""

    def __init__(self, doc):
        self.doc, self.done, self.undone = doc, [], []

    def push(self, e):
        e.apply(self.doc)
        self.done.append(e)
        self.undone.clear()

    def undo(self):
        e = self.done.pop()
        e.revert(self.doc)
        self.undone.append(e)

    def redo(self):
        e = self.undone.pop()
        e.apply(self.doc)
        self.done.append(e)


def test_m1_acceptance(tmp_path):
    store = AssetStore()
    doc = Document(canvas=Size(w=320, h=240), background=(0.1, 0.1, 0.12, 1.0))
    h = History(doc)

    # Import 3 images (like File → Import, centred).
    for i, (w, hh) in enumerate([(200, 150), (120, 90), (64, 64)]):
        info = store.add_file(tmp_png(tmp_path, f"img{i}.png", w, hh, seed=10 + i))
        h.push(edits.AddLayer(ImageLayer(name=info.name, source=info.id,
                                         transform=Transform(x=160, y=120)), asset=info))
    ids = [layer.id for layer in doc.layers]

    snapshots = []
    rng = np.random.default_rng(0)
    for step in range(25):  # arrange: moves, scales, rotations, reorders, visibility
        snapshots.append(copy.deepcopy(doc))
        lid = ids[step % 3]
        kind = step % 5
        if kind == 3:
            h.push(edits.ReorderLayer(lid, int(rng.integers(0, 3))))
        elif kind == 4:
            h.push(edits.SetLayerProps(lid, visible=not doc.layer(lid).visible))
        else:
            t = doc.layer(lid).transform
            h.push(edits.SetTransform(lid, t, replace(
                t,
                x=float(rng.uniform(0, 320)), y=float(rng.uniform(0, 240)),
                scale_x=float(rng.uniform(0.3, 1.5)), scale_y=float(rng.uniform(0.3, 1.5)),
                rotation_deg=float(rng.uniform(-180, 180)), flip_h=bool(rng.integers(0, 2)),
            )))
    final = copy.deepcopy(doc)
    final_px = render(doc, store)

    for _ in range(20):
        h.undo()
    assert doc == snapshots[5]
    for _ in range(20):
        h.redo()
    assert doc == final
    assert np.array_equal(render(doc, store), final_px)

    path = str(tmp_path / "scene.lookbox")
    serialize.save(path, doc, store)
    doc2, store2 = serialize.load(path)
    assert doc2 == final
    px2 = render(doc2, store2)
    assert np.array_equal(px2, final_px)

    out = str(tmp_path / "export.png")
    images.save_png(out, px2)
    with open(out, "rb") as fh:
        exported = images.decode(fh.read(), ".png")
    assert np.array_equal(exported, images.quantize8(final_px).astype(np.float32) / 255.0)
    assert os.path.getsize(out) > 0
