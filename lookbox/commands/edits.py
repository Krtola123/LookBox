"""Undoable edits (ARCHITECTURE §4.2, §16.2).

This is the ONLY place allowed to mutate a Document. Each edit is plain Python
(`apply` / `revert`), so undo is testable without Qt. `commands/qt.py` wraps
these for QUndoStack.
"""

from __future__ import annotations

import copy
from dataclasses import replace

from lookbox.core.model import AssetInfo, Document, Layer, Transform


class Edit:
    text = "Edit"
    # Consecutive edits with the same non-None merge_key collapse into one undo step.
    merge_key: str | None = None

    def apply(self, doc: Document) -> None:
        raise NotImplementedError

    def revert(self, doc: Document) -> None:
        raise NotImplementedError

    def merge(self, newer: "Edit") -> bool:
        """Absorb `newer` (already applied) into self. Return False if not possible."""
        return False


class AddLayer(Edit):
    def __init__(self, layer: Layer, index: int | None = None, asset: AssetInfo | None = None,
                 text: str = "Add layer") -> None:
        self.layer = copy.deepcopy(layer)
        self.index = index
        self.asset = asset
        self.text = text
        self._added_asset = False

    def apply(self, doc: Document) -> None:
        if doc.has_layer(self.layer.id):
            raise ValueError("Layer already exists.")
        if self.asset is not None and self.asset.id not in doc.assets:
            doc.assets[self.asset.id] = self.asset
            self._added_asset = True
        idx = len(doc.layers) if self.index is None else self.index
        doc.layers.insert(idx, copy.deepcopy(self.layer))

    def revert(self, doc: Document) -> None:
        del doc.layers[doc.layer_index(self.layer.id)]
        if self._added_asset:
            del doc.assets[self.asset.id]
            self._added_asset = False


class RemoveLayer(Edit):
    text = "Delete layer"

    def __init__(self, layer_id: str) -> None:
        self.layer_id = layer_id
        self._removed: Layer | None = None
        self._index = -1

    def apply(self, doc: Document) -> None:
        self._index = doc.layer_index(self.layer_id)
        self._removed = doc.layers.pop(self._index)

    def revert(self, doc: Document) -> None:
        doc.layers.insert(self._index, self._removed)


class SetTransform(Edit):
    def __init__(self, layer_id: str, old: Transform, new: Transform,
                 text: str = "Transform", merge_key: str | None = None) -> None:
        self.layer_id = layer_id
        self.old = replace(old)
        self.new = replace(new)
        self.text = text
        self.merge_key = merge_key

    def apply(self, doc: Document) -> None:
        doc.layer(self.layer_id).transform = replace(self.new)

    def revert(self, doc: Document) -> None:
        doc.layer(self.layer_id).transform = replace(self.old)

    def merge(self, newer: Edit) -> bool:
        if not isinstance(newer, SetTransform) or newer.layer_id != self.layer_id:
            return False
        self.new = replace(newer.new)
        return True


class ReorderLayer(Edit):
    text = "Reorder layer"

    def __init__(self, layer_id: str, new_index: int) -> None:
        self.layer_id = layer_id
        self.new_index = new_index
        self._old_index = -1

    def apply(self, doc: Document) -> None:
        self._old_index = doc.layer_index(self.layer_id)
        layer = doc.layers.pop(self._old_index)
        doc.layers.insert(max(0, min(self.new_index, len(doc.layers))), layer)

    def revert(self, doc: Document) -> None:
        layer = doc.layers.pop(doc.layer_index(self.layer_id))
        doc.layers.insert(self._old_index, layer)


_LAYER_PROPS = {"name", "visible", "locked", "opacity", "blend_mode"}


class SetLayerProps(Edit):
    def __init__(self, layer_id: str, text: str = "Change layer", **changes) -> None:
        unknown = set(changes) - _LAYER_PROPS
        if unknown:
            raise ValueError(f"Not editable here: {sorted(unknown)}")
        self.layer_id = layer_id
        self.changes = changes
        self.text = text
        self._old: dict = {}

    def apply(self, doc: Document) -> None:
        layer = doc.layer(self.layer_id)
        self._old = {k: getattr(layer, k) for k in self.changes}
        for k, v in self.changes.items():
            setattr(layer, k, v)

    def revert(self, doc: Document) -> None:
        layer = doc.layer(self.layer_id)
        for k, v in self._old.items():
            setattr(layer, k, v)


def reorder_from(old_ids: list[str], new_ids: list[str]) -> ReorderLayer | None:
    """Find the single-layer move that turns `old_ids` into `new_ids` (both bottom→top).

    Used after a drag-and-drop in the layers list. None if nothing moved or the
    change isn't a single move.
    """
    if old_ids == new_ids or sorted(old_ids) != sorted(new_ids):
        return None
    for i, lid in enumerate(new_ids):
        if old_ids[i] != lid:
            # Either `lid` moved down to i, or old_ids[i] moved up.
            for cand in (lid, old_ids[i]):
                rest_old = [x for x in old_ids if x != cand]
                rest_new = [x for x in new_ids if x != cand]
                if rest_old == rest_new:
                    return ReorderLayer(cand, new_ids.index(cand))
            return None
    return None


def duplicate(doc: Document, layer_id: str, offset: float = 20.0) -> AddLayer:
    """Build (don't apply) an edit that duplicates a layer just above itself.

    The copy references the same asset, so no pixels are copied (§5).
    """
    from lookbox.core.model import new_id

    src = doc.layer(layer_id)
    dup = copy.deepcopy(src)
    dup.id = new_id()
    dup.name = f"{src.name} copy"
    dup.locked = False
    dup.transform = replace(src.transform, x=src.transform.x + offset, y=src.transform.y + offset)
    return AddLayer(dup, index=doc.layer_index(layer_id) + 1, text="Duplicate layer")
