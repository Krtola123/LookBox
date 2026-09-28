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
    """Add a layer, plus the assets it needs that the document doesn't have yet
    (`asset` = its source; `extra_assets` = passes, mask). Undo takes them out again."""

    def __init__(self, layer: Layer, index: int | None = None, asset: AssetInfo | None = None,
                 text: str = "Add layer", extra_assets: tuple[AssetInfo, ...] = ()) -> None:
        self.layer = copy.deepcopy(layer)
        self.index = index
        self.asset = asset
        self.assets = [a for a in (asset, *extra_assets) if a is not None]
        self.text = text
        self._added: list[str] = []

    def apply(self, doc: Document) -> None:
        if doc.has_layer(self.layer.id):
            raise ValueError("Layer already exists.")
        for a in self.assets:
            if a.id not in doc.assets:
                doc.assets[a.id] = a
                self._added.append(a.id)
        idx = len(doc.layers) if self.index is None else self.index
        doc.layers.insert(idx, copy.deepcopy(self.layer))

    def revert(self, doc: Document) -> None:
        del doc.layers[doc.layer_index(self.layer.id)]
        for aid in self._added:
            del doc.assets[aid]
        self._added = []


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


class SetAdjustments(Edit):
    """Replace a layer's Adjustments. Slider drags pass a merge_key unique to that
    drag, so one drag = one undo step (§4.2)."""

    def __init__(self, layer_id: str, old, new, text: str = "Adjust", merge_key: str | None = None) -> None:
        self.layer_id = layer_id
        self.old = copy.deepcopy(old)
        self.new = copy.deepcopy(new)
        self.text = text
        self.merge_key = merge_key

    def apply(self, doc: Document) -> None:
        doc.layer(self.layer_id).adjust = copy.deepcopy(self.new)

    def revert(self, doc: Document) -> None:
        doc.layer(self.layer_id).adjust = copy.deepcopy(self.old)

    def merge(self, newer: Edit) -> bool:
        if not isinstance(newer, SetAdjustments) or newer.layer_id != self.layer_id:
            return False
        self.new = copy.deepcopy(newer.new)
        return True


_STYLE_FIELDS = {"opacity", "blend_mode", "fade", "fill", "width", "height", "effects"}


class SetLayerField(Edit):
    """Change one layer setting (opacity, blend mode, fade, fill, fill size).

    Slider drags pass a per-drag merge_key: one drag = one undo step."""

    def __init__(self, layer_id: str, name: str, old, new, text: str = "Change layer",
                 merge_key: str | None = None) -> None:
        if name not in _STYLE_FIELDS:
            raise ValueError(f"Not editable with SetLayerField: {name}")
        self.layer_id, self.name = layer_id, name
        self.old, self.new = copy.deepcopy(old), copy.deepcopy(new)
        self.text, self.merge_key = text, merge_key

    def apply(self, doc: Document) -> None:
        layer = doc.layer(self.layer_id)
        if not hasattr(layer, self.name):
            raise ValueError(f"A {layer.kind} layer has no '{self.name}'.")
        setattr(layer, self.name, copy.deepcopy(self.new))

    def revert(self, doc: Document) -> None:
        setattr(doc.layer(self.layer_id), self.name, copy.deepcopy(self.old))

    def merge(self, newer: Edit) -> bool:
        if not (isinstance(newer, SetLayerField) and newer.layer_id == self.layer_id
                and newer.name == self.name):
            return False
        self.new = copy.deepcopy(newer.new)
        return True


class SetMask(Edit):
    """Set, change or remove a layer's mask. A new mask image arrives as `asset`
    (added to the document if it isn't there yet; taken out again on undo).
    Edge-slider drags pass a merge_key and no asset, so one drag = one undo step."""

    def __init__(self, layer_id: str, old, new, asset: AssetInfo | None = None, text: str = "Mask",
                 merge_key: str | None = None) -> None:
        self.layer_id = layer_id
        self.old, self.new = copy.deepcopy(old), copy.deepcopy(new)
        self.asset, self.text, self.merge_key = asset, text, merge_key
        self._added_asset = False

    def apply(self, doc: Document) -> None:
        layer = doc.layer(self.layer_id)
        if self.new is not None and layer.kind != "image":
            raise ValueError("Only image layers can have a mask.")
        if self.asset is not None and self.asset.id not in doc.assets:
            doc.assets[self.asset.id] = self.asset
            self._added_asset = True
        layer.mask = copy.deepcopy(self.new)

    def revert(self, doc: Document) -> None:
        doc.layer(self.layer_id).mask = copy.deepcopy(self.old)
        if self._added_asset:
            del doc.assets[self.asset.id]
            self._added_asset = False

    def merge(self, newer: Edit) -> bool:
        if not (isinstance(newer, SetMask) and newer.layer_id == self.layer_id
                and newer.asset is None and self.asset is None):
            return False
        self.new = copy.deepcopy(newer.new)
        return True


class SetPass(Edit):
    """Attach (or with asset None, detach) a render pass on an image layer (§11)."""

    def __init__(self, layer_id: str, kind: str, asset: AssetInfo | None, text: str = "Attach pass") -> None:
        from lookbox.core.io.render_sets import PASS_KINDS

        if kind not in PASS_KINDS:
            raise ValueError(f"Unknown pass: {kind!r}")
        self.layer_id, self.kind, self.asset, self.text = layer_id, kind, asset, text
        self._old: str | None = None
        self._added_asset = False

    def apply(self, doc: Document) -> None:
        layer = doc.layer(self.layer_id)
        if layer.kind != "image":
            raise ValueError("Only image layers have render passes.")
        if self.asset is not None:
            src = doc.assets[layer.source]
            if (self.asset.width, self.asset.height) != (src.width, src.height):
                raise ValueError(f"The pass is {self.asset.width} × {self.asset.height}; the render is "
                                 f"{src.width} × {src.height}. A pass must be the same size as its render.")
        self._old = layer.passes.get(self.kind)
        if self.asset is not None and self.asset.id not in doc.assets:
            doc.assets[self.asset.id] = self.asset
            self._added_asset = True
        if self.asset is None:
            layer.passes.pop(self.kind, None)
        else:
            layer.passes[self.kind] = self.asset.id

    def revert(self, doc: Document) -> None:
        layer = doc.layer(self.layer_id)
        if self._old is None:
            layer.passes.pop(self.kind, None)
        else:
            layer.passes[self.kind] = self._old
        if self._added_asset:
            del doc.assets[self.asset.id]
            self._added_asset = False


class Batch(Edit):
    """Several edits as one undo step (applied in order, reverted in reverse)."""

    def __init__(self, parts: list[Edit], text: str) -> None:
        self.parts, self.text = parts, text

    def apply(self, doc: Document) -> None:
        done = []
        try:
            for part in self.parts:
                part.apply(doc)
                done.append(part)
        except Exception:
            for part in reversed(done):  # leave the document exactly as it was
                part.revert(doc)
            raise

    def revert(self, doc: Document) -> None:
        for part in reversed(self.parts):
            part.revert(doc)


def extract_to_layer(doc: Document, layer_id: str) -> Batch:
    """'Keep the thing': a copy of the layer *with* its mask goes on top as a new layer,
    and the original goes back to showing everything (§9 "Extract to new layer")."""
    from lookbox.core.model import new_id

    src = doc.layer(layer_id)
    if src.mask is None:
        raise ValueError("This layer has no cut-out to extract.")
    cut = copy.deepcopy(src)
    cut.id = new_id()
    cut.name = f"{src.name} cut-out"
    cut.locked = False
    return Batch([AddLayer(cut, index=doc.layer_index(layer_id) + 1),
                  SetMask(layer_id, src.mask, None)], text="Extract to new layer")


def extract_session(doc: Document, layer_id: str, set_mask: "SetMask | None") -> Batch | None:
    """Done + Extract from a mask session as ONE undo step: the session's mask (`set_mask`,
    not applied yet; None = the mask didn't change) goes on a copy above; the original
    shows everything again. None if there's nothing to extract."""
    from lookbox.core.model import new_id

    src = doc.layer(layer_id)
    mask = set_mask.new if set_mask is not None else src.mask
    if mask is None:
        return None
    cut = copy.deepcopy(src)
    cut.id, cut.name, cut.locked = new_id(), f"{src.name} cut-out", False
    cut.mask = copy.deepcopy(mask)
    parts: list[Edit] = [AddLayer(cut, index=doc.layer_index(layer_id) + 1,
                                  asset=set_mask.asset if set_mask is not None else None)]
    if src.mask is not None:
        parts.append(SetMask(layer_id, src.mask, None))
    return Batch(parts, text="Extract selection")


def remove_background(doc: Document, layer_id: str, mask, asset: AssetInfo | None,
                      keep_background: bool) -> Edit:
    """Apply a background-removal mask; optionally keep what was removed as its own
    layer right below (same source, same mask inverted, no effects), so the backdrop
    can be edited, blurred or hidden separately. One undo step either way."""
    from lookbox.core.model import new_id

    layer = doc.layer(layer_id)
    set_mask = SetMask(layer_id, layer.mask, mask, asset=asset, text="Remove background")
    if not keep_background:
        return set_mask
    bg = copy.deepcopy(layer)
    bg.id = new_id()
    bg.name = f"{layer.name} background"
    bg.mask = copy.deepcopy(mask)
    bg.mask.invert = not mask.invert
    bg.effects = type(layer.effects)()  # a shadow/glow on the leftover background makes no sense
    bg.locked = False
    return Batch([set_mask, AddLayer(bg, index=doc.layer_index(layer_id))], text="Remove background")


# Text edits live in their own module (size); re-exported so callers keep using `edits.X`.
from lookbox.commands.text_edits import SetText, change_text, resize_text  # noqa: E402,F401
