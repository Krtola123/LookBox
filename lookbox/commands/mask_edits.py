"""Undoable edits for masks and render passes (M6, M8): set a cut-out, attach a pass,
extract to a new layer, remove background. Plain Python like the rest of commands/;
reached as `edits.SetMask` etc."""

from __future__ import annotations

import copy

from lookbox.commands.edits import AddLayer, Batch, Edit
from lookbox.core.model import AssetInfo, Document


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
