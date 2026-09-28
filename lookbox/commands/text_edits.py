"""Undoable edits for text layers (M7): content and type settings, corner resize.
Plain Python like the rest of commands/; reached as `edits.SetText` etc."""

from __future__ import annotations

import copy
from dataclasses import replace

from lookbox.commands.edits import Edit
from lookbox.core.model import TEXT_ALIGNS, TEXT_FIELDS, Document, Transform


class SetText(Edit):
    """Change a text layer's content or type settings, and (optionally) its transform,
    since a box that changes size is re-placed so the text grows from its top edge and
    its left edge / centre / right edge by alignment (see text.anchored). Typing and slider drags pass a merge_key: one undo step."""

    def __init__(self, layer_id: str, old: dict, new: dict, old_t: Transform | None = None,
                 new_t: Transform | None = None, text: str = "Edit text", merge_key: str | None = None) -> None:
        unknown = (set(old) | set(new)) - set(TEXT_FIELDS)
        if unknown or set(old) != set(new):
            raise ValueError(f"Not text settings (or old/new differ): {sorted(unknown)}")
        if new.get("align", "left") not in TEXT_ALIGNS:
            raise ValueError(f"Unknown alignment: {new['align']!r}")
        if (old_t is None) != (new_t is None):
            raise ValueError("Give both transforms or neither.")
        self.layer_id = layer_id
        self.old, self.new = copy.deepcopy(old), copy.deepcopy(new)
        self.old_t, self.new_t = old_t, new_t
        self.text, self.merge_key = text, merge_key

    def _set(self, doc: Document, values: dict, t: Transform | None) -> None:
        layer = doc.layer(self.layer_id)
        if layer.kind != "text":
            raise ValueError("Not a text layer.")
        for k, v in values.items():
            setattr(layer, k, copy.deepcopy(v))
        if t is not None:
            layer.transform = replace(t)

    def apply(self, doc: Document) -> None:
        self._set(doc, self.new, self.new_t)

    def revert(self, doc: Document) -> None:
        self._set(doc, self.old, self.old_t)

    def merge(self, newer: Edit) -> bool:
        if not (isinstance(newer, SetText) and newer.layer_id == self.layer_id):
            return False
        for k, v in newer.new.items():  # fields only the newer edit touched: its old is the true old
            self.old.setdefault(k, copy.deepcopy(newer.old[k]))
            self.new[k] = copy.deepcopy(v)
        if newer.new_t is not None:
            if self.old_t is None:
                self.old_t = newer.old_t
            self.new_t = newer.new_t
        return True


def resize_text(doc: Document, layer_id: str, t0: Transform, t1: Transform) -> SetText:
    """A corner-drag on text: fold the scale into the font size (text.baked_resize), so
    the text is re-rendered at its new size instead of stretched pixels."""
    from lookbox.core.render.text import baked_resize

    layer = doc.layer(layer_id)
    fields, t = baked_resize(layer, t1)
    return SetText(layer_id, {k: getattr(layer, k) for k in fields}, fields, t0, t, text="Resize text")


def change_text(doc: Document, layer_id: str, label: str = "Edit text", merge_key: str | None = None,
                **fields) -> SetText | None:
    """Build the edit for new text settings, re-placing the box so it grows from its
    anchor (top edge + left/centre/right by alignment). None if nothing changes."""
    from lookbox.core.render.text import anchored, layer_box

    layer = doc.layer(layer_id)
    old = {k: getattr(layer, k) for k in fields}
    if old == fields:
        return None
    after = replace(layer, **fields)
    t = anchored(layer.transform, layer_box(layer), layer_box(after), after.align)
    return SetText(layer_id, old, fields, layer.transform, t, text=label, merge_key=merge_key)
