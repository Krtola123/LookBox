"""Undoable page edits (M14, §5a): add, duplicate, delete, move, resize a page, and
copying layers between pages. Page-list edits apply to the Project; resizing applies
to the page (a Document), like any other edit. Plain Python; reached as `edits.X`."""

from __future__ import annotations

import copy
from dataclasses import replace

from lookbox.commands.edits import AddLayer, Edit
from lookbox.core.model import AssetInfo, Document, Layer, Project, Size, new_id


class AddPage(Edit):
    def __init__(self, page: Document, index: int | None = None, text: str = "Add page") -> None:
        self.page, self.index, self.text = copy.deepcopy(page), index, text

    def apply(self, project: Project) -> None:
        if project.has_page(self.page.id):
            raise ValueError("Page already exists.")
        idx = len(project.pages) if self.index is None else self.index
        project.pages.insert(idx, copy.deepcopy(self.page))

    def revert(self, project: Project) -> None:
        del project.pages[project.page_index(self.page.id)]


class RemovePage(Edit):
    text = "Delete page"

    def __init__(self, page_id: str) -> None:
        self.page_id = page_id
        self._removed: Document | None = None
        self._index = -1

    def apply(self, project: Project) -> None:
        if len(project.pages) <= 1:
            raise ValueError("A project always has at least one page.")
        self._index = project.page_index(self.page_id)
        self._removed = project.pages.pop(self._index)

    def revert(self, project: Project) -> None:
        project.pages.insert(self._index, self._removed)


class MovePage(Edit):
    text = "Move page"

    def __init__(self, page_id: str, new_index: int) -> None:
        self.page_id, self.new_index = page_id, new_index
        self._old = -1

    def apply(self, project: Project) -> None:
        self._old = project.page_index(self.page_id)
        page = project.pages.pop(self._old)
        project.pages.insert(max(0, min(self.new_index, len(project.pages))), page)

    def revert(self, project: Project) -> None:
        page = project.pages.pop(project.page_index(self.page_id))
        project.pages.insert(self._old, page)


class SetCanvas(Edit):
    """Resize a page. Layers keep their canvas positions (like Canva's custom resize);
    with `recentre`, everything shifts so the old centre stays in the middle."""

    def __init__(self, old: Size, new: Size, recentre: bool = True, text: str = "Resize page") -> None:
        self.old, self.new, self.recentre, self.text = replace(old), replace(new), recentre, text

    def _shift(self, doc: Document, dx: float, dy: float) -> None:
        for layer in doc.layers:
            layer.transform = replace(layer.transform, x=layer.transform.x + dx, y=layer.transform.y + dy)

    def apply(self, doc: Document) -> None:
        doc.canvas = replace(self.new)
        if self.recentre:
            self._shift(doc, (self.new.w - self.old.w) / 2.0, (self.new.h - self.old.h) / 2.0)

    def revert(self, doc: Document) -> None:
        doc.canvas = replace(self.old)
        if self.recentre:
            self._shift(doc, (self.old.w - self.new.w) / 2.0, (self.old.h - self.new.h) / 2.0)


def fresh_ids(layer: Layer) -> Layer:
    out = copy.deepcopy(layer)
    out.id = new_id()
    return out


def duplicate_page(project: Project, page_id: str) -> AddPage:
    """A copy right after it: same layers (new ids) and the same images (shared, not copied)."""
    src = project.page(page_id)
    page = copy.deepcopy(src)
    page.id = new_id()
    page.name = f"{src.name} copy" if src.name else ""
    page.layers = [fresh_ids(layer) for layer in src.layers]
    return AddPage(page, index=project.page_index(page_id) + 1, text="Duplicate page")


def layer_assets(doc: Document, layer: Layer) -> list[AssetInfo]:
    """Every asset a layer needs (source, passes, mask, filter), from its page."""
    ids = [getattr(layer, "source", None), *getattr(layer, "passes", {}).values(),
           layer.mask.asset if layer.mask is not None else None,
           layer.lut.asset if layer.lut is not None else None]
    return [doc.assets[i] for i in ids if i is not None and i in doc.assets]


def paste_layer(doc: Document, layer: Layer, assets: list[AssetInfo], offset: float = 0.0) -> AddLayer:
    """Paste a copied layer on top of `doc` (any page), with the assets it needs."""
    new = fresh_ids(layer)
    new.locked = False
    if offset:
        new.transform = replace(new.transform, x=new.transform.x + offset, y=new.transform.y + offset)
    return AddLayer(new, extra_assets=tuple(assets), text="Paste")


class SetPageName(Edit):
    def __init__(self, old: str, new: str) -> None:
        self.old, self.new, self.text = old, new, "Rename page"

    def apply(self, doc: Document) -> None:
        doc.name = self.new

    def revert(self, doc: Document) -> None:
        doc.name = self.old
