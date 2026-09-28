"""Render sets (ARCHITECTURE §11): a beauty render plus its passes.

Renderers name passes differently (Toolbag, Blender, Substance: "shot_objectid.png",
"shot_Object ID.png", "shot-ObjectID.png", "ObjectID_shot.png"...), so names are
compared after dropping case, spaces, dashes, underscores and dots: a file is a
pass of a beauty if its squashed name is the beauty's squashed name plus a known
pass word, before or after it. Anything not detected can be attached by hand.
"""

from __future__ import annotations

import os
import re

import numpy as np

from lookbox.core.io.images import SUPPORTED_EXTS

PASS_KINDS = ("object_id", "material_id", "alpha")
PASS_LABELS = {"object_id": "Object ID", "material_id": "Material ID", "alpha": "Alpha"}
ID_KINDS = ("object_id", "material_id")

# Squashed pass words → kind.
_WORDS = {
    "objectid": "object_id", "objid": "object_id", "object": "object_id", "meshid": "object_id",
    "id": "object_id", "idmap": "object_id", "cryptoobject": "object_id",
    "materialid": "material_id", "matid": "material_id", "material": "material_id",
    "alphamask": "alpha", "alpha": "alpha", "mask": "alpha", "matte": "alpha",
}
_WORD_ORDER = sorted(_WORDS, key=len, reverse=True)  # for guess_kind: longest match first


_FRAME = re.compile(r"^(.*?)[ ._-]+(\d+)$")  # a frame number after a separator: "shot_v02_0001" → 0001


def squash(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def pass_kind(beauty_path: str, other_path: str) -> str | None:
    """The pass kind `other_path` is for `beauty_path`, or None if it isn't one of its passes."""
    bs, os_ = _stem(beauty_path), _stem(other_path)
    b, o = squash(bs), squash(os_)
    if not b or o == b:
        return None
    pairs = [(b, o)]
    bf, of = _FRAME.match(bs), _FRAME.match(os_)
    if bf and of and int(bf.group(2)) == int(of.group(2)):  # same frame: "cam2_0001" / "cam2_objectid_0001"
        pairs.append((squash(bf.group(1)), squash(of.group(1))))
    for bb, oo in pairs:
        for rest in (oo[len(bb):] if oo.startswith(bb) else "", oo[:-len(bb)] if oo.endswith(bb) else ""):
            if rest in _WORDS:
                return _WORDS[rest]
    return None


def guess_kind(path: str) -> str | None:
    """Kind from a lone file name (for attaching by hand): the pass word it ends with
    (frame number aside). None if it doesn't say: the user is asked."""
    stem = _stem(path)
    m = _FRAME.match(stem)
    s = squash(m.group(1) if m else stem)
    for word in _WORD_ORDER:
        if len(word) > 2 and s.endswith(word):
            return _WORDS[word]
    return None


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS


def group(paths: list[str], scan_folders: bool = True) -> list[tuple[str, dict[str, str]]]:
    """Split files into beauty images, each with the passes found for it.

    Passes among `paths` are attached to their beauty instead of becoming layers of
    their own; with `scan_folders`, each beauty's folder is searched for passes too
    (so importing just the beauty brings its ID pass along). The first file found
    wins per kind. A selected file that looks like a pass but isn't attached (a
    second pass of the same kind, say) is imported as a layer: nothing you picked
    disappears. Order follows `paths`.
    """
    paths = [p for p in paths if is_image(p)]
    beauties = [p for p in paths if not any(pass_kind(o, p) for o in paths if o != p)]
    out = []
    attached: set[str] = set()
    for b in beauties:
        passes: dict[str, str] = {}
        candidates = [p for p in paths if p != b]
        if scan_folders:
            folder = os.path.dirname(os.path.abspath(b))
            try:
                names = sorted(os.listdir(folder))
            except OSError:
                names = []
            candidates += [os.path.join(folder, n) for n in names]
        for c in candidates:
            kind = pass_kind(b, c)
            if kind and kind not in passes and is_image(c) and os.path.isfile(c):
                passes[kind] = c
        attached.update(os.path.abspath(p) for p in passes.values())
        out.append((b, passes))
    leftovers = [p for p in paths if p not in beauties and os.path.abspath(p) not in attached]
    out += [(p, {}) for p in leftovers]
    order = {p: i for i, p in enumerate(paths)}
    return sorted(out, key=lambda item: order[item[0]])


def alpha_pass_mask(rgba: np.ndarray, encoded: bool = False) -> np.ndarray:
    """An alpha pass as a mask: its alpha channel if it has one, else its grey level.
    `encoded`: the grey was sRGB-encoded on decode (float/EXR files): undo that, since
    a mask is data (coverage), not a colour."""
    a = rgba[:, :, 3]
    if float(a.min()) < 1.0:
        return np.ascontiguousarray(a, dtype=np.float32)
    g = rgba[:, :, :3].mean(axis=2)
    if encoded:
        g = np.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4)
    return np.ascontiguousarray(g, dtype=np.float32)


def prepare(store, beauty, passes: dict) -> tuple[dict, object, list[str]]:
    """Check a beauty's decoded passes and turn an alpha pass into a mask (runs in the
    import thread). `beauty` and `passes` values are AssetInfo already in `store`.

    Returns (passes to attach {kind: AssetInfo}, mask AssetInfo or None, warnings).
    A pass of the wrong size is dropped with a warning. The alpha pass becomes the
    layer's mask only if the beauty itself is fully opaque (otherwise it already
    carries that alpha, and masking again would double it).
    """
    from lookbox.core.masks.ops import encode_mask_png

    keep, warnings, mask = {}, [], None
    for kind, info in passes.items():
        if (info.width, info.height) != (beauty.width, beauty.height):
            warnings.append(f"{info.name}: {PASS_LABELS[kind]} pass is {info.width} × {info.height}, "
                            f"the render is {beauty.width} × {beauty.height}. Not attached.")
            continue
        keep[kind] = info
    alpha = keep.pop("alpha", None)
    if alpha is not None and float(store.pixels(beauty.id)[:, :, 3].min()) >= 1.0:
        m = alpha_pass_mask(store.pixels(alpha.id), encoded=alpha.ext == ".exr")
        mask = store.add_bytes(encode_mask_png(m), ".png", f"{alpha.name} (mask)")
    return keep, mask, warnings


class ImportItem:
    """One image to add as a layer: its asset, attached passes and (from an alpha pass) mask."""

    def __init__(self, info, passes: dict | None = None, mask=None) -> None:
        self.info, self.passes, self.mask = info, dict(passes or {}), mask

    def extra_assets(self) -> tuple:
        return tuple(self.passes.values()) + ((self.mask,) if self.mask is not None else ())


def import_files(store, paths: list[str], scan_folders: bool = True) -> tuple[list[ImportItem], list[str], list[str]]:
    """Decode files into `store`, grouping render sets (runs in the import thread).
    Returns (items, errors, notes). A pass that fails to load or doesn't fit is
    reported and skipped; the render still imports."""
    from lookbox.core.io.images import ImageError

    items, errors, notes = [], [], []
    for beauty, pass_paths in group(paths, scan_folders):
        try:
            info = store.add_file(beauty)
        except ImageError as exc:
            errors.append(str(exc))
            continue
        except MemoryError:
            errors.append(f"{beauty}: not enough memory to open this image.")
            continue
        loaded = {}
        for kind, p in pass_paths.items():
            try:
                loaded[kind] = store.add_file(p)
            except (ImageError, MemoryError, OSError) as exc:
                errors.append(f"{os.path.basename(p)} ({PASS_LABELS[kind]} pass): {exc}")
        try:
            passes, mask, warnings = prepare(store, info, loaded)
        except MemoryError:  # the render still comes in, just without its passes
            passes, mask, warnings = {}, None, [f"{info.name}: not enough memory to attach its passes."]
        errors.extend(warnings)
        found = [PASS_LABELS[k] for k in passes] + (["Alpha → cut-out"] if mask is not None else [])
        if found:
            notes.append(f"{info.name}: {', '.join(found)}")
        items.append(ImportItem(info, passes, mask))
    return items, errors, notes
