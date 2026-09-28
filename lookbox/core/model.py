"""Document model (ARCHITECTURE §5).

Plain dataclasses, no Qt, no pixels. Pixels live in `core.assets.AssetStore`,
keyed by the asset id stored here. Everything in this module must survive a
JSON round trip and compare equal with `==`, because undo and save tests rely
on that.

M1 subset: Transform, ImageLayer, Document. The remaining Layer fields from §5
(mask, adjust, lut, effects, fade) arrive with their milestones; `from_dict`
fills defaults for anything missing so older files keep loading.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields
from typing import Any

BLEND_MODES = ("normal", "multiply", "screen", "overlay", "add", "soft_light")


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass(kw_only=True)
class Size:
    w: int
    h: int


@dataclass(kw_only=True)
class Transform:
    """Placement of a layer on the canvas.

    x, y: canvas-pixel position of the layer's centre.
    scale_x, scale_y: always > 0. Mirroring is done with flip_h / flip_v.
    rotation_deg: clockwise on screen.
    """

    x: float = 0.0
    y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation_deg: float = 0.0
    flip_h: bool = False
    flip_v: bool = False


@dataclass(kw_only=True)
class AssetInfo:
    """Metadata for an asset. The bytes and decoded pixels are in AssetStore."""

    id: str  # sha256 of the original file bytes
    ext: str  # original file extension incl. dot, e.g. ".png"
    width: int
    height: int
    name: str = ""  # original file name, for display only


@dataclass(kw_only=True)
class Layer:
    id: str = field(default_factory=new_id)
    name: str = "Layer"
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    blend_mode: str = "normal"
    transform: Transform = field(default_factory=Transform)

    kind = "base"  # class attribute, not a field


@dataclass(kw_only=True)
class ImageLayer(Layer):
    source: str = ""  # AssetInfo.id
    crop: tuple[int, int, int, int] | None = None  # x, y, w, h in source pixels
    passes: dict[str, str] = field(default_factory=dict)

    kind = "image"


@dataclass(kw_only=True)
class Document:
    canvas: Size = field(default_factory=lambda: Size(w=1920, h=1080))
    background: tuple[float, float, float, float] | None = None  # None = transparent
    layers: list[Layer] = field(default_factory=list)  # index 0 = bottom
    assets: dict[str, AssetInfo] = field(default_factory=dict)

    # ---- read-only helpers (mutation happens only in commands/) ----
    def layer_index(self, layer_id: str) -> int:
        for i, layer in enumerate(self.layers):
            if layer.id == layer_id:
                return i
        raise KeyError(f"No layer with id {layer_id}")

    def layer(self, layer_id: str) -> Layer:
        return self.layers[self.layer_index(layer_id)]

    def has_layer(self, layer_id: str) -> bool:
        return any(layer.id == layer_id for layer in self.layers)

    def referenced_assets(self) -> set[str]:
        ids: set[str] = set()
        for layer in self.layers:
            if isinstance(layer, ImageLayer):
                ids.add(layer.source)
                ids.update(layer.passes.values())
        return ids


# ---------------------------------------------------------------- JSON


def _plain(obj: Any) -> Any:
    if isinstance(obj, (Size, Transform, AssetInfo)):
        return {f.name: _plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    return obj


def _known(cls: type, data: dict) -> dict:
    """Keep only keys the dataclass knows, so unknown future keys don't crash."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def layer_to_dict(layer: Layer) -> dict:
    out = {"kind": layer.kind}
    for f in fields(layer):
        out[f.name] = _plain(getattr(layer, f.name))
    return out


def layer_from_dict(data: dict) -> Layer:
    kind = data.get("kind")
    if kind != "image":
        raise ValueError(f"Unknown layer kind: {kind!r}")
    d = _known(ImageLayer, data)
    d["transform"] = Transform(**_known(Transform, d.get("transform", {})))
    if d.get("crop") is not None:
        d["crop"] = tuple(int(v) for v in d["crop"])
    d["passes"] = dict(d.get("passes", {}))
    return ImageLayer(**d)


def document_to_dict(doc: Document) -> dict:
    return {
        "canvas": _plain(doc.canvas),
        "background": _plain(doc.background),
        "layers": [layer_to_dict(layer) for layer in doc.layers],
        "assets": {k: _plain(v) for k, v in doc.assets.items()},
    }


def document_from_dict(data: dict) -> Document:
    bg = data.get("background")
    return Document(
        canvas=Size(**_known(Size, data["canvas"])),
        background=tuple(float(v) for v in bg) if bg is not None else None,
        layers=[layer_from_dict(d) for d in data.get("layers", [])],
        assets={
            k: AssetInfo(**_known(AssetInfo, v))
            for k, v in data.get("assets", {}).items()
        },
    )
