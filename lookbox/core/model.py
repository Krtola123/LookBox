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
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

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
class ColorBand:
    """One 'Color edit' swatch (§7 #9): pixels near `hue` get shifted."""

    hue: float  # centre hue of the band, degrees 0–360
    hue_shift: float = 0.0  # −100…100 → ±30°
    saturation: float = 0.0  # −100…100
    lightness: float = 0.0  # −100…100


ADJUST_SLIDERS = ("temperature", "tint", "brightness", "contrast", "highlights", "shadows",
                  "whites", "blacks", "vibrance", "saturation", "clarity", "sharpness", "vignette")


@dataclass(kw_only=True)
class Adjustments:
    """The Adjust panel (§7). Every slider is −100…100, default 0 = no change."""

    temperature: float = 0.0
    tint: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0
    clarity: float = 0.0
    sharpness: float = 0.0
    vignette: float = 0.0
    invert: bool = False
    color_edit: list[ColorBand] = field(default_factory=list)

    def is_identity(self) -> bool:
        return (all(getattr(self, k) == 0 for k in ADJUST_SLIDERS) and not self.invert
                and all(b.hue_shift == 0 and b.saturation == 0 and b.lightness == 0 for b in self.color_edit))


@dataclass(kw_only=True)
class GradientFade:
    """Gradient transparency (§8 'Gradient fade'): multiplies the layer's alpha.

    Parametrised as an axis rather than two points so the panel can drive it with
    sliders: `start`/`end` are positions 0–1 along the axis (linear) or radius
    fractions of the half-diagonal (radial). Fully opaque before `start`, fully
    transparent after `end`.
    """

    kind: str = "linear"  # "linear" | "radial"
    angle_deg: float = 90.0  # linear: direction it fades *towards* (90 = downwards)
    start: float = 0.5
    end: float = 1.0
    invert: bool = False


BLEND_MODES = ("normal", "multiply", "screen", "overlay", "add", "soft_light")
OUTLINE_POSITIONS = ("outside", "center")


# Effects (§8). All sizes are CANVAS pixels and the shadow angle is canvas space
# (a light in the scene), so scaling or rotating a layer doesn't change its shadow.


@dataclass(kw_only=True)
class DropShadow:
    angle_deg: float = 90.0  # direction the shadow falls; 90 = straight down
    distance: float = 20.0
    blur: float = 30.0
    spread: float = 0.0
    squash: float = 0.0  # 0–0.95: flatten towards the layer's bottom edge (floor / contact shadow)
    color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    opacity: float = 0.55


@dataclass(kw_only=True)
class Glow:
    blur: float = 40.0
    spread: float = 0.0
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    opacity: float = 0.6


@dataclass(kw_only=True)
class Outline:
    width: float = 6.0
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    opacity: float = 1.0
    position: str = "outside"  # "outside" | "center"


@dataclass(kw_only=True)
class Effects:
    shadow: DropShadow | None = None
    glow: Glow | None = None
    outline: Outline | None = None
    blur: float = 0.0  # layer blur, canvas px

    def active(self) -> bool:
        return (self.shadow is not None or self.glow is not None or self.outline is not None
                or self.blur > 0)
FILL_KINDS = ("solid", "linear", "radial")


def _default_stops() -> list["GradientStop"]:
    return [GradientStop(pos=0.0, color=(0.23, 0.24, 0.27, 1.0)),
            GradientStop(pos=1.0, color=(0.06, 0.063, 0.075, 1.0))]


@dataclass(kw_only=True)
class GradientStop:
    pos: float  # 0–1
    color: tuple[float, float, float, float]  # straight RGBA, 0–1, sRGB-encoded


@dataclass(kw_only=True)
class Fill:
    """Solid colour or gradient. Solid uses the first stop's colour."""

    kind: str = "radial"
    stops: list[GradientStop] = field(default_factory=_default_stops)
    angle_deg: float = 90.0  # linear: 0 = left→right, 90 = top→bottom
    cx: float = 0.5  # radial centre, layer-local 0–1
    cy: float = 0.45
    radius: float = 0.8  # radial: where the last stop lands, fraction of the half-diagonal


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
    adjust: Adjustments = field(default_factory=Adjustments)
    fade: GradientFade | None = None
    effects: Effects = field(default_factory=Effects)

    kind = "base"  # class attribute, not a field


@dataclass(kw_only=True)
class ImageLayer(Layer):
    source: str = ""  # AssetInfo.id
    crop: tuple[int, int, int, int] | None = None  # x, y, w, h in source pixels
    passes: dict[str, str] = field(default_factory=dict)

    kind = "image"


@dataclass(kw_only=True)
class FillLayer(Layer):
    """A generated backdrop: solid colour or gradient, `width` × `height` px before transform."""

    fill: Fill = field(default_factory=Fill)
    width: int = 1920
    height: int = 1080

    kind = "fill"


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
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [_plain(v) for v in obj]
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


def _layer_common(cls: type, data: dict) -> dict:
    d = _known(cls, data)
    d["transform"] = Transform(**_known(Transform, d.get("transform", {})))
    d["adjust"] = adjustments_from_dict(d.get("adjust", {}))
    fade = d.get("fade")
    d["fade"] = GradientFade(**_known(GradientFade, fade)) if fade is not None else None
    d["effects"] = effects_from_dict(d.get("effects", {}))
    return d


def _colour(v) -> tuple[float, float, float, float]:
    return tuple(float(c) for c in v)


def effects_from_dict(data: dict) -> Effects:
    e = _known(Effects, data)
    for name, cls in (("shadow", DropShadow), ("glow", Glow), ("outline", Outline)):
        if e.get(name) is not None:
            part = _known(cls, e[name])
            if "color" in part:
                part["color"] = _colour(part["color"])
            e[name] = cls(**part)
    return Effects(**e)


def fill_from_dict(data: dict) -> Fill:
    f = _known(Fill, data)
    if "stops" in f:
        f["stops"] = [GradientStop(pos=float(s["pos"]), color=tuple(float(c) for c in s["color"]))
                      for s in f["stops"]]
    return Fill(**f)


def layer_from_dict(data: dict) -> Layer:
    kind = data.get("kind")
    if kind == "image":
        d = _layer_common(ImageLayer, data)
        if d.get("crop") is not None:
            d["crop"] = tuple(int(v) for v in d["crop"])
        d["passes"] = dict(d.get("passes", {}))
        return ImageLayer(**d)
    if kind == "fill":
        d = _layer_common(FillLayer, data)
        d["fill"] = fill_from_dict(d.get("fill", {}))
        return FillLayer(**d)
    raise ValueError(f"Unknown layer kind: {kind!r}")


def adjustments_from_dict(data: dict) -> Adjustments:
    a = _known(Adjustments, data)
    a["color_edit"] = [ColorBand(**_known(ColorBand, b)) for b in a.get("color_edit", [])]
    return Adjustments(**a)


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
