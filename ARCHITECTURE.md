# ARCHITECTURE.md — LookBox (working name)

A personal, offline, Canva-simple image compositor for finishing 3D renders (Marmoset Toolbag) and occasional photos.
This document is the source of truth. Any AI assistant working on this codebase must read it first and follow it.
If a change requires breaking a rule here, **stop and propose an edit to this file first**.

---

## 1. Purpose and non-goals

**Purpose:** take finished renders (plus the odd photo), combine them into a composition, and give it a final look. Keep the UI as friendly as Canva's: big sliders, few choices, no deep menus.

**In scope**
- Layers: position, scale, rotate, flip, z-order, opacity, blend mode
- Adjust panel (see §7)
- Effects: drop shadow, outer glow, outline, blur, gradient transparency (fade)
- Fill layers: solid and gradient backdrops
- Text layers
- LUT filters (.cube)
- Selections and masks: ID-pass pick, lasso, AI click-select (SAM), AI background removal, plus mask refinement
- Upscale (AI, optional feature)
- Import renders plus their passes; export PNG, JPG, TIFF

**Non-goals (do not build)**
- Painting or drawing tools (the only exception is the mask brush, §9)
- Generative fill, outpainting ("Magic Expand"), filling the hole left by an extracted object
- Vector editing, templates, cloud, accounts, collaboration
- Video or animation
- Plugins

---

## 2. Target hardware

The app must run acceptably on **both** of these:

| | Primary | Minimum |
|---|---|---|
| GPU | RTX 4060 (8 GB) | RX 570 (4 GB) |
| CPU | Ryzen 5 3600 | Ryzen 5 3600 |
| RAM | 64 GB | 16 GB |
| OS | Windows 10/11 x64 | Windows 10/11 x64 |

Consequences:
- **No CUDA and no PyTorch at runtime.** All AI goes through **ONNX Runtime + DirectML**, with a CPU fallback.
- Non-AI image processing runs on the CPU (NumPy/OpenCV). It must stay interactive through the preview proxy (§6.3).
- On 4 GB VRAM, only one AI model is loaded at a time (§10).

---

## 3. Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.12 |
| UI | PySide6 (Qt 6). Canvas is `QGraphicsView` / `QGraphicsScene` |
| Image math | NumPy (float32), OpenCV (`opencv-python`) |
| Image I/O | OpenCV (PNG 8/16-bit, JPG, TIFF, EXR). Set `OPENCV_IO_ENABLE_OPENEXR=1` before importing cv2 |
| AI inference | `onnxruntime-directml`. Providers: `["DmlExecutionProvider", "CPUExecutionProvider"]` |
| Text rendering | Qt (`QPainter` into a `QImage`), rasterized into the layer |
| Undo | Qt `QUndoStack` / `QUndoCommand` |
| Packaging | PyInstaller `--onedir`. Models are **not** bundled (§10) |
| Tests | pytest |

Do not add a dependency without asking the user first.

---

## 4. Core principles (non-negotiable)

1. **Non-destructive.** Source pixels are never modified. A layer is a *recipe*: source + parameters. The render re-derives pixels from the recipe.
2. **Every document mutation is an undoable command.** No code outside `commands/` may mutate the `Document`. Slider drags merge into a single command.
3. **The model knows nothing about the UI.** `core/` must not import PySide6. It must be testable headless.
4. **float32 everywhere internally**, values 0–1, straight (un-premultiplied) alpha in storage and premultiplied during compositing. Quantize only on export.
5. **Nothing blocks the UI thread.** Rendering and AI run in workers, are cancellable, and the latest request wins.
6. **The render order is fixed and documented (§6.1).** Don't reorder it ad hoc.

---

## 5. Document model

All model classes are plain `@dataclass`es in `core/model.py`, serializable to JSON.

```
Document
  canvas: Size(w, h)
  background: RGBA | None          # None = transparent
  layers: list[Layer]               # index 0 = bottom
  global_adjust: Adjustments        # applied to the flattened composite
  global_lut: LutRef | None
  assets: dict[AssetId, Asset]      # content-addressed (sha256)

Layer (base)
  id: UUID
  name: str
  visible: bool
  locked: bool
  opacity: float                    # 0–1
  blend_mode: BlendMode             # normal, multiply, screen, overlay, add, soft_light
  transform: Transform              # x, y (canvas px, layer centre), scale_x, scale_y, rotation_deg, flip_h, flip_v
  mask: MaskRef | None              # float32 alpha, asset-backed
  adjust: Adjustments               # per-layer adjust (§7)
  lut: LutRef | None
  effects: Effects                  # §8
  fade: GradientFade | None         # §8

ImageLayer(Layer)
  source: AssetId                   # RGBA float32 once decoded
  crop: Rect | None
  passes: dict[str, AssetId]        # e.g. {"object_id": ..., "material_id": ...} — for ID pick

TextLayer(Layer)
  text, font_family, font_size, weight, italic, color, align, letter_spacing, line_height, box_width

FillLayer(Layer)
  kind: "solid" | "linear" | "radial"
  stops: list[(pos 0–1, RGBA)]
  angle / centre / radius           # depending on kind
```

**Assets** are stored once and referenced by hash. Duplicating a layer never copies pixels.

**Project file:** `.lookbox` = a zip containing `document.json` + `assets/<sha256>.png|exr`. Include `"format_version": 1` and write a migration function whenever the schema changes.

---

## 6. Rendering

### 6.1 Per-layer pipeline (fixed order)

```
source (float32 RGBA, crop applied)
 → mask multiply (alpha *= mask)
 → adjust (§7, colour channels only)
 → LUT
 → layer blur (gaussian, alpha-aware / premultiplied)
 → fade (alpha *= gradient)
 → opacity (alpha *= opacity)
 → effects (§8): shadow and glow are generated from the final alpha and drawn BEHIND the layer; outline is drawn around it
 → transform (affine warp into canvas space, bilinear/area filtering)
 → composite onto the stack using the layer's blend_mode
```

After all layers: `global_adjust` → `global_lut` → output.

### 6.2 Caching
- Each layer caches its pre-transform result, keyed by a hash of (source id, all params that affect it, proxy scale).
- Moving or rotating a layer must NOT invalidate its adjust/effects cache. Only the transform and composite re-run.
- The cache is an LRU with a memory budget (default 4 GB, a setting).

### 6.3 Preview proxy
- Interactive rendering happens at **proxy resolution**: the canvas downscaled so its longest side ≤ viewport size × devicePixelRatio (cap 2048).
- Radius-based parameters (blur, shadow blur, clarity, sharpness, glow) are specified in **full-resolution pixels** and scaled by the proxy factor at render time, so the preview matches the export.
- Export renders at full resolution in a worker with a progress bar.

### 6.4 Threading
- One render worker (`QThread`). The UI posts render requests; any queued request is replaced by the newest one.
- Slider drags are debounced (~30 ms) and render at proxy resolution. The final full-quality proxy render happens on release.

### 6.5 Colour handling (pragmatic, not colour-managed)
- The working space is **display-referred sRGB-encoded float32**. The adjust math in §7 is designed for this.
- 8/16-bit PNG, JPG, TIFF: divide by the max value to reach 0–1.
- EXR (linear): on import, apply exposure (default 0) then the sRGB OETF, and clip to 0–1. Show the exposure slider in the import dialog.
- No ICC profiles. Assume everything is sRGB. Export sRGB.

---

## 7. Adjust panel — exact behaviour

All sliders range −100…+100, default 0. `t = value / 100`. `L` = Rec.709 luma of RGB. Apply in this order:

| # | Control | Implementation |
|---|---|---|
| 1 | Temperature | `R *= 1 + 0.3t`, `B *= 1 − 0.3t`, then renormalize so luma is preserved |
| 2 | Tint | `G *= 1 − 0.3t` (positive = magenta), then renormalize luma |
| 3 | Brightness | Midtone gamma: `x ** (2 ** −t)`. Endpoints stay fixed |
| 4 | Contrast | S-curve around 0.5: blend `x` with `smoothstep(0,1,x)` by t (for t>0); for t<0 lerp toward 0.5 |
| 5 | Highlights | Weight `w = smoothstep(0.5, 1.0, L_blur)`. `L_blur` = luma blurred with sigma ≈ 1% of the image diagonal (gives a local, halo-free-ish result). Gain `x *= 1 + 0.5·t·w` (negative recovers highlights) |
| 6 | Shadows | Weight `w = smoothstep(0.5, 0.0, L_blur)`. Lift/crush: `x = x + 0.5·t·w·(1 − x)` for t>0; `x *= 1 + 0.5·t·w` for t<0 |
| 7 | Whites | Move the white point: levels with `white = 1 − 0.25t` |
| 8 | Blacks | Move the black point: levels with `black = −0.25t` (negative crushes) |
| 9 | Color edit | Per-hue-band HSL: bands at swatch hues with smooth (raised-cosine) falloff; each has hue shift ±30°, saturation, lightness. Swatches are suggested from k-means (k=3–6) on the proxy image, like Canva |
| 10 | Vibrance | Saturation boost weighted by `(1 − s)`, where s is the current HSV saturation: `s' = s + t·0.5·(1 − s)·s_ish` |
| 11 | Saturation | `x = lerp(L, x, 1 + t)` |
| 12 | Clarity | Unsharp mask on L with sigma ≈ 1.5% of the image diagonal, weighted toward midtones (`4·L·(1−L)`), amount `0.6t` |
| 13 | Sharpness | Unsharp mask on L, sigma 1.0 px (full-res), amount `1.5t`, t ≥ 0 only (negative = slight blur) |
| 14 | Vignette | Radial `smoothstep(0.3, 1.0, r)` on the layer bounds; multiply by `1 + 0.8t·v` (negative darkens) |
| 15 | Invert | Toggle: `x = 1 − x` on RGB |

Clip to 0–1 **only at the end**, never between steps.
These formulas are starting points. Tune them against Canva by eye, then **freeze them with golden-image tests** (§14).

UI details: each slider has a numeric box, double-click resets it to 0, a before/after toggle sits in the panel header, and "Reset adjustments" sits at the bottom. **No Auto-adjust in v1.**

---

## 8. Effects

| Effect | Parameters | Notes |
|---|---|---|
| Drop shadow | angle, distance, blur, spread, colour, opacity | From final alpha: dilate by spread, blur, offset, tint, draw behind |
| Outer glow | blur, spread, colour, opacity, blend (screen/add) | Same as shadow with no offset |
| Outline | width, colour, opacity, position (outside/centre) | Distance transform on alpha. Mostly for text |
| Layer blur | radius | Gaussian on premultiplied RGBA |
| Gradient fade | linear/radial, start and end points (layer-local), invert | Multiplies alpha. This is the "gradient transparent" tool |

The shadow canvas must extend beyond layer bounds (pad by blur + distance + spread).
A shadow is a property of its layer. Never implement it as a separate layer.

---

## 9. Selections and masks

All selection tools produce a **float32 mask** at the layer's source resolution.

| Tool | How it works |
|---|---|
| ID pick | Click the canvas → sample the layer's ID pass → mask = pixels matching that colour (tolerance slider, default 0). Shift-click adds, Alt-click subtracts |
| Lasso | Polygonal (click points) and freehand (drag). Rasterize with anti-aliasing |
| Smart select (SAM) | Positive clicks, negative clicks, or a box. Decoder reruns on each click. See §10 |
| Remove background | One click. Whole-image subject mask (§10) |
| Mask brush | Paint add/subtract with a soft round brush of variable size. **Required**: AI masks are never perfect |

**Mask refinement** (applies to any mask): invert, feather (blur), grow/shrink (morphology), smooth, refine edge (guided filter against the source image).

**Actions on a mask:**
- *Apply as layer mask* (non-destructive; this is "remove background").
- *Extract to new layer*: a new ImageLayer referencing the **same source asset** with the mask applied. The original stays untouched underneath. This is our "Magic Grab". The hole is not filled (non-goal).

The mask edit UI is a mode: the canvas shows a red overlay on masked-out areas, with Done and Cancel buttons.

---

## 10. AI subsystem

### 10.1 Interface
```python
class OnnxModel(Protocol):
    key: str                      # "birefnet", "sam_encoder", ...
    vram_class: Literal["small", "medium", "large"]
    def load(self, providers: list[str]) -> None: ...
    def unload(self) -> None: ...
    def run(self, inputs, progress: Callable[[float], None], cancel: Event) -> Any: ...
```
A `ModelManager` owns the sessions. It loads lazily, and in **low-VRAM mode** (auto-detected when VRAM < 6 GB, overridable in Settings) it unloads other models before loading a new one. If creating a DirectML session fails, it retries on CPU and shows a one-line notice.

### 10.2 Models
| Feature | Model | Notes |
|---|---|---|
| Remove background | BiRefNet (general; lite variant in low-VRAM mode), ONNX | Input 1024×1024. Upsample the mask to source size, then refine edge with a guided filter |
| Smart select | SAM 2.1 (tiny or small), ONNX encoder + decoder | Run the encoder **once per image** and cache the embedding by asset hash. The decoder is fast; rerun it per click |
| Upscale | Real-ESRGAN x4plus, ONNX | Tiled: 512 px tiles on ≥6 GB, 256 px in low-VRAM mode, 16 px overlap, feathered blend. Alpha is upscaled separately (bicubic) |

Model sources, filenames, sizes and sha256 values live in `models/models.json`, **not in code**. At build time, verify that the ONNX exports exist and work on DirectML before committing to one. Export from PyTorch yourself if necessary (a dev-only script in `tools/`).

### 10.3 Delivery
- Models download on first use into `%LOCALAPPDATA%\LookBox\models\`, verified by sha256, with a progress dialog and a cancel button.
- The app works fully without any model downloaded. AI buttons show "Download model (xx MB)".

---

## 11. Import and export

**Import**
- Single image: PNG (8/16), JPG, TIFF, EXR, WEBP.
- **Import render set:** pick the beauty image, and the app auto-detects sibling pass files by filename suffix (configurable, defaults `_objectid`, `_materialid`, `_alpha`). Passes attach to `ImageLayer.passes`.
- ID passes must be loaded with **nearest-neighbour** sampling only, never interpolated.
- Drag-and-drop from Explorer onto the canvas.

**Export**
- PNG 8-bit (with **triangular dither** to prevent gradient banding), PNG 16-bit, JPG (quality slider), TIFF 16-bit.
- Scale option: 0.5×, 1×, 2×, or custom.
- Option: export with transparency or flatten onto the background colour.

---

## 12. UI layout (Canva-like)

```
┌──────────────────────────────────────────────────────────────┐
│ Top bar: New · Open · Save · Export · Undo · Redo · Zoom      │
├────┬───────────────────────────────────────────┬─────────────┤
│Tool│                                           │ Context     │
│rail│             Canvas (QGraphicsView)        │ panel:      │
│    │                                           │ Adjust /    │
│ Sel│                                           │ Effects /   │
│ Txt│                                           │ Filters /   │
│ Fil│                                           │ Position /  │
│ Img│                                           │ Layers      │
└────┴───────────────────────────────────────────┴─────────────┘
```
- Dark theme, single accent colour (purple), large hit targets, 8 px spacing grid.
- Selecting a layer shows transform handles on the canvas and its settings in the context panel.
- The Layers list uses thumbnails and drag to reorder, with eye and lock toggles (as in Canva's Position → Layers).
- Shortcuts: Ctrl+Z/Y, Ctrl+S, Ctrl+E (export), Delete, Ctrl+D (duplicate), arrows (nudge 1 px, Shift = 10 px), Space-drag (pan), Ctrl+wheel (zoom).

---

## 13. Project structure

```
lookbox/
  app.py                  # entry point
  core/                   # NO Qt imports allowed
    model.py              # dataclasses (§5)
    assets.py             # content-addressed AssetStore (original bytes + decoded pixels)
    serialize.py          # .lookbox read/write + migrations
    render/
      pipeline.py         # §6.1 orchestration
      adjust.py           # §7, one function per control
      effects.py          # §8
      blend.py
      transform.py
      lut.py
      cache.py
    masks/
      ops.py              # feather, grow, refine edge, ...
      idpick.py
      lasso.py
    io/
      images.py           # load/save, EXR handling, dither
      render_sets.py      # pass detection
  ai/
    manager.py
    birefnet.py
    sam.py
    esrgan.py
  commands/               # the ONLY code that mutates a Document
    edits.py              # plain-Python edits (apply/revert), Qt-free, unit-tested
    qt.py                 # QUndoCommand adapter around edits
  ui/
    main_window.py
    editor.py             # open doc + assets + QUndoStack + selection; the UI's single entry point
    jobs.py               # background workers (export, …)
    canvas/               # QGraphicsView, items, handles, mask overlay
      handles.py          # handle/drag/hit-test geometry — kept Qt-free so it's testable
    panels/               # adjust, effects, filters, position, layers, text
    widgets/              # slider-with-number, colour picker, gradient editor
    theme.qss
  models/models.json
  luts/                   # bundled .cube files
tests/
  golden/                 # reference images
tools/                    # dev-only scripts (onnx export, benchmarks)
```

Keep files under ~400 lines. Split before they grow.

---

## 14. Testing

- **Golden images:** each adjust control and each effect at −100, −50, +50, +100 on 3 fixed test images, compared with a tolerance. Regenerate them only deliberately, with `--update-golden`.
- **Round trip:** save → load → render produces identical pixels.
- **Undo:** for every command, do → undo produces an identical document.
- **Pipeline:** moving a layer does not invalidate its effect cache (assert a cache hit).
- AI tests are skipped automatically if the model isn't downloaded.

---

## 15. Build milestones

Each milestone ends with its acceptance checks passing and a git commit. **Do not start the next milestone early.**

| # | Milestone | Done when |
|---|---|---|
| M1 | Skeleton: window, canvas, import image, ImageLayer, move/scale/rotate handles, layers list, undo/redo, save/load `.lookbox`, PNG export | Import 3 images, arrange them, undo 20 steps, save, reopen, export; everything is identical |
| M2 | Render pipeline + proxy + worker thread + cache | Dragging a layer on a 6000×4000 canvas stays smooth (>30 fps) on the Ryzen 3600 |
| M3 | Adjust panel (§7) + golden tests | All 15 controls work, previews match exports, golden tests pass |
| M4 | Fill layers (solid/gradient), gradient fade, blend modes | A backdrop gradient with no visible banding after 8-bit export |
| M5 | Effects: shadow, glow, outline, blur | The shadow follows the layer when moved, with no re-blur on move |
| M6 | Text layers | Edit text in place, with shadow and outline applied |
| M7 | Render-set import + ID pick + lasso + mask brush + mask refinement + extract to layer | Pick an object from a Toolbag ID pass and extract it to a layer with a clean edge |
| M8 | AI infrastructure + background removal | Works on the RTX 4060 and on CPU fallback; model download flow works |
| M9 | SAM smart select | Click-select in under 300 ms per click after the first encode |
| M10 | LUT filters + global adjust | A .cube file loads; a strength slider works |
| M11 | Upscale | 2× and 4× work with tiling in low-VRAM mode |
| M12 | Packaging | The `--onedir` build runs on a clean Windows machine with no Python installed |

Test on the RX 570 (or force low-VRAM mode + CPU) at M8, M9 and M11, not at the end.

---

## 16. Rules for the AI assistant

1. Read this file before every session. Work on **one milestone** at a time.
2. Never mutate `Document` outside `commands/`. Never import Qt in `core/`.
3. Never modify source assets. Never overwrite the user's files except on an explicit save or export.
4. No new dependencies without asking.
5. No silent fallbacks or bare `except:`. Surface errors to the user in one plain sentence.
6. When you change behaviour covered by a golden test, say so explicitly; don't just regenerate the goldens.
7. If something here is wrong or impractical, propose an edit to this file instead of working around it.
8. At the end of each session: run the tests, summarize what changed, and list the known issues.
