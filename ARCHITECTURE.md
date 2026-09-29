# ARCHITECTURE.md — RRIPP (Reshi Renders' Image Post Processing App; formerly LookBox)

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
| Text rendering | Qt (`QPainter` into a `QImage`), rasterized into the layer. The core owns layout (line breaks, alignment, spacing) and only asks a pluggable engine to measure and draw (§6.1a) |
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
                                    # canvas px at scale 1; box_width None = no wrapping. The box
                                    # size is computed from the layout, never stored. Resizing on
                                    # the canvas bakes the scale into font_size (text stays sharp).

FillLayer(Layer)                    # backdrops; no asset
  fill: Fill                        # kind "solid" | "linear" | "radial", stops [(pos 0–1, RGBA)],
                                    # angle_deg (linear), cx/cy/radius (radial, box-relative)
  width, height                     # size before transform (defaults to the canvas)

Layer.fade: GradientFade | None     # kind linear|radial, angle_deg, start/end (0–1), invert
```

**Assets** are stored once and referenced by hash. Duplicating a layer never copies pixels.

**Project file:** `.rripp` = a zip containing `document.json` + `assets/<sha256>.png|exr`. Include `"format_version": 1` and write a migration function whenever the schema changes.

---

## 6. Rendering

### 6.1 Per-layer pipeline (fixed order)

```
── render_layer(): pre-transform, cacheable (depends only on render_key + level) ──
source (float32 RGBA, crop applied, area-downscaled to the level)
 → mask multiply (alpha *= mask)
 → adjust (§7, colour channels only)
 → LUT
 → fade (alpha *= gradient)
 → premultiply, then pad the image (effects reach past the layer box)
 → layer blur (gaussian on premultiplied RGBA)
 → effects (§8): shadow and glow from the alpha, drawn BEHIND the layer; outline around it
── placement: cheap, never cached ──
 → opacity (whole result × opacity, so fading a layer fades its shadow too, as in Canva/Photoshop)
 → transform (affine warp into canvas space, bilinear/area filtering)
 → composite onto the stack using the layer's blend_mode
```

After all layers: `global_adjust` → `global_lut` → output.

*Changed in M2:* opacity moved after effects (it used to sit before them).
*Changed in M5:* layer blur moved after fade (so it can spread past the box once padded); render_layer returns `Rendered(pixels, pad)` and both placement paths (`warp_to_canvas`, `level_matrix`) take the padding.

### 6.1a Text (M7)
- `core/render/text.py` (Qt-free, tested with a fake engine) does line breaking (greedy, overlong words broken), alignment, line spacing, the box (text + a 0.12 em margin, more for italics) and colouring. A `TextEngine` only measures strings (`metrics`, `advance`) and draws laid-out lines as coverage. `ui/text_engine.py` (Qt) is registered in `app.py` after the `QApplication` exists.
- Qt fonts are resolved against 72-dpi images, so `setPointSizeF(font_size)` is exactly `font_size` px, fractional sizes included; hinting is off so text scales linearly.
- *Found by the Windows CI (M12):* glyphs are filled as outlines (`QPainterPath.addText`, cached per layout), because Qt's `drawText` snaps glyphs per pixel size (a 2× level differed from 1× by 12%). Letter spacing is added by the core (engine advances exclude it; spaced characters are placed at `advance(text[:i]) + i·spacing`), because Qt's font letter spacing didn't show up in its measurements. Qt's offscreen platform on Windows has no fonts (every glyph a box), so Qt tests and the self-test use real windows there.
- Text is the base pixels of the §6.1 pipeline, so adjust, fade, effects, blend modes and opacity apply unchanged.
- Text levels go above 1 (up to 4×, 2× with effects, since blurs cost level² pixels), so zooming in stays crisp; export renders text at the export scale (`export_level`) and `warp_to_canvas(box=...)` places a level ≠ 1 image.
- When text or type settings change the box size, `text.anchored` re-places it so the top edge and the left edge / centre / right edge (by alignment) stay put, rotation and flips included.

### 6.2 Caching
- Each layer caches its `render_layer` result, keyed by `(render_key(layer), level)`. `render_key` hashes every layer field **except** placement ones (id, name, visible, locked, opacity, blend_mode, transform), so new fields invalidate the cache automatically.
- Moving, rotating, hiding or fading a layer never re-renders it.
- The cache is an LRU with a byte budget (default 4 GB, becomes a setting), storing display-ready uint8 BGRA.

### 6.3 Preview levels (replaces the M1 "whole-canvas proxy" plan)
- Measured in M2: compositing the whole canvas on the CPU at proxy size costs ~120 ms/frame, which is far too slow for dragging. So the preview is split:
  - **Per layer, off-thread:** `render_layer` at a power-of-two **level**, the smallest one ≥ the layer's on-screen scale (`zoom × layer scale × devicePixelRatio`). The display never downsamples a level by more than 2×, so there's no shimmer.
  - **On screen, Qt:** each level is a `QGraphicsPixmapItem` placed with `level_matrix` (same matrix as export) and composited by QPainter. Blend modes will use QPainter composition modes (M4).
- While a level renders, the closest cached level stays on screen.
- Radius-based parameters (blur, shadow blur, clarity, sharpness, glow) are specified in **full-resolution pixels** and scaled by the level at render time, so every level looks like a downscaled export.
- **Export** renders everything in float at full resolution (`pipeline.render`), in a cancellable worker with progress. Export is the ground truth; the preview is 8-bit.
- **Blend modes** on screen use QPainter composition modes, which implement the same W3C formulas as `core/render/blend.py` (tested against the spec). Caveat: on a *transparent* canvas the preview blends with the checkerboard while the export blends with transparency; with a backdrop or background colour they match.
- **Dithering:** preview levels are dithered like the PNG export (TPDF noise, cached tile), except untouched images at full resolution, which stay exact.
- *Known future issue:* `global_adjust`/`global_lut` (M10) act on the flattened image, which the Qt composite doesn't have. Plan: preview them with a worker-rendered composite at screen resolution that updates after each change, keeping the Qt composite during drags.

### 6.4 Threading
- Layer renders run in a `QThreadPool` (cores − 2 threads), one job per (layer, level). Results reach the UI thread via queued signals; stale results just sit in the cache.
- Import decoding, project open/save and export are `QThread` jobs (`ui/jobs.py`). Each snapshots what it needs.
- A background save marks the document clean only if nothing changed while it ran (revision counter, not undo index).
- Slider drags are debounced (~30 ms). The canvas reports drag paint cost in the status bar (M2 acceptance readout).

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
| 4 | Contrast | S-curve around 0.5: blend `x` with `smoothstep(0,1,x)` by t (for t>0); for t<0 lerp toward 0.5 by `0.6·|t|` (−100 flattens hard but isn't pure grey) |
| 5 | Highlights | Weight `w = smoothstep(0.5, 1.0, L_blur)`. `L_blur` = luma blurred with sigma ≈ 1% of the image diagonal (gives a local, halo-free-ish result), computed once and shared with Shadows. Gain `x *= 1 + 0.5·t·w` (negative recovers highlights) |
| 6 | Shadows | Weight `w = smoothstep(0.5, 0.0, L_blur)`. Lift/crush: `x = x + 0.5·t·w·(1 − x)` for t>0; `x *= 1 + 0.5·t·w` for t<0 |
| 7 | Whites | Move the white point: levels with `white = 1 − 0.25t` |
| 8 | Blacks | Move the black point: levels with `black = −0.25t` (negative crushes) |
| 9 | Color edit | Per-hue bands at swatch hues, raised-cosine falloff to 0 at ±45°, times "has colour" (greys untouched). Each band: hue shift ±30° (luma-preserving hue rotation), saturation (chroma scale `1 + t`), lightness (`× 1 + 0.5t`). Swatches: k-means (k=4) on hue, circular, from a thumbnail |
| 10 | Vibrance | Chroma scale `1 + t·(1 − s)`, s = HSV saturation: muted colours move most, greys never |
| 11 | Saturation | `x = lerp(L, x, 1 + t)` |
| 12 | Clarity | Unsharp mask on L with sigma ≈ 1.5% of the image diagonal, weighted toward midtones (`4·L·(1−L)`), amount `0.6t` |
| 13 | Sharpness | Unsharp mask on L, sigma 1.0 px (full-res), amount `1.5t`, t ≥ 0 only (negative = slight blur) |
| 14 | Vignette | Radial `smoothstep(0.3, 1.0, r)` on the layer bounds; multiply by `1 + 0.8t·v` (negative darkens) |
| 15 | Invert | Toggle: `x = 1 − x` on RGB |

Clip to 0–1 **only at the end**, never between steps.
These formulas are starting points. Tune them against Canva by eye, then **freeze them with golden-image tests** (§14).

**Alpha-aware blurs.** Highlights/Shadows, Clarity and Sharpness blur with normalised convolution (`blur(x·α)/blur(α)`), so the colour hidden in fully transparent pixels (black in most renders) never bleeds into object edges.

**Resolution independence.** Blur radii are relative to the image diagonal (Highlights/Shadows, Clarity) or scale with the preview level (Sharpness), so every preview level looks like the downscaled export. A test enforces this per control. Sharpness is inherently resolution-bound (a 1 px edge doesn't survive downscaling the same way), so judge it at 100% zoom.

**While dragging a slider**, the canvas renders that layer at half its preview level and swaps in the full level on release. Superseded renders of the same layer are cancelled.

UI details: each slider has a numeric box, double-click resets it to 0, a before/after toggle sits in the panel header, and "Reset adjustments" sits at the bottom. **No Auto-adjust in v1.**

---

## 8. Effects

| Effect | Parameters | Notes |
|---|---|---|
| Drop shadow | direction, distance, blur, spread, floor squash, colour, opacity | From alpha: dilate by spread, squash towards the bottom edge (contact shadow), blur, offset, tint, draw behind. Presets: Soft drop, Contact shadow |
| Outer glow | blur, spread, colour, opacity | Same as shadow with no offset. Drawn normally inside the layer image; for an additive look set the layer's blend mode to Add/Screen (a per-effect blend would need the backdrop, which render_layer doesn't have) |
| Outline | width, colour, opacity, position (outside/centre) | Precise distance transform on alpha, measured from the edge (not pixel centres), anti-aliased |
| Layer blur | radius | Gaussian on premultiplied RGBA, spreads past the box |
| Gradient fade | linear/radial, direction (linear), start and end (0–1 along the axis / radius), invert | Multiplies alpha, smooth ramp. This is the "gradient transparent" tool. *Changed in M4:* an axis + start/end instead of two points, so plain sliders drive it; on-canvas handles can map onto the same fields later |

The render pads the layer by the effects' reach (distance + spread + 3σ of blur) so nothing is clipped.
A shadow is a property of its layer. Never implement it as a separate layer.

**Units:** effect sizes are **canvas pixels** and the shadow direction is **canvas space** (a light in the scene), so resizing or rotating a layer doesn't change its shadow. Consequence: with effects on, `render_key` includes scale/rotation/flip, so rotating or resizing such a layer re-renders it on release (moving still never does). During the drag the preview shows the old render transformed until you let go.

---

## 9. Selections and masks

All selection tools produce a **float32 mask** at the layer's source resolution.

| Tool | How it works |
|---|---|
| ID pick | Click the canvas → sample the layer's ID pass → mask = pixels matching that colour (tolerance slider, default 0). Shift-click adds, Alt-click subtracts. **Soft edges (M8):** renderers anti-alias ID passes, so an exact match sits a pixel inside the beauty's soft edge and looks jagged. Edge pixels are solved as mixes in premultiplied RGBA, p ≈ a·A + c·B1 + (1−a−c)·B2, with B1 the nearest flat other colour and B2 the most different one nearby (transparent if none), so corners where the object meets another object *and* the background work too. Whether the pass was mixed in linear light and sRGB-encoded is measured once from all its edges (mixes are straight lines only in the space they were made in), and the solve runs in that space. Measured on 8×-supersampled synthetic passes (transparent / black / grey backgrounds, both encodings): mean edge error 0.001–0.003 vs ~0.5 for an exact match. `IdPass` stores 16-bit colours, int64 keys and a flat map (≈2 s and ~400 MB for 24 MP, built in the background when Pick starts, freed when the session ends); a click is ≈0.3 s |
| Lasso | Polygonal (click points, Enter or double-click closes) and freehand (drag). Exact scanline fill of sample centres (even-odd) on a supersampled bounding box, area-averaged: true coverage at the edge |
| Smart select (SAM) | Positive clicks, negative clicks, or a box. Decoder reruns on each click. See §10 |
| Remove background | One click. Whole-image subject mask (§10) |
| Mask brush | Paint erase/restore with a soft round brush (size in canvas px, hardness). Works with or without an AI mask (no mask = start fully visible: a manual cut-out). **Required**: AI masks are never perfect |

**Storage (M6):** `Layer.mask = LayerMask(asset, shift, feather, invert)`. The mask image is its own asset (16-bit greyscale PNG, same size as the full source); edge controls are live, in source px, scaled by the preview level, and never baked in.

**Mask refinement:** invert, feather, grow/shrink ("edge shift") are live controls. Refine edge (guided filter against the photo) is optional at removal time, off by default: measured in M6, it makes clean edges slightly *worse* (copies photo noise into the mask) and is only worth trying on hair/fur.

**Actions on a mask:**
- *Apply as layer mask* (non-destructive; this is "remove background").
- *Extract to new layer*: a new ImageLayer referencing the **same source asset** with the mask applied. The original stays untouched underneath. This is our "Magic Grab". The hole is not filled (non-goal).
- *Keep the background* (default on, remembered): Remove background also adds the removed part as a layer right below (same source, same mask inverted, effects cleared) in the same undo step. Known: at soft edges, subject over background isn't perfectly opaque (α + (1−α)² < 1), a faint seam, as in any layer split.

**Mask session (M8):** Pick, Lasso and Brush are tools of one session on the canvas; switch freely. Pick and lasso *replace* the selection, Shift adds, Alt subtracts (`ops.combine`: max / min with the inverse); the brush edits it. Done, or **Extract** (done + extract to new layer, one undo step), or Cancel.

The mask edit UI is a mode: the layer shows unmasked, a red overlay marks hidden areas, Done/Cancel (Enter/Esc), Alt flips erase/restore, [ ] resize. One brush session = one undo step. Switching layers applies the session; so does New/Open/Close (before the save prompt, so painting is never lost silently).

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
As built (M6, `ai/runtime.py`): a `ModelManager` keeps **at most one** model loaded (loading one frees the previous, which suits 4 GB cards), creates sessions lazily with DirectML → CPU fallback, and also falls back to the CPU if the *first GPU run* fails (e.g. out of VRAM), with a one-line notice. Models are plain functions over it (`birefnet.remove_background(run, …)`), so they're testable with a stand-in `run`. Instead of VRAM auto-detection (no dependency-free way on Windows), the user picks the model size once: Best quality (973 MB) or Fast (224 MB).

### 10.2 Models
| Feature | Model | Notes |
|---|---|---|
| Remove background | BiRefNet (onnx-community exports): "Best quality" = BiRefNet-ONNX (973 MB), "Fast" = BiRefNet_lite-ONNX (224 MB), fp32 | Input 1024×1024 RGB, /255, ImageNet mean/std (per the export's preprocessor_config.json). Output logits or probabilities, detected; sigmoid only when needed. Bilinear upsample to source size; refine edge optional (§9) |
| Smart select | SAM 2.1 (tiny or small), ONNX encoder + decoder | Run the encoder **once per image** and cache the embedding by asset hash. The decoder is fast; rerun it per click |
| Upscale | Real-ESRGAN x4plus, ONNX | Tiled: 512 px tiles on ≥6 GB, 256 px in low-VRAM mode, 16 px overlap, feathered blend. Alpha is upscaled separately (bicubic) |

Model sources, filenames, sizes and sha256 values live in `lookbox/models/models.json`, **not in code** (M6 values taken from the Git LFS pointers). At build time, verify that the ONNX exports exist and work on DirectML before committing to one. Export from PyTorch yourself if necessary (a dev-only script in `tools/`).

### 10.3 Delivery
- Models download on first use into `%LOCALAPPDATA%\RRIPP\models\` (override: `LOOKBOX_MODELS_DIR`), streamed and verified by size + sha256, with a progress dialog and a cancel button. Only a verified file takes the final name; failures leave nothing behind. A `.verified` marker avoids re-hashing ~1 GB each launch.
- Runtime dependency: `onnxruntime-directml` on Windows (`onnxruntime` elsewhere). `run.bat` reinstalls requirements whenever requirements.txt changes.
- The app works fully without any model downloaded. AI buttons show "Download model (xx MB)".

---

## 11. Import and export

**Import**
- Single image: PNG (8/16), JPG, TIFF, EXR, WEBP.
- **Render sets (M8), no separate command:** every import looks for passes. Names are compared with case, spaces, dashes, underscores and dots removed; a file is a pass if its name is the beauty's plus a pass word before or after it (`objectid`, `object`, `objid`, `id`, `idmap`, `meshid` → Object ID; `materialid`, `matid`, `material` → Material ID; `alpha`, `alphamask`, `mask`, `matte` → Alpha), frame numbers allowed on both (`shot_objectid_0001`). Passes selected together with their render attach to it instead of becoming layers; the render's folder is also searched. Passes attach to `ImageLayer.passes` (must be the render's size, else skipped with a message). An **alpha pass** becomes the layer's cut-out (a mask asset) when the render itself is opaque. Anything not detected: **Attach ID pass…** in Cut-out. `core/io/render_sets.py`, tested without Qt.
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
- Shortcuts: Ctrl+Z/Y, Ctrl+S, Ctrl+E (export), Delete, Ctrl+D (duplicate), Ctrl+B (backdrop), arrows (nudge 1 px, Shift = 10 px), Space-drag (pan), Ctrl+wheel (zoom).
- Snapping (M5): moving snaps a layer's bounds (edges + centre) to the canvas edges/centre and other visible layers' edges/centres within 8 screen px; resizing snaps the dragged handle when the layer isn't rotated off-axis. Pink guides show the match. Hold Ctrl to place freely.
- Context tabs: Adjust (images), Style (text, fill, opacity, blend, fade, effects), Layers.
- Text (M7): **Text** in the rail (Ctrl+T) adds a text box and puts the cursor in the Style tab's text box; double-clicking text on the canvas does the same. Typing updates the canvas live (the real render, effects included); one typing session = one undo step. Text shows corner handles only: a corner drag scales the font. *Deviation from the M7 row:* the caret is in the panel, not on the canvas; a canvas overlay editor would draw text with a second engine (QTextDocument) that doesn't match the render. Revisit if typing in the panel feels wrong in use.

---

## 13. Project structure

```
lookbox/
  app.py                  # entry point (+ --selftest)
  selftest.py             # build check, run by the exe (§15b)
  core/                   # NO Qt imports allowed
    model.py              # dataclasses (§5)
    assets.py             # content-addressed AssetStore (original bytes + decoded pixels)
    serialize.py          # .lookbox read/write + migrations
    render/
      pipeline.py         # §6.1 orchestration: render_layer, render_key, render (export)
      levels.py           # preview level choice, dithered display quantize, thumbnails
      fill.py             # gradient fills + gradient fade masks
      text.py             # text layout, box, anchoring; pluggable measuring/drawing engine
      effects.py          # layer blur, shadow (+ floor squash), glow, outline, padding
      adjust.py           # §7, one function per control + swatch suggestions
      effects.py          # §8
      blend.py
      transform.py
      lut.py
      cache.py
    masks/
      ops.py              # mask storage, grow/shrink, feather, refine edge, brush stamping
      idpick.py           # ID pass → mask with anti-aliased edges (IdPass)
      lasso.py            # polygon → anti-aliased mask
    io/
      images.py           # load/save, EXR handling, dither
      render_sets.py      # pass detection, grouping, alpha pass → mask, import_files
  ai/                     # Qt-free
    registry.py           # models.json, verified downloads
    runtime.py            # onnxruntime sessions, DirectML → CPU fallback, one model loaded
    birefnet.py           # background removal pre/post-processing
    sam.py                # (M9)
    esrgan.py             # (M11)
  models/models.json      # model urls, sizes, sha256
  commands/               # the ONLY code that mutates a Document
    edits.py              # plain-Python edits (apply/revert), Qt-free, unit-tested
    text_edits.py         # text-layer edits (re-exported by edits.py)
    qt.py                 # QUndoCommand adapter around edits
  ui/
    main_window.py        # layout + wiring only
    documents.py          # new/open/save/import/export flows
    editor.py             # open doc + assets + QUndoStack + selection; the UI's single entry point
    jobs.py               # background workers (import, open, save, export)
    render_service.py     # thread pool + LRU cache for per-layer preview levels
    cutout.py             # remove-background flow (model choice, download, run, apply)
    ai_jobs.py            # download + inference threads
    pixmaps.py            # layer thumbnails
    panels/               # adjust.py (+ color_edit.py, cutout.py), layer_style.py (+ effects.py, text.py), layers.py; later: filters
    text_engine.py        # Qt text engine: measures and draws for core/render/text.py
    widgets/              # slider_row.py, colour_button.py; later: gradient editor
    canvas/               # QGraphicsView, items, handles, mask overlay
      view.py             # mouse/keyboard/zoom
      layer_items.py      # one pixmap item per layer, level choice, placement
      overlay.py          # selection box, handles, dimmed outside
      frame_stats.py      # drag frame timing (Qt-free)
      mask_brush.py       # mask painting mode + red overlay
      file_drop.py        # drag files in from Explorer
      handles.py          # handle/drag/hit-test geometry — kept Qt-free so it's testable
    theme.qss
  luts/                   # bundled .cube files
tests/
  golden/                 # reference data (adjust_<control>.npz); regenerate: pytest --update-golden
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
| M1 | Skeleton: window, canvas, import image, ImageLayer, move/scale/rotate handles, layers list, undo/redo, save/load `.rripp`, PNG export | Import 3 images, arrange them, undo 20 steps, save, reopen, export; everything is identical |
| M2 | Render pipeline + preview levels + workers + cache | Dragging a layer on a 6000×4000 canvas stays smooth: the status bar readout shows ≤33 ms/frame ("OK") on the Ryzen 3600 |
| M3 | Adjust panel (§7) + golden tests | All 15 controls work, previews match exports, golden tests pass |
| M4 | Fill layers (solid/gradient), gradient fade, blend modes | A backdrop gradient with no visible banding after 8-bit export |
| M5 | Effects: shadow, glow, outline, blur | The shadow follows the layer when moved, with no re-blur on move |
| M6 | *(pulled forward)* AI infrastructure + background removal + layer masks + mask brush + edge controls + extract to layer | Remove the background of a photo on the RTX 4060 (and CPU fallback); download flow verifies the model; fix an edge with the brush; extract to a new layer |
| M7 | Text layers | Edit text with shadow and outline applied, live on the canvas (typed in the Style tab, see §12) |
| M8 | Render-set import + ID pick + lasso | Pick an object from a Toolbag ID pass and extract it to a layer with a clean edge (done: soft ID edges, see §9) |
| M9 | SAM smart select | Click-select in under 300 ms per click after the first encode |
| M10 | LUT filters + global adjust | A .cube file loads; a strength slider works |
| M11 | Upscale | 2× and 4× work with tiling in low-VRAM mode |
| M12 | Packaging | The `--onedir` build runs on a clean Windows machine with no Python installed (done: GitHub Actions `windows-latest` builds it and runs `RRIPP.exe --selftest`, see §15b) |

Test on the RX 570 (or force low-VRAM mode + CPU) at M8, M9 and M11, not at the end.

---

## 15b. Packaging (M12)

- `packaging/lookbox.spec` (PyInstaller, one folder, windowed, icon + version info; UPX off because packed DLLs trip antivirus). Data files: theme, icon, models.json. `collect_dynamic_libs("onnxruntime")` brings DirectML.dll. Unused Qt modules are excluded.
- **Self-test:** `RRIPP.exe --selftest [report.txt]` builds the whole main window (minimized), loads bundled data, renders + exports a document with outlined text (real fonts), round-trips a project, runs a tiny ONNX model on the CPU and on DirectML if present. Exit code 0 = pass. This is what catches packaging bugs (missing DLL/plugin/data) that unit tests can't.
- **CI:** `.github/workflows/windows-build.yml` on every push to main: pytest on Windows with real Qt (real windows, not the offscreen platform: on Windows it has no fonts, every glyph is a box; includes `tests/test_qt_ui.py`), build, self-test the exe, upload `RRIPP-windows.zip`. A version (`lookbox.__version__`) that has no release yet is published as release `v<version>`: releasing = bumping the version.
- `build.bat` does the same on the user's PC.
- A windowed exe has `sys.stdout/stderr = None`; `app._harden_frozen` points them at devnull so printing never crashes. `AppUserModelID` gives RRIPP its own taskbar icon.
- Not done (deliberately): an installer, file association for `.rripp`, code signing (unsigned exe → SmartScreen "More info → Run anyway" the first time).

## 15a. Error reporting (added after M6)

`lookbox/crash.py`: Python errors anywhere (UI slots, worker threads) are appended with their traceback to `%LOCALAPPDATA%\RRIPP\logs\rripp.log` and, on the UI thread, shown in a dialog instead of closing the app (repeats within 5 s aren't re-shown). Native crashes write their stack via `faulthandler` to the same file.

## 16. Rules for the AI assistant

1. Read this file before every session. Work on **one milestone** at a time.
2. Never mutate `Document` outside `commands/`. Never import Qt in `core/`.
3. Never modify source assets. Never overwrite the user's files except on an explicit save or export.
4. No new dependencies without asking.
5. No silent fallbacks or bare `except:`. Surface errors to the user in one plain sentence.
6. When you change behaviour covered by a golden test, say so explicitly; don't just regenerate the goldens.
7. If something here is wrong or impractical, propose an edit to this file instead of working around it.
8. At the end of each session: run the tests, summarize what changed, and list the known issues.
