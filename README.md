# RRIPP — Reshi Renders' Image Post Processing App

*(formerly LookBox: old `.lookbox` projects open as they are; the app moves its old data folder over on first start, so AI models aren't downloaded again)*

A personal, offline, Canva-simple compositor for finishing Marmoset Toolbag renders (and the odd photo).
**Read [ARCHITECTURE.md](ARCHITECTURE.md) before changing anything.** It's the source of truth.

## Install (Windows, no Python needed)

1. Download **RRIPP-windows.zip**: from the repo's **Releases** page, or from the latest
   **Actions → Windows build** run (Artifacts, at the bottom; you need to be signed in to GitHub).
2. Unzip it anywhere (e.g. `C:\Tools\RRIPP`) and run **`RRIPP.exe`**. Pin it to the taskbar if you like.
   It's a folder, not a single file: keep `RRIPP.exe` next to its `_internal` folder.

Nothing is installed or written outside `%LOCALAPPDATA%\RRIPP` (AI models, logs, settings in the registry
under `HKCU\Software\RRIPP`). To update: replace the folder.

**Make a release:** raise `__version__` in `lookbox/__init__.py` and push to main; the build publishes
`RRIPP-windows.zip` as release `v<version>` once its tests and self-test pass.
**Build on your own PC:** double-click **`build.bat`** → `dist\RRIPP\RRIPP.exe` (it runs the self-test at the end).
**Check any build:** `RRIPP.exe --selftest report.txt` (no window; exit code 0 = OK).

## Run from source (Windows)

1. Install **Python 3.12** from python.org (tick "Add python.exe to PATH").
2. Double-click **`run.bat`**. The first run creates `.venv` and installs dependencies (a few minutes).

Or manually:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m lookbox
```

You can pass files: `python -m lookbox scene.rripp` or `python -m lookbox a.png b.png`.

**After pulling M6**, start with `run.bat` as usual: it notices the new requirements and installs
`onnxruntime-directml` (the AI runtime) automatically.

## Tests

```bat
.venv\Scripts\activate
pytest
```

`tests/test_qt_ui.py` drives the real widgets (text typing, pick/lasso, zoom, the self-test); it runs
wherever PySide6 is installed, including every GitHub build, and is skipped elsewhere.

Everything under `lookbox/core`, `lookbox/commands/edits.py` and `lookbox/ui/canvas/handles.py` is Qt-free and fully tested
(text layout included, with a stand-in for the Qt font engine).

## Status: 1.2.0 (M13: Fill and Grab)

| Works | How |
|---|---|
| New design | Ctrl+N — size presets (incl. 2160 × 2160), transparent/white/black/custom background |
| Import images | Ctrl+I, the **Image** button, or drag files from Explorer onto the canvas (lands where you drop) |
| Select | Click a layer. Clicks pass through transparent pixels, so full-frame renders don't block layers underneath |
| Move | Drag. Snaps to canvas edges/centre and other layers (pink guides); hold Ctrl to place freely. Shift = lock to one axis. Arrow keys nudge 1 px, Shift+arrow 10 px |
| Resize | Corner handles = proportional (Shift = free), edge handles = one axis, Alt = from centre |
| Rotate | Handle under the layer. Snaps to 0/90/180 within 2°, Shift = 15° steps |
| Layers | Right panel: drag to reorder, 👁 show/hide, 🔒 lock, double-click to rename, Delete removes |
| Duplicate / delete | Ctrl+D / Delete |
| Undo / redo | Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z). Every change is undoable |
| Save / open | Ctrl+S / Ctrl+O — `.rripp` files keep your original images byte-for-byte (16-bit and EXR included) |
| Export PNG | Ctrl+E — full resolution, 8-bit with dithering against gradient banding |
| View | Ctrl+wheel zoom, Space+drag or middle-drag to pan, Ctrl+0 fit, Ctrl+1 100% |
| Smooth previews | Layers render in the background at the resolution your screen needs and stay sharp as you zoom |
| Never freezes | Import, open, save and export run in the background; export shows progress and can be cancelled |
| Adjust | **Adjust** tab: Temperature, Tint, Brightness, Contrast, Highlights, Shadows, Whites, Blacks, Vibrance, Saturation, Color edit (swatches), Sharpness, Clarity, Vignette, Invert. Double-click a slider to reset it; one drag = one undo step |
| Before / after | Toggle in the Adjust header shows the layer without its adjustments (view only) |
| Backdrop | **Backdrop** button (Ctrl+B) adds a gradient behind everything. **Style** tab: solid / linear / radial, two colours (with alpha), angle, centre, size, Fit to canvas |
| Opacity & blend modes | **Style** tab, any layer: Opacity, Blend mode (Normal, Multiply, Screen, Overlay, Add, Soft light) |
| Gradient fade | **Style** tab, any layer: fade to transparent, linear or radial, with direction, start, end, invert |
| Remove background | **Adjust** tab → Cut-out → **Remove background**. First time: pick *Best quality* (973 MB, for 8 GB+ GPUs) or *Fast* (224 MB), downloaded once and checked against its published checksum. Runs on your GPU (DirectML), or the CPU if that fails. Non-destructive: it adds a layer mask |
| Fix the cut-out | **Edge shift** and **Feather** sliders, **Invert**, and **Brush…**: paint Erase/Restore on the canvas (red = hidden; Alt flips the mode, [ ] size, Enter done, Esc cancel). No AI needed: **Cut out by hand…** starts from a fully visible layer |
| Render passes | Import a render and RRIPP finds its passes next to it: `shot_objectid.png`, `shot_Object ID.png`, `shot_MaterialID.png`, `shot_alpha.png` (any case, spaces/dashes/underscores, frame numbers). Picking the passes together with the render works too. Not named like that? **Adjust → Cut-out → Attach ID pass…**. An alpha pass cuts the render out automatically |
| Pick object | **Pick object…** (renders with an ID pass): click an object and it's selected exactly, with the render's own soft edge. Shift+click adds objects, Alt+click removes. Switch between Object ID and Material ID. **Extract** puts it on its own layer |
| Lasso | **Lasso…**: drag around something, or click corner by corner and press Enter (or double-click). Shift adds, Alt removes. Works on any image |
| Fill / Grab | Select something (Pick object, Lasso or Brush), then **Fill** to replace it with what's around it, or **Grab** to lift it onto its own layer and fill the hole behind it. **Fill with:** *AI fill* (208 MB model, downloads once), *Quick fill* (no download, small spots) or *From a clean render* (the same shot rendered without the object: exact, best for renders). The fill is its own layer: hide it to see the original |
| Keep the thing | **Extract to new layer**: the cut-out goes on its own layer above; the original shows everything again |
| Keep the background | On by default ("Keep the background as its own layer"): Remove background also puts what was removed on a layer right below, so you can blur, adjust or hide it. One undo step |
| If something breaks | RRIPP shows a dialog instead of closing and writes the details to `%LOCALAPPDATA%\RRIPP\logs\rripp.log`. Send that file |
| Text | **Text** button (Ctrl+T) adds text and puts the cursor in the **Style** tab's text box; double-click text on the canvas to edit it. Type and watch it update on the canvas, shadow/outline/glow included. Font, weight, italic, colour, alignment, size, letter spacing, line height, and *Wrap at a fixed width*. Drag a corner to resize: the font size changes, so text stays sharp at any size and zoom. Text grows from its left edge (or centre/right edge, by alignment) |
| Filters | **Adjust → Filters**: 8 built-in looks shown on your image (Warm film, Teal & orange, Cool studio, Bleach bypass, Soft matte, Punchy, Mono, Vintage fade), a **Strength** slider, and **Import .cube…** for any LUT from Resolve, Photoshop or a LUT pack (kept for every design afterwards) |
| Whole-design grade | **Adjust → Whole design**: the same sliders and filters, applied to the finished image (everything together, like a final grade). Before/After shows it without. Dragging a layer shows it ungraded for a moment; the grade comes back when you let go |
| Effects | **Style** tab: Shadow (direction, distance, blur, spread, floor squash, colour; presets **Soft drop** and **Contact shadow**), Glow, Outline (outside/centre), Layer blur. Sizes are canvas pixels and the shadow direction stays put when you rotate or resize the layer |

**Tips:** blend modes preview against the checkerboard on a transparent canvas; add a backdrop
(or pick a background colour) to see exactly what exports. Judge Sharpness at 100% zoom (Ctrl+1): sharpening is too fine to show at fit-to-screen.
While you drag a slider the preview is half resolution; it sharpens when you let go.
Tuning a control's look? Regenerate its golden file deliberately: `pytest --update-golden`.

**Checking M2 performance:** drag a layer and watch the status bar. It shows the paint cost per frame
(95th percentile). `OK` means ≤33 ms, i.e. at least 30 fps is sustainable; `SLOW` means it isn't.

Known M1 limitations are listed in the commit message and fixed in later milestones (see ARCHITECTURE §15).
