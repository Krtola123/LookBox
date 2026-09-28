# LookBox

A personal, offline, Canva-simple compositor for finishing Marmoset Toolbag renders (and the odd photo).
**Read [ARCHITECTURE.md](ARCHITECTURE.md) before changing anything.** It's the source of truth.

## Run it (Windows)

1. Install **Python 3.12** from python.org (tick "Add python.exe to PATH").
2. Double-click **`run.bat`**. The first run creates `.venv` and installs dependencies (a few minutes).

Or manually:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m lookbox
```

You can pass files: `python -m lookbox scene.lookbox` or `python -m lookbox a.png b.png`.

## Tests

```bat
.venv\Scripts\activate
pytest
```

Everything under `lookbox/core`, `lookbox/commands/edits.py` and `lookbox/ui/canvas/handles.py` is Qt-free and fully tested.

## Status: milestone M3

| Works | How |
|---|---|
| New design | Ctrl+N — size presets, transparent/white/black/custom background |
| Import images | Ctrl+I, the **Image** button, or drag files from Explorer onto the canvas (lands where you drop) |
| Select | Click a layer. Clicks pass through transparent pixels, so full-frame renders don't block layers underneath |
| Move | Drag. Shift = lock to one axis. Arrow keys nudge 1 px, Shift+arrow 10 px |
| Resize | Corner handles = proportional (Shift = free), edge handles = one axis, Alt = from centre |
| Rotate | Handle under the layer. Snaps to 0/90/180 within 2°, Shift = 15° steps |
| Layers | Right panel: drag to reorder, 👁 show/hide, 🔒 lock, double-click to rename, Delete removes |
| Duplicate / delete | Ctrl+D / Delete |
| Undo / redo | Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z). Every change is undoable |
| Save / open | Ctrl+S / Ctrl+O — `.lookbox` files keep your original images byte-for-byte (16-bit and EXR included) |
| Export PNG | Ctrl+E — full resolution, 8-bit with dithering against gradient banding |
| View | Ctrl+wheel zoom, Space+drag or middle-drag to pan, Ctrl+0 fit, Ctrl+1 100% |
| Smooth previews | Layers render in the background at the resolution your screen needs and stay sharp as you zoom |
| Never freezes | Import, open, save and export run in the background; export shows progress and can be cancelled |
| Adjust | **Adjust** tab: Temperature, Tint, Brightness, Contrast, Highlights, Shadows, Whites, Blacks, Vibrance, Saturation, Color edit (swatches), Sharpness, Clarity, Vignette, Invert. Double-click a slider to reset it; one drag = one undo step |
| Before / after | Toggle in the Adjust header shows the layer without its adjustments (view only) |

**Tips:** judge Sharpness at 100% zoom (Ctrl+1): sharpening is too fine to show at fit-to-screen.
While you drag a slider the preview is half resolution; it sharpens when you let go.
Tuning a control's look? Regenerate its golden file deliberately: `pytest --update-golden`.

**Checking M2 performance:** drag a layer and watch the status bar. It shows the paint cost per frame
(95th percentile). `OK` means ≤33 ms, i.e. at least 30 fps is sustainable; `SLOW` means it isn't.

Known M1 limitations are listed in the commit message and fixed in later milestones (see ARCHITECTURE §15).
