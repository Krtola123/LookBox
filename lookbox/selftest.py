"""`RRIPP.exe --selftest [report.txt]`: prove a build works (window minimized, closes itself).

Packaging bugs don't show up in unit tests: a missing DLL, a data file left out of
the bundle, a Qt plugin that didn't get copied. This runs the real app code the way
the exe does and checks each piece:

- the whole main window builds (every panel constructor runs);
- theme, icon and model list load from inside the bundle;
- a document with a backdrop, an image and outlined text renders and exports,
  with real fonts (the Qt text engine), and survives a save/load round trip;
- onnxruntime loads and runs a tiny model on the CPU, and on the GPU (DirectML)
  where there is one.

Exit code 0 = everything passed. The report goes to the file given (a windowed exe
has no console) and to stdout when there is one.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback

import numpy as np


def _checks():
    from lookbox import __version__

    yield "version", lambda: __version__

    def ui():
        from PySide6.QtWidgets import QApplication

        from lookbox.ui.main_window import MainWindow

        app = QApplication.instance()
        win = MainWindow()
        win.showMinimized()  # a real window (real fonts, real platform), kept out of the way
        app.processEvents()
        win.add_text()  # the text panel, text engine and canvas together
        app.processEvents()
        layers = len(win.editor.doc.layers)
        win.editor.stack.setClean()  # nothing to save: closing must not ask
        win.close()
        app.processEvents()
        return f"main window built, text added ({layers} layer)"

    yield "ui", ui

    def data_files():
        from lookbox.ai.registry import load_registry
        from lookbox.app import resource

        for name in ("theme.qss", "icon.png", "icon.ico"):
            if not os.path.isfile(resource(name)):
                raise FileNotFoundError(resource(name))
        return f"theme + icon present, {len(load_registry())} AI models listed"

    yield "data files", data_files

    def render_export():
        from lookbox.commands import edits
        from lookbox.core import serialize
        from lookbox.core.assets import AssetStore
        from lookbox.core.io import images
        from lookbox.core.model import (Document, Effects, FillLayer, ImageLayer, Outline, Size, TextLayer,
                                        Transform)
        from lookbox.core.render.pipeline import render

        store = AssetStore()
        doc = Document(canvas=Size(w=320, h=200))
        edits.AddLayer(FillLayer(width=320, height=200, transform=Transform(x=160, y=100))).apply(doc)
        img = np.zeros((60, 80, 4), np.float32)
        img[..., 0], img[..., 3] = 0.8, 1.0
        info = store.add_bytes(images.encode_png(img), ".png", "red.png")
        edits.AddLayer(ImageLayer(source=info.id, transform=Transform(x=60, y=60)), asset=info).apply(doc)
        text = TextLayer(text="RRIPP", font_size=40, color=(1, 1, 1, 1), transform=Transform(x=200, y=140),
                         effects=Effects(outline=Outline(width=3, color=(0, 0, 0, 1))))
        edits.AddLayer(text).apply(doc)
        from lookbox.core.render.text import engine

        narrow = engine().advance(text, "iiii")
        wide = engine().advance(text, "WWWW")
        if not wide > 1.5 * narrow:  # a font with real glyphs, not placeholder boxes
            raise AssertionError(f"no real fonts: 'iiii' = {narrow:.0f} px, 'WWWW' = {wide:.0f} px")
        out = render(doc, store)
        ink = out[115:165, 120:280]
        white = (ink[..., :3].min(axis=2) > 0.9).sum()
        if white < 50:
            raise AssertionError(f"text didn't render ({white} white px)")
        with tempfile.TemporaryDirectory() as d:
            png = os.path.join(d, "out.png")
            images.save_png(png, out)
            back = images.decode(open(png, "rb").read(), ".png")
            if back.shape != out.shape:
                raise AssertionError("exported PNG has the wrong size")
            proj = os.path.join(d, "t" + serialize.EXTENSION)
            serialize.save(proj, doc, store)
            doc2, _ = serialize.load(proj)
            if doc2 != doc:
                raise AssertionError("project didn't round-trip")
        return f"rendered + exported 320×200, text {white} px, project round trip ok"

    yield "render + export", render_export

    def filters():
        from lookbox.core.assets import AssetStore
        from lookbox.core.model import Document, FillLayer, LutRef, Size, Transform
        from lookbox.core.render.pipeline import render
        from lookbox.ui.panels.filters import library

        looks = library()
        if len(looks) < 6:
            raise AssertionError(f"only {len(looks)} bundled looks found")
        store = AssetStore()
        doc = Document(canvas=Size(w=64, h=64), layers=[FillLayer(width=64, height=64, transform=Transform(x=32, y=32))])
        plain = render(doc, store)
        info = store.add_bytes(looks[0].data, ".cube", looks[0].name)
        doc.global_lut = LutRef(asset=info.id, strength=1.0)
        graded = render(doc, store)
        if np.array_equal(graded, plain):
            raise AssertionError("the whole-design filter changed nothing")
        return f"{len(looks)} looks, whole-design grade renders"

    yield "filters", filters

    def onnx():
        import onnxruntime as ort

        from lookbox.ai.onnx_tiny import sigmoid_model
        from lookbox.ai.runtime import create_session

        providers = ort.get_available_providers()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tiny.onnx")
            with open(path, "wb") as fh:
                fh.write(sigmoid_model(16))
            x = np.zeros((1, 3, 16, 16), np.float32)
            results = []
            for gpu in (False, True):
                if gpu and "DmlExecutionProvider" not in providers:
                    continue
                s, used, notice = create_session(path, prefer_gpu=gpu)
                y = s.run(None, {"input_image": x})[0]
                if not np.allclose(y, 0.5, atol=1e-3):
                    raise AssertionError(f"{used}: wrong output")
                results.append(used + (f" ({notice})" if notice else ""))
                del s
        return f"onnxruntime {ort.__version__}, providers {providers}, ran on {results}"

    yield "onnxruntime", onnx


def run(report_path: str | None = None) -> int:
    """Run every check (the QApplication must exist). Returns the exit code."""
    lines, failed = [f"RRIPP self-test  {time.strftime('%Y-%m-%d %H:%M:%S')}  python {sys.version.split()[0]}"
                     f"  frozen={getattr(sys, 'frozen', False)}"], 0
    for name, check in _checks():
        t0 = time.perf_counter()
        try:
            detail = check()
            lines.append(f"OK    {name}: {detail}  ({time.perf_counter() - t0:.2f} s)")
        except Exception:
            failed += 1
            lines.append(f"FAIL  {name}:\n" + traceback.format_exc())
    lines.append("PASSED" if failed == 0 else f"FAILED ({failed})")
    text = "\n".join(lines) + "\n"
    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    if sys.stdout is not None:
        sys.stdout.write(text)
    return 0 if failed == 0 else 1
