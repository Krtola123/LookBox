"""Dev/CI: screenshots of the real UI (a sample design, several panels) into a folder.
The Windows build pushes them to the `screens` branch, so UI changes can be looked at
without a Windows PC.  Usage: python tools/screenshot.py <out dir>"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def pump(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    from PySide6.QtWidgets import QApplication, QToolBar

    from lookbox.app import load_theme
    from lookbox.core.io import images
    from lookbox.core.io.render_sets import ImportItem
    from lookbox.core.render import text as text_render
    from lookbox.ui.main_window import MainWindow
    from lookbox.ui.text_engine import QtTextEngine

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(load_theme())
    text_render.set_engine(QtTextEngine())
    win = MainWindow()
    win.resize(1440, 900)
    win.show()
    pump(app, 0.5)

    # A sample design: backdrop, a "render" (shaded sphere on transparent), text.
    ed = win.editor
    win.add_backdrop()
    h = w = 640
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - w / 2, yy - h / 2) / (w * 0.4)
    img = np.zeros((h, w, 4), np.float32)
    shade = np.clip(1.1 - np.hypot(xx - w * 0.4, yy - h * 0.38) / (w * 0.55), 0, 1)
    img[..., 0], img[..., 1], img[..., 2] = 0.85 * shade, 0.55 * shade, 0.25 * shade
    img[..., 3] = np.clip((1 - r) * 40, 0, 1)
    info = ed.store.add_bytes(images.encode_png(img), ".png", "sphere.png")
    ed.add_imported(ed.store, [ImportItem(info)], (960.0, 560.0))
    image_id = ed.selected
    win.add_text()
    pump(app, 1.5)

    def shot(name, widget=None):
        pump(app, 0.8)
        (widget or win).grab().save(os.path.join(out, name))

    ed.select(image_id)
    win.tabs.setCurrentWidget(win.adjust_panel)
    shot("main.png")
    shot("rail.png", win.findChild(QToolBar, "toolRail"))
    win.tabs.setCurrentWidget(win.layer_panel)
    shot("style.png")
    win.tabs.setCurrentIndex(2)
    shot("layers.png")
    # Pages: a square copy below, shown zoomed out.
    win.duplicate_page()
    from lookbox.commands import edits
    from lookbox.core.model import Size
    ed.push(edits.SetCanvas(ed.doc.canvas, Size(w=1080, h=1080)))
    win.pages.relayout()
    ed.set_active(ed.project.pages[0].id)
    win.canvas.set_zoom(win.canvas.zoom() * 0.55)
    pump(app, 2.5)
    shot("pages.png")
    ed.stack.setClean()
    win.close()
    print("screenshots in", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "screens")
