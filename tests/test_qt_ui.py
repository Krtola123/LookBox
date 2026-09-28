"""UI smoke tests with real Qt (offscreen). They run wherever PySide6 is installed:
on your PC (`pytest`) and in the Windows build on GitHub; skipped otherwise.

They drive the real widgets the way a user would, for the parts unit tests can't
reach: the Qt text engine, typing into the text box, the mask session tools, and
Ctrl+wheel zoom.
"""

import os

import numpy as np
import pytest

try:
    if os.name != "nt":  # Windows' offscreen platform has no real fonts; use real windows there
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
except ImportError:  # the sandbox without Qt: skip the whole module
    QApplication = None

from lookbox.core.model import TextLayer
from lookbox.core.render import text as T


def _app():
    if QApplication is None:
        pytest.skip("PySide6 not installed")
    return QApplication.instance() or QApplication([])


class _QtEngine:
    """Use the real Qt text engine for one test, then put back whatever was there
    (test_text.py runs with a fake engine)."""

    def __enter__(self):
        from lookbox.ui.text_engine import QtTextEngine

        self.old = T._engine
        T.set_engine(QtTextEngine())

    def __exit__(self, *exc):
        T.set_engine(self.old)


def _window():
    from lookbox.ui.main_window import MainWindow

    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    return win


def _close(win):
    win.editor.stack.setClean()  # never prompt
    win.close()
    QApplication.processEvents()


def test_qt_text_engine_measures_and_draws():
    _app()
    with _QtEngine():
        layer = TextLayer(text="Hello\nLookBox", font_size=40, font_family="Arial")
        eng = T.engine()
        assert eng.advance(layer, "WWWW") > 1.5 * eng.advance(layer, "iiii"), "no real fonts (glyph boxes)"
        lay = T.layout(layer)
        assert len(lay.lines) == 2 and lay.width > 80 and lay.height > 80
        px = T.render_text(layer, lay.width, lay.height)
        ink = px[..., 3]
        assert ink.max() > 0.99 and 0.05 < (ink > 0.5).mean() < 0.6
        big = T.render_text(layer, lay.width * 2, lay.height * 2)  # a level-2 render is the same picture
        import cv2
        down = cv2.resize(big[..., 3], (lay.width, lay.height), interpolation=cv2.INTER_AREA)

        def box(a):
            ys, xs = np.nonzero(a > 0.5)
            return xs.min(), ys.min(), xs.max(), ys.max(), round(float(a.sum()), 1)

        assert np.abs(down - ink).mean() < 0.03, f"1x {box(ink)} vs 2x→1x {box(down)}"
        wider = T.layout(TextLayer(text="Hello", font_size=40, font_family="Arial", letter_spacing=10))
        assert wider.width >= T.layout(TextLayer(text="Hello", font_size=40, font_family="Arial")).width + 40


def test_add_text_type_and_undo():
    _app()
    with _QtEngine():
        win = _window()
        try:
            win.add_text()
            QApplication.processEvents()
            layer = win.editor.selected_layer()
            assert isinstance(layer, TextLayer)
            box = win.layer_panel.text.box
            box.setPlainText("Render night")
            QApplication.processEvents()
            assert win.editor.doc.layer(layer.id).text == "Render night"
            win.editor.stack.undo()
            assert win.editor.doc.layer(layer.id).text == "Your text"
        finally:
            _close(win)


def _render_set(tmp_path):
    from tests.helpers import write_png

    h, w = 120, 160
    beauty = np.zeros((h, w, 4), np.float32)
    beauty[..., :3], beauty[..., 3] = 0.3, 1.0
    ids = np.zeros((h, w, 4), np.float32)
    ids[..., 3] = 1.0
    ids[30:90, 20:70, 0] = 1.0  # object 1: red block
    ids[30:90, 90:140, 2] = 1.0  # object 2: blue block
    write_png(str(tmp_path / "shot.png"), beauty)
    write_png(str(tmp_path / "shot_objectid.png"), ids)
    return str(tmp_path / "shot.png")


def test_import_render_set_pick_object_and_extract(tmp_path):
    from lookbox.commands import edits
    from lookbox.core.io import render_sets

    _app()
    with _QtEngine():
        win = _window()
        try:
            ed = win.editor
            items, errors, _ = render_sets.import_files(ed.store, [_render_set(tmp_path)])
            assert not errors and ed.add_imported(ed.store, items, (80.0, 60.0))
            layer = ed.selected_layer()
            assert "object_id" in layer.passes
            brush = win.canvas.brush
            brush.tool = "pick"
            brush.start(layer.id)
            # Canvas point over the red block: the layer is centred on (80, 60), 160 × 120.
            brush.press(0 + 45, 0 + 60, shift=False, alt=False)
            brush.release()
            assert brush.mask[60, 45] == 1.0 and brush.mask[60, 115] == 0.0 and brush.mask[5, 5] == 0.0
            brush.press(115, 60, shift=True, alt=False)  # Shift adds the blue one
            brush.release()
            assert brush.mask[60, 115] == 1.0
            batch = edits.extract_session(ed.doc, layer.id, brush.take_edit())
            brush.end()
            ed.push(batch)
            assert len(ed.doc.layers) == 2 and ed.doc.layers[1].mask is not None
            ed.stack.undo()
            assert len(ed.doc.layers) == 1 and ed.doc.layers[0].mask is None
        finally:
            _close(win)


def test_lasso_freehand_and_polygon(tmp_path):
    _app()
    with _QtEngine():
        win = _window()
        try:
            from lookbox.core.io import render_sets

            ed = win.editor
            items, _, _ = render_sets.import_files(ed.store, [_render_set(tmp_path)])
            ed.add_imported(ed.store, items, (80.0, 60.0))
            layer = ed.selected_layer()
            win.canvas.set_zoom(4.0)
            brush = win.canvas.brush
            brush.tool = "lasso"
            brush.start(layer.id)
            # Freehand: a drag around (10..50, 10..50).
            brush.press(10, 10, shift=False, alt=False)
            for x, y in ((50, 10), (50, 50), (10, 50)):
                brush.move(x, y, dragging=True, alt=False)
            brush.release()
            assert brush.mask[30, 30] > 0.99 and brush.mask[80, 100] == 0.0
            # Polygon, adding: click corners, Enter.
            for x, y in ((100, 70), (150, 70), (150, 110)):
                brush.press(x, y, shift=True, alt=False)
                brush.release()
            assert brush.polygon
            brush.key(Qt.Key.Key_Return)
            assert not brush.polygon and brush.mask[100, 145] > 0.99 and brush.mask[30, 30] > 0.99
            brush.finish(apply=True)
            assert ed.doc.layer(layer.id).mask is not None
        finally:
            _close(win)


def test_ctrl_wheel_zooms_around_the_cursor():
    _app()
    win = _window()
    try:
        view = win.canvas
        view.fit()
        QApplication.processEvents()
        pos = QPoint(view.viewport().width() // 3, view.viewport().height() // 3)
        before = view.mapToScene(pos)
        ev = QWheelEvent(QPointF(pos), QPointF(view.viewport().mapToGlobal(pos)), QPoint(0, 0), QPoint(0, 240),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.NoScrollPhase,
                         False)
        z0 = view.zoom()
        view.wheelEvent(ev)
        after = view.mapToScene(pos)
        assert view.zoom() > z0 * 1.2
        assert abs(after.x() - before.x()) * view.zoom() < 1.5 and abs(after.y() - before.y()) * view.zoom() < 1.5
    finally:
        _close(win)


def test_selftest_passes():
    """The same check the built exe runs."""
    from lookbox import selftest

    _app()
    with _QtEngine():
        assert selftest.run(None) == 0

