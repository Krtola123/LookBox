"""One Adjust-panel control: label, numeric box, slider (−100…100).

Double-click the slider to reset it to 0. `pressed`/`released` bracket a drag,
so the panel can make one drag = one undo step and switch the canvas to
half-res previews while dragging.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QSlider, QSpinBox, QWidget

GRADIENTS = {  # Canva-style coloured tracks
    "temperature": "stop:0 #7aa7ff, stop:0.5 #d9d9d9, stop:1 #ffc766",
    "tint": "stop:0 #7ee07e, stop:0.5 #d9d9d9, stop:1 #e27ee6",
}


class _Slider(QSlider):
    reset_requested = Signal()

    def mouseDoubleClickEvent(self, e) -> None:
        self.reset_requested.emit()
        e.accept()


class SliderRow(QWidget):
    edited = Signal(int)  # every change the user makes (drag, typing, reset)
    pressed = Signal()
    released = Signal()

    def __init__(self, key: str, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.setObjectName("sliderRow")
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)

        self.label = QLabel(label)
        self.label.setObjectName("sliderLabel")
        self.slider = _Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(-100, 100)
        self.slider.setPageStep(10)
        self.slider.setObjectName("adjustSlider")
        if key in GRADIENTS:
            self.slider.setStyleSheet(
                "QSlider::groove:horizontal { height: 6px; border-radius: 3px; "
                f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {GRADIENTS[key]}); }}")
        self.box = QSpinBox()
        self.box.setRange(-100, 100)
        self.box.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.box.setFixedWidth(44)
        self.box.setObjectName("sliderBox")

        grid.addWidget(self.label, 0, 0)
        grid.addWidget(self.slider, 1, 0)
        grid.addWidget(self.box, 1, 1)

        self.slider.sliderPressed.connect(self.pressed)
        self.slider.sliderReleased.connect(self.released)
        self.slider.valueChanged.connect(self._from_slider)
        self.box.editingFinished.connect(self._from_box)
        self.slider.reset_requested.connect(self._reset)

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, v: int) -> None:
        """Programmatic update (from the document): never emits `edited`."""
        v = int(round(v))
        for w in (self.slider, self.box):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

    def _from_slider(self, v: int) -> None:
        self.box.blockSignals(True)
        self.box.setValue(v)
        self.box.blockSignals(False)
        self.edited.emit(v)  # covers dragging, clicking the track and keyboard steps

    def _from_box(self) -> None:
        v = self.box.value()
        if v != self.slider.value():
            self.slider.setValue(v)  # → _from_slider → edited

    def _reset(self) -> None:
        if self.slider.value() != 0:
            self.pressed.emit()
            self.slider.setValue(0)
            self.released.emit()
