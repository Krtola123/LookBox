"""Cut-out session → Fill / Grab (M13). Fill replaces what's selected with what's
around it (on a layer of its own); Grab also lifts the selection onto its own layer,
Canva's "Magic Grab". The fill method is remembered."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from lookbox.ui.settings import app_settings

METHODS = (("ai", "AI fill"), ("quick", "Quick fill (no AI)"), ("plate", "From a clean render…"))
SETTING = "fill/method"


class FillRow(QWidget):
    def __init__(self, brush, controller) -> None:
        super().__init__()
        self.brush, self.controller = brush, controller
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 6, 0, 0)
        col.setSpacing(4)
        how = QHBoxLayout()
        lab = QLabel("Fill with")
        lab.setObjectName("sliderLabel")
        self.method = QComboBox()
        for key, label in METHODS:
            self.method.addItem(label, key)
        saved = app_settings().value(SETTING, "ai", type=str)
        self.method.setCurrentIndex(max(0, [k for k, _ in METHODS].index(saved) if saved in dict(METHODS) else 0))
        self.method.currentIndexChanged.connect(
            lambda _i: app_settings().setValue(SETTING, self.method.currentData()))
        self.method.setToolTip("AI fill: for photos and backgrounds (208 MB model, downloads once).\n"
                               "Quick fill: small blemishes, no download.\n"
                               "From a clean render: the same shot rendered without the object; exact.")
        how.addWidget(lab)
        how.addStretch(1)
        how.addWidget(self.method)
        col.addLayout(how)
        row = QHBoxLayout()
        self.fill_btn = QPushButton("Fill")
        self.fill_btn.setToolTip("Replace the selection with what's around it (on its own layer)")
        self.fill_btn.clicked.connect(lambda: self._run(grab=False))
        self.grab_btn = QPushButton("Grab")
        self.grab_btn.setToolTip("Put the selection on its own layer and fill the hole it leaves behind")
        self.grab_btn.clicked.connect(lambda: self._run(grab=True))
        row.addWidget(self.fill_btn)
        row.addWidget(self.grab_btn)
        col.addLayout(row)

    def _run(self, grab: bool) -> None:
        layer_id, mask = self.brush.layer_id, self.brush.mask
        if layer_id is None or mask is None or self.controller.busy:
            return
        mask = mask.copy()
        set_mask = self.brush.take_edit() if grab else None
        self.brush.end()  # the selection was for the fill, not a cut-out of this layer
        self.controller.fill(layer_id, mask, self.method.currentData(), set_mask=set_mask, grab=grab)
