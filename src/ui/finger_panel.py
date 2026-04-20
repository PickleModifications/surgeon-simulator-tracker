"""Five indicator lights showing which fingers are currently detected as down."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.gesture import FINGERS
from src.key_sender import KEY_MAP

_DISPLAY_LABELS: dict[str, str] = {
    "thumb":  "Thumb",
    "index":  "Index",
    "middle": "Middle",
    "ring":   "Ring",
    "pinky":  "Pinky",
}


class _Indicator(QFrame):
    _ON_STYLE = (
        "background-color: #3ddc84;"
        "border-radius: 18px; border: 2px solid #2a9d5e;"
    )
    _OFF_STYLE = (
        "background-color: #2a2a2a;"
        "border-radius: 18px; border: 2px solid #444;"
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(36, 36)
        self.setStyleSheet(self._OFF_STYLE)

    def set_on(self, on: bool) -> None:
        self.setStyleSheet(self._ON_STYLE if on else self._OFF_STYLE)


class FingerPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(16)

        self._indicators: dict[str, _Indicator] = {}

        for finger in FINGERS:
            col = QVBoxLayout()
            col.setSpacing(4)
            col.setAlignment(Qt.AlignmentFlag.AlignHCenter)

            name = QLabel(_DISPLAY_LABELS[finger])
            name.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            name.setStyleSheet("font-weight: bold;")

            key = QLabel(f"({KEY_MAP[finger].upper()})")
            key.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            key.setStyleSheet("color: #888;")

            dot = _Indicator()
            self._indicators[finger] = dot

            col.addWidget(name)
            col.addWidget(key)
            col.addWidget(dot, alignment=Qt.AlignmentFlag.AlignHCenter)
            row.addLayout(col)

    def set_states(self, states: dict[str, bool]) -> None:
        for finger, on in states.items():
            indicator = self._indicators.get(finger)
            if indicator is not None:
                indicator.set_on(on)

    def set_all_off(self) -> None:
        for indicator in self._indicators.values():
            indicator.set_on(False)
