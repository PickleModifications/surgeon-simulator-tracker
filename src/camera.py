"""Webcam enumeration and capture thread.

The capture thread reads frames as fast as the driver delivers them and stores
only the most recent one in a mutex-protected slot. The UI thread polls with
`pop_latest()`. This guarantees the UI never falls behind the camera — stale
frames are dropped rather than queued, so input latency stays bounded even if
a frame takes longer to process.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QThread, pyqtSignal


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str


def list_cameras(max_probe: int = 6) -> list[CameraInfo]:
    """Return available webcams. Prefers DirectShow device names on Windows."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        if names:
            return [CameraInfo(i, n) for i, n in enumerate(names)]
    except Exception:
        pass

    found: list[CameraInfo] = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(CameraInfo(i, f"Camera {i}"))
            cap.release()
    return found


class CameraThread(QThread):
    """Reads from a webcam and exposes the most recent frame on demand."""

    error = pyqtSignal(str)

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self._index = index
        self._running = False
        self._latest: Optional[np.ndarray] = None
        self._lock = QMutex()

    def run(self) -> None:
        cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error.emit(f"Could not open camera index {self._index}.")
            return

        # MJPG over YUY2 is usually 3-5x faster over USB at 30fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Shrink driver buffer so we always read a fresh frame instead of a
        # queued old one. Not all backends honor this; harmless if ignored.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue
                self._lock.lock()
                self._latest = frame
                self._lock.unlock()
        finally:
            cap.release()

    def pop_latest(self) -> Optional[np.ndarray]:
        """Return the most recent unread frame, or None if none is waiting."""
        self._lock.lock()
        frame = self._latest
        self._latest = None
        self._lock.unlock()
        return frame

    def stop(self) -> None:
        self._running = False
        self.wait(2000)
