"""Top-level window tying capture, detection, keystrokes, and UI together."""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.camera import CameraInfo, CameraThread, list_cameras
from src.config import AppConfig, load_config, save_config
from src.gesture import (
    FINGERS,
    CalibrationSample,
    GestureDetector,
    compute_curl_ratios,
    midpoint_thresholds,
)
from src.depth_controller import (
    DescentController,
    STATE_ASCEND,
    STATE_DESCEND,
    STATE_IDLE,
    STATE_MAINTAIN,
    compute_palm_size,
)
from src.hand_tracker import HandResult, HandTracker
from src.key_sender import KeySender
from src.mouse_controller import MouseController, palm_center
from src.ui.finger_panel import FingerPanel
from src.ui.video_widget import VideoWidget

# Slider maps integer ticks [0..1000] to curl-ratio threshold [_SLIDER_MIN.._SLIDER_MAX].
_SLIDER_MIN = 0.2
_SLIDER_MAX = 2.0
_SLIDER_STEPS = 1000


def _ratio_to_slider(v: float) -> int:
    v = max(_SLIDER_MIN, min(_SLIDER_MAX, v))
    return int(round((v - _SLIDER_MIN) / (_SLIDER_MAX - _SLIDER_MIN) * _SLIDER_STEPS))


def _slider_to_ratio(v: int) -> float:
    return _SLIDER_MIN + (v / _SLIDER_STEPS) * (_SLIDER_MAX - _SLIDER_MIN)


class MainWindow(QMainWindow):
    # Emitted from the global keyboard hook (worker thread) so the UI thread
    # handles the toggle; direct GUI calls from a non-Qt thread would crash.
    _hotkey_toggle_signal = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Surgeon Simulator Tracker")
        self.resize(1000, 780)

        self._config: AppConfig = load_config()
        self._cameras: list[CameraInfo] = list_cameras()
        self._camera_thread: Optional[CameraThread] = None
        self._tracker = HandTracker()
        self._gesture = GestureDetector(
            thresholds=dict(self._config.thresholds_for(self._config.hand_mode)),
            debounce_frames=self._config.debounce_frames,
        )
        self._key_sender = KeySender()
        self._mouse = MouseController(
            sensitivity=self._config.mouse_sensitivity,
            deadzone=self._config.mouse_deadzone,
            invert_x=self._config.mouse_invert_x,
            invert_y=self._config.mouse_invert_y,
        )
        if self._config.mouse_center is not None:
            self._mouse.set_center(
                np.array(self._config.mouse_center, dtype=np.float32),
            )
        self._last_tick: Optional[float] = None
        self._latest_palm: Optional[np.ndarray] = None
        self._latest_palm_size: Optional[float] = None

        self._descent = DescentController(
            low_palm_size=self._config.depth_low_palm_size,
            high_palm_size=self._config.depth_high_palm_size,
            maintain_width=self._config.depth_maintain_width,
        )

        self._depth_cal_samples: list[float] = []
        self._depth_cal_low: Optional[float] = None

        self._calibration_mode: Optional[str] = None  # extended|curled|depth_low|depth_high
        self._calibration_sample: Optional[CalibrationSample] = None
        self._calibration_extended_means: Optional[dict[str, float]] = None

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._persist_config)

        # Drives frame processing. Pulls the most recent frame from the camera
        # thread and discards anything older; caps the UI load at ~60 Hz so
        # latency never grows even if inference is briefly slow.
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(16)
        self._frame_timer.timeout.connect(self._tick)

        # High-frequency mouse integrator: moves the cursor in small sub-frame
        # steps so motion is smooth even when the camera only delivers 30 FPS.
        self._mouse_timer = QTimer(self)
        self._mouse_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._mouse_timer.setInterval(5)
        self._mouse_timer.timeout.connect(self._mouse_tick)
        self._last_mouse_tick: Optional[float] = None

        self._build_ui()
        self._apply_config_to_ui()
        self._start_camera()
        self._install_global_hotkey()

        self._hotkey_toggle_signal.connect(self._toggle_enabled)
        self._frame_timer.start()
        self._mouse_timer.start()

    # ---------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # Row 1: camera + hand mode
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Camera:"))
        self._camera_combo = QComboBox()
        for info in self._cameras:
            self._camera_combo.addItem(f"{info.index}: {info.name}", info.index)
        if not self._cameras:
            self._camera_combo.addItem("No cameras found", -1)
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        row1.addWidget(self._camera_combo, 1)

        row1.addSpacing(20)
        row1.addWidget(QLabel("Hand:"))
        self._right_radio = QRadioButton("Right")
        self._left_radio = QRadioButton("Left")
        hand_group = QButtonGroup(self)
        hand_group.addButton(self._right_radio)
        hand_group.addButton(self._left_radio)
        self._right_radio.toggled.connect(self._on_hand_mode_changed)
        row1.addWidget(self._right_radio)
        row1.addWidget(self._left_radio)
        outer.addLayout(row1)

        # Row 2: buttons + status
        row2 = QHBoxLayout()
        self._toggle_btn = QPushButton("Start (F9)")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.clicked.connect(self._toggle_enabled)
        row2.addWidget(self._toggle_btn)

        self._calibrate_btn = QPushButton("Calibrate Fingers")
        self._calibrate_btn.clicked.connect(self._start_calibration)
        row2.addWidget(self._calibrate_btn)

        self._calibrate_center_btn = QPushButton("Calibrate Center")
        self._calibrate_center_btn.clicked.connect(self._calibrate_center)
        row2.addWidget(self._calibrate_center_btn)

        self._calibrate_depth_btn = QPushButton("Calibrate Depth")
        self._calibrate_depth_btn.clicked.connect(self._start_depth_calibration)
        row2.addWidget(self._calibrate_depth_btn)


        self._status_label = QLabel("Disabled — keystrokes not sent.")
        self._status_label.setStyleSheet("color: #bbb;")
        row2.addStretch(1)
        row2.addWidget(self._status_label)
        outer.addLayout(row2)

        # Video
        self._video = VideoWidget()
        outer.addWidget(self._video, 1)

        # Finger panel
        self._fingers = FingerPanel()
        outer.addWidget(self._fingers)

        # Thresholds
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Thresholds:"))
        self._sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        for finger in FINGERS:
            col = QVBoxLayout()
            label = QLabel(finger.capitalize())
            label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            value_label = QLabel("—")
            value_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            value_label.setStyleSheet("color: #888; font-family: monospace;")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(_SLIDER_STEPS)
            slider.valueChanged.connect(
                lambda v, f=finger: self._on_slider_changed(f, v)
            )
            col.addWidget(label)
            col.addWidget(slider)
            col.addWidget(value_label)
            threshold_row.addLayout(col, 1)
            self._sliders[finger] = slider
            self._slider_labels[finger] = value_label
        outer.addLayout(threshold_row)

        # Mouse sensitivity + deadzone row.
        mouse_row = QHBoxLayout()
        mouse_row.addWidget(QLabel("Mouse:"))

        sens_col = QVBoxLayout()
        sens_col.addWidget(QLabel("Sensitivity (px/sec)"))
        self._sens_slider = QSlider(Qt.Orientation.Horizontal)
        self._sens_slider.setRange(100, 3000)
        self._sens_slider.valueChanged.connect(self._on_sens_changed)
        self._sens_label = QLabel("—")
        self._sens_label.setStyleSheet("color: #888; font-family: monospace;")
        sens_col.addWidget(self._sens_slider)
        sens_col.addWidget(self._sens_label)
        mouse_row.addLayout(sens_col, 2)

        dead_col = QVBoxLayout()
        dead_col.addWidget(QLabel("Deadzone"))
        self._dead_slider = QSlider(Qt.Orientation.Horizontal)
        self._dead_slider.setRange(0, 200)  # 0.00 .. 0.20
        self._dead_slider.valueChanged.connect(self._on_dead_changed)
        self._dead_label = QLabel("—")
        self._dead_label.setStyleSheet("color: #888; font-family: monospace;")
        dead_col.addWidget(self._dead_slider)
        dead_col.addWidget(self._dead_label)
        mouse_row.addLayout(dead_col, 1)

        invert_col = QVBoxLayout()
        invert_col.addWidget(QLabel("Invert"))
        self._invert_x_cb = QCheckBox("X")
        self._invert_y_cb = QCheckBox("Y")
        self._invert_x_cb.toggled.connect(self._on_invert_x_changed)
        self._invert_y_cb.toggled.connect(self._on_invert_y_changed)
        invert_col.addWidget(self._invert_x_cb)
        invert_col.addWidget(self._invert_y_cb)
        mouse_row.addLayout(invert_col, 0)

        outer.addLayout(mouse_row)

        # Descent maintain-zone width. Also sets how fast ASCEND/DESCEND can
        # ramp: narrower maintain = steeper duty-cycle curve on either side.
        descent_row = QHBoxLayout()
        descent_row.addWidget(QLabel("Descent:"))
        mw_col = QVBoxLayout()
        mw_col.addWidget(QLabel("Maintain width (fraction of range)"))
        self._mw_slider = QSlider(Qt.Orientation.Horizontal)
        self._mw_slider.setRange(0, 1000)  # 0.00 .. 1.00
        self._mw_slider.valueChanged.connect(self._on_mw_changed)
        self._mw_label = QLabel("—")
        self._mw_label.setStyleSheet("color: #888; font-family: monospace;")
        mw_col.addWidget(self._mw_slider)
        mw_col.addWidget(self._mw_label)
        descent_row.addLayout(mw_col, 1)
        outer.addLayout(descent_row)


    def _apply_config_to_ui(self) -> None:
        # Camera
        idx_to_combo = {
            self._camera_combo.itemData(i): i
            for i in range(self._camera_combo.count())
        }
        target = idx_to_combo.get(self._config.camera_index)
        if target is not None:
            self._camera_combo.setCurrentIndex(target)

        # Hand mode
        if self._config.hand_mode == "left":
            self._left_radio.setChecked(True)
        else:
            self._right_radio.setChecked(True)

        # Sliders reflect current hand mode thresholds
        self._refresh_sliders_from_config()

        # Mouse sliders
        self._sens_slider.blockSignals(True)
        self._sens_slider.setValue(int(round(self._config.mouse_sensitivity)))
        self._sens_slider.blockSignals(False)
        self._sens_label.setText(f"{int(self._config.mouse_sensitivity)}")
        self._dead_slider.blockSignals(True)
        self._dead_slider.setValue(int(round(self._config.mouse_deadzone * 1000)))
        self._dead_slider.blockSignals(False)
        self._dead_label.setText(f"{self._config.mouse_deadzone:.3f}")
        self._invert_x_cb.blockSignals(True)
        self._invert_x_cb.setChecked(self._config.mouse_invert_x)
        self._invert_x_cb.blockSignals(False)
        self._invert_y_cb.blockSignals(True)
        self._invert_y_cb.setChecked(self._config.mouse_invert_y)
        self._invert_y_cb.blockSignals(False)
        self._mw_slider.blockSignals(True)
        self._mw_slider.setValue(int(round(self._config.depth_maintain_width * 1000)))
        self._mw_slider.blockSignals(False)
        self._mw_label.setText(f"{self._config.depth_maintain_width:.2f}")

    def _refresh_sliders_from_config(self) -> None:
        thresholds = self._config.thresholds_for(self._config.hand_mode)
        for finger, slider in self._sliders.items():
            value = thresholds.get(finger, 1.0)
            slider.blockSignals(True)
            slider.setValue(_ratio_to_slider(value))
            slider.blockSignals(False)
            self._slider_labels[finger].setText(f"{value:.2f}")
        self._gesture.thresholds = dict(thresholds)

    # --------------------------------------------------------- camera

    def _start_camera(self) -> None:
        idx = self._camera_combo.currentData()
        if idx is None or idx < 0:
            return
        self._stop_camera()
        thread = CameraThread(int(idx))
        thread.error.connect(self._on_camera_error)
        thread.start()
        self._camera_thread = thread

    def _stop_camera(self) -> None:
        if self._camera_thread is not None:
            try:
                self._camera_thread.error.disconnect(self._on_camera_error)
            except TypeError:
                pass
            self._camera_thread.stop()
            self._camera_thread = None

    def _on_camera_changed(self, _idx: int) -> None:
        self._config.camera_index = int(self._camera_combo.currentData() or 0)
        self._schedule_save()
        self._start_camera()

    def _on_camera_error(self, msg: str) -> None:
        self._status_label.setText(f"Camera error: {msg}")

    # ---------------------------------------------------------- frames

    def _tick(self) -> None:
        if self._camera_thread is None:
            return
        bgr = self._camera_thread.pop_latest()
        if bgr is None:
            return

        now = time.perf_counter()
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now

        result = self._tracker.process(bgr)

        if self._calibration_mode is not None and result is not None:
            ratios = compute_curl_ratios(result.landmarks)
            if self._calibration_sample is not None:
                self._calibration_sample.add(ratios)

        palm: Optional[np.ndarray] = None
        palm_size: Optional[float] = None
        if result is not None:
            palm = palm_center(result.landmarks)
            palm_size = compute_palm_size(result.landmarks, result.world_landmarks)
            self._tracker.draw(bgr, result)
            if self._calibration_mode is None:
                states = self._gesture.update(result.landmarks)
                self._fingers.set_states(states)
                self._key_sender.apply(states)
            else:
                self._fingers.set_all_off()
        else:
            if self._calibration_mode is None:
                self._gesture.reset_states()
                self._key_sender.apply({f: False for f in FINGERS})
            self._fingers.set_all_off()

        self._latest_palm = palm
        self._latest_palm_size = palm_size

        if self._calibration_mode in ("depth_low", "depth_high") and palm_size is not None:
            self._depth_cal_samples.append(palm_size)

        self._draw_mouse_overlay(bgr, palm)
        self._video.show_frame(bgr)

    def _mouse_tick(self) -> None:
        now = time.perf_counter()
        dt = 0.0 if self._last_mouse_tick is None else now - self._last_mouse_tick
        self._last_mouse_tick = now
        if self._calibration_mode is not None:
            return
        self._mouse.update(self._latest_palm, dt)
        self._descent.update(self._latest_palm_size, now)

    def _draw_mouse_overlay(self, bgr: np.ndarray, palm: Optional[np.ndarray]) -> None:
        h, w = bgr.shape[:2]
        if self._mouse.center is not None:
            cx, cy = int(self._mouse.center[0] * w), int(self._mouse.center[1] * h)
            cv2.circle(bgr, (cx, cy), 8, (0, 200, 255), 2)
            dead_px = int(self._mouse.deadzone * max(w, h))
            if dead_px > 2:
                cv2.circle(bgr, (cx, cy), dead_px, (0, 120, 180), 1)
        if palm is not None:
            px, py = int(palm[0] * w), int(palm[1] * h)
            cv2.drawMarker(bgr, (px, py), (60, 230, 90), cv2.MARKER_CROSS, 14, 2)

        # Descent HUD: vertical bar on the right showing live palm_size
        # against the calibrated low/high range.
        bar_x = w - 28
        top = 20
        bottom = h - 40
        cv2.rectangle(bgr, (bar_x - 6, top), (bar_x + 6, bottom), (60, 60, 60), -1)
        cv2.rectangle(bgr, (bar_x - 6, top), (bar_x + 6, bottom), (120, 120, 120), 1)

        if self._descent.is_calibrated():
            def y_for(t: float) -> int:
                return int(bottom - max(0.0, min(1.0, t)) * (bottom - top))
            maint_lo, maint_hi = self._descent.maintain_bounds()
            for frac in (maint_lo, maint_hi):
                y = y_for(frac)
                cv2.line(bgr, (bar_x - 14, y), (bar_x + 14, y), (80, 180, 255), 2)
            # Labels roughly centered in each zone.
            cv2.putText(bgr, "ASCEND",   (bar_x - 88, y_for((1.0 + maint_hi) / 2.0) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 230, 120), 1)
            cv2.putText(bgr, "MAINTAIN", (bar_x - 96, y_for(0.5) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
            cv2.putText(bgr, "DESCEND",  (bar_x - 92, y_for(maint_lo / 2.0) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 230), 1)
            t = self._descent.normalized_t(self._latest_palm_size)
            if t is not None:
                cv2.circle(bgr, (bar_x, y_for(t)), 5, (80, 230, 120), -1)
            duty_pct = int(round(self._descent.duty_cycle * 100))
            cv2.putText(bgr, f"duty {duty_pct:3d}%", (bar_x - 90, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        else:
            cv2.putText(bgr, "depth: not calibrated", (bar_x - 190, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Live palm size text (bottom-left).
        if self._latest_palm_size is not None:
            cv2.putText(bgr, f"size {self._latest_palm_size:.3f}", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        else:
            cv2.putText(bgr, "size: —", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        state = self._descent.state
        colors = {
            STATE_ASCEND:   (120, 230, 120),
            STATE_MAINTAIN: (200, 200, 200),
            STATE_DESCEND:  (120, 120, 230),
            STATE_IDLE:     (120, 120, 120),
        }
        cv2.putText(bgr, f"state: {state.upper()}", (w - 230, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[state], 2)

    # --------------------------------------------------------- hand mode

    def _on_hand_mode_changed(self) -> None:
        new_mode = "left" if self._left_radio.isChecked() else "right"
        if new_mode == self._config.hand_mode:
            return
        self._config.hand_mode = new_mode
        self._refresh_sliders_from_config()
        self._schedule_save()

    # --------------------------------------------------------- sliders

    def _on_slider_changed(self, finger: str, value: int) -> None:
        ratio = _slider_to_ratio(value)
        thresholds = dict(self._config.thresholds_for(self._config.hand_mode))
        thresholds[finger] = ratio
        self._config.set_thresholds_for(self._config.hand_mode, thresholds)
        self._gesture.thresholds[finger] = ratio
        self._slider_labels[finger].setText(f"{ratio:.2f}")
        self._schedule_save()

    # -------------------------------------------------------- enable/disable

    def _toggle_enabled(self) -> None:
        new_state = not self._key_sender.enabled
        self._key_sender.set_enabled(new_state)
        self._mouse.enabled = new_state
        if new_state:
            self._descent.enabled = True
        else:
            self._descent.disable()
        self._toggle_btn.setChecked(new_state)
        if new_state:
            self._toggle_btn.setText("Stop (F9)")
            self._status_label.setText("ENABLED — curls will press game keys.")
            self._status_label.setStyleSheet("color: #3ddc84; font-weight: bold;")
        else:
            self._toggle_btn.setText("Start (F9)")
            self._status_label.setText("Disabled — keystrokes not sent.")
            self._status_label.setStyleSheet("color: #bbb;")

    # -------------------------------------------------------- mouse

    def _calibrate_center(self) -> None:
        palm = self._latest_palm
        if palm is None:
            QMessageBox.warning(
                self,
                "No hand detected",
                "Couldn't find your hand in the live preview. Place it in the "
                "camera view and try again.",
            )
            return
        self._mouse.set_center(palm)
        self._config.mouse_center = (float(palm[0]), float(palm[1]))
        self._persist_config()

    def _on_sens_changed(self, value: int) -> None:
        self._mouse.sensitivity = float(value)
        self._config.mouse_sensitivity = float(value)
        self._sens_label.setText(f"{value}")
        self._schedule_save()

    def _on_dead_changed(self, value: int) -> None:
        d = value / 1000.0
        self._mouse.deadzone = d
        self._config.mouse_deadzone = d
        self._dead_label.setText(f"{d:.3f}")
        self._schedule_save()

    def _on_invert_x_changed(self, checked: bool) -> None:
        self._mouse.invert_x = checked
        self._config.mouse_invert_x = checked
        self._schedule_save()

    def _on_invert_y_changed(self, checked: bool) -> None:
        self._mouse.invert_y = checked
        self._config.mouse_invert_y = checked
        self._schedule_save()

    def _on_mw_changed(self, value: int) -> None:
        w = value / 1000.0
        self._descent.maintain_width = w
        self._config.depth_maintain_width = w
        self._mw_label.setText(f"{w:.2f}")
        self._schedule_save()

    # -------------------------------------------------------- depth calibration

    def _start_depth_calibration(self) -> None:
        self._depth_cal_low = None
        reply = QMessageBox.information(
            self,
            "Calibrate Depth — Step 1 of 2",
            "Place your hand at the LOWEST position (flat on the desk, as low "
            "as it goes).\n\nClick OK to capture a 3-second average.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            return
        self._run_depth_phase("depth_low", on_done=self._depth_step_two)

    def _depth_step_two(self, low: Optional[float]) -> None:
        if low is None:
            QMessageBox.warning(self, "Calibration failed", "No hand visible. Try again.")
            return
        self._depth_cal_low = low
        reply = QMessageBox.information(
            self,
            "Calibrate Depth — Step 2 of 2",
            "Now place your hand at the HIGHEST position (lifted clearly up "
            "off the desk).\n\nClick OK to capture a 3-second average.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            self._depth_cal_low = None
            return
        self._run_depth_phase("depth_high", on_done=self._depth_finish)

    def _depth_finish(self, high: Optional[float]) -> None:
        low = self._depth_cal_low
        if low is None or high is None or abs(high - low) < 1e-4:
            QMessageBox.warning(
                self, "Calibration failed",
                "Low and high palm sizes were too similar. Exaggerate the lift "
                "and try again.",
            )
            self._depth_cal_low = None
            return
        self._descent.set_calibration(low, high)
        self._config.depth_low_palm_size = float(self._descent.low_palm_size)
        self._config.depth_high_palm_size = float(self._descent.high_palm_size)
        self._persist_config()
        self._depth_cal_low = None
        QMessageBox.information(
            self, "Calibration complete",
            f"Low (desk): {self._descent.low_palm_size:.3f}\n"
            f"High (lifted): {self._descent.high_palm_size:.3f}\n\n"
            f"Zones split at {self._descent.low_palm_size + (self._descent.high_palm_size - self._descent.low_palm_size) / 3.0:.3f} "
            f"and {self._descent.low_palm_size + (self._descent.high_palm_size - self._descent.low_palm_size) * 2.0 / 3.0:.3f}.",
        )

    def _run_depth_phase(self, phase: str, on_done) -> None:
        self._calibration_mode = phase
        self._depth_cal_samples = []
        duration_ms = 3000
        progress = QProgressDialog(
            f"Capturing '{phase.replace('depth_', '')}' pose… hold still.",
            None, 0, duration_ms, self,
        )
        progress.setWindowTitle("Calibrating depth")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        timer = QTimer(self)
        timer.setInterval(50)
        elapsed = {"ms": 0}

        def tick() -> None:
            elapsed["ms"] += 50
            progress.setValue(min(elapsed["ms"], duration_ms))
            if elapsed["ms"] >= duration_ms:
                timer.stop()
                progress.close()
                samples = self._depth_cal_samples
                self._calibration_mode = None
                self._depth_cal_samples = []
                mean = float(np.mean(samples)) if samples else None
                on_done(mean)

        timer.timeout.connect(tick)
        timer.start()

    # -------------------------------------------------------- finger calibration

    def _start_calibration(self) -> None:
        self._calibration_extended_means = None
        reply = QMessageBox.information(
            self,
            "Calibration — Step 1 of 2",
            "Hold your hand over the camera with ALL FIVE FINGERS EXTENDED "
            "(flat, relaxed).\n\nClick OK to begin a 3-second capture.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            return
        self._run_calibration_phase("extended", on_done=self._calibration_step_two)

    def _calibration_step_two(self, extended_means: dict[str, float]) -> None:
        self._calibration_extended_means = extended_means
        reply = QMessageBox.information(
            self,
            "Calibration — Step 2 of 2",
            "Now CURL ALL FIVE FINGERS (make a loose fist, thumb tucked in).\n\n"
            "Click OK to begin a 3-second capture.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            self._calibration_mode = None
            return
        self._run_calibration_phase("curled", on_done=self._calibration_finish)

    def _calibration_finish(self, curled_means: dict[str, float]) -> None:
        extended = self._calibration_extended_means or {}
        thresholds = midpoint_thresholds(extended, curled_means)
        self._config.set_thresholds_for(self._config.hand_mode, thresholds)
        self._refresh_sliders_from_config()
        self._persist_config()
        self._calibration_extended_means = None
        QMessageBox.information(
            self,
            "Calibration complete",
            "Thresholds saved. Try flexing each finger to confirm the "
            "indicators respond cleanly.",
        )

    def _run_calibration_phase(self, phase: str, on_done) -> None:
        self._calibration_mode = phase
        self._calibration_sample = CalibrationSample()

        duration_ms = 3000
        progress = QProgressDialog(
            f"Capturing '{phase}' pose… hold still.",
            None,
            0,
            duration_ms,
            self,
        )
        progress.setWindowTitle("Calibrating")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        timer = QTimer(self)
        timer.setInterval(50)
        elapsed = {"ms": 0}

        def tick() -> None:
            elapsed["ms"] += 50
            progress.setValue(min(elapsed["ms"], duration_ms))
            if elapsed["ms"] >= duration_ms:
                timer.stop()
                progress.close()
                sample = self._calibration_sample
                self._calibration_mode = None
                self._calibration_sample = None
                means = sample.means() if sample else {}
                on_done(means)

        timer.timeout.connect(tick)
        timer.start()

    # --------------------------------------------------------- hotkey

    def _install_global_hotkey(self) -> None:
        try:
            import keyboard
        except Exception:
            return
        try:
            keyboard.add_hotkey(
                self._config.toggle_hotkey,
                self._hotkey_toggle_signal.emit,
            )
        except Exception:
            pass

    def _uninstall_global_hotkey(self) -> None:
        try:
            import keyboard
            keyboard.remove_all_hotkeys()
        except Exception:
            pass

    # -------------------------------------------------------- persistence

    def _schedule_save(self) -> None:
        self._save_timer.start()

    def _persist_config(self) -> None:
        save_config(self._config)

    # -------------------------------------------------------- lifecycle

    def closeEvent(self, event: QCloseEvent) -> None:
        self._frame_timer.stop()
        self._mouse_timer.stop()
        self._uninstall_global_hotkey()
        self._key_sender.set_enabled(False)
        self._mouse.enabled = False
        self._descent.disable()
        self._stop_camera()
        self._tracker.close()
        self._persist_config()
        super().closeEvent(event)
