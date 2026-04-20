"""Palm-size based descent with explicit two-point calibration.

Top-down camera pointed at a desk: the hand stays essentially flat and
parallel to the sensor. The perceived hand size (wrist-to-middle-MCP
distance in the image) therefore varies almost entirely with vertical lift,
not with lateral desk motion.

Calibration captures two snapshots:
    low_palm_size  = hand resting on the desk  (smallest)
    high_palm_size = hand lifted up clearly    (largest)

Everything between those two is split into equal thirds:
    bottom third -> DESCEND (LMB held)
    middle third -> MAINTAIN (LMB tapped at ~10 Hz)
    top third    -> ASCEND  (LMB released)

Hysteresis (5% of the range by default) keeps state changes clean at the
boundaries.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pydirectinput

STATE_IDLE = "idle"
STATE_ASCEND = "ascend"
STATE_MAINTAIN = "maintain"
STATE_DESCEND = "descend"

_PALM_IDS = (0, 5, 9, 13, 17)  # wrist + index/middle/ring/pinky MCPs


def compute_palm_size(
    landmarks: np.ndarray,
    world_landmarks: Optional[np.ndarray] = None,
) -> float:
    """Palm pentagon perimeter in normalized image coords.

    Sums the 5 edges of the closed polygon through (wrist, index_MCP,
    middle_MCP, ring_MCP, pinky_MCP). All five of those landmarks are on
    the palm skeleton — they do NOT move when the user curls/extends
    fingers or lifts the thumb, so finger actions can't spoof depth
    changes. Averaging 5 edges damps single-landmark jitter compared to
    relying on one bone.

    No tilt correction: the camera is top-down and the hand stays flat on
    the desk, so `cos(tilt) ≈ 1` in practice; the correction added noise
    whenever MediaPipe's 3D inference jittered (e.g. during thumb raises).
    """
    total = 0.0
    n = len(_PALM_IDS)
    for i in range(n):
        a = landmarks[_PALM_IDS[i], :2]
        b = landmarks[_PALM_IDS[(i + 1) % n], :2]
        total += float(np.linalg.norm(a - b))
    return total


class DescentController:
    def __init__(
        self,
        low_palm_size: Optional[float] = None,
        high_palm_size: Optional[float] = None,
        maintain_width: float = 0.33,    # fraction of [low..high] that counts as MAINTAIN
        tap_period: float = 0.10,        # PWM period, seconds
    ) -> None:
        self.enabled: bool = False
        self.low_palm_size: Optional[float] = low_palm_size
        self.high_palm_size: Optional[float] = high_palm_size
        self.maintain_width: float = maintain_width
        self.tap_period: float = tap_period

        self._state: str = STATE_IDLE
        self._duty: float = 0.5
        self._lmb_pressed: bool = False
        self._phase_start: Optional[float] = None
        pydirectinput.PAUSE = 0

    def set_calibration(self, low: float, high: float) -> None:
        if abs(high - low) < 1e-4:
            return  # degenerate
        self.low_palm_size = float(min(low, high))
        self.high_palm_size = float(max(low, high))

    def is_calibrated(self) -> bool:
        return self.low_palm_size is not None and self.high_palm_size is not None

    def disable(self) -> None:
        self._release_if_needed()
        self.enabled = False
        self._state = STATE_IDLE
        self._phase_start = None

    def _release_if_needed(self) -> None:
        if self._lmb_pressed:
            try:
                pydirectinput.mouseUp(button="left")
            except Exception:
                pass
            self._lmb_pressed = False

    def _press_if_needed(self) -> None:
        if not self._lmb_pressed:
            try:
                pydirectinput.mouseDown(button="left")
            except Exception:
                pass
            self._lmb_pressed = True

    def maintain_bounds(self) -> tuple[float, float]:
        """Return (lo, hi) in normalized-t space that constitute the MAINTAIN zone."""
        half = max(0.0, min(1.0, self.maintain_width)) / 2.0
        return (0.5 - half, 0.5 + half)

    def compute_duty(self, palm_size: Optional[float]) -> float:
        """Duty cycle in [0, 1]. 1.0 = LMB fully held, 0.0 = fully released,
        0.5 = balanced (maintain). Ramps linearly from the maintain boundary
        toward each calibrated extreme."""
        if palm_size is None or not self.is_calibrated():
            return 0.0
        span = self.high_palm_size - self.low_palm_size
        if span < 1e-4:
            return 0.5
        t = (palm_size - self.low_palm_size) / span
        t = max(0.0, min(1.0, t))
        lo, hi = self.maintain_bounds()
        if t < lo:
            # DESCEND side: 0.5 at boundary, 1.0 at t = 0.
            denom = lo if lo > 1e-6 else 1.0
            frac = (lo - t) / denom
            return 0.5 + 0.5 * max(0.0, min(1.0, frac))
        if t > hi:
            # ASCEND side: 0.5 at boundary, 0.0 at t = 1.
            denom = (1.0 - hi) if (1.0 - hi) > 1e-6 else 1.0
            frac = (t - hi) / denom
            return 0.5 - 0.5 * max(0.0, min(1.0, frac))
        return 0.5

    def classify_state(self, palm_size: Optional[float]) -> str:
        if palm_size is None or not self.is_calibrated():
            return STATE_IDLE
        span = self.high_palm_size - self.low_palm_size
        if span < 1e-4:
            return STATE_IDLE
        t = (palm_size - self.low_palm_size) / span
        lo, hi = self.maintain_bounds()
        if t < lo:
            return STATE_DESCEND
        if t > hi:
            return STATE_ASCEND
        return STATE_MAINTAIN

    def update(self, palm_size: Optional[float], now: float) -> None:
        self._state = self.classify_state(palm_size)
        duty = self.compute_duty(palm_size) if self.is_calibrated() else 0.0
        self._duty = duty

        if not self.enabled or palm_size is None or not self.is_calibrated():
            self._release_if_needed()
            self._phase_start = None
            return

        # Full hold / full release shortcuts — avoid PWM edge events at the
        # endpoints.
        if duty >= 0.999:
            self._press_if_needed()
            self._phase_start = None
            return
        if duty <= 0.001:
            self._release_if_needed()
            self._phase_start = None
            return

        # PWM the LMB at the current duty cycle.
        if self._phase_start is None:
            self._phase_start = now
        cycle_elapsed = now - self._phase_start
        if cycle_elapsed >= self.tap_period:
            self._phase_start = now
            cycle_elapsed = 0.0
        should_press = cycle_elapsed < (duty * self.tap_period)
        if should_press:
            self._press_if_needed()
        else:
            self._release_if_needed()

    @property
    def state(self) -> str:
        return self._state

    @property
    def duty_cycle(self) -> float:
        return self._duty

    def normalized_t(self, palm_size: Optional[float]) -> Optional[float]:
        """Return t in [0..1] for HUD rendering, or None if uncalibrated."""
        if palm_size is None or not self.is_calibrated():
            return None
        span = self.high_palm_size - self.low_palm_size
        if span < 1e-4:
            return None
        return max(0.0, min(1.0, (palm_size - self.low_palm_size) / span))
