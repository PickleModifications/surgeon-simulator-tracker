"""Joystick-style mouse control from a tracked hand position.

Each frame we get the palm position in normalized camera coordinates (0..1
on both axes). Relative to a user-calibrated center, we compute a velocity
vector whose magnitude grows with the offset. The cursor is moved by that
velocity on every tick. Fractional pixels accumulate so very slow drifts
still register eventually.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pydirectinput

# MediaPipe landmark IDs used for a robust palm-center estimate.
_PALM_IDS = (0, 5, 9, 13, 17)  # wrist + MCPs of index/middle/ring/pinky


def palm_center(landmarks: np.ndarray) -> np.ndarray:
    """Return the 2D palm center in normalized image coords."""
    pts = landmarks[list(_PALM_IDS), :2]
    return pts.mean(axis=0)


class MouseController:
    def __init__(
        self,
        sensitivity: float = 900.0,  # px/sec at full-frame deflection
        deadzone: float = 0.04,      # normalized radius; inside = no movement
        invert_x: bool = False,
        invert_y: bool = False,
    ) -> None:
        self.enabled: bool = False
        self.sensitivity: float = sensitivity
        self.deadzone: float = deadzone
        self.invert_x: bool = invert_x
        self.invert_y: bool = invert_y
        self.center: Optional[np.ndarray] = None
        self._accum_x: float = 0.0
        self._accum_y: float = 0.0
        pydirectinput.PAUSE = 0

    def set_center(self, pos: np.ndarray) -> None:
        self.center = np.array(pos[:2], dtype=np.float32)

    def clear_center(self) -> None:
        self.center = None

    def update(self, palm_pos: Optional[np.ndarray], dt: float) -> None:
        """Move the mouse by one tick of velocity. No-op if disabled or
        uncalibrated or the hand is missing."""
        if (
            not self.enabled
            or self.center is None
            or palm_pos is None
            or dt <= 0
        ):
            # Reset fractional accumulator so residual drift doesn't jump the
            # cursor the next time we engage.
            self._accum_x = 0.0
            self._accum_y = 0.0
            return

        offset = np.asarray(palm_pos[:2], dtype=np.float32) - self.center
        dist = float(np.linalg.norm(offset))
        if dist < self.deadzone:
            return

        # Scale so that at the deadzone boundary velocity is 0 and it ramps
        # linearly up to full sensitivity at offset==0.5 (edge of frame).
        effective = (dist - self.deadzone) / max(1e-6, 0.5 - self.deadzone)
        effective = min(1.0, effective)
        unit = offset / (dist + 1e-9)
        velocity = unit * effective * self.sensitivity  # px/sec

        sx = -1.0 if self.invert_x else 1.0
        sy = -1.0 if self.invert_y else 1.0
        self._accum_x += velocity[0] * dt * sx
        self._accum_y += velocity[1] * dt * sy

        step_x = int(self._accum_x)
        step_y = int(self._accum_y)
        if step_x == 0 and step_y == 0:
            return
        self._accum_x -= step_x
        self._accum_y -= step_y
        try:
            pydirectinput.moveRel(step_x, step_y, relative=True)
        except Exception:
            pass
