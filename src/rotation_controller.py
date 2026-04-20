"""Hand-rotation -> right-click rotation-mode driver.

Uses the palm normal (from MediaPipe's world_landmarks) as a joystick:

    palm flat (normal pointing at camera)  -> rotation mode OFF
                                             (RMB released)
    palm tilted past the deadzone          -> rotation mode ON
                                             (RMB held; cursor velocity
                                              driven by tilt magnitude)

The normal's X component drives horizontal cursor velocity; Y drives
vertical. Further from flat = faster. Return to flat = cursor stops and
RMB releases. Surgeon Simulator persists the in-game hand rotation after
RMB releases, so we don't need to re-track it.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pydirectinput

_WRIST = 0
_INDEX_MCP = 5
_PINKY_MCP = 17


def palm_normal_world(world_landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Unit normal of the palm triangle (wrist, index_MCP, pinky_MCP) in
    camera-aligned 3D coords. Returns None if world landmarks are missing.

    Components:
        x: palm yaw (rotation around vertical) - horizontal tilt
        y: palm pitch (rotation around horizontal) - forward/back tilt
        z: how flat the palm is (|z| ~ 1 when facing the camera squarely)
    """
    if world_landmarks is None or world_landmarks.shape[0] < 18:
        return None
    w = world_landmarks[_WRIST]
    i = world_landmarks[_INDEX_MCP]
    p = world_landmarks[_PINKY_MCP]
    n = np.cross(i - w, p - w)
    mag = float(np.linalg.norm(n))
    if mag < 1e-6:
        return None
    return (n / mag).astype(np.float32)


class RotationController:
    def __init__(
        self,
        sensitivity: float = 1500.0,   # px per unit of tilt change
        deadzone: float = 0.20,        # min tilt magnitude (after neutral) to engage
        rmb_hold_sec: float = 0.10,    # RMB auto-releases this long after last delta
        invert_x: bool = False,
        invert_y: bool = False,
        neutral_x: float = 0.0,
        neutral_y: float = 0.0,
    ) -> None:
        self.enabled: bool = False
        self.sensitivity: float = sensitivity
        self.deadzone: float = deadzone
        self.rmb_hold_sec: float = rmb_hold_sec
        self.invert_x: bool = invert_x
        self.invert_y: bool = invert_y
        self.neutral_x: float = neutral_x
        self.neutral_y: float = neutral_y

        self._rmb_pressed: bool = False
        self._accum_x: float = 0.0
        self._accum_y: float = 0.0
        self._last_tilt: float = 0.0
        self._last_tilt_vec: tuple[float, float] = (0.0, 0.0)
        # Reference tilt: when a rotation "session" begins, we snapshot the
        # current tilt and emit deltas relative to that. A session ends when
        # no delta has been sent for `rmb_hold_sec`; the next change in tilt
        # starts a new session with the current tilt as the new reference.
        self._ref_tilt_vec: Optional[tuple[float, float]] = None
        self._last_delta_time: Optional[float] = None
        self._past_deadzone: bool = False
        self._is_actuating: bool = False
        pydirectinput.PAUSE = 0

    def set_neutral(self, palm_normal: Optional[np.ndarray]) -> None:
        if palm_normal is None:
            self.neutral_x = 0.0
            self.neutral_y = 0.0
            return
        self.neutral_x = float(palm_normal[0])
        self.neutral_y = float(palm_normal[1])

    def _press_rmb(self) -> None:
        if not self._rmb_pressed:
            try:
                pydirectinput.mouseDown(button="right")
            except Exception:
                pass
            self._rmb_pressed = True

    def _release_rmb(self) -> None:
        if self._rmb_pressed:
            try:
                pydirectinput.mouseUp(button="right")
            except Exception:
                pass
            self._rmb_pressed = False

    def disable(self) -> None:
        self._release_rmb()
        self.enabled = False
        self._accum_x = 0.0
        self._accum_y = 0.0
        self._ref_tilt_vec = None
        self._last_delta_time = None
        self._is_actuating = False

    def update(
        self,
        palm_normal: Optional[np.ndarray],
        dt: float,
    ) -> bool:
        """Process one tick. Computes the tilt status for the HUD regardless
        of whether `enabled` is set. Only presses RMB + moves the cursor
        when both `enabled` is true AND the tilt is past the deadzone.

        Returns True iff rotation mode is currently ACTUATING (so the caller
        knows to suppress translation-based cursor motion)."""
        # --- compute tilt (always, for HUD) ------------------------------
        if palm_normal is None:
            self._last_tilt_vec = (0.0, 0.0)
            self._last_tilt = 0.0
            self._past_deadzone = False
        else:
            nx = float(palm_normal[0]) - self.neutral_x
            ny = float(palm_normal[1]) - self.neutral_y
            tilt = float(np.hypot(nx, ny))
            self._last_tilt_vec = (nx, ny)
            self._last_tilt = tilt
            self._past_deadzone = tilt >= self.deadzone

        # --- actuation (only when enabled and past deadzone) ------------
        if (
            not self.enabled
            or palm_normal is None
            or not self._past_deadzone
        ):
            self._release_rmb()
            self._is_actuating = False
            self._accum_x = 0.0
            self._accum_y = 0.0
            self._ref_tilt_vec = None
            self._last_delta_time = None
            return False

        now = time.perf_counter()
        nx, ny = self._last_tilt_vec

        # Entering a fresh rotation session: snapshot current tilt as the
        # reference. No delta this tick (we want zero movement on the
        # RMB press itself; deltas only come from subsequent tilt changes).
        if self._ref_tilt_vec is None:
            self._ref_tilt_vec = (nx, ny)
            self._accum_x = 0.0
            self._accum_y = 0.0
            self._last_delta_time = now
            self._press_rmb()
            self._is_actuating = True
            return True

        ref_x, ref_y = self._ref_tilt_vec
        sx = -1.0 if self.invert_x else 1.0
        sy = -1.0 if self.invert_y else 1.0
        target_px_x = (nx - ref_x) * self.sensitivity * sx
        target_px_y = (ny - ref_y) * self.sensitivity * sy

        delta_x = target_px_x - self._accum_x
        delta_y = target_px_y - self._accum_y

        step_x = int(delta_x) if abs(delta_x) >= 1.0 else 0
        step_y = int(delta_y) if abs(delta_y) >= 1.0 else 0

        if step_x != 0 or step_y != 0:
            self._accum_x += step_x
            self._accum_y += step_y
            self._last_delta_time = now
            self._press_rmb()
            self._is_actuating = True
            try:
                pydirectinput.moveRel(step_x, step_y, relative=True)
            except Exception:
                pass
            return True

        # No delta this tick. If it's been quiet for longer than the hold
        # window, release RMB and park the session; the next change will
        # open a new session with a fresh reference. This frees translation
        # to take over while the hand is held steady at a tilted pose.
        if (
            self._last_delta_time is not None
            and (now - self._last_delta_time) > self.rmb_hold_sec
        ):
            self._release_rmb()
            self._is_actuating = False
            self._ref_tilt_vec = None
            self._accum_x = 0.0
            self._accum_y = 0.0
            self._last_delta_time = None
            return False

        return self._rmb_pressed

    @property
    def is_actuating(self) -> bool:
        """True iff RMB is currently held and cursor deltas are being sent."""
        return self._is_actuating

    @property
    def past_deadzone(self) -> bool:
        """True iff tilt is past the deadzone right now, regardless of enable.
        Use this for HUD/preview so the user can test alignment before F9."""
        return self._past_deadzone

    @property
    def last_tilt(self) -> float:
        return self._last_tilt

    @property
    def last_tilt_vec(self) -> tuple[float, float]:
        return self._last_tilt_vec
