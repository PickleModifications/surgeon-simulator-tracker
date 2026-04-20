"""Press/release game keys via pydirectinput (SendInput scan codes)."""
from __future__ import annotations

import pydirectinput

KEY_MAP: dict[str, str] = {
    "thumb":  "space",
    "index":  "r",
    "middle": "e",
    "ring":   "w",
    "pinky":  "a",
}


class KeySender:
    def __init__(self) -> None:
        self.enabled: bool = False
        self._pressed: dict[str, bool] = {f: False for f in KEY_MAP}
        pydirectinput.PAUSE = 0  # no artificial delay between injected events

    def apply(self, states: dict[str, bool]) -> None:
        """Press/release keys to match the provided finger states.

        When disabled, tracks nothing and ensures all keys are released.
        """
        if not self.enabled:
            return
        for finger, want_down in states.items():
            key = KEY_MAP.get(finger)
            if not key:
                continue
            was_down = self._pressed[finger]
            if want_down and not was_down:
                pydirectinput.keyDown(key)
                self._pressed[finger] = True
            elif not want_down and was_down:
                pydirectinput.keyUp(key)
                self._pressed[finger] = False

    def release_all(self) -> None:
        """Release any keys we believe are held. Safe to call repeatedly."""
        for finger, pressed in list(self._pressed.items()):
            if pressed:
                try:
                    pydirectinput.keyUp(KEY_MAP[finger])
                except Exception:
                    pass
                self._pressed[finger] = False

    def set_enabled(self, enabled: bool) -> None:
        if self.enabled and not enabled:
            self.release_all()
        self.enabled = enabled

    @property
    def pressed_snapshot(self) -> dict[str, bool]:
        return dict(self._pressed)
