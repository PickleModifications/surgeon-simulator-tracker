"""Finger curl detection with debounce + calibration."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")

# MediaPipe landmark indices
_WRIST = 0
_LONG_FINGER_IDS: dict[str, tuple[int, int]] = {
    # finger: (tip_id, pip_id)
    "index":  (8, 6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}
_THUMB_CMC = 1
_THUMB_MCP = 2
_THUMB_IP = 3
_THUMB_TIP = 4

DEFAULT_THRESHOLDS: dict[str, float] = {
    # Thumb: "straightness" ratio (direct base->tip / summed segments).
    # Extended ~1.0, curled drops toward ~0.7. Below threshold = curled.
    "thumb":  0.88,
    "index":  1.25,
    "middle": 1.25,
    "ring":   1.25,
    "pinky":  1.20,
}


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_curl_ratios(landmarks: np.ndarray) -> dict[str, float]:
    """Compute per-finger curl ratio from 21x3 normalized landmarks.

    For the long fingers: ratio = d(tip, wrist) / d(pip, wrist).
        Extended: ~1.5+ (tip far from wrist), curled: <1.0.
    For the thumb: "straightness" = d(tip, cmc) / (sum of segment lengths).
        Extended thumb ~= 1.0 (segments are colinear, direct distance == sum).
        Curled/bent thumb drops toward ~0.7 as the thumb folds.
        This is rotation-invariant and works for both abduction and flexion.
    Only the first two coordinates are used (x, y); z is noisy under a
    downward-facing camera.
    """
    p = landmarks[:, :2]
    wrist = p[_WRIST]

    ratios: dict[str, float] = {}
    for name, (tip, pip) in _LONG_FINGER_IDS.items():
        ratios[name] = _dist(p[tip], wrist) / (_dist(p[pip], wrist) + 1e-6)

    seg_sum = (
        _dist(p[_THUMB_CMC], p[_THUMB_MCP])
        + _dist(p[_THUMB_MCP], p[_THUMB_IP])
        + _dist(p[_THUMB_IP], p[_THUMB_TIP])
    )
    direct = _dist(p[_THUMB_TIP], p[_THUMB_CMC])
    ratios["thumb"] = direct / (seg_sum + 1e-6)
    return ratios


def is_curled(finger: str, ratio: float, threshold: float) -> bool:
    """A finger is 'down' when its curl ratio drops below its threshold."""
    return ratio < threshold


@dataclass
class GestureDetector:
    thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )
    debounce_frames: int = 2

    _last_stable_state: dict[str, bool] = field(
        default_factory=lambda: {f: False for f in FINGERS}, init=False
    )
    _candidate_state: dict[str, bool] = field(
        default_factory=lambda: {f: False for f in FINGERS}, init=False
    )
    _candidate_count: dict[str, int] = field(
        default_factory=lambda: {f: 0 for f in FINGERS}, init=False
    )
    _last_ratios: dict[str, float] = field(default_factory=dict, init=False)

    def update(self, landmarks: np.ndarray) -> dict[str, bool]:
        ratios = compute_curl_ratios(landmarks)
        self._last_ratios = ratios

        for f in FINGERS:
            observed = is_curled(f, ratios[f], self.thresholds[f])
            if observed == self._last_stable_state[f]:
                self._candidate_count[f] = 0
                self._candidate_state[f] = observed
            else:
                if observed == self._candidate_state[f]:
                    self._candidate_count[f] += 1
                else:
                    self._candidate_state[f] = observed
                    self._candidate_count[f] = 1

                if self._candidate_count[f] >= self.debounce_frames:
                    self._last_stable_state[f] = observed
                    self._candidate_count[f] = 0

        return dict(self._last_stable_state)

    def reset_states(self) -> None:
        self._last_stable_state = {f: False for f in FINGERS}
        self._candidate_state = {f: False for f in FINGERS}
        self._candidate_count = {f: 0 for f in FINGERS}

    @property
    def last_ratios(self) -> dict[str, float]:
        return dict(self._last_ratios)


@dataclass
class CalibrationSample:
    """Container for calibration measurements from one pose."""
    ratios: dict[str, list[float]] = field(
        default_factory=lambda: {f: [] for f in FINGERS}
    )

    def add(self, ratios: dict[str, float]) -> None:
        for f in FINGERS:
            self.ratios[f].append(ratios[f])

    def means(self) -> dict[str, float]:
        return {
            f: (float(np.mean(v)) if v else 0.0)
            for f, v in self.ratios.items()
        }


def midpoint_thresholds(
    extended: dict[str, float], curled: dict[str, float]
) -> dict[str, float]:
    """Threshold is the midpoint between the extended and curled means."""
    out: dict[str, float] = {}
    for f in FINGERS:
        e = extended.get(f, DEFAULT_THRESHOLDS[f])
        c = curled.get(f, DEFAULT_THRESHOLDS[f])
        out[f] = (e + c) / 2.0
    return out
