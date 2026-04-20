"""MediaPipe Hands wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandResult:
    landmarks: np.ndarray                  # (21, 3) normalized image coords
    world_landmarks: Optional[np.ndarray]  # (21, 3) in meters, origin at hand center
    handedness: str                        # "Left" or "Right" as reported by MediaPipe
    raw: object                            # raw mp result for drawing utilities


class HandTracker:
    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 0,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles

    def process(self, bgr_frame: np.ndarray) -> Optional[HandResult]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        landmarks_proto = result.multi_hand_landmarks[0]
        handedness_label = "Right"
        if result.multi_handedness:
            handedness_label = result.multi_handedness[0].classification[0].label

        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks_proto.landmark],
            dtype=np.float32,
        )

        world_pts: Optional[np.ndarray] = None
        world_list = getattr(result, "multi_hand_world_landmarks", None)
        if world_list:
            world_lm = world_list[0]
            world_pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in world_lm.landmark],
                dtype=np.float32,
            )

        return HandResult(
            landmarks=pts,
            world_landmarks=world_pts,
            handedness=handedness_label,
            raw=result,
        )

    def draw(self, bgr_frame: np.ndarray, result: HandResult) -> None:
        """Draw landmarks + connections onto the frame in-place."""
        raw = result.raw
        if not raw.multi_hand_landmarks:
            return
        for hand_landmarks in raw.multi_hand_landmarks:
            self._drawing.draw_landmarks(
                bgr_frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._styles.get_default_hand_landmarks_style(),
                self._styles.get_default_hand_connections_style(),
            )

    def close(self) -> None:
        self._hands.close()
