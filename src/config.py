"""Persistent user config stored as JSON next to main.py."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src.gesture import DEFAULT_THRESHOLDS

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass
class AppConfig:
    camera_index: int = 0
    hand_mode: str = "right"  # "right" or "left"
    thresholds_right: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )
    thresholds_left: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )
    debounce_frames: int = 2
    toggle_hotkey: str = "f9"
    mouse_center: Optional[tuple[float, float]] = None
    mouse_sensitivity: float = 900.0
    mouse_deadzone: float = 0.04
    mouse_invert_x: bool = False
    mouse_invert_y: bool = False
    depth_low_palm_size: Optional[float] = None
    depth_high_palm_size: Optional[float] = None
    depth_maintain_width: float = 0.33

    def thresholds_for(self, hand_mode: str) -> dict[str, float]:
        return (
            self.thresholds_right
            if hand_mode == "right"
            else self.thresholds_left
        )

    def set_thresholds_for(self, hand_mode: str, values: dict[str, float]) -> None:
        if hand_mode == "right":
            self.thresholds_right = dict(values)
        else:
            self.thresholds_left = dict(values)


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return AppConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cfg = AppConfig()
        cfg.camera_index = int(raw.get("camera_index", cfg.camera_index))
        cfg.hand_mode = str(raw.get("hand_mode", cfg.hand_mode))
        cfg.thresholds_right = {
            **cfg.thresholds_right,
            **(raw.get("thresholds_right") or {}),
        }
        cfg.thresholds_left = {
            **cfg.thresholds_left,
            **(raw.get("thresholds_left") or {}),
        }
        cfg.debounce_frames = int(raw.get("debounce_frames", cfg.debounce_frames))
        cfg.toggle_hotkey = str(raw.get("toggle_hotkey", cfg.toggle_hotkey))
        center = raw.get("mouse_center")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            cfg.mouse_center = (float(center[0]), float(center[1]))
        cfg.mouse_sensitivity = float(
            raw.get("mouse_sensitivity", cfg.mouse_sensitivity)
        )
        cfg.mouse_deadzone = float(
            raw.get("mouse_deadzone", cfg.mouse_deadzone)
        )
        cfg.mouse_invert_x = bool(raw.get("mouse_invert_x", cfg.mouse_invert_x))
        cfg.mouse_invert_y = bool(raw.get("mouse_invert_y", cfg.mouse_invert_y))
        lo = raw.get("depth_low_palm_size")
        hi = raw.get("depth_high_palm_size")
        cfg.depth_low_palm_size = float(lo) if lo is not None else None
        cfg.depth_high_palm_size = float(hi) if hi is not None else None
        cfg.depth_maintain_width = float(
            raw.get("depth_maintain_width", cfg.depth_maintain_width)
        )
        return cfg
    except (json.JSONDecodeError, ValueError, TypeError):
        return AppConfig()


def save_config(cfg: AppConfig, path: Path = CONFIG_PATH) -> None:
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
