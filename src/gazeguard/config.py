"""Validated, user-editable configuration with safe JSON persistence."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path

DEFAULT_HOME = Path.home() / ".gazeguard_v1"


@dataclass(frozen=True)
class Settings:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    mirror: bool = True
    sensitivity: float = 0.55
    yaw_limit: float = 0.085
    pitch_limit: float = 0.095
    smoothing: float = 0.28
    absence_seconds: float = 1.0
    position_shift: float = 0.24
    size_change: float = 0.40
    speech_window: int = 15
    speech_variation: float = 0.012
    gesture_frames: int = 5
    gesture_cooldown_seconds: float = 2.0
    animation_seconds: float = 1.5
    state_hold_seconds: float = 0.14
    save_session: bool = True
    show_landmarks: bool = True
    alerts_enabled: bool = True
    alert_after_seconds: float = 4.0
    alert_cooldown_seconds: float = 15.0
    focus_target_percent: float = 80.0

    def validate(self) -> "Settings":
        if self.camera_index < 0 or self.camera_width < 320 or self.camera_height < 240:
            raise ValueError("Invalid camera index or dimensions")
        for name in ("sensitivity", "yaw_limit", "pitch_limit", "absence_seconds",
                     "position_shift", "size_change", "speech_variation",
                     "gesture_cooldown_seconds", "animation_seconds", "state_hold_seconds",
                     "alert_after_seconds", "alert_cooldown_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.smoothing < 1 or self.sensitivity > 1:
            raise ValueError("smoothing must be [0,1) and sensitivity <= 1")
        if self.speech_window < 3 or self.gesture_frames < 1:
            raise ValueError("Invalid detection window")
        if not 0 <= self.focus_target_percent <= 100:
            raise ValueError("focus_target_percent must be in [0,100]")
        return self


def load_settings(path: Path) -> Settings:
    if not path.exists():
        return Settings().validate()
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    allowed = {f.name for f in fields(Settings)}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("Settings JSON has unknown keys or incorrect format")
    return Settings(**raw).validate()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_default_settings(path: Path) -> None:
    if not path.exists():
        atomic_json(path, asdict(Settings()))
