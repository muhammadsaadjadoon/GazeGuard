"""Monotonic, rate-limited attention alerts; intentionally no audio capture."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Alert:
    code: str
    message: str
    when: float


class AlertEngine:
    def __init__(self, after_seconds: float = 4, cooldown_seconds: float = 15):
        if after_seconds <= 0 or cooldown_seconds < 0:
            raise ValueError("invalid alert timing")
        self.after_seconds = after_seconds
        self.cooldown_seconds = cooldown_seconds
        self.active_status: str | None = None
        self.active_since: float | None = None
        self.last_alert: dict[str, float] = {}

    def update(self, status: str, now: float) -> Alert | None:
        if status != self.active_status:
            self.active_status = status
            self.active_since = now
        if status in ("ATTENTIVE", "SEARCHING"):
            return None
        assert self.active_since is not None
        if now - self.active_since < self.after_seconds:
            return None
        last = self.last_alert.get(status)
        if last is not None and now - last < self.cooldown_seconds:
            return None
        self.last_alert[status] = now
        msg = {"LOOKING_AWAY": "Look toward the camera",
               "ABSENT": "Return to the camera view",
               "HANDS_OBSTRUCTING": "Please clear the camera view",
               "OUT_OF_POSITION": "Return to calibrated position"}.get(status, status)
        return Alert(status, msg, now)
