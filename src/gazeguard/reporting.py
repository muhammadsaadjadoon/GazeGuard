"""Privacy-safe local session reporting: states only, no video or face embeddings."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
from .config import atomic_json


class SessionRecorder:
    def __init__(self, started: float):
        self.started = started
        self.previous_time = started
        self.previous_state: str | None = None
        self.durations: dict[str, float] = defaultdict(float)
        self.events: list[dict] = []
        self.gestures: dict[str, int] = defaultdict(int)
        self.speech_alerts = 0
        self._previous_speech_alert = False

    def record(self, now: float, state: str, gesture: str | None = None,
               speech_alert: bool = False) -> None:
        if self.previous_state is not None:
            self.durations[self.previous_state] += max(0.0, now - self.previous_time)
        if state != self.previous_state:
            self.events.append({"elapsed_seconds": round(max(0, now - self.started), 2),
                                "event": "status", "value": state})
        if gesture is not None:
            self.gestures[gesture] += 1
            self.events.append({"elapsed_seconds": round(now - self.started, 2),
                                "event": "gesture", "value": gesture})
        if speech_alert and not self._previous_speech_alert:
            self.speech_alerts += 1
            self.events.append({"elapsed_seconds": round(now - self.started, 2),
                                "event": "speech_without_eye_contact", "value": "likely"})
        self._previous_speech_alert = speech_alert
        self.previous_time = now
        self.previous_state = state

    def summary(self, end: float) -> dict:
        durations = dict(self.durations)
        if self.previous_state is not None:
            durations[self.previous_state] = durations.get(self.previous_state, 0) + max(0, end - self.previous_time)
        observed = sum(durations.get(key, 0) for key in ("ATTENTIVE", "LOOKING_AWAY",
                        "OUT_OF_POSITION", "HANDS_OBSTRUCTING", "UNVERIFIED_GAZE"))
        attentive = durations.get("ATTENTIVE", 0)
        return {"duration_seconds": round(max(0, end - self.started), 2),
                "observable_seconds": round(observed, 2),
                "attentive_percent_observable": round(100 * attentive / observed, 1) if observed else None,
                "states_seconds": {key: round(val, 2) for key, val in sorted(durations.items())},
                "gestures": dict(self.gestures), "speech_alerts": self.speech_alerts,
                "privacy": "No video, audio, images, or face landmark data saved"}

    def export(self, folder: Path, end: float) -> tuple[Path, Path]:
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
        summary_path = folder / f"session_{stamp}.json"
        events_path = folder / f"session_{stamp}.csv"
        atomic_json(summary_path, self.summary(end))
        with events_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["elapsed_seconds", "event", "value"])
            writer.writeheader()
            writer.writerows(self.events)
        return summary_path, events_path
