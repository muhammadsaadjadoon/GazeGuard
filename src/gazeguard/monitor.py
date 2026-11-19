"""Timestamp-driven monitoring and classification state machine."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
from .calibration import Calibration
from .config import Settings
from .geometry import FaceMetrics


@dataclass(frozen=True)
class Observation:
    status: str
    eyes: str
    head: str
    speaking_likely: bool
    face_present: bool
    seat_ok: bool
    gaze_dx: float = 0.0
    gaze_dy: float = 0.0
    yaw_delta: float = 0.0
    pitch_delta: float = 0.0
    reason: str = ""

    @property
    def attentive(self) -> bool:
        return self.status == "ATTENTIVE"


def direction(dx: float, dy: float, thresholds: dict[str, float]) -> str:
    parts = []
    if dy < -thresholds["up"]:
        parts.append("UP")
    elif dy > thresholds["down"]:
        parts.append("DOWN")
    if dx < -thresholds["left"]:
        parts.append("LEFT")
    elif dx > thresholds["right"]:
        parts.append("RIGHT")
    return "-".join(parts) or "CENTER"


class Monitor:
    def __init__(self, settings: Settings, calibration: Calibration):
        self.settings = settings.validate()
        self.calibration = calibration.validate()
        self.thresholds = calibration.thresholds(settings.sensitivity)
        self.smooth: np.ndarray | None = None
        self.mouth_samples: deque[float] = deque(maxlen=settings.speech_window)
        self.last_face_at: float | None = None
        self.last_status: str | None = None
        self.pending_status: str | None = None
        self.pending_since: float = 0.0
        self.good_frames = 0
        self.raw_window: deque[np.ndarray] = deque(maxlen=5)
        self.ambiguous_frames = 0

    def _screen_like(self, raw: np.ndarray) -> bool:
        if self.calibration.screen_reference is None:
            return False
        ref = np.asarray(self.calibration.screen_reference, dtype=float)
        center = self.calibration.center
        noise = np.maximum(np.asarray(self.calibration.center_noise or [.015, .015, .01, .01]), [.012, .012, .010, .010])
        weights = np.asarray([1.3, 1.3, 0.7, 0.8], dtype=float)
        lens_vec = (raw - center) / noise
        screen_vec = (raw - ref) / noise
        separation_vec = (ref - center) / noise
        lens_distance = float(np.linalg.norm(lens_vec * weights))
        screen_distance = float(np.linalg.norm(screen_vec * weights))
        separation = float(np.linalg.norm(separation_vec * weights))
        if separation < 2.6:
            return False
        # Project onto the learned lens->screen axis. This reduces false CAMERA
        # classifications when the user is reading the screen near the webcam.
        axis = separation_vec[:2]
        axis_norm = max(float(np.linalg.norm(axis)), 1e-6)
        progress = float(np.dot(lens_vec[:2], axis / axis_norm))
        return (
            screen_distance + 0.45 < lens_distance
            and progress > separation * 0.55
            and lens_distance > separation * 0.45
        )

    def _between_lens_and_screen(self, raw: np.ndarray) -> bool:
        """Reject sustained ambiguous gaze between lens and taught screen target.

        Does not declare exact screen gaze; a webcam may not resolve the two.
        """
        if self.calibration.screen_reference is None:
            return False
        delta = np.asarray(self.calibration.screen_reference)[:2] - self.calibration.center[:2]
        norm2 = float(np.dot(delta, delta))
        if norm2 < 1e-7:
            return False
        progress = float(np.dot(raw[:2] - self.calibration.center[:2], delta) / norm2)
        return 0.38 < progress < 0.62

    def update(self, face: FaceMetrics | None, now: float,
               unrecognized_hands: bool = False) -> Observation:
        if face is None:
            self.good_frames = 0
            self.ambiguous_frames = 0
            self.smooth = None
            self.mouth_samples.clear()
            self.raw_window.clear()
            absent = self.last_face_at is None or now - self.last_face_at >= self.settings.absence_seconds
            status = "ABSENT" if absent else "SEARCHING"
            self.last_status = status
            self.pending_status = None
            return Observation(status, "UNKNOWN", "UNKNOWN", False, False, False)

        self.last_face_at = now
        if not np.all(np.isfinite(face.vector)):
            self.smooth = None
            self.raw_window.clear()
            self.good_frames = 0
            self.last_status = "SEARCHING"
            self.pending_status = None
            return Observation("SEARCHING", "UNKNOWN", "UNKNOWN", False, True, False,
                               reason="Unreliable eye landmarks; hold still")

        raw = face.vector
        self.raw_window.append(raw.copy())
        robust = np.median(np.asarray(self.raw_window), axis=0)
        self.smooth = raw if self.smooth is None else self.settings.smoothing * self.smooth + (1 - self.settings.smoothing) * robust
        dx, dy = (self.smooth - self.calibration.center)[:2]
        yaw, pitch = (self.smooth - self.calibration.center)[2:]
        eyes = direction(float(dx), float(dy), self.thresholds)
        head = direction(
            float(yaw),
            float(pitch),
            {"left": self.settings.yaw_limit, "right": self.settings.yaw_limit,
             "up": self.settings.pitch_limit, "down": self.settings.pitch_limit},
        )

        x, y, width = self.calibration.home
        shift = float(np.hypot(face.center_x - x, face.center_y - y))
        size_change = abs(face.width - width) / max(width, 1e-6)
        seat_ok = shift <= self.settings.position_shift and size_change <= self.settings.size_change

        self.mouth_samples.append(face.mouth_ratio)
        speaking = (
            len(self.mouth_samples) == self.mouth_samples.maxlen
            and float(np.std(self.mouth_samples)) > self.settings.speech_variation
        )

        individual_eyes_ok = True
        eye_centers = self.calibration.eye_centers
        if eye_centers is not None:
            if face.eye_left is None or face.eye_right is None:
                individual_eyes_ok = False
            else:
                for idx, (eye, reference) in enumerate(zip((face.eye_left, face.eye_right), eye_centers)):
                    delta = np.asarray(eye, dtype=float) - np.asarray(reference, dtype=float)
                    x_limit = self.thresholds['left'] if delta[0] < 0 else self.thresholds['right']
                    y_limit = self.thresholds['up'] if delta[1] < 0 else self.thresholds['down']
                    eye_noise = self.calibration.eye_noise
                    if eye_noise is not None:
                        x_limit = max(x_limit, 4.6 * eye_noise[idx][0])
                        y_limit = max(y_limit, 4.6 * eye_noise[idx][1])
                    x_limit = max(x_limit, 0.082)
                    y_limit = max(y_limit, 0.070)
                    if abs(delta[0]) > x_limit or abs(delta[1]) > y_limit:
                        individual_eyes_ok = False

        raw_dx, raw_dy, raw_yaw, raw_pitch = raw - self.calibration.center
        raw_off_target = (
            abs(raw_dx) > max(self.thresholds['left'], self.thresholds['right']) * 1.10
            or abs(raw_dy) > max(self.thresholds['up'], self.thresholds['down']) * 1.10
            or abs(raw_yaw) > self.settings.yaw_limit * 1.55
            or abs(raw_pitch) > self.settings.pitch_limit * 1.55
        )
        screen_like = self._screen_like(raw)
        if self._between_lens_and_screen(raw):
            self.ambiguous_frames += 1
        else:
            self.ambiguous_frames = 0
        ambiguous = self.ambiguous_frames >= 3
        setup_required = self.calibration.screen_reference is None

        if not seat_ok:
            candidate, reason = "OUT_OF_POSITION", "Move into the camera frame"
        elif screen_like:
            candidate, reason = "LOOKING_AWAY", "Screen gaze detected; direct camera-lens gaze is required for ACTIVE"
        elif unrecognized_hands:
            candidate, reason = "HANDS_OBSTRUCTING", "Keep hands clear of the face"
        elif ambiguous:
            candidate, reason = "UNVERIFIED_GAZE", "Gaze is between screen and camera lens; cannot verify direct lens contact"
        elif eyes == "CENTER" and head == "CENTER" and individual_eyes_ok and not raw_off_target:
            if setup_required:
                candidate, reason = "SETUP_REQUIRED", "Run Teach screen so the app can separate screen gaze from camera-lens gaze"
            else:
                candidate, reason = "ATTENTIVE", "Direct camera-lens gaze verified"
        elif not individual_eyes_ok:
            candidate, reason = "LOOKING_AWAY", "Eye landmarks differ from lens calibration"
        elif eyes != "CENTER":
            candidate, reason = "LOOKING_AWAY", f"Eyes pointing {eyes.lower()}"
        elif head != "CENTER":
            candidate, reason = "LOOKING_AWAY", f"Head orientation {head.lower()}"
        else:
            candidate, reason = "LOOKING_AWAY", "Sudden gaze change detected"

        if candidate == "ATTENTIVE":
            self.good_frames += 1
        else:
            self.good_frames = 0
        if candidate != self.pending_status:
            self.pending_status = candidate
            self.pending_since = now

        if self.last_status is None or self.last_status in ("SEARCHING", "ABSENT"):
            self.last_status = candidate
        elif candidate != "ATTENTIVE" and (raw_off_target or not individual_eyes_ok or not seat_ok or screen_like or unrecognized_hands or setup_required or ambiguous):
            self.last_status = candidate
        elif now - self.pending_since >= self.settings.state_hold_seconds and (candidate != "ATTENTIVE" or self.good_frames >= 4):
            self.last_status = candidate

        return Observation(
            self.last_status,
            eyes,
            head,
            bool(speaking),
            True,
            seat_ok,
            float(dx),
            float(dy),
            float(yaw),
            float(pitch),
            reason if self.last_status == candidate else "Stabilizing gaze classification",
        )
