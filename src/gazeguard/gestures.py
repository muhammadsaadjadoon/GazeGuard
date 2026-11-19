"""Heuristic gesture recognition; no identity or gesture model training required."""
from __future__ import annotations
import numpy as np


def hand_points(hand, width: int, height: int) -> np.ndarray:
    return np.array([[p.x * width, p.y * height] for p in (hand.landmark if hasattr(hand, "landmark") else hand)], dtype=float)


def folded(points: np.ndarray, tip: int, pip: int) -> bool:
    return np.linalg.norm(points[tip] - points[0]) < np.linalg.norm(points[pip] - points[0])


def thumbs_up(points: np.ndarray) -> bool:
    size = np.linalg.norm(points[9] - points[0])
    if size < 1e-6:
        return False
    closed = all(folded(points, tip, pip) for tip, pip in ((8, 6), (12, 10),
                                                         (16, 14), (20, 18)))
    return bool(closed and points[4, 1] < points[3, 1] < points[2, 1]
                and (points[2, 1] - points[4, 1]) > 0.5 * size)


def finger_heart(points: np.ndarray) -> bool:
    size = np.linalg.norm(points[9] - points[0])
    if size < 1e-6:
        return False
    pinched = np.linalg.norm(points[4] - points[8]) < 0.30 * size
    closed = all(folded(points, tip, pip) for tip, pip in ((12, 10), (16, 14), (20, 18)))
    return bool(pinched and closed)


def two_hand_heart(a: np.ndarray, b: np.ndarray) -> bool:
    size = (np.linalg.norm(a[9] - a[0]) + np.linalg.norm(b[9] - b[0])) / 2
    if size < 1e-6:
        return False
    return bool(np.linalg.norm(a[8] - b[8]) < 0.45 * size
                and np.linalg.norm(a[4] - b[4]) < 0.45 * size
                and (a[8, 1] + b[8, 1]) < (a[4, 1] + b[4, 1]))


def detect_gesture(points: list[np.ndarray]) -> str | None:
    if len(points) >= 2 and two_hand_heart(points[0], points[1]):
        return "HEART"
    for hand in points:
        if thumbs_up(hand):
            return "THUMBS_UP"
        if finger_heart(hand):
            return "HEART"
    return None


class GestureDebouncer:
    """Fire once per stable presentation; avoid repeat-trigger from a held pose."""
    def __init__(self, frames: int, cooldown_seconds: float):
        self.frames = frames
        self.cooldown_seconds = cooldown_seconds
        self.current: str | None = None
        self.count = 0
        self.last_event_at = -float("inf")
        self.fired_for_hold = False

    def update(self, gesture: str | None, now: float) -> str | None:
        if gesture != self.current:
            self.current = gesture
            self.count = 0
            self.fired_for_hold = False
        if gesture is None:
            return None
        self.count += 1
        if (self.count >= self.frames and not self.fired_for_hold
                and now - self.last_event_at >= self.cooldown_seconds):
            self.fired_for_hold = True
            self.last_event_at = now
            return gesture
        return None


# A hand appearing in the frame is NOT automatically a facial obstruction.
def hand_overlaps_face(hand: np.ndarray, face_box: tuple[int, int, int, int],
                       min_fraction: float = 0.25) -> bool:
    """Return whether enough landmarks lie in the face rectangle to impair gaze."""
    arr = np.asarray(hand, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 5 or not np.all(np.isfinite(arr)):
        return False
    x1, y1, x2, y2 = face_box
    inside = ((arr[:, 0] >= x1) & (arr[:, 0] <= x2) &
              (arr[:, 1] >= y1) & (arr[:, 1] <= y2))
    return bool(float(np.mean(inside)) >= min_fraction)
