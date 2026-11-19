"""Normalized landmark measurements independent of the webcam/UI layer."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

L_IRIS, R_IRIS = 468, 473
EYE_A = (33, 133, 159, 145)
EYE_B = (263, 362, 386, 374)
NOSE, FACE_L, FACE_R, FOREHEAD, CHIN = 1, 234, 454, 10, 152
LIP_TOP, LIP_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 78, 308


@dataclass(frozen=True)
class FaceMetrics:
    gaze_x: float
    gaze_y: float
    yaw: float
    pitch: float
    center_x: float
    center_y: float
    width: float
    mouth_ratio: float
    eye_left: tuple[float, float] | None = None
    eye_right: tuple[float, float] | None = None

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.gaze_x, self.gaze_y, self.yaw, self.pitch], dtype=float)

    @property
    def home(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y, self.width], dtype=float)


def point(landmarks, i: int, width: int, height: int) -> np.ndarray:
    p = landmarks[i]
    return np.array([p.x * width, p.y * height], dtype=float)


def eye_gaze(landmarks, width, height, iris_idx, corners) -> tuple[float, float]:
    outer, inner, top, bottom = (point(landmarks, i, width, height) for i in corners)
    iris = point(landmarks, iris_idx, width, height)
    left, right = sorted((outer, inner), key=lambda p: p[0])
    axis = right - left
    length = max(float(np.linalg.norm(axis)), 1e-6)
    u = axis / length
    gh = float(np.dot(iris - left, u) / length - 0.5)
    perp = np.array([-u[1], u[0]])
    # Consistent screen-space Y: top of image is negative, bottom positive.
    gv = float((iris[1] - (top[1] + bottom[1]) / 2) / length)
    return gh, gv


def measure_face(landmarks, width: int, height: int) -> FaceMetrics:
    if len(landmarks) <= R_IRIS:
        raise ValueError("Refined iris landmarks are required (468+ points)")
    # Discard blinks and severely occluded/degenerate eyes rather than making
    # an incorrect gaze decision. Median calibration samples only use valid eyes.
    for corners in (EYE_A, EYE_B):
        a, b, top, bottom = (point(landmarks, j, width, height) for j in corners)
        eye_width = float(np.linalg.norm(a - b))
        eye_opening = float(np.linalg.norm(top - bottom))
        if eye_width < 3 or eye_opening / eye_width < 0.085:
            raise ValueError("Blink or unreliable eye landmarks")
    ah, av = eye_gaze(landmarks, width, height, L_IRIS, EYE_A)
    bh, bv = eye_gaze(landmarks, width, height, R_IRIS, EYE_B)
    nose = point(landmarks, NOSE, width, height)
    fl, fr = point(landmarks, FACE_L, width, height), point(landmarks, FACE_R, width, height)
    forehead = point(landmarks, FOREHEAD, width, height)
    chin = point(landmarks, CHIN, width, height)
    lo, hi = sorted((fl[0], fr[0]))
    yaw = float((nose[0] - lo) / max(hi - lo, 1e-6) - 0.5)
    pitch = float((nose[1] - forehead[1]) / max(chin[1] - forehead[1], 1e-6) - 0.5)
    xs = np.array([p.x for p in landmarks])
    ys = np.array([p.y for p in landmarks])
    mouth_gap = float(np.linalg.norm(point(landmarks, LIP_TOP, width, height)
                                     - point(landmarks, LIP_BOTTOM, width, height)))
    mouth_width = float(np.linalg.norm(point(landmarks, MOUTH_LEFT, width, height)
                                       - point(landmarks, MOUTH_RIGHT, width, height)))
    return FaceMetrics((ah + bh) / 2, (av + bv) / 2, yaw, pitch,
                       float((xs.min() + xs.max()) / 2),
                       float((ys.min() + ys.max()) / 2), float(xs.max() - xs.min()),
                       mouth_gap / max(mouth_width, 1e-6), (ah, av), (bh, bv))


def face_bounds(landmarks, width: int, height: int) -> tuple[int, int, int, int]:
    xs = np.array([p.x * width for p in landmarks])
    ys = np.array([p.y * height for p in landmarks])
    fw, fh = xs.max() - xs.min(), ys.max() - ys.min()
    return (max(0, int(xs.min() - 0.15 * fw)),
            max(0, int(ys.min() - 0.30 * fh)),
            min(width - 1, int(xs.max() + 0.15 * fw)),
            min(height - 1, int(ys.max() + 0.10 * fh)))
