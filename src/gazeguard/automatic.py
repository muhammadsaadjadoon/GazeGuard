"""Calibration-free, binary *estimated* camera-axis gaze classification.

ACTIVE is a broad, short-term near-camera-axis estimate. This module does NOT
claim exact physical-lens fixation or reliably separate a close screen target.
Only explicit manual lens + screen calibration can improve that distinction.
Unknown/invalid frames are conservatively marked LOOKING_AWAY, with a reason.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from .geometry import FaceMetrics
from .monitor import Observation


class AutomaticMonitor:
    def __init__(self):
        self.history = deque(maxlen=3)
        self.last_status = 'LOOKING_AWAY'

    def update(self, face: FaceMetrics | None, now: float, obstructed: bool = False) -> Observation:
        if face is None:
            self.history.clear()
            self.last_status = 'LOOKING_AWAY'
            return Observation('LOOKING_AWAY', 'UNKNOWN', 'UNKNOWN', False, False, False,
                               reason='Face not visible; cannot establish camera gaze')
        values = np.asarray(face.vector, dtype=float)
        eyes = (face.eye_left, face.eye_right)
        if not np.all(np.isfinite(values)) or any(e is None or not np.all(np.isfinite(e)) for e in eyes):
            self.history.clear()
            self.last_status = 'LOOKING_AWAY'
            return Observation('LOOKING_AWAY', 'UNKNOWN', 'UNKNOWN', False, True, True,
                               reason='Eye landmarks not reliable; lens gaze not detected')
        gaze_x, gaze_y, yaw, pitch = values
        eye_x_delta = abs(eyes[0][0] - eyes[1][0])
        eye_y_delta = abs(eyes[0][1] - eyes[1][1])
        gaze_label = ('LEFT' if gaze_x < -.075 else 'RIGHT' if gaze_x > .075 else
                      'UP' if gaze_y < -.065 else 'DOWN' if gaze_y > .075 else 'CENTER REGION')
        head_label = ('TURNED' if abs(yaw) > .085 or abs(pitch) > .14 else 'FRONTAL')
        if obstructed:
            near = False
            reason = 'Hand is covering facial landmarks'
        elif eye_x_delta > .10 or eye_y_delta > .09:
            near = False
            reason = 'Two-eye measurements disagree'
        elif abs(gaze_x) > .075 or gaze_y < -.065 or gaze_y > .075 or abs(yaw) > .085 or abs(pitch) > .14:
            near = False
            reason = 'Eyes or head oriented away from the camera axis'
        else:
            near = True
            reason = ('Eyes and head appear directed near the camera axis (estimate); '
                      'lens versus adjacent screen cannot be guaranteed on a standard webcam')
        self.history.append(near)
        # Exit ACTIVE immediately; enter after two consecutive valid observations.
        # This short stability gate avoids one-frame false positives without
        # adding hundreds of milliseconds to every LOOKING AWAY transition.
        active = near and len(self.history) >= 2 and self.history[-2]
        status = 'ATTENTIVE' if active else 'LOOKING_AWAY'
        if near and not active:
            reason = 'Checking stable camera-axis gaze'
        self.last_status = status
        return Observation(status, gaze_label, head_label, False, True, True,
                           float(gaze_x), float(gaze_y), float(yaw), float(pitch), reason)
