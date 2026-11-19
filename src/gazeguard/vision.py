"""MediaPipe Tasks VIDEO pipeline (Python 3.14); models fetched on first use.

Model assets are downloaded from Google's public MediaPipe model storage.
They are kept locally under the user data directory, not bundled as code.
"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import logging
import os
import time
from urllib.request import urlopen, Request

LOG = logging.getLogger("gazeguard")
MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def ensure_models(data_dir: Path) -> dict[str, Path]:
    """Download missing model assets once. Do not overwrite a valid existing file."""
    models = data_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MODEL_URLS.items():
        dest = models / name
        if dest.is_file() and dest.stat().st_size > 100_000:
            paths[name] = dest
            continue
        temp = models / (name + ".download")
        LOG.info("Downloading first-run model %s from Google (internet required)...", name)
        try:
            request = Request(url, headers={"User-Agent": "GazeGuard/5.0 Python314"})
            with urlopen(request, timeout=90) as response, temp.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if temp.stat().st_size <= 100_000:
                raise RuntimeError(f"Downloaded model {name} is incomplete")
            os.replace(temp, dest)
        except Exception as exc:
            temp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Cannot download {name}: {exc}. Check internet/firewall or manually "
                f"place the official .task model in: {models}"
            ) from exc
        paths[name] = dest
    return paths


class TaskVision:
    """Face/hand landmarks returned as lists compatible with legacy geometry."""
    def __init__(self, models: dict[str, Path]):
        self.models = models
        self.stack = ExitStack()
        self.face = None
        self.hands = None
        self.last_timestamp_ms = -1

    def __enter__(self):
        import mediapipe as mp
        try:
            vision = mp.tasks.vision
            base = mp.tasks.BaseOptions
            self.face = self.stack.enter_context(vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=base(model_asset_path=str(self.models["face_landmarker.task"])),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.55,
                    min_face_presence_confidence=0.55,
                    min_tracking_confidence=0.55,
                )
            ))
            self.hands = self.stack.enter_context(vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=base(model_asset_path=str(self.models["hand_landmarker.task"])),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.45,
                    min_hand_presence_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
            ))
            return self
        except BaseException:
            self.stack.close()
            raise

    def process(self, bgr_frame):
        import cv2
        import mediapipe as mp
        import numpy as np
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(time.monotonic() * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        face_result = self.face.detect_for_video(img, timestamp_ms)
        hand_result = self.hands.detect_for_video(img, timestamp_ms)
        return face_result.face_landmarks or [], hand_result.hand_landmarks or []

    def __exit__(self, exc_type, exc, tb):
        return self.stack.__exit__(exc_type, exc, tb)
