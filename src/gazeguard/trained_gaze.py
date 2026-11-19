"""Small, portable binary classifier trained from consented webcam landmarks.

It estimates camera-lens gaze versus non-lens gaze and is NOT a calibrated
angular eye tracker. The model file is plain JSON, not executable pickle.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

FEATURE_NAMES = (
    'gaze_x', 'gaze_y', 'head_yaw', 'head_pitch',
    'left_iris_x', 'left_iris_y', 'right_iris_x', 'right_iris_y',
)


def extract_features(metrics) -> np.ndarray:
    if metrics is None or metrics.eye_left is None or metrics.eye_right is None:
        raise ValueError('Face and both irises must be visible')
    v = np.asarray([*metrics.vector, *metrics.eye_left, *metrics.eye_right], dtype=float)
    if v.shape != (8,) or not np.all(np.isfinite(v)):
        raise ValueError('Invalid gaze feature vector')
    return v


def sigmoid(v):
    x = np.asarray(v, dtype=float)
    return 1. / (1. + np.exp(-np.clip(x, -40, 40)))


@dataclass(frozen=True)
class TrainedGazeModel:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float
    validation: dict

    def probability(self, features: np.ndarray) -> float:
        x = np.asarray(features, dtype=float)
        if x.shape != (8,) or not np.all(np.isfinite(x)):
            raise ValueError('Expected 8 finite landmark features')
        norm = np.clip((x - self.mean) / self.std, -12, 12)
        basis = np.concatenate([norm, norm ** 2])
        return float(sigmoid(np.dot(basis, self.weights) + self.bias))

    def classify(self, features: np.ndarray) -> bool:
        return self.probability(features) >= self.threshold


def load_model(path: Path) -> TrainedGazeModel:
    with Path(path).open('r', encoding='utf-8') as fh:
        value = json.load(fh)
    if value.get('format') != 'gazeguard-landmark-binary-v1' or tuple(value.get('feature_names', [])) != FEATURE_NAMES:
        raise ValueError('Not a compatible GazeGuard model')
    mean, std, weights = (np.asarray(value[key], dtype=float) for key in ('mean','std','weights'))
    if (mean.shape != (8,) or std.shape != (8,) or weights.shape != (16,)
            or not all(np.all(np.isfinite(v)) for v in (mean, std, weights)) or np.any(std <= 0)):
        raise ValueError('Invalid model arrays')
    bias, threshold = float(value['bias']), float(value['threshold'])
    if not (np.isfinite(bias) and np.isfinite(threshold) and 0 < threshold <= 1):
        raise ValueError('Invalid model parameters')
    return TrainedGazeModel(mean,std,weights,bias,threshold,value.get('validation', {}))


def save_model(path: Path, model: TrainedGazeModel) -> None:
    payload = dict(format='gazeguard-landmark-binary-v1', feature_names=list(FEATURE_NAMES),
                   mean=model.mean.tolist(), std=model.std.tolist(),
                   weights=model.weights.tolist(), bias=model.bias,
                   threshold=model.threshold, validation=model.validation)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    temp.replace(path)
