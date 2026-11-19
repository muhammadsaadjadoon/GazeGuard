"""Five-point gaze calibration with explicit schema and sanity checks."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .config import atomic_json

STEPS = (
    ("center", "Look directly at the CAMERA, then press SPACE"),
    ("left", "Keep head still; look to your LEFT, then press SPACE"),
    ("right", "Keep head still; look to your RIGHT, then press SPACE"),
    ("up", "Keep head still; look UP, then press SPACE"),
    ("down", "Keep head still; look DOWN, then press SPACE"),
)


@dataclass(frozen=True)
class Calibration:
    samples: dict[str, list[float]]
    home: list[float]
    version: int = 1
    eye_centers: list[list[float]] | None = None
    center_noise: list[float] | None = None
    eye_noise: list[list[float]] | None = None
    screen_reference: list[float] | None = None

    def validate(self) -> "Calibration":
        if self.version not in (1, 2) or set(self.samples) != {name for name, _ in STEPS}:
            raise ValueError("Calibration steps/version mismatch")
        for value in self.samples.values():
            if len(value) != 4 or not np.all(np.isfinite(value)):
                raise ValueError("Calibration measurements are malformed")
        if len(self.home) != 3 or not np.all(np.isfinite(self.home)) or self.home[2] <= 0:
            raise ValueError("Calibration position is malformed")
        if self.eye_centers is not None:
            a = np.asarray(self.eye_centers, dtype=float)
            if a.shape != (2, 2) or not np.all(np.isfinite(a)):
                raise ValueError("Invalid individual-eye calibration")
        if self.center_noise is not None:
            a = np.asarray(self.center_noise, dtype=float)
            if a.shape != (4,) or not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError("Invalid center measurement spread")
        if self.eye_noise is not None:
            a = np.asarray(self.eye_noise, dtype=float)
            if a.shape != (2, 2) or not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError("Invalid eye measurement spread")
        if self.screen_reference is not None:
            a = np.asarray(self.screen_reference, dtype=float)
            if a.shape != (4,) or not np.all(np.isfinite(a)):
                raise ValueError("Invalid screen-gaze reference")
        return self

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.samples["center"], dtype=float)

    def thresholds(self, sensitivity: float) -> dict[str, float]:
        """Balanced, personalized gaze limits (not a validated gaze accuracy metric).

        Old code used 28% of a side-look excursion, which treated normal
        eye-tracker landmark jitter as Looking Away. Take a moderate share of
        each measured excursion, and include 3-sigma-ish center noise margin.
        """
        base = self.center
        center_noise = np.asarray(self.center_noise if self.center_noise is not None else [0.012, 0.015, 0, 0])
        result = {}
        for name, axis, floor in (("left", 0, 0.062), ("right", 0, 0.062),
                                  ("up", 1, 0.055), ("down", 1, 0.055)):
            distance = abs(self.samples[name][axis] - base[axis])
            # A weak or accidental side calibration should not collapse the gate.
            limit = max(floor, distance * sensitivity, center_noise[axis] * 4.0)
            result[name] = min(limit, 0.165 if axis == 0 else 0.145)
        return result


def load_calibration(path: Path) -> Calibration | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
        return Calibration(**payload).validate()
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError):
        return None


def save_calibration(path: Path, calibration: Calibration) -> None:
    calibration.validate()
    atomic_json(path, {"version": calibration.version, "samples": calibration.samples,
                       "home": calibration.home, "eye_centers": calibration.eye_centers,
                       "center_noise": calibration.center_noise, "eye_noise": calibration.eye_noise,
                       "screen_reference": calibration.screen_reference})


class CalibrationWizard:
    def __init__(self, sample_count: int = 12):
        self.index = 0
        self.sample_count = sample_count
        self.recent: list[np.ndarray] = []
        self.recent_home: list[np.ndarray] = []
        self.recent_eyes: list[np.ndarray] = []
        self.eye_centers: list[list[float]] | None = None
        self.center_noise: list[float] | None = None
        self.eye_noise: list[list[float]] | None = None
        self.samples: dict[str, list[float]] = {}
        self.home: list[float] | None = None

    @property
    def instruction(self) -> str:
        return STEPS[self.index][1]

    def observe(self, vector: np.ndarray, home: np.ndarray, eyes=None) -> None:
        if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(home)):
            self.clear_recent()
            return
        # Legacy vector-only callers are supported, but a mixed eye/no-eye
        # window is never allowed to form a personal two-eye reference.
        if eyes is None and self.recent_eyes:
            self.clear_recent()
            return
        self.recent.append(np.asarray(vector, dtype=float).copy())
        self.recent_home.append(np.asarray(home, dtype=float).copy())
        self.recent = self.recent[-self.sample_count:]
        self.recent_home = self.recent_home[-self.sample_count:]
        # Eye samples must correspond to exactly the same frames as face samples.
        # If an iris is missing, discard the entire sample window instead of
        # mixing eye and face observations from different timestamps.
        if eyes is None:
            return
        eye_values = np.asarray(eyes, dtype=float)
        if eye_values.shape != (2, 2) or not np.all(np.isfinite(eye_values)):
            self.clear_recent()
            return
        self.recent_eyes.append(eye_values.copy())
        self.recent_eyes = self.recent_eyes[-self.sample_count:]

    def clear_recent(self) -> None:
        self.recent.clear()
        self.recent_home.clear()
        self.recent_eyes.clear()

    def confirm(self) -> Calibration | None:
        if len(self.recent) < self.sample_count:
            return None
        # Reject unstable samples (excess head/eye motion); collect again.
        # Tolerances chosen to reject gross motion without requiring a tripod.
        values = np.asarray(self.recent)
        if self.sample_count >= 10:
            spread = np.percentile(values, 90, axis=0) - np.percentile(values, 10, axis=0)
            if spread[0] > .11 or spread[1] > .09 or spread[2] > .10 or spread[3] > .10:
                self.clear_recent()
                return None
        name = STEPS[self.index][0]
        self.samples[name] = np.median(self.recent, axis=0).tolist()
        if name == "center":
            self.home = np.median(self.recent_home, axis=0).tolist()
            self.center_noise = np.maximum(1.4826 * np.median(np.abs(values - np.median(values, axis=0)), axis=0), .003).tolist()
            if len(self.recent_eyes) == self.sample_count:
                eyes = np.asarray(self.recent_eyes)
                eye_spread = np.percentile(eyes, 90, axis=0) - np.percentile(eyes, 10, axis=0)
                if np.any(eye_spread > .12):
                    self.samples.pop(name, None)
                    self.home = None
                    self.clear_recent()
                    return None
                self.eye_centers = np.median(eyes, axis=0).tolist()
                self.eye_noise = np.maximum(1.4826 * np.median(np.abs(eyes - np.median(eyes, axis=0)), axis=0), .003).tolist()
        self.clear_recent()
        self.index += 1
        if self.index == len(STEPS):
            return Calibration(self.samples, self.home, version=2 if self.eye_centers is not None else 1, eye_centers=self.eye_centers,
                               center_noise=self.center_noise, eye_noise=self.eye_noise).validate()
        return None


def align_lens_reference(calibration: Calibration, vectors, homes, eyes) -> Calibration:
    """Explicit user-initiated lens alignment; rejects moving/invalid samples.

    Preserve side-gaze distances relative to the old center; only shift the
    estimated lens origin, and never learn from normal monitoring frames.
    """
    from dataclasses import replace
    v = np.asarray(vectors, dtype=float)
    h = np.asarray(homes, dtype=float)
    e = np.asarray(eyes, dtype=float)
    if v.ndim != 2 or v.shape[1] != 4 or h.shape != (len(v), 3) or e.shape != (len(v), 2, 2) or len(v) < 20:
        raise ValueError("Hold your gaze on the physical lens for at least 20 valid frames")
    if not all(np.all(np.isfinite(a)) for a in (v, h, e)):
        raise ValueError("Invalid eye landmarks; please retry lens alignment")
    span = np.percentile(v, 90, axis=0) - np.percentile(v, 10, axis=0)
    esp = np.percentile(e, 90, axis=0) - np.percentile(e, 10, axis=0)
    if np.any(span > [0.095, 0.085, 0.085, 0.09]) or np.any(esp > 0.095):
        raise ValueError("Eye or head movement was unstable; hold still and retry")
    center = np.median(v, axis=0)
    old = calibration.center
    delta = center - old
    samples = {name: (np.asarray(value, dtype=float) + delta).tolist()
               for name, value in calibration.samples.items()}
    samples["center"] = center.tolist()
    screen_reference = None
    if calibration.screen_reference is not None:
        screen_reference = (np.asarray(calibration.screen_reference, dtype=float) + delta).tolist()
    aligned = replace(calibration, samples=samples, home=np.median(h, axis=0).tolist(),
                      eye_centers=np.median(e, axis=0).tolist(), version=2,
                      center_noise=np.maximum(np.std(v, axis=0), .004).tolist(),
                      eye_noise=np.maximum(np.std(e, axis=0), .004).tolist(),
                      screen_reference=screen_reference)
    return aligned.validate()


def teach_screen_reference(calibration: Calibration, vectors) -> Calibration:
    """Capture an explicit negative example (screen gaze); refuse non-separable samples.

    A normal webcam may not distinguish the display from the camera lens. In that
    case do NOT fabricate accuracy: retain prior calibration and explain why.
    """
    from dataclasses import replace
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] != 4 or len(v) < 20 or not np.all(np.isfinite(v)):
        raise ValueError("Look at screen center for 20 valid frames and retry")
    span = np.percentile(v, 90, axis=0) - np.percentile(v, 10, axis=0)
    if np.any(span > [0.11, 0.10, 0.10, 0.10]):
        raise ValueError("Screen reference is unstable; keep your head still")
    screen = np.median(v, axis=0)
    center = calibration.center
    noise = np.asarray(calibration.center_noise or [.015, .015, .01, .01])
    # Both iris coordinates must be measurably different, otherwise a negative
    # label would invent evidence and increase false confidence.
    distance = float(np.linalg.norm((screen[:2] - center[:2]) /
                                    np.maximum(noise[:2], [.012, .012])))
    if distance < 3.5:
        raise ValueError("Lens and screen look too similar for this camera. Move the lens farther from the screen center or adjust lighting; cannot reliably distinguish them.")
    return replace(calibration, screen_reference=screen.tolist()).validate()
