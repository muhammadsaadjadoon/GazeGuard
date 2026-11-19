from pathlib import Path
import numpy as np
import pytest
from gazeguard.calibration import Calibration, CalibrationWizard, load_calibration, save_calibration
from gazeguard.config import Settings, load_settings, atomic_json
from gazeguard.geometry import FaceMetrics
from gazeguard.gestures import GestureDebouncer, detect_gesture
from gazeguard.monitor import Monitor, direction
from gazeguard.reporting import SessionRecorder


def calibration():
    return Calibration({"center": [0, 0, 0, 0], "left": [-0.2, 0, 0, 0],
                        "right": [0.2, 0, 0, 0], "up": [0, -0.2, 0, 0],
                        "down": [0, 0.2, 0, 0]}, [0.5, 0.5, 0.3], screen_reference=[0.08, 0, 0, 0])


def face(x=0, y=0, yaw=0, pitch=0, cx=0.5, cy=0.5, width=0.3, mouth=0):
    return FaceMetrics(x, y, yaw, pitch, cx, cy, width, mouth)


def test_calibration_roundtrip(tmp_path):
    path = tmp_path / "cal.json"
    save_calibration(path, calibration())
    assert load_calibration(path) == calibration()
    path.write_text("{invalid")
    assert load_calibration(path) is None


def test_calibration_thresholds_and_directions():
    t = calibration().thresholds(0.35)
    assert t["left"] == pytest.approx(0.07)
    assert direction(-0.08, 0, t) == "LEFT"
    assert direction(0, -0.08, t) == "UP"
    assert direction(0, 0, t) == "CENTER"


def test_calibration_wizard():
    wiz = CalibrationWizard(sample_count=3)
    for i in range(5):
        for _ in range(3):
            wiz.observe(np.array([i * 0.1, 0, 0, 0]), np.array([0.5, 0.5, 0.3]))
        result = wiz.confirm()
    assert result is not None and result.samples["down"][0] == pytest.approx(0.4)


def test_monitor_states_and_reset():
    settings = Settings(smoothing=0, state_hold_seconds=0.01, absence_seconds=0.1)
    mon = Monitor(settings, calibration())
    assert mon.update(face(), 0).status == "ATTENTIVE"
    assert mon.update(face(x=0.2), 0.02).status == "LOOKING_AWAY"
    assert mon.update(face(x=0.2), 0.05).status == "LOOKING_AWAY"
    assert mon.update(None, 0.08).status == "SEARCHING"
    assert mon.update(None, 0.21).status == "ABSENT"
    assert mon.update(face(cx=0.9), 0.3).status == "OUT_OF_POSITION"


def test_monitor_speech_heuristic():
    mon = Monitor(Settings(smoothing=0, speech_window=3), calibration())
    for i, gap in enumerate([0.0, 0.05, 0.0]):
        obs = mon.update(face(mouth=gap), i * 0.04)
    assert obs.speaking_likely


def test_debounce_one_event_per_hold():
    d = GestureDebouncer(3, 1.0)
    assert [d.update("HEART", t) for t in (0, 0.1, 0.2, 0.3)] == [None, None, "HEART", None]
    d.update(None, 1.5)
    assert d.update("HEART", 1.6) is None
    assert d.update("HEART", 1.7) is None
    assert d.update("HEART", 1.8) == "HEART"
    assert detect_gesture([]) is None


def test_reports(tmp_path):
    r = SessionRecorder(0)
    r.record(0, "ATTENTIVE")
    r.record(2, "LOOKING_AWAY", "HEART", True)
    r.record(4, "LOOKING_AWAY", speech_alert=True)
    result = r.summary(5)
    assert result["attentive_percent_observable"] == 40
    assert result["gestures"]["HEART"] == 1
    assert result["speech_alerts"] == 1
    summary, events = r.export(tmp_path, 5)
    assert summary.exists() and events.exists()
    assert "No video" in summary.read_text()


def test_settings_validation(tmp_path):
    with pytest.raises(ValueError):
        Settings(smoothing=1).validate()
    p = tmp_path / "settings.json"
    atomic_json(p, {"camera_index": 1})
    assert load_settings(p).camera_index == 1
    atomic_json(p, {"extra": 1})
    with pytest.raises(ValueError):
        load_settings(p)
