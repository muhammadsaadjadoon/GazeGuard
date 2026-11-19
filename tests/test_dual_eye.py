import numpy as np
from gazeguard.calibration import Calibration, CalibrationWizard, load_calibration, save_calibration
from gazeguard.geometry import FaceMetrics
from gazeguard.config import Settings
from gazeguard.monitor import Monitor


def calibration():
    return Calibration({"center": [0, 0, 0, 0], "left": [-.2, 0, 0, 0],
                        "right": [.2, 0, 0, 0], "up": [0, -.2, 0, 0],
                        "down": [0, .2, 0, 0]}, [.5, .5, .3],
                       version=2, eye_centers=[[0, 0], [0, 0]], screen_reference=[0.08, 0, 0, 0])


def face(left=(0, 0), right=(0, 0)):
    return FaceMetrics(0, 0, 0, 0, .5, .5, .3, 0, left, right)


def test_individual_eye_guard_rejects_cancelled_eyes():
    mon = Monitor(Settings(smoothing=.8), calibration())
    assert mon.update(face(), 0).status == "ATTENTIVE"
    assert mon.update(face((.12, 0), (-.12, 0)), .03).status == "LOOKING_AWAY"


def test_legacy_metrics_fail_closed_against_dual_eye_calibration():
    mon = Monitor(Settings(), calibration())
    assert mon.update(FaceMetrics(0, 0, 0, 0, .5, .5, .3, 0), 0).status == "LOOKING_AWAY"


def test_dual_eye_calibration_roundtrip(tmp_path):
    file = tmp_path / "profile.json"
    save_calibration(file, calibration())
    assert load_calibration(file) == calibration()


def test_wizard_saves_individual_eye_centers():
    w = CalibrationWizard(sample_count=12)
    result = None
    for i in range(5):
        for _ in range(12):
            v = np.array([0.1 * i, 0, 0, 0])
            w.observe(v, np.array([.5, .5, .3]), [[.02, .01], [-.02, .01]])
        result = w.confirm()
    assert result.version == 2
    assert result.eye_centers == [[.02, .01], [-.02, .01]]
