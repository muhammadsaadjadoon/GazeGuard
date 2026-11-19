"""Regression scenarios for frequent false Looking Away states."""
import numpy as np
from gazeguard.calibration import Calibration, CalibrationWizard
from gazeguard.config import Settings
from gazeguard.geometry import FaceMetrics
from gazeguard.monitor import Monitor


def profile():
    return Calibration(
        samples={"center": [0.01, -0.01, 0, 0], "left": [-0.15, -0.01, 0, 0],
                 "right": [0.17, -0.01, 0, 0], "up": [0.01, -0.16, 0, 0],
                 "down": [0.01, 0.14, 0, 0]},
        home=[.5, .5, .3], version=2,
        eye_centers=[[.01, -.01], [.01, -.01]],
        center_noise=[.016, .012, .01, .01],
        eye_noise=[[.018, .014], [.018, .014]], screen_reference=[.08, -.01, 0, 0])


def frame(offset=0., head=0., left=None, right=None, x=.5):
    g = .01 + offset
    e = (g, -.01)
    return FaceMetrics(g, -.01, head, 0., x, .5, .3, 0., left or e, right or e)


def test_direct_lens_with_natural_iris_jitter_stays_attentive():
    monitor = Monitor(Settings(), profile())
    statuses = [monitor.update(frame(offset=v), i * .05).status for i, v in enumerate(
        [0, .024, -.028, .031, -.018, .028, -.024, .020])]
    assert statuses == ['ATTENTIVE'] * len(statuses)


def test_large_side_glance_is_rejected():
    monitor = Monitor(Settings(), profile())
    monitor.update(frame(), 0.)
    out = monitor.update(frame(offset=.12), .06)
    assert out.status == 'LOOKING_AWAY'
    assert 'Eyes' in out.reason or 'landmarks' in out.reason or 'Screen gaze' in out.reason


def test_out_of_position_is_explained():
    monitor = Monitor(Settings(), profile())
    out = monitor.update(frame(x=.92), 0.)
    assert out.status == 'OUT_OF_POSITION'
    assert 'frame' in out.reason


def test_center_noise_is_stored_from_calibration():
    wizard = CalibrationWizard(sample_count=12)
    result = None
    for i in range(5):
        for j in range(12):
            v = [0.03 * i + (j % 3 - 1)*.005, -.01, 0, 0]
            wizard.observe(np.array(v), np.array([.5, .5, .3]), [[.01, -.01], [.01, -.01]])
        result = wizard.confirm()
    assert result is not None
    assert len(result.center_noise) == 4
    assert len(result.eye_noise) == 2
