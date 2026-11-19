"""Regression tests for visible hands independent of gestures and screen negatives."""
from dataclasses import replace
import numpy as np
import pytest
from gazeguard.calibration import Calibration, teach_screen_reference
from gazeguard.config import Settings
from gazeguard.geometry import FaceMetrics
from gazeguard.gestures import hand_overlaps_face
from gazeguard.monitor import Monitor


def profile():
    return Calibration(samples={
        'center': [0, 0, 0, 0], 'left': [-.2, 0, 0, 0],
        'right': [.2, 0, 0, 0], 'up': [0, -.2, 0, 0],
        'down': [0, .2, 0, 0]}, home=[.5, .5, .3], version=2,
        eye_centers=[[0, 0], [0, 0]],
        center_noise=[.006, .006, .006, .006],
        eye_noise=[[.006, .006], [.006, .006]])


def face(x):
    return FaceMetrics(x, 0, 0, 0, .5, .5, .3, 0, (x, 0), (x, 0))


def test_hand_visible_not_necessarily_face_occlusion():
    palm=np.column_stack((np.linspace(200, 260, 21), np.linspace(200, 270, 21)))
    assert not hand_overlaps_face(palm,(20,20,150,150))
    assert hand_overlaps_face(palm,(195,195,280,290))


def test_reject_invalid_hand_points():
    assert not hand_overlaps_face(np.ones((3, 2)), (0, 0, 200, 200))


def test_screen_reference_saved_if_separable():
    p=profile()
    samples=[np.array([.08+(i%3-1)*.001, .0, 0, 0]) for i in range(30)]
    fitted=teach_screen_reference(p, samples)
    assert fitted.screen_reference is not None
    assert p.screen_reference is None


def test_indistinguishable_screen_rejected():
    with pytest.raises(ValueError,match='too similar'):
        teach_screen_reference(profile(), [np.array([.003, 0, 0, 0])]*30)


def test_screen_negative_changes_candidate_not_center():
    m=Monitor(Settings(),replace(profile(),screen_reference=[.08,0,0,0]))
    assert m.update(face(0), 0).status == 'ATTENTIVE'
    assert m.update(face(.08), .2).status == 'LOOKING_AWAY'
    assert 'Screen gaze' in m.update(face(.08), .4).reason


def test_hand_obstruction_status_distinct_from_hand_presence():
    m=Monitor(Settings(),replace(profile(), screen_reference=[.08, 0, 0, 0]))
    assert m.update(face(0), 0, False).status == 'ATTENTIVE'
    assert m.update(face(0), .2, True).status == 'HANDS_OBSTRUCTING'
