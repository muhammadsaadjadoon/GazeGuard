"""Calibration-free launch, conservative estimation and optional manual mode."""
from gazeguard.automatic import AutomaticMonitor
from gazeguard.geometry import FaceMetrics
from gazeguard.webapp import Engine, ASSETS


def sample(x=0.0, y=0.0, yaw=0.0, pitch=0.0):
    return FaceMetrics(x, y, yaw, pitch, .5, .5, .35, .01, (x,y), (x,y))


def test_launch_never_requires_personal_calibration(tmp_path):
    e = Engine(tmp_path)
    assert e.wizard is None
    assert e.snapshot()['mode'] == 'AUTOMATIC'
    assert isinstance(e.monitor, AutomaticMonitor)


def test_near_axis_is_estimate_not_verified_lens_contact():
    m = AutomaticMonitor()
    obs = [m.update(sample(), i * .05) for i in range(4)]
    assert obs[-1].status == 'ATTENTIVE'
    assert obs[-1].attentive
    assert 'cannot be guaranteed' in obs[-1].reason


def test_clear_side_gaze_and_missing_face():
    m = AutomaticMonitor()
    assert m.update(sample(x=.19), 0).status == 'LOOKING_AWAY'
    assert m.update(sample(y=-.14), .1).status == 'LOOKING_AWAY'
    assert m.update(None, .2).status == 'LOOKING_AWAY'


def test_obstructing_hand_and_eye_disagreement():
    m=AutomaticMonitor()
    assert m.update(sample(), 0, True).status == 'LOOKING_AWAY'
    face=FaceMetrics(0,0,0,0,.5,.5,.35,.01,(-.09,0),(.09,0))
    assert m.update(face,.1).status == 'LOOKING_AWAY'


def test_advanced_calibration_is_optional_and_reversible(tmp_path):
    e=Engine(tmp_path)
    e.reset()
    assert e.wizard is not None
    assert e.snapshot()['mode'] == 'MANUAL'
    e.automatic()
    assert e.wizard is None
    assert isinstance(e.monitor, AutomaticMonitor)
    assert e.snapshot()['mode']=='AUTOMATIC'


def test_user_facing_layout_compact_and_advanced_hidden():
    page=(ASSETS/'index.html').read_text()
    assert 'Live attention status' in page
    assert '<details id="advanced"' in page
    assert 'Session' in page and 'Hands visible' in page
    assert 'Your focus, clearly' not in page
    assert 'EXTRA FEATURES' not in page
