"""Prevent a sustained screen-adjacent gaze from being counted as lens attention."""
from dataclasses import replace
from gazeguard.calibration import Calibration
from gazeguard.config import Settings
from gazeguard.geometry import FaceMetrics
from gazeguard.monitor import Monitor


def profile():
    return Calibration(
        samples={'center': [0,0,0,0], 'left': [-.2,0,0,0], 'right': [.2,0,0,0],
                 'up': [0,-.2,0,0], 'down': [0,.2,0,0]},
        home=[.5,.5,.3], eye_centers=[[0,0],[0,0]], version=2,
        center_noise=[.008,.008,.01,.01], eye_noise=[[.008,.008],[.008,.008]],
        screen_reference=[.08,0,0,0],
    )


def f(x):
    return FaceMetrics(x, 0, 0, 0, .5,.5,.3, 0, (x,0), (x,0))


def test_sustained_midpoint_does_not_become_green_active():
    m=Monitor(Settings(), profile())
    assert m.update(f(0),0).attentive
    for t in (.05,.10,.15):
        out=m.update(f(.04),t)
    assert out.status == 'UNVERIFIED_GAZE'
    assert not out.attentive


def test_clear_lens_gaze_still_active_and_screen_rejected():
    m=Monitor(Settings(), profile())
    assert m.update(f(0),0).attentive
    assert m.update(f(.08),.1).status == 'LOOKING_AWAY'


def test_without_screen_reference_requires_setup():
    m=Monitor(Settings(), replace(profile(), screen_reference=None))
    assert m.update(f(0),0).status == 'SETUP_REQUIRED'


def test_ambiguous_time_is_not_counted_as_attention():
    from gazeguard.reporting import SessionRecorder
    r = SessionRecorder(0)
    r.record(0, 'ATTENTIVE')
    r.record(1, 'UNVERIFIED_GAZE')
    result = r.summary(3)
    assert result['observable_seconds'] == 3
    assert result['attentive_percent_observable'] == 33.3
