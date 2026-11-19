"""Binary automatic gaze, stability and UI regressions."""
from gazeguard.automatic import AutomaticMonitor
from gazeguard.geometry import FaceMetrics
from gazeguard.webapp import ASSETS


def face(x=0, y=0, yaw=0, pitch=0):
    return FaceMetrics(x, y, yaw, pitch, .5, .5, .35, 0, (x, y), (x, y))


def test_binary_statuses_in_automatic_mode():
    m = AutomaticMonitor()
    values = [face(), face(), face(x=.085), face(), face(), None, face(y=.11)]
    outputs = [m.update(value, i*.05).status for i, value in enumerate(values)]
    assert outputs == ['LOOKING_AWAY', 'ATTENTIVE', 'LOOKING_AWAY',
                       'LOOKING_AWAY', 'ATTENTIVE', 'LOOKING_AWAY', 'LOOKING_AWAY']
    assert set(outputs) == {'LOOKING_AWAY', 'ATTENTIVE'}


def test_obstructed_face_is_not_active():
    m = AutomaticMonitor()
    m.update(face(), 0)
    m.update(face(), .05)
    assert m.update(face(), .10, obstructed=True).status == 'LOOKING_AWAY'


def test_ui_only_exposes_active_or_looking_away_during_tracking():
    js = (ASSETS / 'app.js').read_text(encoding='utf-8')
    assert "d.status==='ATTENTIVE'?'ACTIVE':d.camera?'LOOKING AWAY'" in js
    assert 'Likely active (estimate)' not in js
    assert "put('signal-attention',d.status==='ATTENTIVE'?'ACTIVE'" in js


def test_does_not_claim_exact_lens_contact():
    m = AutomaticMonitor()
    m.update(face(), 0)
    result = m.update(face(), .05)
    assert result.status == 'ATTENTIVE'
    assert 'cannot be guaranteed' in result.reason
