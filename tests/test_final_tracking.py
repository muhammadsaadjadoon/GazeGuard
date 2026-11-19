"""High-jitter center and off-lens regression checks (synthetic samples)."""
from gazeguard.calibration import Calibration
from gazeguard.config import Settings
from gazeguard.geometry import FaceMetrics
from gazeguard.monitor import Monitor


def profile():
    return Calibration(
        samples={'center':[0,0,0,0], 'left':[-.20,0,0,0], 'right':[.20,0,0,0],
                 'up':[0,-.20,0,0], 'down':[0,.20,0,0]},
        home=[.5,.5,.3], version=2,
        eye_centers=[[0,0],[0,0]], center_noise=[.018,.016,.009,.009], screen_reference=[.08,0,0,0],
        eye_noise=[[.016,.013],[.016,.013]])


def sample(x, left=None, right=None):
    return FaceMetrics(x, 0, 0, 0, .5,.5,.3,0, left or (x,0), right or (x,0))


def test_center_jitter_remains_attentive():
    m=Monitor(Settings(), profile())
    assert all(m.update(sample(x), i*.06).status=='ATTENTIVE'
               for i,x in enumerate((0,.025,-.033,.044,-.041,.036,0,-.025)))


def test_both_eyes_sustained_off_lens_rejected():
    m=Monitor(Settings(),profile())
    m.update(sample(0),0)
    assert m.update(sample(.17),.06).status=='LOOKING_AWAY'


def test_divergent_eyes_rejected_even_when_average_centered():
    m=Monitor(Settings(),profile())
    m.update(sample(0),0)
    assert m.update(sample(0,left=(.17,0),right=(-.17,0)),.06).status=='LOOKING_AWAY'
