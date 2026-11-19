import numpy as np
import pytest
from gazeguard.calibration import Calibration, align_lens_reference
from gazeguard.monitor import Monitor
from gazeguard.config import Settings
from gazeguard.geometry import FaceMetrics

def initial():
    return Calibration(
        samples={"center": [0., 0., 0., 0.], "left": [-.18, 0., 0., 0.],
                 "right": [.18, 0., 0., 0.], "up": [0., -.17, 0., 0.],
                 "down": [0., .17, 0., 0.]}, home=[.5, .5, .3], version=2,
        eye_centers=[[0., 0.], [0., 0.]], screen_reference=[.08,0.,0.,0.])

def lens_data(x=.12):
    vectors=[]; homes=[]; eyes=[]
    for j in range(30):
        noise=(j % 5 - 2)*.001
        vectors.append([x + noise, -.01 + noise, .008, 0])
        homes.append([.52, .48, .31])
        eyes.append([[x + noise, -.01 + noise], [x + noise, -.01 + noise]])
    return vectors, homes, eyes

def test_verified_lens_fixes_false_away():
    cal=initial()
    v,h,e=lens_data()
    face=FaceMetrics(.12, -.01, .008, 0., .52, .48, .31, 0., (.12,-.01), (.12,-.01))
    before=Monitor(Settings(),cal).update(face,0)
    assert before.status=='LOOKING_AWAY'
    fixed=align_lens_reference(cal,v,h,e)
    after=Monitor(Settings(),fixed).update(face,0)
    assert after.status=='ATTENTIVE'
    assert fixed.samples['right'][0] - fixed.samples['center'][0] == pytest.approx(.18)

def test_cannot_align_without_stationary_reference():
    v,h,e=lens_data()
    for j in range(30):
        v[j][0] += j*.012
        e[j][0][0] += j*.012
    with pytest.raises(ValueError, match='unstable'):
        align_lens_reference(initial(),v,h,e)

def test_cannot_align_without_valid_samples():
    with pytest.raises(ValueError):
        align_lens_reference(initial(), [], [], [])
