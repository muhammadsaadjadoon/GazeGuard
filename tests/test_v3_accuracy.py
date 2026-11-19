import numpy as np
from gazeguard.calibration import CalibrationWizard
from gazeguard.geometry import eye_gaze, measure_face

class Pt:
    def __init__(self,x,y): self.x=x; self.y=y

def test_gaze_y_same_sign_for_both_eyes():
    points=[Pt(.5,.5) for _ in range(478)]
    for corner in ((33,133,159,145,468),(263,362,386,374,473)):
        l,r,t,b,iris=corner
        points[l]=Pt(.2,.5); points[r]=Pt(.4,.5)
        points[t]=Pt(.3,.45); points[b]=Pt(.3,.55)
        points[iris]=Pt(.3,.53)
        dx,dy=eye_gaze(points,1000,1000,iris,(l,r,t,b))
        assert dy>0

def test_reject_blink_landmarks():
    pts=[Pt(.5,.5) for _ in range(478)]
    pts[33]=Pt(.3,.5); pts[133]=Pt(.4,.5)
    pts[159]=Pt(.35,.5); pts[145]=Pt(.35,.5)
    try: measure_face(pts,1000,1000)
    except ValueError as e: assert 'Blink' in str(e)
    else: raise AssertionError('must reject closed eyes')

def test_unstable_calibration_rejected():
    wiz=CalibrationWizard(sample_count=20)
    for i in range(20):
        wiz.observe(np.array([-.2 if i%2 else .2,0,0,0]),np.array([.5,.5,.3]))
    assert wiz.confirm() is None
    assert wiz.index==0
    assert len(wiz.recent)==0
