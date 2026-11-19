import numpy as np
from gazeguard.calibration import CalibrationWizard

def test_missing_eye_breaks_same_frame_sample_window():
    wizard = CalibrationWizard(sample_count=4)
    vector = np.array([0., 0., 0., 0.])
    home = np.array([.5, .5, .3])
    eyes = ((.01, .02), (.01, .02))
    wizard.observe(vector, home, eyes)
    wizard.observe(vector, home, None)
    assert len(wizard.recent) == len(wizard.recent_eyes) == 0
    for _ in range(4): wizard.observe(vector, home, eyes)
    assert len(wizard.recent) == len(wizard.recent_eyes) == 4

def test_corrupt_eye_sample_does_not_poison_calibration():
    wizard = CalibrationWizard(sample_count=3)
    wizard.observe(np.zeros(4), np.array([.5,.5,.4]), ((float('nan'),0),(0,0)))
    assert wizard.recent == wizard.recent_eyes == []
