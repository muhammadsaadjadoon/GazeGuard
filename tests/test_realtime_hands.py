"""Real-time capture and complete 21-landmark hand skeleton regressions."""
import time
import numpy as np
from gazeguard.live_capture import LatestFrameCamera
from gazeguard.webapp import HAND_EDGES, draw_hand_skeleton


class CameraStub:
    def __init__(self):
        self.count = 0
        self.released = False

    def read(self):
        self.count += 1
        return True, np.full((8, 8, 3), self.count % 255, dtype=np.uint8)

    def release(self):
        self.released = True


def test_reader_keeps_only_newest_frame():
    cam = CameraStub()
    reader = LatestFrameCamera(cam, idle_sleep=.001)
    reader.start()
    first = reader.next(0, timeout=.25)
    assert first is not None
    time.sleep(.025)
    later = reader.next(first.number, timeout=.25)
    reader.close()
    assert later is not None
    assert later.number > first.number
    assert cam.released
    assert reader.error is None


def test_hand_skeleton_draws_lines_and_joints():
    assert len(HAND_EDGES) >= 20
    image = np.zeros((180, 220, 3), dtype=np.uint8)
    # Nonoverlapping rows; a line midpoint must be colored even if it is not a landmark.
    hand = np.asarray([(10 + i * 7, 10 + i * 5) for i in range(21)])
    result = draw_hand_skeleton(image, hand)
    assert result is image
    assert np.count_nonzero(image) > 150
    assert np.any(image[12:15, 12:15])


def test_missing_hand_landmarks_do_not_crash():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert draw_hand_skeleton(image, []) is image
    assert np.count_nonzero(image) == 0
