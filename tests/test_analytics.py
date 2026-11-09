import numpy as np
from remote_ai.analytics import MotionDetector

def test_motion_detector_initializes_without_motion():
    detector = MotionDetector()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    _, motion, area = detector.apply(frame)
    assert motion is False
    assert area == 0
