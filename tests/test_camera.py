import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watch_dial_capture.capture import Camera


class _FakeVideoCapture:
    """Stand-in for cv2.VideoCapture that reports as never opened."""

    instances = []

    def __init__(self, device_index):
        self.device_index = device_index
        self.released = False
        _FakeVideoCapture.instances.append(self)

    def isOpened(self):
        return False

    def release(self):
        self.released = True


def test_camera_releases_capture_when_open_fails(monkeypatch):
    _FakeVideoCapture.instances = []
    monkeypatch.setattr(
        "watch_dial_capture.capture.cv2.VideoCapture", _FakeVideoCapture
    )

    with pytest.raises(RuntimeError):
        Camera(device_index=3)

    assert len(_FakeVideoCapture.instances) == 1
    assert _FakeVideoCapture.instances[0].released is True
