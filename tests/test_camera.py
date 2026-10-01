import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from watch_dial_capture.capture import (
    DEFAULT_FOCUS_WARMUP_FRAMES,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    Camera,
)


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

    def set(self, prop_id, value):
        pass


def test_camera_releases_capture_when_open_fails(monkeypatch):
    _FakeVideoCapture.instances = []
    monkeypatch.setattr(
        "watch_dial_capture.capture.cv2.VideoCapture", _FakeVideoCapture
    )

    with pytest.raises(RuntimeError):
        Camera(device_index=3)

    assert len(_FakeVideoCapture.instances) == 1
    assert _FakeVideoCapture.instances[0].released is True


class _RecordingVideoCapture:
    """Stand-in for cv2.VideoCapture that records calls and reads frames."""

    def __init__(self, device_index, frames):
        self.device_index = device_index
        self._frames = list(frames)
        self.set_calls = []
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop_id, value):
        self.set_calls.append((prop_id, value))

    def read(self):
        self.read_count += 1
        frame = self._frames[min(self.read_count, len(self._frames)) - 1]
        return True, frame

    def release(self):
        self.released = True


def test_camera_requests_max_resolution_and_autofocus(monkeypatch):
    frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(1, 3)]
    fake = _RecordingVideoCapture(0, frames)
    monkeypatch.setattr(
        "watch_dial_capture.capture.cv2.VideoCapture", lambda index: fake
    )

    Camera(device_index=0, focus_warmup_frames=1, sleep_fn=lambda _: None)

    assert (cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH) in fake.set_calls
    assert (cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT) in fake.set_calls
    assert (cv2.CAP_PROP_AUTOFOCUS, 1) in fake.set_calls


def test_camera_read_discards_warmup_frames(monkeypatch):
    frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(1, 6)]
    fake = _RecordingVideoCapture(0, frames)
    monkeypatch.setattr(
        "watch_dial_capture.capture.cv2.VideoCapture", lambda index: fake
    )
    sleeps = []

    camera = Camera(device_index=0, focus_warmup_frames=5, sleep_fn=sleeps.append)
    result = camera.read()

    # Five frames were read (four warm-up + the kept one) and the last
    # one is what's returned.
    assert fake.read_count == 5
    assert np.array_equal(result, frames[-1])
    # A short pause happens between each warm-up read.
    assert len(sleeps) == 5


def test_camera_read_defaults_to_several_warmup_frames(monkeypatch):
    frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(1, 50)]
    fake = _RecordingVideoCapture(0, frames)
    monkeypatch.setattr(
        "watch_dial_capture.capture.cv2.VideoCapture", lambda index: fake
    )

    camera = Camera(device_index=0, sleep_fn=lambda _: None)
    camera.read()

    assert fake.read_count == DEFAULT_FOCUS_WARMUP_FRAMES
