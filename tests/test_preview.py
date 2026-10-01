import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watch_dial_capture import preview
from watch_dial_capture.detection import Detection


class _FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.properties = []

    def isOpened(self):
        return self.opened

    def set(self, property_id, value):
        self.properties.append((property_id, value))

    def read(self):
        return True, np.zeros((120, 160, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.mark.parametrize(
    ("key", "accepted"),
    [(ord("q"), False), (27, False), (13, True), (32, True)],
)
def test_preview_keyboard_controls_release_camera_and_draw_detection(
    monkeypatch, key, accepted
):
    capture = _FakeCapture()
    windows = []
    displayed = []
    detection_arguments = []

    def detect(frame, **kwargs):
        detection_arguments.append(kwargs)
        return [Detection(center_x=80, center_y=60, radius=20)]

    monkeypatch.setattr(preview.cv2, "VideoCapture", lambda *args: capture)
    monkeypatch.setattr(preview.cv2, "namedWindow", windows.append)
    monkeypatch.setattr(
        preview.cv2, "imshow", lambda name, frame: displayed.append((name, frame))
    )
    monkeypatch.setattr(preview.cv2, "waitKey", lambda delay: key)
    monkeypatch.setattr(preview.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(preview, "detect_watch_regions", detect)

    result = preview.run_camera_preview(
        3,
        backend=123,
        frame_width=1280,
        frame_height=720,
        min_radius=50,
        max_radius=200,
        dp=1.5,
        param1=110,
        param2=60,
    )

    assert result is accepted
    assert capture.released
    assert capture.properties == [
        (cv2.CAP_PROP_FRAME_WIDTH, 1280),
        (cv2.CAP_PROP_FRAME_HEIGHT, 720),
        (cv2.CAP_PROP_AUTOFOCUS, 1),
    ]
    assert windows == [preview.PREVIEW_WINDOW_NAME]
    assert displayed[0][0] == preview.PREVIEW_WINDOW_NAME
    assert detection_arguments == [
        {
            "min_radius": 50,
            "max_radius": 200,
            "dp": 1.5,
            "param1": 110,
            "param2": 60,
        }
    ]


def test_preview_camera_open_failure_releases_camera_and_windows(monkeypatch):
    capture = _FakeCapture(opened=False)
    destroyed = []

    monkeypatch.setattr(preview.cv2, "VideoCapture", lambda *args: capture)
    monkeypatch.setattr(preview.cv2, "destroyAllWindows", lambda: destroyed.append(True))

    with pytest.raises(RuntimeError, match="Could not open camera"):
        preview.run_camera_preview(
            0,
            backend=0,
            frame_width=640,
            frame_height=480,
            min_radius=20,
            max_radius=100,
            dp=1.2,
            param1=100,
            param2=40,
        )

    assert capture.released
    assert destroyed == [True]


def test_preview_display_failure_reports_gui_error_and_cleans_up(monkeypatch):
    capture = _FakeCapture()
    destroyed = []

    monkeypatch.setattr(preview.cv2, "VideoCapture", lambda *args: capture)
    monkeypatch.setattr(preview.cv2, "namedWindow", lambda name: None)
    monkeypatch.setattr(
        preview.cv2, "imshow", lambda *args: (_ for _ in ()).throw(cv2.error("no GUI"))
    )
    monkeypatch.setattr(preview.cv2, "destroyAllWindows", lambda: destroyed.append(True))
    monkeypatch.setattr(preview, "detect_watch_regions", lambda *args, **kwargs: [])

    with pytest.raises(RuntimeError, match="GUI-enabled OpenCV"):
        preview.run_camera_preview(
            0,
            backend=0,
            frame_width=640,
            frame_height=480,
            min_radius=20,
            max_radius=100,
            dp=1.2,
            param1=100,
            param2=40,
        )

    assert capture.released
    assert destroyed == [True]


def test_preview_read_failure_releases_camera_and_windows(monkeypatch):
    capture = _FakeCapture()
    capture.read = lambda: (False, None)
    destroyed = []

    monkeypatch.setattr(preview.cv2, "VideoCapture", lambda *args: capture)
    monkeypatch.setattr(preview.cv2, "namedWindow", lambda name: None)
    monkeypatch.setattr(preview.cv2, "destroyAllWindows", lambda: destroyed.append(True))

    with pytest.raises(RuntimeError, match="Failed to read frame"):
        preview.run_camera_preview(
            0,
            backend=0,
            frame_width=640,
            frame_height=480,
            min_radius=20,
            max_radius=100,
            dp=1.2,
            param1=100,
            param2=40,
        )

    assert capture.released
    assert destroyed == [True]
