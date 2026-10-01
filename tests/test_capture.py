import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import make_synthetic_watches_image
from watch_dial_capture import capture as capture_module
from watch_dial_capture.capture import MIN_CAPTURE_INTERVAL_SECONDS, run_capture_loop


class _FakeCamera:
    def __init__(self, frame):
        self._frame = frame
        self.read_count = 0

    def read(self):
        self.read_count += 1
        return self._frame


class _FlakyCamera:
    """A fake camera whose first read fails, to exercise error recovery."""

    def __init__(self, frame):
        self._frame = frame
        self.read_count = 0

    def read(self):
        self.read_count += 1
        if self.read_count == 1:
            raise RuntimeError("simulated camera glitch")
        return self._frame


class _AlwaysFailingCamera:
    """A fake camera that never succeeds, to exercise the failure limit."""

    def __init__(self):
        self.read_count = 0

    def read(self):
        self.read_count += 1
        raise RuntimeError("camera permanently disconnected")


def test_run_capture_loop_continues_after_a_failed_capture(tmp_path):
    frame = make_synthetic_watches_image()
    camera = _FlakyCamera(frame)
    sleeps = []

    results = list(
        run_capture_loop(
            camera,
            tmp_path,
            interval_seconds=5,
            count=2,
            min_radius=50,
            max_radius=120,
            sleep_fn=sleeps.append,
        )
    )

    # First read fails (retried), then two successful reads satisfy count=2.
    assert camera.read_count == 3
    assert len(results) == 2


def test_run_capture_loop_runs_requested_number_of_captures(tmp_path):
    frame = make_synthetic_watches_image()
    camera = _FakeCamera(frame)
    sleeps = []

    results = list(
        run_capture_loop(
            camera,
            tmp_path,
            interval_seconds=90,
            count=3,
            min_radius=50,
            max_radius=120,
            sleep_fn=sleeps.append,
        )
    )

    assert camera.read_count == 3
    assert len(results) == 3
    # Sleep is called between captures, not after the last one.
    assert sleeps == [90, 90]
    for result in results:
        assert result.annotated_path.exists()
        assert len(result.watch_paths) == 2


def test_run_capture_loop_enforces_minimum_interval(tmp_path):
    frame = make_synthetic_watches_image()
    camera = _FakeCamera(frame)
    sleeps = []

    list(
        run_capture_loop(
            camera,
            tmp_path,
            interval_seconds=5,
            count=2,
            min_radius=50,
            max_radius=120,
            sleep_fn=sleeps.append,
        )
    )

    # A requested interval below the one-minute floor is clamped up.
    assert sleeps == [MIN_CAPTURE_INTERVAL_SECONDS]


def test_run_capture_loop_raises_after_max_consecutive_failures(tmp_path):
    camera = _AlwaysFailingCamera()
    sleeps = []

    with pytest.raises(RuntimeError):
        list(
            run_capture_loop(
                camera,
                tmp_path,
                interval_seconds=90,
                max_consecutive_failures=3,
                sleep_fn=sleeps.append,
            )
        )

    assert camera.read_count == 3
    assert sleeps == [90, 90]


def test_run_capture_loop_forwards_hough_params_to_detection(tmp_path, monkeypatch):
    frame = make_synthetic_watches_image()
    camera = _FakeCamera(frame)
    captured_kwargs = {}
    real_detect = capture_module.detect_watch_regions

    def fake_detect(image, **kwargs):
        captured_kwargs.update(kwargs)
        return real_detect(image, **kwargs)

    monkeypatch.setattr(capture_module, "detect_watch_regions", fake_detect)

    list(
        run_capture_loop(
            camera,
            tmp_path,
            interval_seconds=90,
            count=1,
            min_radius=50,
            max_radius=120,
            dp=1.5,
            param1=90,
            param2=55,
            sleep_fn=lambda _: None,
        )
    )

    assert captured_kwargs["min_radius"] == 50
    assert captured_kwargs["max_radius"] == 120
    assert captured_kwargs["dp"] == 1.5
    assert captured_kwargs["param1"] == 90
    assert captured_kwargs["param2"] == 55


def test_run_capture_loop_default_radius_matches_detection_defaults():
    import inspect

    signature = inspect.signature(run_capture_loop)

    assert signature.parameters["min_radius"].default == 150
    assert signature.parameters["max_radius"].default == 350

