import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import make_synthetic_watches_image
from watch_dial_capture.capture import run_capture_loop


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

    assert camera.read_count == 2
    # Only the second attempt succeeded, so only one result is yielded,
    # but the loop did not stop/crash after the first failure.
    assert len(results) == 1


def test_run_capture_loop_runs_requested_number_of_captures(tmp_path):
    frame = make_synthetic_watches_image()
    camera = _FakeCamera(frame)
    sleeps = []

    results = list(
        run_capture_loop(
            camera,
            tmp_path,
            interval_seconds=5,
            count=3,
            min_radius=50,
            max_radius=120,
            sleep_fn=sleeps.append,
        )
    )

    assert camera.read_count == 3
    assert len(results) == 3
    # Sleep is called between captures, not after the last one.
    assert sleeps == [5, 5]
    for result in results:
        assert result.annotated_path.exists()
        assert len(result.watch_paths) == 2
