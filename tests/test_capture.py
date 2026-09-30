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
