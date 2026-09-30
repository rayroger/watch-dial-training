import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import make_synthetic_watches_image
from watch_dial_capture.dataset import save_capture
from watch_dial_capture.detection import detect_watch_regions


def test_save_capture_writes_annotated_and_cropped_images(tmp_path):
    image = make_synthetic_watches_image()
    detections = detect_watch_regions(image, min_radius=50, max_radius=120)
    assert len(detections) == 2

    result = save_capture(
        image, detections, tmp_path, timestamp=datetime(2024, 1, 2, 15, 30, 0)
    )

    assert result.annotated_path.exists()
    assert result.annotated_path.parent.name == "annotated"
    assert result.annotated_path.name == "20240102-153000.jpg"

    assert len(result.watch_paths) == 2
    for index, watch_path in enumerate(result.watch_paths):
        assert watch_path.exists()
        assert watch_path.parent.name == "watches"
        assert watch_path.name == f"20240102-153000_watch_{index}.jpg"

    assert result.metadata_path.exists()
    assert "detections" in result.metadata_path.read_text()


def test_save_capture_with_no_detections_still_writes_annotated_image(tmp_path):
    image = make_synthetic_watches_image(centers_and_radii=())

    result = save_capture(image, [], tmp_path, timestamp=datetime(2024, 1, 2, 15, 30, 0))

    assert result.annotated_path.exists()
    assert result.watch_paths == []
