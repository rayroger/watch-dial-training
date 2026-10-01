import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import make_synthetic_watches_image
from watch_dial_capture.dataset import _write_image, save_capture
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
    assert result.annotated_path.name == "20240102-153000-000.jpg"

    assert len(result.watch_paths) == 2
    for index, watch_path in enumerate(result.watch_paths):
        assert watch_path.exists()
        assert watch_path.parent.name == "watches"
        assert watch_path.name == f"20240102-153000-000_watch_{index}.jpg"

    assert result.metadata_path.exists()
    assert result.metadata_path.parent.name == "metadata"
    assert result.metadata_path.name == "20240102-153000-000.json"
    assert "detections" in result.metadata_path.read_text()


def test_save_capture_with_no_detections_still_writes_annotated_image(tmp_path):
    image = make_synthetic_watches_image(centers_and_radii=())

    result = save_capture(image, [], tmp_path, timestamp=datetime(2024, 1, 2, 15, 30, 0))

    assert result.annotated_path.exists()
    assert result.watch_paths == []


def test_write_image_rejects_empty_image(tmp_path):
    empty_image = np.empty((0, 0, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        _write_image(tmp_path / "empty.jpg", empty_image)


def test_write_image_raises_io_error_on_write_failure(tmp_path, monkeypatch):
    image = make_synthetic_watches_image()

    monkeypatch.setattr("watch_dial_capture.dataset.cv2.imwrite", lambda *a, **k: False)

    with pytest.raises(IOError):
        _write_image(tmp_path / "will_fail.jpg", image)


def test_save_capture_cleans_up_partial_writes_on_failure(tmp_path, monkeypatch):
    image = make_synthetic_watches_image()
    detections = detect_watch_regions(image, min_radius=50, max_radius=120)
    assert len(detections) == 2

    real_write_image = _write_image
    call_count = {"n": 0}

    def flaky_write_image(path, img):
        call_count["n"] += 1
        # Let the annotated image and the first watch crop succeed,
        # then fail on the second watch crop.
        if call_count["n"] == 3:
            raise IOError("simulated write failure")
        return real_write_image(path, img)

    monkeypatch.setattr("watch_dial_capture.dataset._write_image", flaky_write_image)

    with pytest.raises(IOError):
        save_capture(image, detections, tmp_path, timestamp=datetime(2024, 1, 2, 15, 30, 0))

    # None of the files from this (failed) capture should remain behind.
    remaining = list(tmp_path.rglob("*"))
    remaining_files = [p for p in remaining if p.is_file()]
    assert remaining_files == []
