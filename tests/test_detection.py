import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import make_synthetic_watches_image
from watch_dial_capture.detection import Detection, detect_watch_regions


def test_detect_watch_regions_finds_multiple_watches():
    image = make_synthetic_watches_image()

    detections = detect_watch_regions(image, min_radius=50, max_radius=120)

    assert len(detections) == 2
    # Sorted left-to-right.
    assert detections[0].center_x < detections[1].center_x
    for detection in detections:
        assert 50 <= detection.radius <= 120


def test_detect_watch_regions_respects_radius_bounds():
    image = make_synthetic_watches_image()

    # Circles are drawn with radius 80; nothing should match a much
    # smaller radius bound.
    detections = detect_watch_regions(image, min_radius=5, max_radius=20)

    assert detections == []


def test_detect_watch_regions_rejects_empty_image():
    with pytest.raises(ValueError):
        detect_watch_regions(__import__("numpy").empty((0, 0)))


def test_detection_bounding_box_is_clipped_to_zero():
    detection = Detection(center_x=5, center_y=5, radius=20)

    x1, y1, x2, y2 = detection.bounding_box

    assert x1 == 0
    assert y1 == 0
    assert x2 == 25
    assert y2 == 25
