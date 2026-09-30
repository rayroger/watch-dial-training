import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import make_synthetic_watches_image
from watch_dial_capture.annotate import annotate_image
from watch_dial_capture.detection import detect_watch_regions


def test_annotate_image_draws_labels_and_timestamp():
    image = make_synthetic_watches_image()
    detections = detect_watch_regions(image, min_radius=50, max_radius=120)
    assert detections, "test setup should always detect the synthetic watches"

    annotated = annotate_image(image, detections, timestamp="2024-01-02 15:30:00")

    # Annotation must not mutate the caller's image and must be the same shape.
    assert annotated.shape == image.shape
    assert not (annotated == image).all()


def test_annotate_image_handles_no_detections():
    image = make_synthetic_watches_image(centers_and_radii=())

    annotated = annotate_image(image, [])

    assert annotated.shape == image.shape
