"""Shared test fixtures/utilities."""

from __future__ import annotations

import numpy as np


def make_synthetic_watches_image(
    width: int = 640,
    height: int = 480,
    centers_and_radii=((150, 240, 80), (450, 240, 80)),
) -> np.ndarray:
    """Build a simple synthetic BGR image containing drawn circles.

    This stands in for a photo of one or more analog wrist watches: a
    dark background with light, sharp-edged circles that a Hough
    circle transform can reliably pick up, so the detection/annotation
    pipeline can be exercised without a real camera or dataset.
    """
    import cv2

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (20, 20, 20)
    for x, y, r in centers_and_radii:
        cv2.circle(image, (x, y), r, (200, 200, 200), thickness=-1)
        cv2.circle(image, (x, y), r, (0, 0, 0), thickness=2)
    return image
