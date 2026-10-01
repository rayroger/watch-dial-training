"""Detection of circular watch dials in a picture.

Analog wrist watches are (almost) always circular, so a classic
Hough-circle transform is a lightweight, dependency-free way to locate
candidate watch faces in a photo without needing a trained model
first (we are, after all, trying to *build* the training set for that
model). Each detected circle is returned together with a square
bounding box that can be used to crop the watch out of the original
picture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detected watch face.

    Attributes:
        center_x: X coordinate (pixels) of the circle center.
        center_y: Y coordinate (pixels) of the circle center.
        radius: Radius (pixels) of the detected circle.
    """

    center_x: int
    center_y: int
    radius: int

    @property
    def bounding_box(self) -> "tuple[int, int, int, int]":
        """Return ``(x1, y1, x2, y2)`` for the square crop around the watch."""
        x1 = max(self.center_x - self.radius, 0)
        y1 = max(self.center_y - self.radius, 0)
        x2 = self.center_x + self.radius
        y2 = self.center_y + self.radius
        return x1, y1, x2, y2


def detect_watch_regions(
    image: np.ndarray,
    *,
    min_radius: int = 150,
    max_radius: int = 350,
    min_distance: int | None = None,
    dp: float = 1.2,
    param1: float = 100,
    param2: float = 40,
) -> List[Detection]:
    """Detect circular watch dials within ``image``.

    Args:
        image: A BGR (as produced by OpenCV) or grayscale image.
        min_radius: Smallest watch-face radius (in pixels) to detect.
        max_radius: Largest watch-face radius (in pixels) to detect.
        min_distance: Minimum distance between the centers of two
            detected circles. Defaults to ``2 * min_radius`` so that
            watches placed next to each other are not merged together.
        dp: Inverse ratio of the accumulator resolution, forwarded to
            :func:`cv2.HoughCircles`.
        param1: Higher Canny edge-detector threshold, forwarded to
            :func:`cv2.HoughCircles`.
        param2: Accumulator threshold for circle centers; lower values
            detect more (possibly false) circles, forwarded to
            :func:`cv2.HoughCircles`.

    Returns:
        A list of :class:`Detection` sorted left-to-right so that the
        same physical watch tends to keep the same index across
        successive captures.
    """
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty array")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = cv2.medianBlur(gray, 5)

    if min_distance is None:
        min_distance = max(2 * min_radius, 1)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_distance,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    detections = [
        Detection(center_x=int(round(x)), center_y=int(round(y)), radius=int(round(r)))
        for x, y, r in circles[0]
    ]
    detections.sort(key=lambda d: d.center_x)
    return detections


def crop_detections(image: np.ndarray, detections: Sequence[Detection]) -> List[np.ndarray]:
    """Return a list of image crops, one per detection, clipped to bounds."""
    height, width = image.shape[:2]
    crops = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bounding_box
        x2 = min(x2, width)
        y2 = min(y2, height)
        crops.append(image[y1:y2, x1:x2].copy())
    return crops
