"""Annotate captured pictures with the detected watch regions."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .detection import Detection


def annotate_image(
    image: np.ndarray,
    detections: Sequence[Detection],
    *,
    timestamp: str | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of ``image`` annotated with the detected watches.

    Each detection is outlined with a circle and a numbered label
    (``watch_0``, ``watch_1``, ...) so the annotated picture and the
    per-watch crops produced by :func:`watch_dial_capture.dataset.save_capture`
    can be cross-referenced. When ``timestamp`` is provided it is
    stamped in the top-left corner of the image, which is useful when
    reviewing a dataset of periodic captures.
    """
    annotated = image.copy()

    for index, detection in enumerate(detections):
        center = (detection.center_x, detection.center_y)
        cv2.circle(annotated, center, detection.radius, color, thickness)
        cv2.circle(annotated, center, 3, color, -1)

        label = f"watch_{index}"
        label_origin = (
            max(detection.center_x - detection.radius, 0),
            max(detection.center_y - detection.radius - 10, 15),
        )
        cv2.putText(
            annotated,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if timestamp:
        cv2.putText(
            annotated,
            timestamp,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated
