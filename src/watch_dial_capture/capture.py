"""Periodic camera capture loop.

Opens a camera device with OpenCV and, every ``interval`` seconds,
grabs a frame, detects watch faces in it, annotates the frame and
saves the annotated picture plus per-watch crops via
:func:`watch_dial_capture.dataset.save_capture`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator, Optional

from .dataset import CaptureResult, save_capture
from .detection import detect_watch_regions

logger = logging.getLogger(__name__)


class Camera:
    """Thin wrapper around :class:`cv2.VideoCapture` for easier testing."""

    def __init__(self, device_index: int = 0):
        self._capture = cv2.VideoCapture(device_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera at index {device_index}")

    def read(self):
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


def run_capture_loop(
    camera: Camera,
    output_dir: str | Path,
    *,
    interval_seconds: float = 60.0,
    count: Optional[int] = None,
    min_radius: int = 40,
    max_radius: int = 400,
    sleep_fn=time.sleep,
) -> Iterator[CaptureResult]:
    """Periodically capture, annotate and save pictures.

    Args:
        camera: An open :class:`Camera` (or any object exposing a
            ``read()`` method returning a BGR image).
        output_dir: Directory passed to
            :func:`watch_dial_capture.dataset.save_capture`.
        interval_seconds: Delay between successive captures.
        count: Number of captures to take. ``None`` means run forever
            (until interrupted).
        min_radius: Forwarded to
            :func:`watch_dial_capture.detection.detect_watch_regions`.
        max_radius: Forwarded to
            :func:`watch_dial_capture.detection.detect_watch_regions`.
        sleep_fn: Sleep function, overridable for tests.

    Yields:
        The :class:`~watch_dial_capture.dataset.CaptureResult` for each
        capture, as they happen.
    """
    captured = 0
    while count is None or captured < count:
        frame = camera.read()
        detections = detect_watch_regions(
            frame, min_radius=min_radius, max_radius=max_radius
        )
        logger.info("Captured frame with %d watch(es) detected", len(detections))
        result = save_capture(frame, detections, output_dir)
        yield result

        captured += 1
        if count is None or captured < count:
            sleep_fn(interval_seconds)
