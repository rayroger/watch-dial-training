"""Periodic camera capture loop.

Opens a camera device with OpenCV in a *photo* configuration (highest
supported resolution, with time given for autofocus/auto-exposure to
settle before each shot) rather than treating it as a live video
stream, and every ``interval`` seconds (at most once a minute) grabs a
still, detects watch faces in it, annotates the frame and saves the
annotated picture plus per-watch crops via
:func:`watch_dial_capture.dataset.save_capture`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2

from .dataset import CaptureResult, save_capture
from .detection import detect_watch_regions

logger = logging.getLogger(__name__)

# Exceptions expected from a real camera/detection pipeline (a dropped
# frame, a disconnected device, an unreadable/corrupt image, ...). Any
# other exception type is treated as a programming error and is allowed
# to propagate instead of being silently retried.
RECOVERABLE_ERRORS = (RuntimeError, OSError, ValueError, cv2.error)

# Watches don't move and don't need to be photographed more than once a
# minute, so this is enforced as a hard floor on the capture interval
# regardless of what a caller requests.
MIN_CAPTURE_INTERVAL_SECONDS = 60.0

# Deliberately higher than any consumer webcam's native resolution: cv2
# clamps requests like this down to the highest mode the device
# actually supports, which is how you ask a ``cv2.VideoCapture`` for
# its best still-photo quality instead of a low-res video/preview
# stream.
DEFAULT_FRAME_WIDTH = 7680
DEFAULT_FRAME_HEIGHT = 4320

# Number of frames to read (and discard) before keeping one, so that a
# camera's continuous autofocus/auto-exposure has time to lock onto the
# watches instead of capturing whatever blurry frame happens to be in
# the buffer right after the resolution/focus mode changes.
DEFAULT_FOCUS_WARMUP_FRAMES = 15
DEFAULT_FOCUS_WARMUP_DELAY_SECONDS = 0.1


class Camera:
    """A :class:`cv2.VideoCapture` wrapper that behaves like a still camera.

    Plain video-streaming usage of OpenCV (read a frame as soon as it's
    available, repeatedly, at whatever resolution the driver defaults
    to) tends to produce lower-resolution, poorly-focused images
    because the camera is optimised for a smooth low-latency preview
    rather than a single sharp photo. This wrapper instead requests the
    camera's maximum resolution and, each time a photo is taken,
    discards a handful of warm-up frames first to give the hardware
    autofocus/auto-exposure a chance to settle.
    """

    def __init__(
        self,
        device_index: int = 0,
        *,
        backend: int = cv2.CAP_ANY,
        frame_width: int = DEFAULT_FRAME_WIDTH,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        focus_warmup_frames: int = DEFAULT_FOCUS_WARMUP_FRAMES,
        focus_warmup_delay_seconds: float = DEFAULT_FOCUS_WARMUP_DELAY_SECONDS,
        sleep_fn=time.sleep,
    ):
        self._capture = cv2.VideoCapture(device_index, backend)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open camera at index {device_index}")

        # Ask for the highest resolution the device supports (photo
        # mode) instead of whatever lower-res default the driver
        # streams for live video/preview.
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        # Enable continuous autofocus where supported so the warm-up
        # frames below actually give the lens time to focus.
        self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        self._focus_warmup_frames = max(focus_warmup_frames, 1)
        self._focus_warmup_delay_seconds = focus_warmup_delay_seconds
        self._sleep_fn = sleep_fn

    def read(self):
        """Capture a single still photo.

        Unlike a raw video-stream ``read()``, this discards
        ``focus_warmup_frames - 1`` frames first (pausing briefly
        between each) so the camera's autofocus/auto-exposure has time
        to settle on the watches before the frame that is actually kept
        is grabbed.
        """
        frame = None
        for _ in range(self._focus_warmup_frames):
            ok, frame = self._capture.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")
            if self._focus_warmup_delay_seconds:
                self._sleep_fn(self._focus_warmup_delay_seconds)
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
    min_radius: int = 150,
    max_radius: int = 350,
    dp: float = 1.2,
    param1: float = 100,
    param2: float = 40,
    max_consecutive_failures: Optional[int] = 5,
    sleep_fn=time.sleep,
) -> Iterator[CaptureResult]:
    """Periodically capture, annotate and save pictures.

    Args:
        camera: An open :class:`Camera` (or any object exposing a
            ``read()`` method returning a BGR image).
        output_dir: Directory passed to
            :func:`watch_dial_capture.dataset.save_capture`.
        interval_seconds: Delay between successive captures. Clamped up
            to :data:`MIN_CAPTURE_INTERVAL_SECONDS` (one minute) if a
            smaller value is requested, since watches don't need to be
            photographed any more often than that.
        count: Number of *successful* captures to take before returning.
            A failed attempt (e.g. a transient camera glitch) does not
            count against this total; the loop simply retries after
            the usual interval. ``None`` means run forever (until
            interrupted).
        min_radius: Forwarded to
            :func:`watch_dial_capture.detection.detect_watch_regions`.
        max_radius: Forwarded to
            :func:`watch_dial_capture.detection.detect_watch_regions`.
        dp: Forwarded to
            :func:`watch_dial_capture.detection.detect_watch_regions`.
        param1: Forwarded to
            :func:`watch_dial_capture.detection.detect_watch_regions`.
        param2: Forwarded to
            :func:`watch_dial_capture.detection.detect_watch_regions`.
        max_consecutive_failures: Raise :class:`RuntimeError` once this
            many capture attempts in a row have failed (e.g. the camera
            was unplugged), instead of retrying forever. ``None``
            disables this limit.
        sleep_fn: Sleep function, overridable for tests.

    Yields:
        The :class:`~watch_dial_capture.dataset.CaptureResult` for each
        capture, as they happen.
    """
    if interval_seconds < MIN_CAPTURE_INTERVAL_SECONDS:
        logger.warning(
            "Requested interval of %.1fs is below the %.0fs minimum; "
            "using %.0fs instead.",
            interval_seconds,
            MIN_CAPTURE_INTERVAL_SECONDS,
            MIN_CAPTURE_INTERVAL_SECONDS,
        )
        interval_seconds = MIN_CAPTURE_INTERVAL_SECONDS

    captured = 0
    consecutive_failures = 0
    while count is None or captured < count:
        try:
            frame = camera.read()
            detections = detect_watch_regions(
                frame,
                min_radius=min_radius,
                max_radius=max_radius,
                dp=dp,
                param1=param1,
                param2=param2,
            )
            logger.info("Captured frame with %d watch(es) detected", len(detections))
            result = save_capture(frame, detections, output_dir)
        except RECOVERABLE_ERRORS:
            consecutive_failures += 1
            logger.exception(
                "Capture attempt failed (%d consecutive failure(s)); will retry "
                "on the next interval",
                consecutive_failures,
            )
            if (
                max_consecutive_failures is not None
                and consecutive_failures >= max_consecutive_failures
            ):
                raise RuntimeError(
                    f"Capture failed {consecutive_failures} times in a row; giving up"
                ) from None
            sleep_fn(interval_seconds)
            continue

        consecutive_failures = 0
        captured += 1
        yield result
        if count is None or captured < count:
            sleep_fn(interval_seconds)
