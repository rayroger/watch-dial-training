"""Live camera preview for framing and tuning watch detection."""

from __future__ import annotations

import time

import cv2

from .detection import Detection, detect_watch_regions

PREVIEW_WINDOW_NAME = "Watch dial preview"
DETECTION_INTERVAL_SECONDS = 0.1


def run_camera_preview(
    device_index: int,
    *,
    backend: int,
    frame_width: int,
    frame_height: int,
    min_radius: int,
    max_radius: int,
    dp: float,
    param1: float,
    param2: float,
) -> bool:
    """Show live detections; return True to continue to capture, False to quit."""
    capture = None
    try:
        capture = cv2.VideoCapture(device_index, backend)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera at index {device_index}")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cv2.namedWindow(PREVIEW_WINDOW_NAME)

        detections: list[Detection] = []
        last_detection_time = float("-inf")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    "Failed to read frame from camera during preview"
                )

            now = time.monotonic()
            if now - last_detection_time >= DETECTION_INTERVAL_SECONDS:
                detections = detect_watch_regions(
                    frame,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    dp=dp,
                    param1=param1,
                    param2=param2,
                )
                last_detection_time = now

            display_frame = frame.copy()
            height, width = display_frame.shape[:2]
            for detection in detections:
                cv2.circle(
                    display_frame,
                    (detection.center_x, detection.center_y),
                    detection.radius,
                    (0, 255, 0),
                    2,
                )
                x1, y1, x2, y2 = detection.bounding_box
                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (min(x2, width - 1), min(y2, height - 1)),
                    (255, 0, 0),
                    1,
                )
            cv2.putText(
                display_frame,
                (
                    f"Detected watches: {len(detections)}  |  "
                    "Enter/Space: continue  |  Q/Esc: quit"
                ),
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.imshow(PREVIEW_WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return False
            if key in (10, 13, 32):
                return True
    except cv2.error as exc:
        raise RuntimeError(
            "Could not open or display the camera preview. Install a GUI-enabled "
            "OpenCV build and ensure a display is available."
        ) from exc
    finally:
        if capture is not None:
            try:
                capture.release()
            except cv2.error:
                pass
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
