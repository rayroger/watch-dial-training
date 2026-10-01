"""Command-line entry point for periodic watch-dial capture.

Example::

    python -m watch_dial_capture.cli --output-dir ./dataset --interval 60

Run ``python -m watch_dial_capture.cli --help`` for the full list of
options.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from typing import Optional, Sequence

import cv2

from .capture import (
    Camera,
    DEFAULT_FOCUS_WARMUP_FRAMES,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    MIN_CAPTURE_INTERVAL_SECONDS,
    run_capture_loop,
)

CAMERA_BACKENDS = {
    "any": ("CAP_ANY", False),
    "dshow": ("CAP_DSHOW", True),
    "msmf": ("CAP_MSMF", True),
}

CAMERA_PROPERTY_NAMES = (
    "FRAME_WIDTH",
    "FRAME_HEIGHT",
    "FPS",
    "FOURCC",
    "BRIGHTNESS",
    "CONTRAST",
    "SATURATION",
    "HUE",
    "GAIN",
    "EXPOSURE",
    "AUTO_EXPOSURE",
    "FOCUS",
    "AUTOFOCUS",
    "ZOOM",
)


def resolve_camera_backend(name: str) -> int:
    """Resolve a friendly backend name to an available OpenCV constant."""
    constant_name, windows_only = CAMERA_BACKENDS[name]
    if windows_only and sys.platform != "win32":
        raise ValueError(f"Camera backend '{name}' is only available on Windows")
    backend = getattr(cv2, constant_name, None)
    if backend is None:
        raise ValueError(
            f"Camera backend '{name}' is not available in this OpenCV build"
        )
    return backend


def _camera_backend_name(capture) -> str:
    try:
        return capture.getBackendName()
    except (AttributeError, cv2.error):
        return "unknown"


def list_cameras(backend: int, max_indices: int = 10) -> list[tuple[int, str]]:
    """Probe a bounded range of camera indices, releasing every opened handle."""
    cameras = []
    for index in range(max_indices):
        capture = cv2.VideoCapture(index, backend)
        try:
            if capture.isOpened():
                cameras.append((index, _camera_backend_name(capture)))
        finally:
            capture.release()
    return cameras


def dump_camera_properties(device_index: int, backend: int) -> list[tuple[str, str]]:
    """Read a useful set of common properties from one camera."""
    capture = cv2.VideoCapture(device_index, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera at index {device_index}")

    properties = [("BACKEND", _camera_backend_name(capture))]
    for name in CAMERA_PROPERTY_NAMES:
        property_id = getattr(cv2, f"CAP_PROP_{name}", None)
        if property_id is None:
            properties.append((name, "unavailable in this OpenCV build"))
            continue
        try:
            value = capture.get(property_id)
        except cv2.error:
            properties.append((name, "unsupported/unreadable"))
            continue
        if not math.isfinite(value) or value < 0:
            rendered = f"unsupported/unreadable ({value})"
        elif name == "FOURCC":
            code = int(value)
            fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
            rendered = f"{fourcc} ({code})" if fourcc.isprintable() else str(value)
        else:
            rendered = str(value)
        properties.append((name, rendered))
    capture.release()
    return properties


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Periodically photograph one or more analog wrist watches, "
            "annotate the detected watch faces, and save per-watch crops "
            "for use as TensorFlow training data."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="./captures",
        help="Directory where annotated pictures and watch crops are saved.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of the camera device to use (default: 0).",
    )
    parser.add_argument(
        "--camera-backend",
        choices=CAMERA_BACKENDS,
        default="any",
        help="OpenCV camera backend: any (default), dshow, or msmf (Windows only).",
    )
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe camera indices 0–9 with the selected backend and exit.",
    )
    diagnostics.add_argument(
        "--dump-camera-props",
        action="store_true",
        help="Print common properties for the selected camera and exit.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help=(
            "Seconds to wait between captures (default: 60). Values below "
            f"{MIN_CAPTURE_INTERVAL_SECONDS:.0f} are clamped up to that "
            "minimum: watches don't need to be photographed more than once "
            "a minute."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of captures to take before exiting. Omit to run forever.",
    )
    parser.add_argument(
        "--min-radius",
        type=int,
        default=150,
        help="Smallest expected watch-face radius in pixels (default: 150).",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=350,
        help="Largest expected watch-face radius in pixels (default: 350).",
    )
    parser.add_argument(
        "--dp",
        type=float,
        default=1.2,
        help=(
            "Inverse ratio of the Hough-circle accumulator resolution "
            "(default: 1.2), forwarded to cv2.HoughCircles."
        ),
    )
    parser.add_argument(
        "--param1",
        type=float,
        default=100,
        help=(
            "Higher Canny edge-detector threshold (default: 100), "
            "forwarded to cv2.HoughCircles."
        ),
    )
    parser.add_argument(
        "--param2",
        type=float,
        default=40,
        help=(
            "Accumulator threshold for circle centers (default: 40); "
            "raise this to reject weaker/accidental circular edges "
            "(e.g. cables, shadows) and reduce false positives, "
            "forwarded to cv2.HoughCircles."
        ),
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=5,
        help=(
            "Stop with an error after this many capture attempts in a row "
            "fail, e.g. if the camera is disconnected (default: 5). Use 0 "
            "to retry forever."
        ),
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help=(
            "Requested capture width in pixels (default: "
            f"{DEFAULT_FRAME_WIDTH}). Set higher than any webcam actually "
            "supports to make it use its maximum (photo-mode) resolution "
            "instead of a lower-res video/preview stream."
        ),
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help=f"Requested capture height in pixels (default: {DEFAULT_FRAME_HEIGHT}).",
    )
    parser.add_argument(
        "--focus-warmup-frames",
        type=int,
        default=DEFAULT_FOCUS_WARMUP_FRAMES,
        help=(
            "Number of frames to discard before keeping one, giving the "
            f"camera's autofocus time to settle (default: "
            f"{DEFAULT_FOCUS_WARMUP_FRAMES})."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        backend = resolve_camera_backend(args.camera_backend)
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_cameras:
        cameras = list_cameras(backend)
        if cameras:
            for index, backend_name in cameras:
                print(f"Camera {index}: open (backend: {backend_name})")
        else:
            print("No cameras found for the selected backend.")
        return 0

    if args.dump_camera_props:
        try:
            properties = dump_camera_properties(args.camera_index, backend)
        except RuntimeError as exc:
            parser.error(str(exc))
        for name, value in properties:
            print(f"{name}: {value}")
        return 0

    with Camera(
        args.camera_index,
        backend=backend,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        focus_warmup_frames=args.focus_warmup_frames,
    ) as camera:
        try:
            for result in run_capture_loop(
                camera,
                args.output_dir,
                interval_seconds=args.interval,
                count=args.count,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
                dp=args.dp,
                param1=args.param1,
                param2=args.param2,
                max_consecutive_failures=args.max_consecutive_failures or None,
            ):
                print(f"Saved {result.annotated_path} ({len(result.watch_paths)} watch crop(s))")
        except KeyboardInterrupt:
            print("Capture stopped by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
