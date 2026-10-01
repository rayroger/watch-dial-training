"""Command-line entry point for periodic watch-dial capture.

Example::

    python -m watch_dial_capture.cli --output-dir ./dataset --interval 60

Run ``python -m watch_dial_capture.cli --help`` for the full list of
options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from .capture import (
    Camera,
    DEFAULT_FOCUS_WARMUP_FRAMES,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    MIN_CAPTURE_INTERVAL_SECONDS,
    run_capture_loop,
)


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
        default=40,
        help="Smallest expected watch-face radius in pixels (default: 40).",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=400,
        help="Largest expected watch-face radius in pixels (default: 400).",
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

    with Camera(
        args.camera_index,
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
                max_consecutive_failures=args.max_consecutive_failures or None,
            ):
                print(f"Saved {result.annotated_path} ({len(result.watch_paths)} watch crop(s))")
        except KeyboardInterrupt:
            print("Capture stopped by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
