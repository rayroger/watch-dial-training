"""Command-line entry point for periodic watch-dial capture.

Example::

    python -m watch_dial_capture.cli --output-dir ./dataset --interval 30

Run ``python -m watch_dial_capture.cli --help`` for the full list of
options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from .capture import Camera, run_capture_loop


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
        help="Seconds to wait between captures (default: 60).",
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

    with Camera(args.camera_index) as camera:
        try:
            for result in run_capture_loop(
                camera,
                args.output_dir,
                interval_seconds=args.interval,
                count=args.count,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
            ):
                print(f"Saved {result.annotated_path} ({len(result.watch_paths)} watch crop(s))")
        except KeyboardInterrupt:
            print("Capture stopped by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
