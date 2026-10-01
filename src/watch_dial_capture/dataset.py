"""Persist captured pictures to disk in a layout suited for model training.

The output directory produced by :func:`save_capture` looks like::

    output_dir/
        annotated/
            20240102-153000-000.jpg          # full picture with boxes/labels
        watches/
            20240102-153000-000_watch_0.jpg  # cropped, per-watch dials
            20240102-153000-000_watch_1.jpg
        metadata/
            20240102-153000-000.json         # detection metadata for this capture

Keeping the annotated (human-review) pictures and metadata separate
from the per-watch crops (the actual TensorFlow training inputs) makes
it easy to point a training pipeline at just the ``watches`` directory
without needing to filter out non-image files.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from .annotate import annotate_image
from .detection import Detection, crop_detections

ANNOTATED_DIR_NAME = "annotated"
WATCHES_DIR_NAME = "watches"
METADATA_DIR_NAME = "metadata"


def save_capture(
    image: np.ndarray,
    detections: Sequence[Detection],
    output_dir: str | Path,
    *,
    timestamp: datetime | None = None,
) -> "CaptureResult":
    """Save an annotated picture plus one cropped image per detected watch.

    Args:
        image: The full picture as captured from the camera (BGR).
        detections: Watch regions found by
            :func:`watch_dial_capture.detection.detect_watch_regions`.
        output_dir: Root directory where ``annotated/``, ``watches/``
            and ``metadata/`` sub-directories will be created.
        timestamp: Capture time used to name the files and to stamp the
            annotated picture. Defaults to ``datetime.now()``.

    Returns:
        A :class:`CaptureResult` describing where each file was written.
    """
    timestamp = timestamp or datetime.now()
    # Include millisecond precision so that multiple watches captured in
    # the same second (or successive runs invoked in quick succession)
    # do not silently overwrite each other's files.
    stamp = timestamp.strftime("%Y%m%d-%H%M%S-%f")[:-3]

    output_dir = Path(output_dir)
    annotated_dir = output_dir / ANNOTATED_DIR_NAME
    watches_dir = output_dir / WATCHES_DIR_NAME
    metadata_dir = output_dir / METADATA_DIR_NAME
    annotated_dir.mkdir(parents=True, exist_ok=True)
    watches_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    written_paths: List[Path] = []
    try:
        annotated_image = annotate_image(image, detections, timestamp=stamp)
        annotated_path = annotated_dir / f"{stamp}.jpg"
        _write_image(annotated_path, annotated_image)
        written_paths.append(annotated_path)

        watch_paths: List[Path] = []
        for index, crop in enumerate(crop_detections(image, detections)):
            watch_path = watches_dir / f"{stamp}_watch_{index}.jpg"
            _write_image(watch_path, crop)
            watch_paths.append(watch_path)
            written_paths.append(watch_path)

        metadata_path = metadata_dir / f"{stamp}.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "timestamp": timestamp.isoformat(),
                    "detections": [asdict(d) for d in detections],
                },
                indent=2,
            )
        )
        written_paths.append(metadata_path)
    except Exception:
        # Don't leave a partially-written capture (e.g. the annotated
        # picture and some, but not all, watch crops) on disk: a
        # training pipeline scanning these directories should never see
        # an incomplete/inconsistent set of files for a given timestamp.
        for path in written_paths:
            path.unlink(missing_ok=True)
        raise

    return CaptureResult(
        annotated_path=annotated_path,
        watch_paths=watch_paths,
        metadata_path=metadata_path,
    )


class CaptureResult:
    """Paths written by a single call to :func:`save_capture`."""

    def __init__(
        self,
        annotated_path: Path,
        watch_paths: List[Path],
        metadata_path: Path,
    ) -> None:
        self.annotated_path = annotated_path
        self.watch_paths = watch_paths
        self.metadata_path = metadata_path


def _write_image(path: Path, image: np.ndarray) -> None:
    if image.size == 0:
        raise ValueError(f"cannot write empty image to {path}")
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"failed to write image to {path}")
