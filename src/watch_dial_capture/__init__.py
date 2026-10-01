"""watch_dial_capture

A small toolkit for periodically photographing multiple analog wrist
watches, annotating the resulting picture with a bounding box/label per
watch, and extracting each watch dial into its own cropped image. The
collected images are intended to be used as training data for a
TensorFlow model that reads the time from an analog watch face.
"""

from .detection import Detection, detect_watch_regions
from .annotate import annotate_image
from .dataset import save_capture

__all__ = [
    "Detection",
    "detect_watch_regions",
    "annotate_image",
    "save_capture",
]
