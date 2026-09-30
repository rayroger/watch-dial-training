# watch-dial-training

A small desktop (Python) application that periodically photographs one
or more running analog wrist watches, annotates each detected watch
face on the picture, and crops out each watch into its own image.
The resulting pictures/crops are intended to be used as training data
for a TensorFlow model that reads the time from an analog watch dial.

## How it works

1. A frame is grabbed from a webcam (or any camera supported by
   OpenCV) at a configurable interval.
2. Circular watch faces in the frame are located with a Hough-circle
   transform (`watch_dial_capture.detection`) — no trained model is
   required to bootstrap the very first training set.
3. The full frame is annotated with a circle/label per detected watch
   plus a timestamp (`watch_dial_capture.annotate`).
4. The annotated picture and one cropped image per detected watch are
   saved to disk (`watch_dial_capture.dataset`), e.g.:

   ```
   captures/
     annotated/20240102-153000.jpg          # full picture, for review
     watches/20240102-153000_watch_0.jpg    # per-watch crop, for training
     watches/20240102-153000_watch_1.jpg
     watches/20240102-153000_metadata.json  # detection metadata
   ```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m watch_dial_capture.cli \
  --output-dir ./captures \
  --interval 30 \
  --camera-index 0
```

Run `python -m watch_dial_capture.cli --help` for all options
(capture interval, camera index, expected watch radius range, number
of captures to take, etc.).

## Tests

```bash
pip install -r requirements.txt
pip install pytest
python -m pytest tests/
```
