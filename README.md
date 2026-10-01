# watch-dial-training

A small desktop (Python) application that periodically photographs one
or more running analog wrist watches, annotates each detected watch
face on the picture, and crops out each watch into its own image.
The resulting pictures/crops are intended to be used as training data
for a TensorFlow model that reads the time from an analog watch dial.

## How it works

1. A still photo is captured from a webcam (or any camera supported by
   OpenCV) at a configurable interval — at most once a minute, since
   watches don't need to be photographed any more often than that.
   Rather than treating the camera as a live video stream, the
   capture requests the device's highest supported resolution and
   discards a few warm-up frames before keeping one, giving the
   camera's autofocus/auto-exposure time to settle on the watches
   (`watch_dial_capture.capture.Camera`).
2. Circular watch faces in the frame are located with a Hough-circle
   transform (`watch_dial_capture.detection`) — no trained model is
   required to bootstrap the very first training set.
3. The full frame is annotated with a circle/label per detected watch
   plus a timestamp (`watch_dial_capture.annotate`).
4. The annotated picture and one cropped image per detected watch are
   saved to disk (`watch_dial_capture.dataset`), e.g.:

   ```
   captures/
     annotated/20240102-153000-000.jpg          # full picture, for review
     watches/20240102-153000-000_watch_0.jpg    # per-watch crop, for training
     watches/20240102-153000-000_watch_1.jpg
     metadata/20240102-153000-000.json          # detection metadata
   ```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m watch_dial_capture.cli \
  --output-dir ./captures \
  --interval 60 \
  --camera-index 0
```

Run `python -m watch_dial_capture.cli --help` for all options
(capture interval, camera index, expected watch radius range, number
of captures to take, photo-mode resolution and autofocus warm-up,
etc.).

### Choosing and inspecting a camera (Windows)

List camera indices available through the automatic OpenCV backend:

```powershell
python -m watch_dial_capture.cli --list-cameras
```

To use camera index 0 or 1 with DirectShow (DSHOW) or Media Foundation
(MSMF), select the index and backend explicitly:

```powershell
python -m watch_dial_capture.cli --camera-index 0 --camera-backend dshow --output-dir ./captures
python -m watch_dial_capture.cli --camera-index 1 --camera-backend msmf --output-dir ./captures
```

List cameras through a specific backend, or inspect common properties for
one camera without starting capture:

```powershell
python -m watch_dial_capture.cli --camera-backend dshow --list-cameras
python -m watch_dial_capture.cli --camera-index 0 --camera-backend dshow --dump-camera-props
```

Property reporting depends on the OpenCV backend and camera device; an
unsupported or unreadable value is marked in the output. Native driver
settings may expose more controls and capability details than OpenCV's
`get()`/`set()` interface.

## Tests

```bash
pip install -r requirements.txt
pip install pytest
python -m pytest tests/
```
