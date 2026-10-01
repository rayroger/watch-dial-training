import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watch_dial_capture import cli


def test_resolve_camera_backend_maps_friendly_name(monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.cv2, "CAP_DSHOW", 700, raising=False)

    assert cli.resolve_camera_backend("dshow") == 700
    assert cli.resolve_camera_backend("any") == cv2.CAP_ANY


def test_resolve_windows_backend_errors_on_non_windows(monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "linux")

    with pytest.raises(ValueError, match="only available on Windows"):
        cli.resolve_camera_backend("msmf")


class _DiagnosticCapture:
    def __init__(self, index, backend):
        self.index = index
        self.backend = backend
        self.released = False

    def isOpened(self):
        return self.index == 1

    def getBackendName(self):
        return "DSHOW"

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if property_id == cv2.CAP_PROP_EXPOSURE:
            return -1
        return 0

    def release(self):
        self.released = True


def test_list_cameras_probes_bounded_indices_and_skips_capture(
    monkeypatch, tmp_path, capsys
):
    captures = []

    def video_capture(index, backend):
        capture = _DiagnosticCapture(index, backend)
        captures.append(capture)
        return capture

    monkeypatch.setattr(cli.cv2, "VideoCapture", video_capture)
    monkeypatch.setattr(cli, "resolve_camera_backend", lambda _: 123)

    assert cli.main(["--list-cameras", "--output-dir", str(tmp_path / "unused")]) == 0

    assert len(captures) == 10
    assert all(capture.released for capture in captures)
    assert all(capture.backend == 123 for capture in captures)
    assert "Camera 1: open (backend: DSHOW)" in capsys.readouterr().out
    assert not (tmp_path / "unused").exists()


def test_dump_camera_properties_reports_unreadable_values_without_capture(
    monkeypatch, tmp_path, capsys
):
    captures = []

    def video_capture(index, backend):
        capture = _DiagnosticCapture(index, backend)
        capture.isOpened = lambda: True
        captures.append(capture)
        return capture

    monkeypatch.setattr(cli.cv2, "VideoCapture", video_capture)
    monkeypatch.setattr(cli, "resolve_camera_backend", lambda _: 123)

    assert cli.main(
        [
            "--dump-camera-props",
            "--camera-index",
            "1",
            "--output-dir",
            str(tmp_path / "unused"),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "BACKEND: DSHOW" in output
    assert "FRAME_WIDTH: 640" in output
    assert "EXPOSURE: unsupported/unreadable (-1)" in output
    assert captures[0].released
    assert not (tmp_path / "unused").exists()


def test_unsupported_backend_is_a_clear_cli_error():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--camera-backend", "unknown", "--list-cameras"])

    assert exc_info.value.code == 2


def test_preview_and_hough_options_are_parsed():
    args = cli.build_arg_parser().parse_args(
        [
            "--preview",
            "--hough-dp",
            "1.5",
            "--hough-param1",
            "120",
            "--hough-param2",
            "55",
        ]
    )

    assert args.preview
    assert args.hough_dp == 1.5
    assert args.hough_param1 == 120
    assert args.hough_param2 == 55


def test_preview_cancel_skips_capture(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_camera_preview", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        cli, "Camera", lambda *args, **kwargs: pytest.fail("capture should not start")
    )

    assert cli.main(["--preview"]) == 0

    assert "Preview cancelled" in capsys.readouterr().out


def test_accepted_preview_uses_cli_detection_settings(monkeypatch):
    preview_args = {}
    capture_args = {}

    def preview(device_index, **kwargs):
        preview_args["device_index"] = device_index
        preview_args.update(kwargs)
        return True

    class CameraContext:
        def __init__(self, device_index, **kwargs):
            capture_args["device_index"] = device_index
            capture_args.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            pass

    def capture_loop(camera, output_dir, **kwargs):
        capture_args["loop"] = kwargs
        return []

    monkeypatch.setattr(cli, "run_camera_preview", preview)
    monkeypatch.setattr(cli, "Camera", CameraContext)
    monkeypatch.setattr(cli, "run_capture_loop", capture_loop)
    assert (
        cli.main(
            [
                "--preview",
                "--camera-index",
                "2",
                "--camera-backend",
                "any",
                "--frame-width",
                "1280",
                "--frame-height",
                "720",
                "--min-radius",
                "80",
                "--max-radius",
                "220",
                "--hough-dp",
                "1.4",
                "--hough-param1",
                "110",
                "--hough-param2",
                "60",
            ]
        )
        == 0
    )

    assert preview_args["device_index"] == 2
    assert preview_args["frame_width"] == 1280
    assert preview_args["frame_height"] == 720
    assert preview_args["min_radius"] == 80
    assert preview_args["max_radius"] == 220
    assert preview_args["dp"] == 1.4
    assert preview_args["param1"] == 110
    assert preview_args["param2"] == 60
    assert capture_args["loop"]["dp"] == 1.4
    assert capture_args["loop"]["param1"] == 110
    assert capture_args["loop"]["param2"] == 60


def test_camera_diagnostics_do_not_open_preview(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_cameras", lambda backend: [])
    monkeypatch.setattr(
        cli, "dump_camera_properties", lambda *args: [("BACKEND", "fake")]
    )
    monkeypatch.setattr(
        cli,
        "run_camera_preview",
        lambda *args, **kwargs: pytest.fail("diagnostics should skip preview"),
    )

    assert cli.main(["--preview", "--list-cameras"]) == 0
    assert "No cameras found" in capsys.readouterr().out
    assert cli.main(["--preview", "--dump-camera-props"]) == 0
    assert "BACKEND: fake" in capsys.readouterr().out
