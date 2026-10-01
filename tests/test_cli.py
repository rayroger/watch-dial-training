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


def test_parser_defaults_match_expected_radius_and_hough_params():
    args = cli.build_arg_parser().parse_args([])

    assert args.min_radius == 150
    assert args.max_radius == 350
    assert args.dp == 1.2
    assert args.param1 == 100
    assert args.param2 == 40


def test_parser_accepts_explicit_radius_and_hough_overrides():
    args = cli.build_arg_parser().parse_args(
        [
            "--min-radius",
            "10",
            "--max-radius",
            "20",
            "--dp",
            "2.0",
            "--param1",
            "50",
            "--param2",
            "75",
        ]
    )

    assert args.min_radius == 10
    assert args.max_radius == 20
    assert args.dp == 2.0
    assert args.param1 == 50
    assert args.param2 == 75


def test_main_threads_radius_and_hough_params_into_capture_loop(monkeypatch, tmp_path):
    captured_kwargs = {}

    class _FakeCamera:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

    def fake_camera_factory(*args, **kwargs):
        return _FakeCamera()

    def fake_run_capture_loop(camera, output_dir, **kwargs):
        captured_kwargs.update(kwargs)
        return iter(())

    monkeypatch.setattr(cli, "Camera", fake_camera_factory)
    monkeypatch.setattr(cli, "run_capture_loop", fake_run_capture_loop)

    assert (
        cli.main(
            [
                "--output-dir",
                str(tmp_path / "unused"),
                "--min-radius",
                "123",
                "--max-radius",
                "321",
                "--dp",
                "1.5",
                "--param1",
                "90",
                "--param2",
                "55",
            ]
        )
        == 0
    )

    assert captured_kwargs["min_radius"] == 123
    assert captured_kwargs["max_radius"] == 321
    assert captured_kwargs["dp"] == 1.5
    assert captured_kwargs["param1"] == 90
    assert captured_kwargs["param2"] == 55
