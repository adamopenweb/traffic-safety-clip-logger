"""Tests for the v4l2-ctl output parser and format selection."""

from __future__ import annotations

from traffic_logger.capture.camera_probe import parse_v4l2_formats, select_format

# Representative `v4l2-ctl --list-formats-ext` output for an Astra-like camera.
SAMPLE = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'YUYV' (YUYV 4:2:2)
\t\tSize: Discrete 1280x960
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\t\tInterval: Discrete 0.067s (15.000 fps)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t[1]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 1280x960
\t\t\tInterval: Discrete 0.033s (30.000 fps)
"""


def test_parse_formats_structure():
    formats = parse_v4l2_formats(SAMPLE)
    assert [f.pixel_format for f in formats] == ["YUYV", "MJPG"]

    yuyv = formats[0]
    assert yuyv.description.startswith("YUYV")
    assert len(yuyv.sizes) == 2
    assert yuyv.sizes[0]["width"] == 1280 and yuyv.sizes[0]["height"] == 960
    assert 30.0 in yuyv.sizes[0]["fps"]
    assert 15.0 in yuyv.sizes[0]["fps"]
    assert yuyv.sizes[1]["width"] == 640


def test_parse_empty():
    assert parse_v4l2_formats("") == []


def test_select_format_prefers_first_available():
    formats = parse_v4l2_formats(SAMPLE)
    # YUYV is first preference and supports 1280x960@30 -> chosen.
    assert select_format(formats, ["YUYV", "MJPG"], [1280, 960], 30) == "YUYV"


def test_select_format_falls_through_to_supported():
    formats = parse_v4l2_formats(SAMPLE)
    # RGB3 not available -> falls through to MJPG which supports 1280x960@30.
    assert select_format(formats, ["RGB3", "MJPG"], [1280, 960], 30) == "MJPG"


def test_select_format_resolution_mismatch_returns_none():
    formats = parse_v4l2_formats(SAMPLE)
    assert select_format(formats, ["YUYV"], [9999, 9999], 30) is None
