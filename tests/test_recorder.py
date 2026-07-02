"""Tests for the recorder's pure command-building and finalize helpers."""

from __future__ import annotations

from pathlib import Path

from traffic_logger.capture import recorder
from traffic_logger.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _config():
    return load_config(CONFIG_DIR / "config.mini_pc.yaml")


def test_build_input_args_maps_pixel_format():
    args = recorder.build_input_args(
        {"source": "/dev/video0", "capture_resolution": [1280, 960],
         "capture_fps": 30, "pixel_format_preference": ["YUYV", "MJPG"]}
    )
    assert "-f" in args and "v4l2" in args
    assert "-input_format" in args
    assert args[args.index("-input_format") + 1] == "yuyv422"  # YUYV -> yuyv422
    assert "1280x960" in args
    assert args[-1] == "/dev/video0"


def test_build_input_args_mjpg_and_rgb():
    a = recorder.build_input_args({"pixel_format_preference": ["MJPG"]})
    assert a[a.index("-input_format") + 1] == "mjpeg"
    b = recorder.build_input_args({"pixel_format_preference": ["RGB3"]})
    assert b[b.index("-input_format") + 1] == "rgb24"


def test_build_capture_command_reencodes_raw_input(tmp_path):
    cfg = _config()
    cfg.raw["camera"]["pixel_format_preference"] = ["YUYV"]   # raw -> must re-encode
    cmd = recorder.build_capture_command(cfg, incoming_dir=tmp_path)
    assert "libx264" in cmd
    assert "-f" in cmd and "segment" in cmd
    seg_seconds = str(cfg.recording["segment_seconds"])
    assert cmd[cmd.index("-segment_time") + 1] == seg_seconds
    assert "-strftime" in cmd
    assert "-an" in cmd  # no audio
    # Output pattern writes unix-ms-named segments into the incoming dir.
    assert cmd[-1].endswith("segment_%Y%m%d-%H%M%S.mp4")
    assert str(tmp_path) in cmd[-1]


def test_build_capture_command_copies_h264_camera_input(tmp_path):
    # When the camera supplies H.264 (hardware encoder), stream-copy it so the
    # mini-PC does no CPU encoding (the cause of frozen-frame captures).
    cfg = _config()
    cfg.raw["camera"]["pixel_format_preference"] = ["H264", "MJPG"]
    cmd = recorder.build_capture_command(cfg, incoming_dir=tmp_path)
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"
    assert "libx264" not in cmd
    assert "-input_format" in cmd and cmd[cmd.index("-input_format") + 1] == "h264"
    # Still segmented with unix-ms names.
    assert "segment" in cmd and cmd[-1].endswith("segment_%Y%m%d-%H%M%S.mp4")


def test_encode_mode_force_overrides(tmp_path):
    cfg = _config()
    cfg.raw["camera"]["pixel_format_preference"] = ["H264"]
    cfg.raw["recording"]["encode_mode"] = "h264"   # force re-encode even of H264 input
    cmd = recorder.build_capture_command(cfg, incoming_dir=tmp_path)
    assert "libx264" in cmd

    cfg.raw["recording"]["encode_mode"] = "copy"    # force copy
    cfg.raw["camera"]["pixel_format_preference"] = ["MJPG"]
    cmd2 = recorder.build_capture_command(cfg, incoming_dir=tmp_path)
    assert cmd2[cmd2.index("-c") + 1] == "copy"


def test_is_rtsp_url():
    assert recorder.is_rtsp_url("rtsp://cam/stream") is True
    assert recorder.is_rtsp_url("rtmp://host/live") is True
    assert recorder.is_rtsp_url("http://host:8000/s.mjpg") is True
    assert recorder.is_rtsp_url("/dev/video0") is False
    assert recorder.is_rtsp_url("samples/x.mp4") is False


def test_build_rtsp_input_args_uses_tcp():
    args = recorder.build_rtsp_input_args("rtsp://u:p@cam:554/main")
    assert args == ["-rtsp_transport", "tcp", "-i", "rtsp://u:p@cam:554/main"]


def test_build_capture_command_records_rtsp_stream_copy(tmp_path):
    # Single-box: record the camera's 4K main RTSP stream by stream-copy. The
    # recording.source overrides camera.source (analyzer's sub-stream).
    cfg = _config()
    cfg.raw["recording"]["source"] = "rtsp://u:p@cam:554/cam/realmonitor?channel=1&subtype=0"
    cfg.raw["recording"]["encode_mode"] = "copy"
    cmd = recorder.build_capture_command(cfg, incoming_dir=tmp_path)
    assert cmd[cmd.index("-rtsp_transport") + 1] == "tcp"
    assert cmd[cmd.index("-i") + 1].endswith("subtype=0")
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert "libx264" not in cmd
    assert "v4l2" not in cmd  # network input, not a v4l2 device
    assert "segment" in cmd and cmd[-1].endswith("segment_%Y%m%d-%H%M%S.mp4")


def test_rtsp_auto_mode_copies_but_honors_forced_reencode(tmp_path):
    cfg = _config()
    cfg.raw["recording"]["source"] = "rtsp://cam:554/main"
    cfg.raw["recording"]["encode_mode"] = "auto"  # network stream already encoded
    cmd = recorder.build_capture_command(cfg, incoming_dir=tmp_path)
    assert cmd[cmd.index("-c") + 1] == "copy"

    cfg.raw["recording"]["encode_mode"] = "h264"  # explicit re-encode honored
    cmd2 = recorder.build_capture_command(cfg, incoming_dir=tmp_path)
    assert "libx264" in cmd2


def test_build_capture_command_accepts_synthetic_input(tmp_path):
    cfg = _config()
    cmd = recorder.build_capture_command(
        cfg, incoming_dir=tmp_path,
        input_args=["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10"],
    )
    assert "lavfi" in cmd
    assert "/dev/video0" not in " ".join(cmd)


def test_parse_start_ms():
    assert recorder.parse_start_ms("segment_1780923590000.mp4") == 1780923590000
    assert recorder.parse_start_ms("/x/y/segment_42.mp4") == 42
    assert recorder.parse_start_ms("not_a_segment.mp4") is None
    assert recorder.parse_start_ms("segment_abc.mp4") is None


def test_parse_start_ms_portable_datetime():
    # Portable local-wall-clock form (used when recording natively on Windows).
    import datetime as _dt

    expected = int(_dt.datetime(2026, 6, 14, 13, 12, 25).timestamp() * 1000)
    assert recorder.parse_start_ms("segment_20260614-131225.mp4") == expected
    assert recorder.parse_start_ms("/x/segment_20260614-131225.mp4") == expected


def test_parse_start_ms_decodes_in_configured_tz():
    # With an explicit tz the wall-clock stamp is bound to THAT zone (not the host's),
    # so the epoch is correct regardless of the host tz. 12:00 UTC on 2026-07-01:
    import datetime as _dt
    from zoneinfo import ZoneInfo

    expected = int(_dt.datetime(2026, 7, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC")).timestamp() * 1000)
    assert recorder.parse_start_ms("segment_20260701-120000.mp4", tz="UTC") == expected
    # The legacy unix-ms form ignores tz (already absolute).
    assert recorder.parse_start_ms("segment_1780923590000.mp4", tz="UTC") == 1780923590000


def test_completed_segments_excludes_newest():
    files = [
        Path("segment_300.mp4"),
        Path("segment_100.mp4"),
        Path("segment_200.mp4"),
    ]
    completed = recorder.completed_segments(files)
    # Newest (300) is still being written -> excluded; rest in time order.
    assert [p.name for p in completed] == ["segment_100.mp4", "segment_200.mp4"]


def test_completed_segments_single_file_is_active():
    assert recorder.completed_segments([Path("segment_100.mp4")]) == []
    assert recorder.completed_segments([]) == []
