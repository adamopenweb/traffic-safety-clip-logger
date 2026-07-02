"""Live(ish) Recorder.run loop test.

Drives the real supervise + indexer + prune loop of ``Recorder.run`` against a
synthetic ffmpeg ``lavfi`` source (no camera). The production command builder
uses strftime ``%s`` for unix-ms names (glibc/Linux); here we monkeypatch in a
portable numeric segment pattern so the *loop wiring* is exercised on any OS.
The real ``build_capture_command`` is covered separately in test_recorder.py.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from traffic_logger.capture import recorder as rec_mod
from traffic_logger.capture.recorder import Recorder
from traffic_logger.capture.segment_index import SegmentIndex
from traffic_logger.config import load_config
from traffic_logger.util.ffmpeg import ffmpeg_available, ffmpeg_path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")


def _portable_command_factory(ffmpeg):
    def fake_build(config, *, incoming_dir, input_args=None, pixel_format=None):
        # 1-second segments named segment_178092359XXXX.mp4 (parseable as ms),
        # incrementing via the segment-index format token (portable on Windows).
        pattern = str(Path(incoming_dir) / "segment_178092359%04d.mp4")
        return [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            # -re throttles the synthetic source to real time (a real camera
            # already delivers frames at native fps, so production omits it).
            "-re", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-g", "10",
            "-f", "segment", "-segment_time", "1", "-segment_format", "mp4",
            "-reset_timestamps", "1",
            pattern,
        ]
    return fake_build


def test_recorder_loop_records_and_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rec_mod, "build_capture_command", _portable_command_factory(ffmpeg_path())
    )

    cfg = load_config(CONFIG_DIR / "config.mini_pc.yaml")
    # Point storage at the temp dir and shorten segments for a fast test.
    cfg.raw["recording"]["ring_path"] = str(tmp_path / "ring")
    cfg.raw["recording"]["segment_index_path"] = str(tmp_path / "index" / "segments.sqlite")
    cfg.raw["recording"]["segment_seconds"] = 1

    recorder = Recorder(cfg, input_args=["-f", "lavfi", "-i", "testsrc"])

    thread = threading.Thread(target=recorder.run, daemon=True)
    thread.start()
    # Let a few 1-second segments be produced and finalized.
    time.sleep(6)
    recorder.request_stop()
    thread.join(timeout=15)
    assert not thread.is_alive()

    index_path = cfg.raw["recording"]["segment_index_path"]
    with SegmentIndex(index_path) as idx:
        segments = idx.get_all()

    # At least one completed segment was indexed with sane metadata.
    assert len(segments) >= 1
    for seg in segments:
        assert Path(seg.path).exists()
        assert seg.codec == "h264"
        assert seg.duration > 0
        assert seg.start_ts < seg.end_ts
        # Stored in a dated folder under the ring root.
        assert "ring" in seg.path
