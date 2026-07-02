"""Tests for clip export, thumbnail, and metadata."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from traffic_logger.util.ffmpeg import (
    ffmpeg_available,
    ffmpeg_path,
    ffprobe_available,
    ffprobe_segment,
)

_HAS_FF = ffmpeg_available() and ffprobe_available()
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _make_video(path, seconds=5):
    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# -- clip window (pure) ----------------------------------------------------
def test_clip_window():
    from traffic_logger.events.exporter import clip_window

    w = clip_window(100.0, pre_roll=10, post_roll=20)
    assert (w.start, w.trigger, w.end) == (90.0, 100.0, 120.0)
    assert w.duration == 30.0


@pytest.mark.skipif(not _HAS_FF, reason="ffmpeg/ffprobe not available")
def test_export_from_source(tmp_path):
    from traffic_logger.events.exporter import export_from_source

    video = tmp_path / "src.mp4"
    _make_video(video, seconds=5)
    out = export_from_source(video, start_offset=1.0, duration=2.0, out_path=tmp_path / "clip.mp4")
    assert out.exists() and out.stat().st_size > 0
    assert ffprobe_segment(out)["duration"] == pytest.approx(2.0, abs=0.4)


@pytest.mark.skipif(not _HAS_FF, reason="ffmpeg/ffprobe not available")
def test_export_from_segments(tmp_path):
    from traffic_logger.capture.segment_index import SegmentRecord
    from traffic_logger.events.exporter import export_from_segments

    segs = []
    for i in range(3):
        p = tmp_path / f"segment_{i}.mp4"
        _make_video(p, seconds=2)
        segs.append(SegmentRecord(
            path=str(p), start_ts=100.0 + i * 2, end_ts=102.0 + i * 2, duration=2.0,
            size_bytes=p.stat().st_size, codec="h264", width=160, height=120, fps=10.0,
        ))
    # Window spanning the middle of segment 0 through the middle of segment 2.
    out = export_from_segments(segs, abs_start=101.0, abs_end=104.0, out_path=tmp_path / "clip.mp4")
    assert out.exists()
    assert ffprobe_segment(out)["duration"] == pytest.approx(3.0, abs=0.6)


@pytest.mark.skipif(not _HAS_FF, reason="ffmpeg not available")
def test_generate_thumbnail(tmp_path):
    from traffic_logger.events.thumbnail import generate_thumbnail

    video = tmp_path / "src.mp4"
    _make_video(video, seconds=5)
    thumb = generate_thumbnail(video, tmp_path / "t.jpg", offset_seconds=2.0)
    assert thumb.exists() and thumb.stat().st_size > 0


# -- metadata (pure) -------------------------------------------------------
def test_build_metadata_matches_schema():
    from traffic_logger.analyze.rules.base import CandidateEvent
    from traffic_logger.config import load_config
    from traffic_logger.events.manager import FinalEvent
    from traffic_logger.events.metadata import build_metadata

    cand = CandidateEvent(
        event_type="center_lane_pass", trigger_ts=100.0, primary_track_id=42, score=0.9,
        evidence={"candidate_track_id": 42, "direction": "left_to_right",
                  "lane_sequence": ["travel_lane_a", "center_turn_lane"],
                  "speed_percentile": 0.96, "passed_track_id": 7},
        track_ids=[42, 7],
    )
    fe = FinalEvent(
        event_id="abc12345", event_type="center_lane_pass",
        event_types=["center_lane_pass", "relative_speeding"], trigger_ts=100.0,
        score=0.9, primary_track_id=42, track_ids=[7, 42], candidates=[cand],
    )
    cfg = load_config(CONFIG_DIR / "config.dev.yaml")
    meta = build_metadata(fe, "/x/clip.mp4", "/x/t.jpg", config=cfg,
                          created_at="2026-06-08T13:00:00-04:00",
                          start_ts=90.0, trigger_ts=100.0, end_ts=120.0)

    assert meta["event_id"] == "abc12345"
    assert meta["event_type"] == "center_lane_pass"
    assert set(meta["event_types"]) == {"center_lane_pass", "relative_speeding"}
    assert meta["clip_path"] == "/x/clip.mp4"
    assert meta["thumbnail_path"] == "/x/t.jpg"
    assert meta["primary_track_id"] == 42
    assert meta["start_ts"] == 90.0 and meta["end_ts"] == 120.0
    track = next(t for t in meta["tracks"] if t["track_id"] == 42)
    assert track["direction"] == "left_to_right"
    assert track["lane_band_sequence"][-1] == "center_turn_lane"
    assert track["speed"]["percentile"] == 0.96
    assert meta["evidence"]["triggers"][0]["event_type"] == "center_lane_pass"
    assert meta["config_snapshot"]["clip_total_seconds"] == 30


def test_write_metadata_roundtrip(tmp_path):
    import json

    from traffic_logger.events.metadata import write_metadata

    out = write_metadata({"event_id": "x", "score": 0.5}, tmp_path / "e.json")
    assert json.loads(out.read_text(encoding="utf-8"))["event_id"] == "x"
