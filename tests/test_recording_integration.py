"""End-to-end recording integration test.

Exercises the real ffprobe -> index -> prune path on genuine MP4 files produced
by ffmpeg's synthetic ``lavfi`` source, so no camera is required (spec: "Do not
require live camera for automated tests"). Skipped if ffmpeg/ffprobe are absent.
"""

from __future__ import annotations

import subprocess

import pytest

from traffic_logger.capture.recorder import finalize_segment
from traffic_logger.capture.ring_pruner import prune_ring
from traffic_logger.capture.segment_index import SegmentIndex
from traffic_logger.util.ffmpeg import (
    ffmpeg_available,
    ffmpeg_path,
    ffprobe_available,
)

pytestmark = pytest.mark.skipif(
    not (ffmpeg_available() and ffprobe_available()),
    reason="ffmpeg/ffprobe not available",
)


def _make_segment(path, seconds=1):
    """Generate a real ~`seconds`-long H.264 MP4 with ffmpeg lavfi."""
    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"testsrc=size=320x240:rate=10:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_finalize_indexes_real_segment(tmp_path):
    incoming = tmp_path / "ring" / "incoming"
    incoming.mkdir(parents=True)
    ring_root = tmp_path / "ring"

    # start ts 1780923590000 ms -> 2026-06-08 (UTC) date folder.
    seg = incoming / "segment_1780923590000.mp4"
    _make_segment(seg, seconds=1)

    index = SegmentIndex(":memory:")
    record = finalize_segment(index, seg, ring_root, timezone="UTC")

    assert record is not None
    # File was moved out of incoming into a dated ring folder and indexed.
    assert not seg.exists()
    assert record.path.endswith("segment_1780923590000.mp4")
    assert "2026-06-08" in record.path
    assert record.codec == "h264"
    assert record.width == 320 and record.height == 240
    assert record.duration > 0
    assert record.start_ts == pytest.approx(1780923590.0)
    assert record.end_ts > record.start_ts
    assert index.count() == 1
    index.close()


def test_record_index_prune_end_to_end(tmp_path):
    incoming = tmp_path / "ring" / "incoming"
    incoming.mkdir(parents=True)
    ring_root = tmp_path / "ring"
    index = SegmentIndex(":memory:")

    # Three real segments with increasing start timestamps.
    records = []
    for i in range(3):
        start_ms = 1780923590000 + i * 10000
        seg = incoming / f"segment_{start_ms}.mp4"
        _make_segment(seg, seconds=1)
        rec = finalize_segment(index, seg, ring_root, timezone="UTC")
        assert rec is not None
        records.append(rec)

    assert index.count() == 3
    total = index.total_bytes()
    assert total > 0

    # Cap just below total -> only the oldest segment is pruned.
    cap = total - records[0].size_bytes // 2
    result = prune_ring(index, max_bytes=cap)

    assert len(result.deleted_paths) >= 1
    assert records[0].path in result.deleted_paths
    # Oldest file is gone from disk; index reflects the deletion.
    from pathlib import Path
    assert not Path(records[0].path).exists()
    assert index.total_bytes() <= cap
    index.close()
