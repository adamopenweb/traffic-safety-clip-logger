"""Tests for ring buffer prune selection (pure logic shipped in M0)."""

from __future__ import annotations

import pytest

from traffic_logger.capture.ring_pruner import (
    SegmentInfo,
    prune_ring,
    select_segments_to_delete,
    total_bytes,
)
from traffic_logger.capture.segment_index import SegmentIndex, SegmentRecord

GB = 1024 ** 3


def _seg(ts: float, gb: float, name: str | None = None) -> SegmentInfo:
    return SegmentInfo(path=name or f"seg_{ts}.mp4", start_ts=ts, size_bytes=int(gb * GB))


def test_no_deletion_when_under_cap():
    segs = [_seg(1, 50), _seg(2, 50)]
    assert select_segments_to_delete(segs, max_bytes=200 * GB) == []


def test_deletes_oldest_first_until_under_cap():
    segs = [_seg(3, 80), _seg(1, 80), _seg(2, 80)]  # intentionally unordered
    to_delete = select_segments_to_delete(segs, max_bytes=200 * GB)
    # 240GB total, cap 200GB -> must drop the single oldest (ts=1, 80GB) -> 160GB.
    assert [s.start_ts for s in to_delete] == [1]
    remaining = [s for s in segs if s not in to_delete]
    assert total_bytes(remaining) <= 200 * GB


def test_never_deletes_active_segment():
    segs = [_seg(1, 150, "active.mp4"), _seg(2, 150, "old.mp4")]
    to_delete = select_segments_to_delete(
        segs, max_bytes=200 * GB, active_path="active.mp4"
    )
    # Oldest is the active segment; it must be skipped and the next oldest taken.
    paths = [s.path for s in to_delete]
    assert "active.mp4" not in paths
    assert "old.mp4" in paths


def test_stays_over_cap_rather_than_delete_active():
    # Only segment over the cap is the active one -> nothing can be deleted.
    segs = [_seg(1, 300, "active.mp4")]
    to_delete = select_segments_to_delete(
        segs, max_bytes=200 * GB, active_path="active.mp4"
    )
    assert to_delete == []


def test_prune_ring_deletes_files_and_index_rows(tmp_path):
    # Three real files; cap forces the oldest to be removed.
    index = SegmentIndex(":memory:")
    paths = []
    for i, start in enumerate([100.0, 200.0, 300.0]):
        f = tmp_path / f"segment_{int(start)}.mp4"
        f.write_bytes(b"x" * (80 * 1024 * 1024))  # 80 MB each
        paths.append(f)
        index.add_segment(SegmentRecord(
            path=str(f), start_ts=start, end_ts=start + 10, duration=10.0,
            size_bytes=f.stat().st_size, codec="h264", width=1280, height=960, fps=30.0,
        ))

    result = prune_ring(index, max_bytes=200 * 1024 * 1024)  # 200 MB cap, 240 MB present

    # Oldest (start=100) deleted from disk and index; newer kept.
    assert str(paths[0]) in result.deleted_paths
    assert not paths[0].exists()
    assert paths[1].exists() and paths[2].exists()
    assert index.count() == 2
    assert index.total_bytes() <= 200 * 1024 * 1024
    index.close()


def test_prune_ring_missing_file_still_removes_row(tmp_path):
    index = SegmentIndex(":memory:")
    ghost = tmp_path / "segment_100.mp4"  # never created on disk
    index.add_segment(SegmentRecord(
        path=str(ghost), start_ts=100.0, end_ts=110.0, duration=10.0,
        size_bytes=300 * 1024 * 1024, codec="h264", width=1280, height=960, fps=30.0,
    ))
    result = prune_ring(index, max_bytes=10 * 1024 * 1024)
    assert str(ghost) in result.deleted_paths
    assert index.count() == 0
    index.close()


def test_prune_ring_noop_when_under_cap():
    index = SegmentIndex(":memory:")
    index.add_segment(SegmentRecord(
        path="a.mp4", start_ts=1.0, end_ts=11.0, duration=10.0,
        size_bytes=50, codec="h264", width=1, height=1, fps=30.0,
    ))
    result = prune_ring(index, max_bytes=1000)
    assert result.deleted_paths == []
    assert index.count() == 1
    index.close()
