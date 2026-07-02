"""Tests for the SQLite segment index."""

from __future__ import annotations

from traffic_logger.capture.segment_index import SegmentIndex, SegmentRecord


def _rec(path: str, start: float, dur: float = 10.0, size: int = 100) -> SegmentRecord:
    return SegmentRecord(
        path=path, start_ts=start, end_ts=start + dur, duration=dur,
        size_bytes=size, codec="h264", width=1280, height=960, fps=30.0,
    )


def test_add_and_get_roundtrip():
    with SegmentIndex(":memory:") as idx:
        rec = _rec("/data/ring/2026-06-08/segment_1780923590000.mp4", 1780923590.0)
        idx.add_segment(rec)
        got = idx.get_segment(rec.path)
        assert got == rec
        assert idx.has(rec.path)
        assert idx.count() == 1


def test_get_all_ordered_oldest_first():
    with SegmentIndex(":memory:") as idx:
        idx.add_segment(_rec("c.mp4", 30))
        idx.add_segment(_rec("a.mp4", 10))
        idx.add_segment(_rec("b.mp4", 20))
        order = [s.path for s in idx.get_all()]
        assert order == ["a.mp4", "b.mp4", "c.mp4"]


def test_total_bytes_and_delete():
    with SegmentIndex(":memory:") as idx:
        idx.add_segment(_rec("a.mp4", 10, size=100))
        idx.add_segment(_rec("b.mp4", 20, size=250))
        assert idx.total_bytes() == 350
        idx.delete_segment("a.mp4")
        assert idx.total_bytes() == 250
        assert not idx.has("a.mp4")
        assert idx.count() == 1


def test_add_is_idempotent_on_path():
    with SegmentIndex(":memory:") as idx:
        idx.add_segment(_rec("a.mp4", 10, size=100))
        idx.add_segment(_rec("a.mp4", 10, size=999))  # same path -> replace
        assert idx.count() == 1
        assert idx.total_bytes() == 999


def test_get_overlapping_window():
    with SegmentIndex(":memory:") as idx:
        idx.add_segment(_rec("a.mp4", 0, dur=10))    # [0, 10]
        idx.add_segment(_rec("b.mp4", 10, dur=10))   # [10, 20]
        idx.add_segment(_rec("c.mp4", 20, dur=10))   # [20, 30]
        overlapping = [s.path for s in idx.get_overlapping(12, 22)]
        assert overlapping == ["b.mp4", "c.mp4"]


def test_persists_to_disk(tmp_path):
    db = tmp_path / "idx" / "segments.sqlite"
    with SegmentIndex(db) as idx:
        idx.add_segment(_rec("a.mp4", 10))
    assert db.exists()
    with SegmentIndex(db) as idx2:
        assert idx2.count() == 1
