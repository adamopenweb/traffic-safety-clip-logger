"""Tests for the capture health check."""

from __future__ import annotations

from traffic_logger.capture.health import max_segment_age, recording_health
from traffic_logger.capture.segment_index import SegmentIndex, SegmentRecord


def test_max_segment_age_floor_and_scaling():
    assert max_segment_age(10) == 60.0          # 3*10+30 = 60, floored at 60
    assert max_segment_age(30) == 120.0         # 3*30+30 = 120


def test_recording_health_no_segments():
    healthy, reason = recording_health(None, now=1000.0, max_age_seconds=60)
    assert healthy is False
    assert "no segments" in reason


def test_recording_health_fresh():
    healthy, reason = recording_health(latest_end_ts=970.0, now=1000.0, max_age_seconds=60)
    assert healthy is True


def test_recording_health_stale():
    healthy, reason = recording_health(latest_end_ts=900.0, now=1000.0, max_age_seconds=60)
    assert healthy is False
    assert "old" in reason


def test_segment_index_latest_end_ts():
    with SegmentIndex(":memory:") as idx:
        assert idx.latest_end_ts() is None
        idx.add_segment(SegmentRecord("a.mp4", 10.0, 20.0, 10.0, 100))
        idx.add_segment(SegmentRecord("b.mp4", 30.0, 41.0, 11.0, 100))
        assert idx.latest_end_ts() == 41.0
