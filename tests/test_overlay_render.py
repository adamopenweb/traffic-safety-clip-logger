"""Annotation labels: one stable speed per car (median), gate speed for the primary."""

from __future__ import annotations

from traffic_logger.events.overlay_buffer import OverlayBox, OverlaySnapshot
from traffic_logger.events.overlay_render import _box_label, aggregate_track_speeds


def _box(tid, kmh):
    return OverlayBox(track_id=tid, bbox=(0.0, 0.0, 10.0, 10.0), speed_kmh=kmh)


def _snap(ts, *boxes):
    return OverlaySnapshot(ts=ts, boxes=tuple(boxes))


def test_aggregate_is_per_track_median():
    snaps = [
        _snap(0.0, _box(1, 60.0), _box(2, 40.0)),
        _snap(0.1, _box(1, 90.0), _box(2, 42.0)),   # track 1 momentarily spikes to 90
        _snap(0.2, _box(1, 64.0), _box(2, 41.0)),
    ]
    agg = aggregate_track_speeds(snaps)
    assert agg[1] == 64.0    # median of 60/90/64 -> the spike is ignored
    assert agg[2] == 41.0


def test_aggregate_ignores_missing_speeds():
    snaps = [_snap(0.0, _box(1, None)), _snap(0.1, _box(1, 50.0))]
    assert aggregate_track_speeds(snaps) == {1: 50.0}
    assert aggregate_track_speeds([_snap(0.0, _box(9, None))]) == {}   # all None -> no entry


def test_box_label_override_then_fallback():
    b = _box(7, 55.0)
    assert _box_label(b, 72.0) == "#7 72km/h"     # override (the gate speed) wins
    assert _box_label(b) == "#7 55km/h"           # else the box's own speed
    rel = OverlayBox(track_id=8, bbox=(0, 0, 1, 1), speed_kmh=None, speed_rel=0.5)
    assert _box_label(rel) == "#8 v=0.50"         # relative fallback when no km/h
