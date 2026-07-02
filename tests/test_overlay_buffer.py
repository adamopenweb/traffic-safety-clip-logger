"""Tests for the annotation overlay buffer and snapshot selection (Approach B)."""

from __future__ import annotations

import pytest

from traffic_logger.events.overlay_buffer import (
    OverlayBox,
    OverlayBuffer,
    OverlaySnapshot,
    deserialize_snapshots,
    nearest_snapshot,
    serialize_snapshots,
)


def _box(track_id=1):
    return OverlayBox(track_id=track_id, bbox=(10.0, 20.0, 30.0, 40.0), speed_rel=0.3)


def test_append_and_slice_window():
    buf = OverlayBuffer(capacity_seconds=100.0)
    for t in range(10):
        buf.append(float(t), [_box(t)])
    window = buf.slice(3.0, 6.0)
    assert [s.ts for s in window] == [3.0, 4.0, 5.0, 6.0]
    # Boxes are carried through.
    assert window[0].boxes[0].track_id == 3


def test_capacity_evicts_old_snapshots():
    buf = OverlayBuffer(capacity_seconds=5.0)
    for t in range(20):
        buf.append(float(t), [_box(t)])
    # Newest ts is 19; anything with ts < 14 must have been evicted.
    assert len(buf) == 6  # ts 14..19 inclusive
    assert buf.slice(0.0, 13.0) == []


def test_frame_size_set_once():
    buf = OverlayBuffer(capacity_seconds=10.0)
    assert buf.frame_size is None
    buf.set_frame_size(704, 480)
    buf.set_frame_size(1920, 1080)  # ignored once set
    assert buf.frame_size == (704, 480)


def test_nearest_snapshot_picks_closest_within_tolerance():
    snaps = [OverlaySnapshot(ts=float(t), boxes=()) for t in (0.0, 1.0, 2.0)]
    assert nearest_snapshot(snaps, 1.1, tolerance=0.2).ts == 1.0
    assert nearest_snapshot(snaps, 0.9, tolerance=0.2).ts == 1.0
    # Exactly between -> the earlier one (tie resolves to the lower index probe).
    assert nearest_snapshot(snaps, 1.5, tolerance=0.6).ts in (1.0, 2.0)


def test_nearest_snapshot_returns_none_outside_tolerance():
    snaps = [OverlaySnapshot(ts=0.0, boxes=()), OverlaySnapshot(ts=5.0, boxes=())]
    assert nearest_snapshot(snaps, 2.5, tolerance=0.2) is None
    assert nearest_snapshot([], 1.0, tolerance=1.0) is None


def test_snapshot_serialize_roundtrip():
    snaps = [
        OverlaySnapshot(ts=1.5, boxes=(
            OverlayBox(track_id=7, bbox=(1.0, 2.0, 3.0, 4.0), speed_kmh=58.0,
                       speed_rel=0.8, lane="center_turn_lane", direction="left_to_right"),
            OverlayBox(track_id=8, bbox=(5.0, 6.0, 7.0, 8.0)),
        )),
        OverlaySnapshot(ts=2.0, boxes=()),
    ]
    out = deserialize_snapshots(serialize_snapshots(snaps))
    assert out == snaps  # frozen dataclasses compare by value


def test_search_offset_recovers_known_shift():
    from traffic_logger.events.overlay_render import _search_offset

    start_ts = 1000.0
    true_delta = 0.60

    # The primary's projected centre sweeps in x over its on-screen second.
    def center(ts):
        return (100.0 + 600.0 * (ts - (start_ts + 9.0)), 200.0)

    t = start_ts + 9.0
    primary_track = []
    while t <= start_ts + 10.5:
        primary_track.append((t, center(t)))
        t += 0.03

    # 4K detections: the real car (== primary centre at clip_time+true_delta) plus
    # a far-off distractor car the matcher must ignore.
    samples = []
    ct = 8.7
    while ct <= 9.6:
        car = center(start_ts + ct + true_delta)
        samples.append((ct, [car, (5000.0, 5000.0)]))
        ct += 0.15

    est = _search_offset(samples, primary_track, start_ts=start_ts,
                         search_center=0.75, radius=0.6, step=0.05, gate_px=400.0)
    assert abs(est - true_delta) <= 0.05


def test_search_offset_falls_back_when_unmatched():
    from traffic_logger.events.overlay_render import _search_offset

    # No detections in any sample -> nothing matches -> return the search centre.
    primary_track = [(1000.0 + i * 0.05, (float(i), 0.0)) for i in range(40)]
    samples = [(float(t), []) for t in range(5)]
    assert _search_offset(samples, primary_track, start_ts=1000.0,
                          search_center=0.7, radius=0.5, step=0.05) == 0.7


def test_stream_projector_scale_and_center():
    pytest.importorskip("cv2")
    from traffic_logger.events.overlay_render import StreamProjector

    # No distortion -> stage 1 is identity, stage 2 is pure resolution scale.
    proj = StreamProjector((704, 480), (3840, 2160), k1=0.0)
    # Frame centre maps to frame centre.
    cx, cy = proj.project(352.0, 240.0)
    assert abs(cx - 1920) <= 1 and abs(cy - 1080) <= 1
    # A sub point scales by the resolution ratio.
    x, y = proj.project(176.0, 120.0)
    assert abs(x - 176.0 * 3840 / 704) <= 1
    assert abs(y - 120.0 * 2160 / 480) <= 1
    # project_bbox returns 4 corners.
    assert len(proj.project_bbox((10.0, 20.0, 30.0, 40.0))) == 4
