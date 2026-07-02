"""Tests for the pure Track / TrackStore history logic (no CV deps)."""

from __future__ import annotations

from traffic_logger.analyze.tracker import (
    LEFT_TO_RIGHT,
    RIGHT_TO_LEFT,
    Observation,
    Track,
    TrackStore,
    bottom_center,
)


def test_bottom_center():
    assert bottom_center((10, 20, 30, 60)) == (20.0, 60.0)


def test_track_records_history_and_age():
    track = Track(track_id=7)
    track.add(Observation(ts=0.0, bbox=(0, 0, 10, 10), confidence=0.9))
    track.add(Observation(ts=0.1, bbox=(2, 0, 12, 10), confidence=0.8))
    assert track.age == 2
    assert track.first_ts == 0.0
    assert track.last_ts == 0.1
    assert track.latest_bbox == (2, 0, 12, 10)
    assert track.latest_confidence == 0.8
    assert track.latest_bottom_center == (7.0, 10.0)
    hist = track.bottom_center_history()
    assert hist[0] == (0.0, 5.0, 10.0)
    assert hist[1] == (0.1, 7.0, 10.0)


def test_direction_left_to_right():
    track = Track(1)
    for i in range(5):
        x = i * 20
        track.add(Observation(ts=i * 0.1, bbox=(x, 0, x + 10, 10), confidence=1.0))
    assert track.direction() == LEFT_TO_RIGHT


def test_direction_right_to_left():
    track = Track(1)
    for i in range(5):
        x = 200 - i * 20
        track.add(Observation(ts=i * 0.1, bbox=(x, 0, x + 10, 10), confidence=1.0))
    assert track.direction() == RIGHT_TO_LEFT


def test_direction_none_when_barely_moving():
    track = Track(1)
    for i in range(5):
        track.add(Observation(ts=i * 0.1, bbox=(0, 0, 10, 10), confidence=1.0))
    assert track.direction() is None
    # single observation -> undecidable
    assert Track(2).direction() is None


def test_track_history_capped():
    track = Track(1, max_history=3)
    for i in range(10):
        track.add(Observation(ts=float(i), bbox=(i, 0, i + 1, 1), confidence=1.0))
    assert track.age == 3
    assert track.first_ts == 7.0  # only the last 3 kept


def test_track_store_update_and_active():
    store = TrackStore()
    store.update([(1, (0, 0, 10, 10), 0.9), (2, (5, 5, 15, 15), 0.8)], ts=0.0)
    store.update([(1, (2, 0, 12, 10), 0.9)], ts=0.1)
    assert len(store) == 2
    assert store.get(1).age == 2
    assert store.get(2).age == 1
    # min_observations filters short tracks; ordered by id.
    active = store.active_tracks(min_observations=2)
    assert [t.track_id for t in active] == [1]


def test_track_store_sweep_evicts_idle_tracks():
    store = TrackStore()
    store.update([(1, (0, 0, 10, 10), 0.9), (2, (5, 5, 15, 15), 0.8)], ts=0.0)
    store.update([(1, (2, 0, 12, 10), 0.9)], ts=100.0)  # track 1 still active at t=100
    # At t=100, track 2 (last seen t=0) is 100s idle; track 1 (last seen t=100) is fresh.
    evicted = store.sweep(now_ts=100.0, max_idle_seconds=60.0)
    assert evicted == [2]
    assert len(store) == 1 and store.get(2) is None and store.get(1) is not None
    # A non-positive threshold disables the sweep (no eviction).
    assert store.sweep(now_ts=1e9, max_idle_seconds=0.0) == []
    assert len(store) == 1
