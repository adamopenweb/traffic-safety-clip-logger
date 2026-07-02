"""Tests for the event manager's dedup/merge logic."""

from __future__ import annotations

from traffic_logger.analyze.rules.base import CandidateEvent
from traffic_logger.events.manager import EventManager


def _cand(event_type, ts, tid, score=0.5, extra_tracks=None):
    return CandidateEvent(
        event_type=event_type, trigger_ts=ts, primary_track_id=tid, score=score,
        evidence={"track_id": tid}, track_ids=[tid] + (extra_tracks or []),
    )


def _ids():
    counter = {"n": 0}

    def factory():
        counter["n"] += 1
        return f"e{counter['n'] - 1}"

    return factory


def test_same_track_within_window_merges_types():
    m = EventManager(merge_window_seconds=12, id_factory=_ids())
    m.add(_cand("relative_speeding", 10.0, 42, score=0.6), 10.0)
    m.add(_cand("center_lane_pass", 12.0, 42, score=0.9), 12.0)
    finals = m.flush_all()

    assert len(finals) == 1
    fe = finals[0]
    assert set(fe.event_types) == {"relative_speeding", "center_lane_pass"}
    assert fe.event_type == "center_lane_pass"   # highest score wins primary
    assert fe.score == 0.9
    assert fe.primary_track_id == 42
    assert len(fe.candidates) == 2
    assert fe.event_id == "e0"


def test_different_tracks_stay_separate():
    m = EventManager(merge_window_seconds=12, id_factory=_ids())
    m.add(_cand("relative_speeding", 10.0, 1), 10.0)
    m.add(_cand("relative_speeding", 11.0, 2), 11.0)
    finals = m.flush_all()
    assert {fe.primary_track_id for fe in finals} == {1, 2}
    assert len(finals) == 2


def test_same_track_far_apart_separate():
    m = EventManager(merge_window_seconds=12, id_factory=_ids())
    m.add(_cand("relative_speeding", 10.0, 42), 10.0)
    m.add(_cand("relative_speeding", 30.0, 42), 30.0)  # 20s later > 12s window
    finals = m.flush_all()
    assert len(finals) == 2


def test_overtake_shared_track_id_merges():
    # Center-lane overtake lists the passed vehicle in track_ids; a speeding
    # event on that passed vehicle shares the track and merges.
    m = EventManager(merge_window_seconds=12, id_factory=_ids())
    m.add(_cand("center_lane_pass", 10.0, 42, score=0.9, extra_tracks=[7]), 10.0)
    m.add(_cand("relative_speeding", 11.0, 7, score=0.5), 11.0)
    finals = m.flush_all()
    assert len(finals) == 1
    assert finals[0].track_ids == [7, 42]


def test_flush_respects_merge_window_timing():
    m = EventManager(merge_window_seconds=12, id_factory=_ids())
    m.add(_cand("relative_speeding", 10.0, 42), 10.0)
    assert m.flush(15.0) == []          # 5s < window, still open
    ready = m.flush(25.0)               # 15s > window, finalize
    assert len(ready) == 1
    assert m.flush(30.0) == []          # nothing left
