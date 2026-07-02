"""Tests for the center-lane passing rule (Patterns A and B)."""

from __future__ import annotations

import pytest

from traffic_logger.analyze.metrics import SpeedEstimator
from traffic_logger.analyze.rules.center_lane_pass import (
    CenterLanePassRule,
    center_dwell_seconds,
    compress_lane_sequence,
    detect_overtake,
)
from traffic_logger.analyze.tracker import LEFT_TO_RIGHT, RIGHT_TO_LEFT, Observation, Track

CL_CFG = {
    "enabled": True,
    "center_lane_min_time_seconds_strict": 0.8,
    "center_lane_min_time_seconds_sensitive": 0.4,
    "speed_percentile_threshold_strict": 0.90,
    "speed_percentile_threshold_sensitive": 0.80,
    "detect_overtake": True,
    "overtake_window_seconds": 6,
}
CENTER = "center_turn_lane"


def _track(track_id, gy_values, gx=0.5, lane=CENTER, dt=0.1):
    """Build a track travelling along the road (normalized y) within a lane."""
    t = Track(track_id)
    for i, gy in enumerate(gy_values):
        t.add(Observation(ts=i * dt, bbox=(95, 0, 105, 10), confidence=0.9,
                          ground_point=(gx, gy), lane_band=lane))
    return t


def _slow_baseline(direction=LEFT_TO_RIGHT, n=20, speed=0.2):
    est = SpeedEstimator(window_seconds=3600)
    for i in range(n):
        est.observe(direction, 0.0, speed, track_id=1000 + i)
    return est


def _rule(aggr=0.3):
    return CenterLanePassRule(CL_CFG, aggressiveness=aggr, cooldown_seconds=8.0, min_track_age=3)


# -- pure helpers ----------------------------------------------------------
def test_compress_lane_sequence():
    assert compress_lane_sequence(["a", "a", "b", None, None, "a"]) == ["a", "b", "off_road", "a"]
    assert compress_lane_sequence([]) == []


def test_center_dwell_seconds():
    hist = [(0.0, "travel_lane_a"), (0.1, CENTER), (0.2, CENTER), (0.3, CENTER)]
    assert center_dwell_seconds(hist) == pytest.approx(0.2)  # 0.3 - 0.1
    # Not currently in center -> no dwell.
    assert center_dwell_seconds(hist + [(0.4, "travel_lane_b")]) == 0.0
    assert center_dwell_seconds([]) == 0.0


def test_detect_overtake_progress():
    cand = [(0.0, 0.1), (1.0, 0.9)]   # starts behind, ends ahead
    other = [(0.0, 0.3), (1.0, 0.5)]
    detected, before, after, n = detect_overtake(cand, other, dir_sign=1, window_seconds=6, ts=1.0)
    assert detected is True and before < 0 < after and n == 2
    # No flip -> not an overtake.
    detected2, *_ = detect_overtake(other, cand, dir_sign=1, window_seconds=6, ts=1.0)
    assert detected2 is False


# -- Pattern A: center-lane pass = center car + same-direction travel companion
def test_center_pass_fires_with_same_direction_travel_companion():
    est, rule = _slow_baseline(), _rule()
    cand = _track(7, [i * 0.1 for i in range(10)])  # center lane, fast along road
    companion = _track(8, [i * 0.05 for i in range(10)], gx=0.2, lane="travel_lane_a")  # same dir, travel
    events = [e for e in rule.evaluate([cand, companion], est, ts=0.9) if e.primary_track_id == 7]
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "center_lane_pass"
    assert ev.evidence["rule"] == "center_lane_pass"
    assert ev.evidence["overtake_detected"] is False
    assert ev.evidence["passed_track_id"] == 8       # the car being passed
    assert ev.evidence["center_lane_time_seconds"] >= rule.center_min_time
    assert ev.evidence["speed_percentile"] >= rule.speed_pct_threshold
    assert 8 in ev.track_ids


def test_no_trigger_for_lone_center_track():
    # A single fast car in center with no same-direction travel companion (e.g. a
    # far-lane car misclassified into the center band) must NOT fire.
    est, rule = _slow_baseline(), _rule()
    track = _track(7, [i * 0.1 for i in range(10)])
    assert rule.evaluate([track], est, ts=0.9) == []


def test_no_trigger_when_companion_is_opposite_direction():
    # An oncoming car in a travel lane (e.g. while a vehicle waits in center to
    # turn) is not a "pass" -- only a same-direction companion counts.
    est, rule = _slow_baseline(), _rule()
    cand = _track(7, [i * 0.1 for i in range(10)])  # center, left_to_right
    oncoming = _track(8, [0.9 - i * 0.05 for i in range(10)], gx=0.2, lane="travel_lane_a")  # right_to_left
    assert rule.evaluate([cand, oncoming], est, ts=0.9) == []


def test_directions_filter_rejects_unlisted_direction():
    # With a directions allow-list, a same-direction center pass in the *other*
    # direction must not fire (right-to-left traffic legitimately uses the turn
    # lane), while a left-to-right one still fires.
    cfg = dict(CL_CFG, directions=["left_to_right"])
    rule = CenterLanePassRule(cfg, aggressiveness=0.3, cooldown_seconds=8.0, min_track_age=3)

    # Right-to-left candidate + companion (gy decreasing) -> rejected.
    est = _slow_baseline(direction=RIGHT_TO_LEFT)
    cand = _track(7, [0.9 - i * 0.1 for i in range(10)])
    comp = _track(8, [0.9 - i * 0.05 for i in range(10)], gx=0.2, lane="travel_lane_a")
    assert rule.evaluate([cand, comp], est, ts=0.9) == []

    # Left-to-right still fires.
    est2 = _slow_baseline(direction=LEFT_TO_RIGHT)
    cand2 = _track(7, [i * 0.1 for i in range(10)])
    comp2 = _track(8, [i * 0.05 for i in range(10)], gx=0.2, lane="travel_lane_a")
    fired = [e for e in rule.evaluate([cand2, comp2], est2, ts=0.9) if e.primary_track_id == 7]
    assert len(fired) == 1


def test_no_trigger_when_not_in_center():
    est, rule = _slow_baseline(), _rule()
    track = _track(7, [i * 0.1 for i in range(10)], lane="travel_lane_a")
    assert rule.evaluate([track], est, ts=0.9) == []


def test_no_trigger_when_dwell_too_short():
    est, rule = _slow_baseline(), _rule()
    # Only the last two obs are in the center lane -> dwell ~0.1s < 0.68s.
    track = Track(7)
    bands = ["travel_lane_a"] * 8 + [CENTER, CENTER]
    for i, lane in enumerate(bands):
        track.add(Observation(ts=i * 0.1, bbox=(95, 0, 105, 10),
                              ground_point=(0.5, i * 0.1), lane_band=lane))
    assert rule.evaluate([track], est, ts=0.9) == []


def test_no_trigger_when_slow():
    # Baseline is fast, so the candidate's speed is a low percentile.
    est = _slow_baseline(speed=5.0)
    rule = _rule()
    track = _track(7, [i * 0.02 for i in range(10)])  # slow, in center
    assert rule.evaluate([track], est, ts=0.9) == []


# -- Pattern B: overtake through the center lane ---------------------------
def test_pattern_b_overtake_is_stronger_event():
    est, rule = _slow_baseline(), _rule()
    cand = _track(1, [0.1 + i * 0.08 for i in range(11)], gx=0.5, lane=CENTER)        # center, passing
    other = _track(2, [0.3 + i * 0.02 for i in range(11)], gx=0.2, lane="travel_lane_a")  # being passed
    events = rule.evaluate([cand, other], est, ts=1.0)

    cand_events = [e for e in events if e.primary_track_id == 1]
    assert len(cand_events) == 1
    ev = cand_events[0]
    assert ev.evidence["rule"] == "center_lane_overtake"
    assert ev.evidence["overtake_detected"] is True
    assert ev.evidence["passed_track_id"] == 2
    assert ev.evidence["relative_position_before"] < 0 < ev.evidence["relative_position_after"]
    assert ev.score >= 0.85
    assert 2 in ev.track_ids
    # The passed vehicle (in a travel lane) is not itself flagged.
    assert all(e.primary_track_id != 2 for e in events)


def test_cooldown_suppresses_repeat():
    est, rule = _slow_baseline(), _rule()
    cand = _track(7, [i * 0.1 for i in range(10)])
    comp = _track(8, [i * 0.05 for i in range(10)], gx=0.2, lane="travel_lane_a")
    tracks = [cand, comp]
    fired = lambda t: [e for e in rule.evaluate(tracks, est, ts=t) if e.primary_track_id == 7]
    assert len(fired(0.9)) == 1
    assert fired(1.0) == []          # within cooldown
    assert len(fired(10.0)) == 1     # after cooldown


def test_thresholds_resolved_by_aggressiveness():
    rule = _rule(aggr=0.3)
    assert rule.center_min_time == pytest.approx(0.68)       # lerp(0.8, 0.4, 0.3)
    assert rule.speed_pct_threshold == pytest.approx(0.87)   # lerp(0.90, 0.80, 0.3)
