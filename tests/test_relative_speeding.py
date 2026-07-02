"""Tests for the relative speeding rule."""

from __future__ import annotations

import pytest

from traffic_logger.analyze.metrics import SpeedEstimator
from traffic_logger.analyze.rules.relative_speeding import RelativeSpeedingRule
from traffic_logger.analyze.tracker import LEFT_TO_RIGHT, Observation, Track

RS_CFG = {
    "enabled": True,
    "percentile_threshold_strict": 0.97,
    "percentile_threshold_sensitive": 0.90,
    "min_duration_seconds_strict": 0.8,
    "min_duration_seconds_sensitive": 0.4,
    "min_tracks_for_baseline": 5,
    "rolling_window_minutes": 60,
}


def _track(track_id: int, age: int = 5) -> Track:
    t = Track(track_id)
    for i in range(age):
        t.add(Observation(ts=i * 0.1, bbox=(0, 0, 10, 10), ground_point=(0.5, i * 0.1)))
    return t


def _baseline(distinct: int = 20) -> SpeedEstimator:
    """A spread of normal speeds (0.5..1.5) from `distinct` separate tracks."""
    est = SpeedEstimator(window_seconds=3600)
    for i in range(distinct):
        speed = 0.5 + i * (1.0 / max(1, distinct - 1))
        est.observe(LEFT_TO_RIGHT, ts=0.0, speed=speed, track_id=1000 + i)
    return est


def _rule(aggr: float = 0.3, min_tracks: int = 5) -> RelativeSpeedingRule:
    cfg = dict(RS_CFG, min_tracks_for_baseline=min_tracks)
    return RelativeSpeedingRule(cfg, aggressiveness=aggr, cooldown_seconds=8.0, min_track_age=3)


def test_evidence_includes_kmh_in_metric_mode():
    est = _baseline()
    # 10 m across x 40 m along per normalized unit.
    rule = RelativeSpeedingRule(
        dict(RS_CFG, min_tracks_for_baseline=5), aggressiveness=0.3,
        cooldown_seconds=8.0, min_track_age=3, meters_per_unit=(10.0, 40.0),
    )
    track = _track(42)
    rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.0)
    events = rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.8)
    assert events[0].evidence["speed_kmh"] is not None
    assert events[0].evidence["speed_kmh"] > 0


def test_evidence_kmh_none_in_relative_mode():
    est, rule = _baseline(), _rule()   # no meters_per_unit
    track = _track(42)
    rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.0)
    events = rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.8)
    assert events[0].evidence["speed_kmh"] is None


def test_thresholds_resolved_by_aggressiveness():
    rule = _rule(aggr=0.3)
    # lerp(0.97, 0.90, 0.3) and lerp(0.8, 0.4, 0.3)
    assert rule.percentile_threshold == pytest.approx(0.949)
    assert rule.min_duration == pytest.approx(0.68)


def test_fast_vehicle_triggers_with_evidence():
    est, rule = _baseline(), _rule()
    track = _track(42)

    # First call arms the persistence timer (duration 0 < min_duration).
    assert rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.0) == []
    # After min_duration the over-threshold condition emits a candidate.
    events = rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.8)
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "relative_speeding"
    assert ev.primary_track_id == 42
    assert ev.evidence["percentile"] == pytest.approx(1.0)
    assert ev.evidence["direction"] == LEFT_TO_RIGHT
    assert ev.evidence["warmup"] is False
    assert ev.evidence["duration_seconds"] >= rule.min_duration
    assert ev.evidence["rolling_median"] is not None
    assert ev.score > 0


def test_normal_speed_does_not_trigger():
    est, rule = _baseline(), _rule()
    track = _track(7)
    # Median-ish speed -> ~0.5 percentile, below threshold.
    assert rule.evaluate(track, 1.0, LEFT_TO_RIGHT, est, ts=10.0) == []
    assert rule.evaluate(track, 1.0, LEFT_TO_RIGHT, est, ts=11.0) == []


def test_no_trigger_before_min_duration():
    est, rule = _baseline(), _rule()
    track = _track(7)
    rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.0)        # arm
    # Still inside the persistence window -> nothing yet.
    assert rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.3) == []


def test_cooldown_suppresses_repeat_then_reopens():
    est, rule = _baseline(), _rule()
    track = _track(7)
    rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.0)
    assert len(rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.8)) == 1
    # Within cooldown (8s) -> suppressed.
    assert rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=12.0) == []
    # After cooldown -> emits again.
    assert len(rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=20.0)) == 1


def test_warmup_lowers_confidence():
    # distinct tracks (10) below min_tracks_for_baseline (50) -> warmup.
    est, rule = _baseline(distinct=10), _rule(min_tracks=50)
    track = _track(7)
    rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.0)
    events = rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.8)
    assert len(events) == 1
    ev = events[0]
    assert ev.evidence["warmup"] is True
    assert ev.score == pytest.approx(1.0 * 0.6)  # WARMUP_SCORE_FACTOR


def _abs_rule(threshold, **kw):
    return RelativeSpeedingRule(
        dict(RS_CFG, absolute_kmh_threshold=threshold, absolute_min_duration_seconds=0.3),
        aggressiveness=0.3, cooldown_seconds=8.0, min_track_age=3,
        meters_per_unit=(10.0, 40.0), **kw)


def test_absolute_gate_triggers_over_threshold():
    est = _baseline()
    rule = _abs_rule(100)            # _track moves ~144 km/h at this scale
    track = _track(42, age=12)       # enough history for the steady estimate
    assert rule.evaluate(track, 1.0, LEFT_TO_RIGHT, est, ts=10.0) == []   # arms timer
    events = rule.evaluate(track, 1.0, LEFT_TO_RIGHT, est, ts=10.4)
    assert len(events) == 1
    ev = events[0]
    assert ev.evidence["rule"] == "absolute_speeding"
    assert ev.evidence["speed_kmh"] > 100
    assert ev.evidence["threshold_kmh"] == 100
    assert ev.evidence["over_by_kmh"] > 0
    assert ev.score > 0.5


def test_absolute_gate_below_threshold_no_trigger():
    est = _baseline()
    rule = _abs_rule(200)            # ~144 km/h < 200 -> never fires
    track = _track(42, age=12)
    assert rule.evaluate(track, 1.0, LEFT_TO_RIGHT, est, ts=10.0) == []
    assert rule.evaluate(track, 1.0, LEFT_TO_RIGHT, est, ts=10.4) == []


def test_absolute_threshold_ignored_without_metric_mode():
    # Threshold set but no meters_per_unit -> falls back to the percentile rule.
    est = _baseline()
    rule = RelativeSpeedingRule(
        dict(RS_CFG, absolute_kmh_threshold=50), aggressiveness=0.3,
        cooldown_seconds=8.0, min_track_age=3)  # no meters_per_unit
    track = _track(42)
    rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.0)
    events = rule.evaluate(track, 2.0, LEFT_TO_RIGHT, est, ts=10.8)
    assert events and events[0].evidence["rule"] == "relative_speeding"


def test_unknown_direction_or_thin_baseline_skips():
    est, rule = _baseline(), _rule()
    track = _track(7)
    assert rule.evaluate(track, 2.0, None, est, ts=10.0) == []        # no direction
    assert rule.evaluate(track, None, LEFT_TO_RIGHT, est, ts=10.0) == []  # no speed
    thin = SpeedEstimator(3600)
    thin.observe(LEFT_TO_RIGHT, 0.0, 1.0, track_id=1)  # only 1 sample (<5)
    assert rule.evaluate(track, 9.0, LEFT_TO_RIGHT, thin, ts=10.0) == []
