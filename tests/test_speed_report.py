"""Tests for the speeding-report aggregation (pure logic, no filesystem)."""

from __future__ import annotations

from traffic_logger.events.speed_report import (
    Violation,
    aggregate,
    violation_from_metadata,
)


def _meta(rule, speed, over=None, direction="left_to_right", ts=1000.0):
    ev = {"rule": rule, "direction": direction}
    if speed is not None:
        ev["speed_kmh"] = speed
    if over is not None:
        ev["over_by_kmh"] = over
    return {"trigger_ts": ts, "clip_path": "data/events/x.mp4",
            "evidence": {"triggers": [{"evidence": ev}]}}


def test_violation_from_absolute_event():
    v = violation_from_metadata(_meta("absolute_speeding", 56.1, over=1.1), limit_kmh=50.0)
    assert v is not None
    assert v.speed_kmh == 56.1
    assert v.over_limit_kmh == 6.1   # speed - posted limit, not the gate margin
    assert v.direction == "left_to_right"


def test_legacy_relative_event_is_ignored():
    assert violation_from_metadata(_meta("relative_speeding", 80.0), 50.0) is None
    assert violation_from_metadata({"evidence": {"triggers": []}}, 50.0) is None


def test_picks_highest_speed_trigger():
    meta = {"trigger_ts": 1.0, "clip_path": "c.mp4", "evidence": {"triggers": [
        {"evidence": {"rule": "absolute_speeding", "speed_kmh": 58.0, "direction": "a"}},
        {"evidence": {"rule": "absolute_speeding", "speed_kmh": 64.0, "direction": "a"}},
    ]}}
    assert violation_from_metadata(meta, 50.0).speed_kmh == 64.0


def test_aggregate_distribution_and_worst():
    vs = [
        Violation(ts=100.0, speed_kmh=57.0, over_limit_kmh=7.0, direction="left_to_right",
                  clipped=False, vehicle_type="car"),
        Violation(ts=200.0, speed_kmh=62.0, over_limit_kmh=12.0, direction="left_to_right",
                  clipped=True, vehicle_type="truck"),
        Violation(ts=300.0, speed_kmh=72.0, over_limit_kmh=22.0, direction="right_to_left",
                  clipped=True, vehicle_type="car"),
    ]
    st = aggregate(vs, "America/Toronto", top=2)
    assert st.count == 3
    assert st.max_kmh == 72.0
    assert dict(st.by_speed_bin)["55-59"] == 1
    assert dict(st.by_speed_bin)["60-64"] == 1
    assert dict(st.by_speed_bin)["70-79"] == 1
    assert dict(st.by_direction)["left_to_right"] == 2
    assert dict(st.by_vehicle_type)["car"] == 2 and dict(st.by_vehicle_type)["truck"] == 1
    assert [v.speed_kmh for v in st.worst] == [72.0, 62.0]  # top 2, fastest first


def test_aggregate_empty():
    st = aggregate([], "America/Toronto")
    assert st.count == 0 and st.worst == []
