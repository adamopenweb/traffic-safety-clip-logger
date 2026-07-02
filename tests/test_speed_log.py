"""Tests for the lightweight speed log + event speed extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from traffic_logger.events.speed_log import (
    SpeedLog,
    SpeedRecord,
    event_speed_and_direction,
)
from traffic_logger.events.speed_report import violation_from_record


def test_speedlog_add_and_window():
    with SpeedLog(":memory:") as log:
        log.add(SpeedRecord(ts=1000.0, speed_kmh=57.0, direction="left_to_right",
                            clipped=False, vehicle_type="car"))
        log.add(SpeedRecord(ts=2000.0, speed_kmh=68.0, direction="right_to_left",
                            clipped=True, vehicle_type="truck"))
        log.add(SpeedRecord(ts=3000.0, speed_kmh=60.0, direction=None, clipped=False))
        assert len(log.in_window(0, 5000)) == 3
        mid = log.in_window(1500, 5000)
        assert [r.speed_kmh for r in mid] == [68.0, 60.0]
        assert mid[0].clipped is True and mid[0].vehicle_type == "truck"


def test_violation_from_record():
    rec = SpeedRecord(ts=10.0, speed_kmh=63.0, direction="left_to_right", clipped=True)
    v = violation_from_record(rec, limit_kmh=50.0)
    assert v.speed_kmh == 63.0 and v.over_limit_kmh == 13.0 and v.clipped is True


# --- event_speed_and_direction (reads FinalEvent candidate evidence) ----------

@dataclass
class _Cand:
    evidence: dict


@dataclass
class _FE:
    candidates: List[_Cand]


def test_event_speed_picks_absolute_trigger_max():
    fe = _FE([
        _Cand({"rule": "absolute_speeding", "speed_kmh": 58.0, "direction": "left_to_right"}),
        _Cand({"rule": "absolute_speeding", "speed_kmh": 64.0, "direction": "left_to_right",
               "vehicle_type": "truck"}),
    ])
    assert event_speed_and_direction(fe) == (64.0, "left_to_right", "truck")


def test_event_speed_none_for_non_speeding():
    fe = _FE([_Cand({"rule": "center_lane_pass", "track_id": 1})])
    assert event_speed_and_direction(fe) is None
    assert event_speed_and_direction(_FE([])) is None
