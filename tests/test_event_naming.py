"""Tests for descriptive event-clip filenames (descriptor_tokens / event_stem)."""

from __future__ import annotations

from traffic_logger.events.metadata import descriptor_tokens, event_stem


def _ev(**kw):
    return kw


def test_speeding_tokens_full():
    evs = [_ev(rule="absolute_speeding", speed_kmh=86.1, direction="left_to_right",
               vehicle_type="car")]
    assert descriptor_tokens(evs) == ["86kmh", "car", "LtR"]


def test_speed_rounds_and_takes_max():
    evs = [
        _ev(rule="absolute_speeding", speed_kmh=56.4, direction="right_to_left"),
        _ev(rule="absolute_speeding", speed_kmh=57.8, direction="right_to_left"),
    ]
    assert descriptor_tokens(evs) == ["58kmh", "RtL"]  # max, rounded


def test_unknown_vehicle_type_omitted():
    evs = [_ev(rule="absolute_speeding", speed_kmh=60.0, vehicle_type="unknown",
               direction="left_to_right")]
    assert descriptor_tokens(evs) == ["60kmh", "LtR"]


def test_non_speeding_event_keeps_dir_and_type():
    evs = [_ev(track_id=1, direction="left_to_right", vehicle_type="car",
               lane_sequence=["center"])]  # center-lane: no speed_kmh
    assert descriptor_tokens(evs) == ["car", "LtR"]


def test_empty_evidence_yields_no_tokens():
    assert descriptor_tokens([]) == []
    assert descriptor_tokens([{}]) == []


def test_event_stem_layout():
    evs = [_ev(rule="absolute_speeding", speed_kmh=86.1, direction="left_to_right",
               vehicle_type="car")]
    stem = event_stem("20260620_082834", "relative_speeding", "9f77faa0", evs)
    assert stem == "20260620_082834_86kmh_car_LtR_relative_speeding_9f77faa0"


def test_event_stem_fallback_when_no_tokens():
    stem = event_stem("20260620_073011", "center_lane_pass", "77be1234", [{}])
    assert stem == "20260620_073011_center_lane_pass_77be1234"
