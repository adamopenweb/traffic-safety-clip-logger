"""Tests for the lane band model (the pure ratio math shipped in M0)."""

from __future__ import annotations

import pytest

from traffic_logger.analyze import lane_model

DEFAULT_CFG = {
    "bike_lane_width_ratio": 0.12,
    "travel_lane_width_ratio": 0.28,
    "center_lane_width_ratio": 0.20,
}


def test_five_bands_in_order():
    bands = lane_model.normalize_lane_ratios(DEFAULT_CFG)
    names = [name for name, _ in bands]
    assert names == [
        "bike_lane_a",
        "travel_lane_a",
        "center_turn_lane",
        "travel_lane_b",
        "bike_lane_b",
    ]


def test_widths_sum_to_one():
    bands = lane_model.normalize_lane_ratios(DEFAULT_CFG)
    assert sum(w for _, w in bands) == pytest.approx(1.0)


def test_normalization_handles_unnormalized_input():
    # Ratios that don't sum to 1.0 should still normalize cleanly.
    cfg = {
        "bike_lane_width_ratio": 1.0,
        "travel_lane_width_ratio": 2.0,
        "center_lane_width_ratio": 4.0,
    }
    bands = dict(lane_model.normalize_lane_ratios(cfg))
    # total raw = 1 + 2 + 4 + 2 + 1 = 10
    assert bands["center_turn_lane"] == pytest.approx(0.4)
    assert bands["bike_lane_a"] == pytest.approx(0.1)


def test_edges_are_contiguous():
    edges = lane_model.lane_band_edges(DEFAULT_CFG)
    assert edges[0][1] == pytest.approx(0.0)
    assert edges[-1][2] == pytest.approx(1.0)
    # Each band's end equals the next band's start.
    for (_, _, end), (_, start, _) in zip(edges, edges[1:]):
        assert end == pytest.approx(start)


def test_zero_ratios_raise():
    with pytest.raises(ValueError):
        lane_model.normalize_lane_ratios(
            {"bike_lane_width_ratio": 0, "travel_lane_width_ratio": 0,
             "center_lane_width_ratio": 0}
        )


def test_band_widths_override_allows_asymmetric_lanes():
    # Explicit per-band widths (far->near) override the mirrored ratios.
    bands = dict(lane_model.normalize_lane_ratios(
        {"band_widths": [0.05, 0.10, 0.25, 0.35, 0.25]}))
    assert bands["bike_lane_a"] == pytest.approx(0.05)
    assert bands["travel_lane_b"] == pytest.approx(0.35)
    assert bands["bike_lane_b"] == pytest.approx(0.25)
    assert sum(bands.values()) == pytest.approx(1.0)


def test_band_widths_are_normalized():
    bands = dict(lane_model.normalize_lane_ratios({"band_widths": [1, 2, 4, 2, 1]}))
    assert bands["center_turn_lane"] == pytest.approx(0.4)  # 4/10


def test_band_widths_wrong_count_raises():
    with pytest.raises(ValueError):
        lane_model.normalize_lane_ratios({"band_widths": [0.5, 0.5]})


def test_assign_lane_band_maps_positions_to_bands():
    # Default edges: bike 0-.12, travel .12-.40, center .40-.60, travel .60-.88, bike .88-1.0
    assign = lambda nx: lane_model.assign_lane_band(nx, DEFAULT_CFG)
    assert assign(0.05) == "bike_lane_a"
    assert assign(0.20) == "travel_lane_a"
    assert assign(0.50) == "center_turn_lane"
    assert assign(0.70) == "travel_lane_b"
    assert assign(0.95) == "bike_lane_b"


def test_assign_lane_band_off_road_is_none():
    assert lane_model.assign_lane_band(-0.1, DEFAULT_CFG) is None
    assert lane_model.assign_lane_band(1.0, DEFAULT_CFG) is None
    assert lane_model.assign_lane_band(1.5, DEFAULT_CFG) is None


def test_assign_lane_band_boundaries_are_left_closed():
    # Exactly on an edge belongs to the band starting at that edge.
    assert lane_model.assign_lane_band(0.12, DEFAULT_CFG) == "travel_lane_a"
    assert lane_model.assign_lane_band(0.40, DEFAULT_CFG) == "center_turn_lane"
    assert lane_model.assign_lane_band(0.0, DEFAULT_CFG) == "bike_lane_a"
