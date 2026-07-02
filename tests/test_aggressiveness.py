"""Tests for the aggressiveness -> threshold mapping."""

from __future__ import annotations

import pytest

from traffic_logger.util.aggressiveness import clamp01, lerp, resolve_thresholds


def test_lerp_endpoints_and_midpoint():
    assert lerp(0.97, 0.90, 0.0) == pytest.approx(0.97)  # strict
    assert lerp(0.97, 0.90, 1.0) == pytest.approx(0.90)  # sensitive
    assert lerp(0.97, 0.90, 0.5) == pytest.approx(0.935)


def test_lerp_clamps_out_of_range_aggressiveness():
    assert lerp(0.8, 0.4, -1.0) == pytest.approx(0.8)
    assert lerp(0.8, 0.4, 2.0) == pytest.approx(0.4)


def test_clamp01():
    assert clamp01(-0.2) == 0.0
    assert clamp01(0.5) == 0.5
    assert clamp01(1.7) == 1.0


def test_resolve_thresholds_collapses_pairs():
    section = {
        "enabled": True,
        "percentile_threshold_strict": 0.97,
        "percentile_threshold_sensitive": 0.90,
        "min_duration_seconds_strict": 0.8,
        "min_duration_seconds_sensitive": 0.4,
        "min_tracks_for_baseline": 20,
    }
    resolved = resolve_thresholds(section, aggressiveness=0.0)
    # Pairs collapse to a single base key...
    assert resolved["percentile_threshold"] == pytest.approx(0.97)
    assert resolved["min_duration_seconds"] == pytest.approx(0.8)
    # ...unpaired keys pass through unchanged, and the _strict/_sensitive
    # keys are consumed (not left behind).
    assert resolved["enabled"] is True
    assert resolved["min_tracks_for_baseline"] == 20
    assert "percentile_threshold_strict" not in resolved
    assert "percentile_threshold_sensitive" not in resolved


def test_resolve_thresholds_sensitive_end():
    section = {
        "speed_percentile_threshold_strict": 0.90,
        "speed_percentile_threshold_sensitive": 0.80,
    }
    resolved = resolve_thresholds(section, aggressiveness=1.0)
    assert resolved["speed_percentile_threshold"] == pytest.approx(0.80)
