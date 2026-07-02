"""Tests for speed estimation and per-direction baselines (pure math)."""

from __future__ import annotations

import pytest

from traffic_logger.analyze.metrics import (
    DirectionSpeedBaseline,
    SpeedEstimator,
    across_speed_factor,
    metric_scale,
    speed_kmh,
    speed_kmh_calibrated,
    steady_speed_kmh,
    track_speed,
)


_ACROSS_CAL = {
    "speed_across_correction": {
        "enabled": True,
        "near_gx": 0.71, "near_factor": 0.925,
        "far_gx": 0.16, "far_factor": 0.667,
    }
}


def _gh_at(gx):
    return [(0.0, gx, 0.0), (0.1, gx, 0.2)]


def test_across_factor_disabled_or_empty():
    assert across_speed_factor(_gh_at(0.71), {}) == 1.0
    off = {"speed_across_correction": {"enabled": False, "near_gx": 0.7, "near_factor": 0.9,
                                       "far_gx": 0.1, "far_factor": 0.6}}
    assert across_speed_factor(_gh_at(0.71), off) == 1.0
    assert across_speed_factor([], _ACROSS_CAL) == 1.0


def test_across_factor_at_anchors_and_midpoint():
    assert across_speed_factor(_gh_at(0.71), _ACROSS_CAL) == pytest.approx(1 / 0.925)
    assert across_speed_factor(_gh_at(0.16), _ACROSS_CAL) == pytest.approx(1 / 0.667)
    mid_factor = (0.925 + 0.667) / 2
    assert across_speed_factor(_gh_at((0.71 + 0.16) / 2), _ACROSS_CAL) == pytest.approx(1 / mid_factor)


def test_across_factor_clamps_outside_anchor_range():
    # gx beyond the near anchor clamps to the near factor (no wild extrapolation).
    assert across_speed_factor(_gh_at(0.95), _ACROSS_CAL) == pytest.approx(1 / 0.925)
    assert across_speed_factor(_gh_at(0.0), _ACROSS_CAL) == pytest.approx(1 / 0.667)


def test_steady_speed_kmh_constant_track():
    gh = [(i * 0.1, 0.0, i * 0.1) for i in range(12)]  # 1.0 unit/s along
    assert steady_speed_kmh(gh, (10.0, 40.0), {}) == pytest.approx(144.0)  # 40 m/s


def test_steady_speed_kmh_rejects_jitter_peak():
    # Steady ~1.0 unit/s with one spiked frame; the trimmed end-to-end ignores it,
    # unlike a short rolling window which would peak on the spike.
    gh = [(i * 0.1, 0.0, i * 0.1) for i in range(12)]
    gh[6] = (0.6, 0.0, 0.95)  # mid spike
    steady = steady_speed_kmh(gh, (1.0, 1.0), {})
    assert steady == pytest.approx(3.6, abs=0.2)  # ~1.0 unit/s -> 3.6 km/h, not the spike


def test_steady_speed_kmh_needs_history():
    assert steady_speed_kmh([(0.0, 0.0, 0.0), (0.1, 0.0, 0.1)], (10.0, 40.0), {}) is None


def test_steady_speed_kmh_recovers_from_endpoint_jump():
    # ~1.0 unit/s clean; the first RETAINED point leaps back, which would inflate the
    # end-to-end estimate (the real-world 111 km/h artifact). The robust per-frame
    # median pulls it back to the true speed instead of reporting the inflated value.
    gh = [(i / 30.0, 0.5, 0.05 + 0.55 * i / 17) for i in range(18)]
    true = steady_speed_kmh(gh, (1.0, 1.0), {})
    gh[1] = (gh[1][0], 0.5, gh[1][2] - 0.45)  # endpoint detection jump -> inflates e2e
    assert steady_speed_kmh(gh, (1.0, 1.0), {}) == pytest.approx(true, rel=0.1)


def test_steady_speed_kmh_rejects_impossible_span():
    # A track whose retained endpoints span >1.6x the calibrated road can't be a real
    # crossing (the car left the field of view) -- a multi-frame jump. Rejected.
    gh = [(i / 30.0, 0.16, i * (1.95 / 13)) for i in range(15)]  # spans ~1.95 lengths
    assert steady_speed_kmh(gh, (11.46, 14.94), {}) is None
    # a genuinely fast car spanning ONE length (a clean full crossing) is kept
    ok = [(i / 30.0, 0.16, i * (1.0 / 13)) for i in range(15)]
    assert steady_speed_kmh(ok, (11.46, 14.94), {}) is not None


def test_steady_speed_kmh_raw_disables_span_guard():
    # max_span=inf yields the RAW measurement (no span rejection) -- the value the pass
    # recorder stores for forensics/validity, vs the guarded default that returns None.
    gh = [(i / 30.0, 0.16, i * (1.95 / 13)) for i in range(15)]  # spans ~1.95 lengths
    assert steady_speed_kmh(gh, (11.46, 14.94), {}) is None            # guarded -> rejected
    assert steady_speed_kmh(gh, (11.46, 14.94), {}, max_span=float("inf")) is not None  # raw


def test_steady_speed_kmh_keeps_fast_clean_track():
    # A genuinely fast car has a short but CLEAN track; the guard must leave it alone
    # (its e2e agrees with the per-frame median, so nothing is substituted).
    gh = [(i / 30.0, 0.5, 0.02 + 0.95 * i / 14) for i in range(15)]
    guarded = steady_speed_kmh(gh, (1.0, 1.0), {})
    unguarded = steady_speed_kmh(gh, (1.0, 1.0), {}, jump_ratio=1e9)  # guard disabled
    assert guarded == pytest.approx(unguarded, rel=0.01)


def test_speed_kmh_calibrated_applies_correction():
    gh = [(0.0, 0.16, 0.0), (0.1, 0.16, 0.2)]  # far lane
    scale = (10.0, 10.0)
    raw = speed_kmh(gh, 0.5, scale)
    cal = speed_kmh_calibrated(gh, 0.5, scale, _ACROSS_CAL)
    assert cal == pytest.approx(raw / 0.667)


def test_track_speed_basic():
    # Moves 0.4 normalized units along y over 0.2s -> 2.0 units/sec.
    history = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.2), (0.2, 0.0, 0.4)]
    assert track_speed(history, window_seconds=0.5) == pytest.approx(2.0)


def test_track_speed_window_uses_recent_points():
    # Long history; only the last 0.5s should count for the smoothed speed.
    history = [(t / 10.0, 0.0, t / 10.0) for t in range(0, 30)]  # 3s, 0.1 units/0.1s = 1.0/s
    assert track_speed(history, window_seconds=0.5) == pytest.approx(1.0, abs=1e-6)


def test_track_speed_scaled_to_meters():
    # Move 0.5 normalized units along the road over 1s; the road spans 40 m
    # along -> 20 m travelled -> 20 m/s.
    history = [(0.0, 0.5, 0.0), (1.0, 0.5, 0.5)]
    assert track_speed(history, 2.0, scale=(10.0, 40.0)) == pytest.approx(20.0)


def test_speed_kmh():
    # 20 m/s -> 72 km/h.
    history = [(0.0, 0.5, 0.0), (1.0, 0.5, 0.5)]
    assert speed_kmh(history, 2.0, scale=(10.0, 40.0)) == pytest.approx(72.0)


def test_metric_scale_resolution():
    assert metric_scale({"units": "relative"}) is None
    assert metric_scale({}) is None
    assert metric_scale(
        {"units": "meters", "target_width_units": 11, "target_length_units": 45}
    ) == (11.0, 45.0)
    # Degenerate spans -> disabled.
    assert metric_scale({"units": "meters", "target_width_units": 0, "target_length_units": 45}) is None


def test_track_speed_insufficient_or_zero_dt():
    assert track_speed([(0.0, 0.0, 0.0)]) is None
    assert track_speed([]) is None
    assert track_speed([(1.0, 0.0, 0.0), (1.0, 1.0, 1.0)]) is None  # dt == 0


def test_baseline_count_and_prune():
    b = DirectionSpeedBaseline(window_seconds=10.0)
    b.add(0.0, 1.0, track_id=1)
    b.add(5.0, 2.0, track_id=2)
    assert b.count == 2
    assert b.distinct_tracks == 2
    # A sample far in the future prunes the oldest beyond the window.
    b.add(20.0, 3.0, track_id=3)
    assert b.count == 1  # only ts=20 remains (0 and 5 are >10s before 20)


def test_baseline_percentile_of():
    b = DirectionSpeedBaseline(window_seconds=3600)
    for i, s in enumerate([1.0, 2.0, 3.0, 4.0]):
        b.add(0.0, s, track_id=i)
    assert b.percentile_of(2.0) == pytest.approx(0.5)
    assert b.percentile_of(4.0) == pytest.approx(1.0)
    assert b.percentile_of(0.5) == pytest.approx(0.0)


def test_baseline_quantiles_and_stats():
    b = DirectionSpeedBaseline(window_seconds=3600)
    for i in range(1, 101):  # speeds 1..100
        b.add(0.0, float(i), track_id=i)
    assert b.quantile(0.5) == pytest.approx(50, abs=1)
    stats = b.stats()
    assert stats["median"] == pytest.approx(50, abs=1)
    assert stats["p97"] == pytest.approx(97, abs=1)
    assert stats["p90"] < stats["p95"] < stats["p97"]


def test_speed_estimator_separates_directions():
    est = SpeedEstimator(window_seconds=3600)
    est.observe("left_to_right", 0.0, 1.0, track_id=1)
    est.observe("left_to_right", 0.0, 1.0, track_id=2)
    est.observe("right_to_left", 0.0, 9.0, track_id=3)
    assert est.count("left_to_right") == 2
    assert est.count("right_to_left") == 1
    # A 5.0 speed is fast among the L2R baseline but slow among R2L's 9.0.
    assert est.percentile("left_to_right", 5.0) == pytest.approx(1.0)
    assert est.percentile("right_to_left", 5.0) == pytest.approx(0.0)
