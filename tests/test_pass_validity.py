"""Physical-plausibility invariants for a pass's steady speed (pure)."""

from __future__ import annotations

from traffic_logger.analyze.pass_validity import (
    PassGeometry, ValidityThresholds, speed_validity)

THR = ValidityThresholds()


def _valid(**kw):
    base = dict(steady_kmh=60.0, track_seconds=0.8, ground_span=1.0, n_points=20)
    base.update(kw)
    return speed_validity(PassGeometry(**base), THR)


def test_clean_pass_is_valid():
    assert _valid() == (True, None)


def test_no_speed():
    assert _valid(steady_kmh=None) == (False, "no_speed")


def test_fragment_track_rejected():
    assert _valid(track_seconds=0.3) == (False, "fragment_track")
    # the validated fast-motorcycle track (0.5s) is above the floor -> kept
    assert _valid(track_seconds=0.5)[0] is True


def test_too_few_points_rejected():
    assert _valid(n_points=4) == (False, "too_few_points")


def test_over_max_kmh_rejected():
    assert _valid(steady_kmh=140.0) == (False, "over_max_kmh")


def test_span_exceeds_road_rejected():
    # A stored span > max_span == the endpoints jumped across the FOV.
    assert _valid(ground_span=2.3) == (False, "span_exceeds_road")
    assert _valid(ground_span=1.5)[0] is True


def test_implied_distance_proxy_when_span_missing():
    # Legacy rows have no stored span -> fall back to implied ground distance.
    # 100 km/h over 1.25s -> ~34.7 m, ~2.3x the 14.94 m road -> rejected.
    assert speed_validity(
        PassGeometry(steady_kmh=100.0, track_seconds=1.25, ground_span=None, n_points=None),
        THR) == (False, "implied_distance")
    # 80 km/h over 0.7s -> ~15.6 m (spans ~the road) -> kept
    assert speed_validity(
        PassGeometry(steady_kmh=80.0, track_seconds=0.7, ground_span=None, n_points=None),
        THR)[0] is True


def test_stored_span_takes_precedence_over_proxy():
    # With a real span present, a long-but-clean track that the crude implied-distance
    # proxy would over-reject is correctly kept (the span is the true invariant).
    assert speed_validity(
        PassGeometry(steady_kmh=100.0, track_seconds=1.25, ground_span=1.1, n_points=30),
        THR)[0] is True
