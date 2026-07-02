"""Tests for ground_span -- the stationary-vehicle test used to keep parked cars
(neighbours' cars across the street) out of the annotation overlay."""

from __future__ import annotations

from traffic_logger.analyze.metrics import ground_span


def test_parked_car_spans_near_zero():
    # Many samples, all at essentially the same ground point (slight bbox jitter).
    gh = [(float(i) * 0.1, 0.80 + 0.002 * (i % 2), 0.30 - 0.001 * (i % 3))
          for i in range(40)]
    assert ground_span(gh) < 0.02


def test_moving_car_spans_the_road():
    # A car traversing the frame along the road axis.
    gh = [(float(i) * 0.1, 0.05 * i, 0.5 + 0.01 * i) for i in range(20)]
    assert ground_span(gh) > 0.5


def test_too_short_history_is_zero():
    assert ground_span([]) == 0.0
    assert ground_span([(0.0, 0.5, 0.5)]) == 0.0


def test_span_is_bbox_diagonal_not_path_length():
    # Out-and-back: large path length but small net bbox -> span stays small,
    # so a car that wiggles in place is still treated as stationary.
    gh = [(0.0, 0.50, 0.50), (0.1, 0.55, 0.50), (0.2, 0.50, 0.50),
          (0.3, 0.55, 0.50), (0.4, 0.50, 0.50)]
    assert ground_span(gh) < 0.06
