"""Tests for the perspective transform (numpy DLT homography)."""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None, reason="numpy not installed"
)


def test_rectangle_corners_map_to_unit_square():
    from traffic_logger.analyze.project import PerspectiveTransform

    # A rectangle clicked as TL, TR, BR, BL.
    src = [(100, 100), (500, 100), (500, 400), (100, 400)]
    t = PerspectiveTransform(src)

    assert t.project(100, 100) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert t.project(500, 100) == pytest.approx((1.0, 0.0), abs=1e-6)
    assert t.project(500, 400) == pytest.approx((1.0, 1.0), abs=1e-6)
    assert t.project(100, 400) == pytest.approx((0.0, 1.0), abs=1e-6)
    # Center maps to the middle of the plane.
    assert t.project(300, 250) == pytest.approx((0.5, 0.5), abs=1e-6)


def test_round_trip_project_unproject():
    from traffic_logger.analyze.project import PerspectiveTransform

    src = [(120, 90), (640, 110), (700, 470), (60, 430)]  # irregular quad
    t = PerspectiveTransform(src)
    for x, y in [(300, 250), (200, 400), (500, 150)]:
        nx, ny = t.project(x, y)
        bx, by = t.unproject(nx, ny)
        assert (bx, by) == pytest.approx((x, y), abs=1e-6)


def test_point_ordering_is_robust_to_click_order():
    from traffic_logger.analyze.project import PerspectiveTransform

    ordered = [(100, 100), (500, 100), (500, 400), (100, 400)]
    scrambled = [(500, 400), (100, 100), (100, 400), (500, 100)]
    a = PerspectiveTransform(ordered)
    b = PerspectiveTransform(scrambled)
    # Same canonical ordering -> identical projection of the center.
    assert a.project(300, 250) == pytest.approx(b.project(300, 250), abs=1e-9)
    assert b.project(100, 100) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_swap_axes_puts_across_road_on_x():
    from traffic_logger.analyze.project import build_transform

    # A road running horizontally across the frame: wide quad, short height.
    # Far edge ~y=300, near edge ~y=440, full width.
    pts = [[0, 300], [1280, 300], [1280, 440], [0, 440]]

    # Without swap, the long (travel) axis maps to projected x.
    plain = build_transform({"source_points": pts, "swap_axes": False})
    # Auto should detect the horizontal road and swap so across-road -> x.
    auto = build_transform({"source_points": pts, "swap_axes": "auto"})
    assert auto.swap_xy is True
    assert plain.swap_xy is False

    # A point at the near edge / mid-width: across-road coordinate should be ~1
    # under swap (near edge), and the along coordinate ~0.5 (mid road length).
    across, along = auto.project(640, 440)
    assert across == pytest.approx(1.0, abs=1e-6)
    assert along == pytest.approx(0.5, abs=1e-6)
    # Round trip still holds with swap.
    assert auto.unproject(across, along) == pytest.approx((640, 440), abs=1e-6)


def test_build_transform_requires_four_points():
    from traffic_logger.analyze.project import build_transform

    assert build_transform({"source_points": []}) is None
    assert build_transform({"source_points": [(0, 0), (1, 0), (1, 1)]}) is None
    t = build_transform({"source_points": [(0, 0), (10, 0), (10, 10), (0, 10)]})
    assert t is not None


def test_lane_band_polygons_cover_image_region():
    from traffic_logger.analyze.lane_model import lane_band_polygons
    from traffic_logger.analyze.project import PerspectiveTransform

    src = [(100, 100), (500, 100), (500, 400), (100, 400)]
    t = PerspectiveTransform(src)
    polys = lane_band_polygons(t, {})
    assert [name for name, _ in polys] == [
        "bike_lane_a", "travel_lane_a", "center_turn_lane",
        "travel_lane_b", "bike_lane_b",
    ]
    # First band starts at the left road edge, last ends at the right edge.
    first_corners = polys[0][1]
    assert first_corners[0] == pytest.approx((100, 100), abs=1e-6)   # x=0,y=0
    last_corners = polys[-1][1]
    assert last_corners[1] == pytest.approx((500, 100), abs=1e-6)    # x=1,y=0
