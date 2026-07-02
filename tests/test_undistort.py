"""Tests for the radial lens de-warp (analyze/undistort.py)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from traffic_logger.analyze.undistort import Undistorter, build_undistorter


def test_build_returns_none_when_absent_or_disabled():
    assert build_undistorter(None) is None
    assert build_undistorter({}) is None
    assert build_undistorter({"undistort": {"enabled": False, "k1": -0.3}}) is None


def test_build_returns_none_for_zero_coeffs():
    # Enabled but a no-op (all coefficients zero) -> skip the step.
    assert build_undistorter({"undistort": {"enabled": True, "k1": 0.0, "k2": 0.0}}) is None


def test_build_returns_undistorter_when_enabled():
    u = build_undistorter({"undistort": {"enabled": True, "k1": -0.35, "k2": 0.0}})
    assert isinstance(u, Undistorter)
    assert u.k1 == pytest.approx(-0.35)
    assert u.k2 == pytest.approx(0.0)


def test_build_returns_undistorter_for_roll_only():
    # Roll alone (no radial coeffs) is still a real transform.
    u = build_undistorter({"undistort": {"enabled": True, "k1": 0.0, "roll_degrees": 1.7}})
    assert isinstance(u, Undistorter)
    assert u.roll_degrees == pytest.approx(1.7)


def test_roll_applies_without_error():
    pytest.importorskip("cv2")
    u = Undistorter(0.0, roll_degrees=2.0)
    frame = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    out = u(frame)
    assert out.shape == frame.shape
    assert u._size == (64, 48)


def test_call_preserves_shape_and_dtype():
    pytest.importorskip("cv2")
    u = Undistorter(-0.35)
    frame = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    out = u(frame)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    # Remap tables get cached for the frame size on first call.
    assert u._size == (64, 48)


def test_maps_recomputed_on_size_change():
    pytest.importorskip("cv2")
    u = Undistorter(-0.2)
    u(np.zeros((48, 64, 3), dtype=np.uint8))
    assert u._size == (64, 48)
    u(np.zeros((96, 128, 3), dtype=np.uint8))
    assert u._size == (128, 96)
