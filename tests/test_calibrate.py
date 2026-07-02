"""Tests for calibration helpers."""

from __future__ import annotations

import importlib.util

import pytest

from traffic_logger.analyze import calibrate as cal

_HAS_CV = importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None


def test_parse_points_valid():
    pts = cal.parse_points("100,100 500,100 500,400 100,400")
    assert pts == [(100, 100), (500, 100), (500, 400), (100, 400)]


def test_parse_points_invalid():
    with pytest.raises(ValueError):
        cal.parse_points("100,100 500,100")           # too few
    with pytest.raises(ValueError):
        cal.parse_points("100 200 300 400")           # missing commas


def test_points_yaml_snippet():
    snippet = cal.points_yaml_snippet([(1, 2), (3, 4), (5, 6), (7, 8)])
    assert "source_points" in snippet
    assert "- [1.0, 2.0]" in snippet


def test_write_source_points_roundtrip(tmp_path):
    import yaml

    cfg = tmp_path / "c.yaml"
    cfg.write_text("calibration:\n  mode: relative\n  source_points: []\n", encoding="utf-8")
    cal.write_source_points(cfg, [(10, 20), (30, 40), (50, 60), (70, 80)])
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["calibration"]["source_points"] == [[10, 20], [30, 40], [50, 60], [70, 80]]
    assert data["calibration"]["mode"] == "relative"  # other keys preserved


@pytest.mark.skipif(not _HAS_CV, reason="cv2/numpy not installed")
def test_render_preview_draws_overlay(tmp_path):
    import cv2
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    points = [(100, 120), (520, 120), (560, 420), (60, 420)]
    preview, projector = cal.render_preview(frame, points, {"lane_model": {}})

    assert preview.shape == frame.shape
    # Something was drawn (overlay + quad lines) over the black frame.
    assert preview.sum() > 0
    assert len(projector.source_points) == 4

    out = tmp_path / "preview.jpg"
    cv2.imwrite(str(out), preview)
    assert out.exists() and out.stat().st_size > 0
