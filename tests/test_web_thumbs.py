"""Thumbnail downscaling (best-effort, cached)."""

from __future__ import annotations

import pytest

from traffic_logger.web import thumbs


def test_non_image_returns_none(tmp_path):
    src = tmp_path / "x.jpg"
    src.write_bytes(b"not really an image")
    assert thumbs.downscaled_thumb(src, tmp_path / "cache") is None


def test_caps_width_and_caches(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    src = tmp_path / "big.jpg"
    cv2.imwrite(str(src), np.zeros((600, 1200, 3), dtype=np.uint8))

    out = thumbs.downscaled_thumb(src, tmp_path / "cache", max_w=480)
    assert out is not None and out.exists()
    assert cv2.imread(str(out)).shape[1] == 480          # width capped
    # second call hits the cache and returns the same path
    assert thumbs.downscaled_thumb(src, tmp_path / "cache", max_w=480) == out


def test_small_image_not_upscaled(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    src = tmp_path / "small.jpg"
    cv2.imwrite(str(src), np.zeros((120, 300, 3), dtype=np.uint8))
    out = thumbs.downscaled_thumb(src, tmp_path / "cache", max_w=480)
    assert out is not None and cv2.imread(str(out)).shape[1] == 300  # unchanged
