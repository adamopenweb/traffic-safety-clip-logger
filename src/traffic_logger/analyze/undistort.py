"""Radial lens-distortion correction (de-warp), applied before detection.

The wide motorized lens leaves residual barrel distortion that bows the
straight road/curb lines. That hurts us twice:

* it breaks the flat-plane assumption the perspective homography relies on for
  speed, and
* it makes a 4-point road quad unable to follow the curved curbs, so the lane
  bands never line up.

We correct it once with a single radial coefficient (OpenCV's distortion
model). The camera matrix is built from the frame size with ``fx = fy = width``
and the image centre, which makes the coefficient **resolution-independent**:
the same ``k1`` straightens both the 704x480 sub-stream and the 4K main
(identical field of view, different pixel counts). The de-warp maps are
precomputed once per frame size and reused via ``cv2.remap``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_undistorter(calibration_cfg: Optional[Dict[str, Any]]) -> "Optional[Undistorter]":
    """Construct an :class:`Undistorter` from ``calibration.undistort`` config.

    Returns ``None`` when undistortion is disabled or a no-op (all coeffs 0),
    so callers can cheaply skip the step.
    """
    cfg = (calibration_cfg or {}).get("undistort") or {}
    if not cfg.get("enabled", False):
        return None
    k1 = float(cfg.get("k1", 0.0))
    k2 = float(cfg.get("k2", 0.0))
    roll = float(cfg.get("roll_degrees", 0.0))
    if k1 == 0.0 and k2 == 0.0 and roll == 0.0:
        return None
    return Undistorter(k1, k2, roll_degrees=roll)


class Undistorter:
    """Radial de-warp (+ optional roll level) of BGR frames; caches remap tables.

    ``roll_degrees`` rotates the image about its centre to cancel residual camera
    roll (a uniform tilt the radial model can't fix), baked into the same remap
    via the rectification rotation so the road comes out level. Positive degrees
    rotate counter-clockwise.
    """

    def __init__(self, k1: float, k2: float = 0.0, roll_degrees: float = 0.0):
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.roll_degrees = float(roll_degrees)
        self._size: Optional[tuple[int, int]] = None
        self._map1 = None
        self._map2 = None

    def _ensure_maps(self, w: int, h: int) -> None:
        if self._size == (w, h):
            return
        import math

        import cv2
        import numpy as np

        fx = fy = float(w)
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.array([self.k1, self.k2, 0.0, 0.0, 0.0], dtype=np.float64)
        R = None
        if self.roll_degrees:
            a = math.radians(self.roll_degrees)
            c, s = math.cos(a), math.sin(a)
            # Rotation about the optical (z) axis -> rolls the rectified image.
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K, D, R, K, (w, h), cv2.CV_16SC2
        )
        self._size = (w, h)

    def __call__(self, frame):
        """Return a de-warped copy of ``frame`` (same dimensions)."""
        import cv2

        h, w = frame.shape[:2]
        self._ensure_maps(w, h)
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
