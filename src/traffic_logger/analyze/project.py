"""Perspective projection.

Builds a homography from the four clicked road-surface corners to a normalized
target plane (the unit square), and projects vehicle bottom-center points into
that plane. In the normalized plane:

* x runs across the road (0 = one edge, 1 = the other) and drives lane-band
  classification;
* y runs along the road (0 = far edge of the quad, 1 = near edge) and drives
  same-direction speed comparisons (Milestone 4).

The homography is solved with a small NumPy DLT (no OpenCV dependency) so the
transform math is light and unit-testable. ``target_width_units`` /
``target_length_units`` are kept as metadata for a later metric rescale; for the
MVP the plane is the unit square and speed stays relative.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def order_points(points: Sequence[Point]) -> "list":
    """Order four points canonically as top-left, top-right, bottom-right,
    bottom-left, so calibration is robust to the order the user clicked.
    """
    import numpy as np

    pts = np.asarray(points, dtype=float)
    if pts.shape != (4, 2):
        raise ValueError("Exactly four (x, y) points are required")
    s = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]  # y - x
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmin(diff))]
    bl = pts[int(np.argmax(diff))]
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def _homography(src: "list", dst: "list"):
    """Solve the 3x3 homography mapping src -> dst from 4 correspondences (DLT)."""
    import numpy as np

    a = []
    b = []
    for (x, y), (X, Y) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        b.append(X)
        a.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        b.append(Y)
    h = np.linalg.solve(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ], dtype=float)


class PerspectiveTransform:
    """Maps image pixels to/from the normalized road plane (unit square)."""

    # Unit-square corners matching order_points' TL, TR, BR, BL.
    _UNIT_SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    def __init__(
        self,
        source_points: Sequence[Point],
        target_width_units: float = 1.0,
        target_length_units: float = 1.0,
        reorder: bool = True,
        swap_xy: bool = False,
    ) -> None:
        import numpy as np

        self.target_width_units = float(target_width_units)
        self.target_length_units = float(target_length_units)
        # When True, the projected axes are swapped so output[0] is always the
        # across-road (lane) axis and output[1] the along-road (travel) axis,
        # regardless of how the road is oriented in the image. See
        # build_transform for auto-detection.
        self.swap_xy = bool(swap_xy)
        ordered = order_points(source_points) if reorder else [tuple(p) for p in source_points]
        self.source_points = ordered
        self._H = _homography(ordered, self._UNIT_SQUARE)        # image -> unit square
        self._H_inv = np.linalg.inv(self._H)                     # unit square -> image

    def project(self, x: float, y: float) -> Point:
        """Image pixel -> normalized (across, along) in the unit square."""
        nx, ny = self._apply(self._H, x, y)
        return (ny, nx) if self.swap_xy else (nx, ny)

    def unproject(self, across: float, along: float) -> Point:
        """Normalized (across, along) -> image pixel."""
        raw = (along, across) if self.swap_xy else (across, along)
        return self._apply(self._H_inv, raw[0], raw[1])

    def project_scaled(self, x: float, y: float) -> Point:
        """Image pixel -> plane coords scaled by the target unit dimensions."""
        across, along = self.project(x, y)
        return (across * self.target_width_units, along * self.target_length_units)

    @staticmethod
    def _apply(matrix, x: float, y: float) -> Point:
        denom = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        if denom == 0:
            denom = 1e-12
        px = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denom
        py = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denom
        return (float(px), float(py))


def build_transform(calibration_cfg: dict) -> Optional[PerspectiveTransform]:
    """Build a transform from config, or None if calibration is incomplete.

    Returns None when ``source_points`` is missing or doesn't have 4 points, so
    callers can degrade gracefully (lane bands disabled, warn).
    """
    points = calibration_cfg.get("source_points") or []
    if len(points) != 4:
        return None
    return PerspectiveTransform(
        points,
        target_width_units=float(calibration_cfg.get("target_width_units", 1.0)),
        target_length_units=float(calibration_cfg.get("target_length_units", 1.0)),
        swap_xy=_resolve_swap(calibration_cfg.get("swap_axes", "auto"), points),
    )


def _resolve_swap(swap_axes, points: Sequence[Point]) -> bool:
    """Decide whether to swap the projected axes.

    ``swap_axes`` may be True/False, or "auto" (default): with auto, the road's
    travel direction is taken to be the longer visual extent of the calibration
    quad (camera perpendicular to the road), so the across-road (lane) axis maps
    to projected x. A road spanning the frame horizontally -> swap; a road
    running vertically -> no swap.
    """
    if isinstance(swap_axes, bool):
        return swap_axes
    if swap_axes in (None, "auto"):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (max(xs) - min(xs)) > (max(ys) - min(ys))
    return bool(swap_axes)
