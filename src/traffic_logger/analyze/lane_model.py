"""Lane band model.

After the perspective transform, the road width is divided into five bands
(spec section "Lane Band Model"):

    bike_lane_a, travel_lane_a, center_turn_lane, travel_lane_b, bike_lane_b

Milestone 0 ships the pure-math ``normalize_lane_ratios`` helper (real, tested
now). Per-frame band assignment from a projected point arrives with the
calibration milestone (M3).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Canonical band order across the road, left edge to right edge (camera-relative).
LANE_BANDS: Tuple[str, ...] = (
    "bike_lane_a",
    "travel_lane_a",
    "center_turn_lane",
    "travel_lane_b",
    "bike_lane_b",
)


def normalize_lane_ratios(lane_model_cfg: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Build normalized (band_name, width_fraction) pairs that sum to 1.0.

    Two ways to declare widths (far edge -> near edge), highest precedence first:

    * ``band_widths``: an explicit list of five raw widths, one per band in
      ``LANE_BANDS`` order. Use this when the road is asymmetric or the lanes
      don't fit the mirrored model (e.g. a wider near lane, or no bike lane on
      one side -> set that width to ~0).
    * the mirrored ``*_width_ratio`` trio, which assumes a symmetric road:

          bike_lane_width_ratio    -> bike_lane_a, bike_lane_b
          travel_lane_width_ratio  -> travel_lane_a, travel_lane_b
          center_lane_width_ratio  -> center_turn_lane

    Raw widths need not sum to 1.0; they are normalized here so classification
    is stable regardless of the absolute numbers chosen.
    """
    explicit = lane_model_cfg.get("band_widths")
    if explicit:
        raw = [float(w) for w in explicit]
        if len(raw) != len(LANE_BANDS):
            raise ValueError(
                f"band_widths must have {len(LANE_BANDS)} values "
                f"(far->near: {', '.join(LANE_BANDS)})"
            )
    else:
        bike = float(lane_model_cfg.get("bike_lane_width_ratio", 0.12))
        travel = float(lane_model_cfg.get("travel_lane_width_ratio", 0.28))
        center = float(lane_model_cfg.get("center_lane_width_ratio", 0.20))
        raw = [bike, travel, center, travel, bike]

    total = sum(raw)
    if total <= 0:
        raise ValueError("Lane width ratios must sum to a positive value")

    return [(name, value / total) for name, value in zip(LANE_BANDS, raw)]


def lane_band_edges(lane_model_cfg: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    """Cumulative band boundaries as (band_name, start_frac, end_frac).

    Useful for both classification and drawing the overlay preview.
    """
    edges: List[Tuple[str, float, float]] = []
    cursor = 0.0
    for name, width in normalize_lane_ratios(lane_model_cfg):
        edges.append((name, cursor, cursor + width))
        cursor += width
    return edges


def assign_lane_band(
    normalized_x: float, lane_model_cfg: Dict[str, Any]
) -> Optional[str]:
    """Classify a normalized cross-road position into a lane band.

    ``normalized_x`` is the projected point's x-coordinate in the road plane
    (0 = one road edge, 1 = the other). Returns the band name, or None if the
    point lies outside the calibrated road surface ([0, 1)).
    """
    if normalized_x < 0.0 or normalized_x >= 1.0:
        return None
    for name, start, end in lane_band_edges(lane_model_cfg):
        if start <= normalized_x < end:
            return name
    # Floating-point edge: a value of exactly 1.0 is handled above; this is a
    # safety net for the final band's closed upper edge.
    return LANE_BANDS[-1]


def assign_lane_band_for_point(
    projector, x: float, y: float, lane_model_cfg: Dict[str, Any]
) -> Optional[str]:
    """Project an image point and classify its lane band (None if off-road)."""
    nx, _ny = projector.project(x, y)
    return assign_lane_band(nx, lane_model_cfg)


def lane_band_polygons(
    projector, lane_model_cfg: Dict[str, Any]
) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Image-space quadrilaterals for each lane band, for overlay drawing.

    Each band occupies a vertical strip ``x in [start, end], y in [0, 1]`` of
    the normalized plane; its four corners are unprojected back into image
    coordinates. Pure (uses only ``projector.unproject``).
    """
    polygons: List[Tuple[str, List[Tuple[float, float]]]] = []
    for name, start, end in lane_band_edges(lane_model_cfg):
        corners = [
            projector.unproject(start, 0.0),
            projector.unproject(end, 0.0),
            projector.unproject(end, 1.0),
            projector.unproject(start, 1.0),
        ]
        polygons.append((name, corners))
    return polygons


# BGR colors per band for overlays (bike / travel / center / travel / bike).
LANE_COLORS = {
    "bike_lane_a": (0, 200, 200),
    "travel_lane_a": (0, 180, 0),
    "center_turn_lane": (0, 0, 220),
    "travel_lane_b": (0, 180, 0),
    "bike_lane_b": (0, 200, 200),
}


def draw_lane_overlay(frame, projector, lane_model_cfg: Dict[str, Any], alpha: float = 0.35):
    """Blend translucent colored lane-band strips onto a BGR frame.

    Returns the same frame (modified). Lazily imports cv2/numpy so this module
    stays importable without the CV stack.
    """
    import cv2
    import numpy as np

    overlay = frame.copy()
    for name, corners in lane_band_polygons(projector, lane_model_cfg):
        pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], LANE_COLORS.get(name, (200, 200, 200)))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
