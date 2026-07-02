"""Calibration helper for the `traffic-log calibrate` command.

Loads a frame (sample image or first video frame), collects the four
road-surface corners (interactively, or non-interactively via ``--points``),
renders a preview image with the projected five lane bands, and can write the
points back into the config.

The pure pieces (``parse_points``) are unit-tested; image/HighGUI pieces lazily
import cv2 so this module imports without the CV stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from ..util.logging import get_logger

log = get_logger(__name__)

Point = Tuple[float, float]
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_points(text: str) -> List[Point]:
    """Parse ``"x1,y1 x2,y2 x3,y3 x4,y4"`` into four (x, y) points."""
    tokens = text.replace(";", " ").split()
    points: List[Point] = []
    for tok in tokens:
        if "," not in tok:
            raise ValueError(f"Bad point '{tok}'; expected 'x,y'")
        xs, ys = tok.split(",", 1)
        points.append((float(xs), float(ys)))
    if len(points) != 4:
        raise ValueError(f"Expected 4 points, got {len(points)}")
    return points


def load_frame(source: str | Path):
    """Load a single BGR frame from an image file or the first video frame."""
    import cv2

    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Calibration source not found: {src}")
    if src.suffix.lower() in _IMAGE_EXTS:
        frame = cv2.imread(str(src))
        if frame is None:
            raise RuntimeError(f"Could not read image: {src}")
        return frame
    cap = cv2.VideoCapture(str(src))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from video: {src}")
    return frame


def render_preview(frame, points: Sequence[Point], calibration_cfg: dict):
    """Render the calibration preview: lane bands + the clicked quadrilateral."""
    import cv2
    import numpy as np

    from .lane_model import draw_lane_overlay
    from .project import PerspectiveTransform, _resolve_swap

    projector = PerspectiveTransform(
        points,
        target_width_units=float(calibration_cfg.get("target_width_units", 1.0)),
        target_length_units=float(calibration_cfg.get("target_length_units", 1.0)),
        swap_xy=_resolve_swap(calibration_cfg.get("swap_axes", "auto"), points),
    )
    img = draw_lane_overlay(frame.copy(), projector, calibration_cfg.get("lane_model", {}))

    ordered = np.array(projector.source_points, dtype=np.int32)
    cv2.polylines(img, [ordered.reshape(-1, 1, 2)], True, (255, 255, 0), 2)
    for i, (x, y) in enumerate(projector.source_points):
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(img, str(i), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img, projector


def collect_points_interactive(frame) -> List[Point]:
    """Open a window and collect four clicks. Raises if HighGUI is unavailable."""
    import cv2

    points: List[Point] = []
    window = "calibrate: click 4 road corners (any order), then any key"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow(window, frame)

    cv2.imshow(window, frame)
    cv2.setMouseCallback(window, on_mouse)
    while len(points) < 4:
        if cv2.waitKey(20) & 0xFF == 27:  # Esc aborts
            break
    cv2.destroyWindow(window)
    if len(points) != 4:
        raise RuntimeError("Calibration aborted before 4 points were collected")
    return points


def write_source_points(config_path: str | Path, points: Sequence[Point]) -> None:
    """Write ``calibration.source_points`` back into a YAML config file.

    Note: this round-trips through PyYAML and does not preserve comments. The
    user is advised to review the file afterward.
    """
    import yaml

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data.setdefault("calibration", {})["source_points"] = [
        [float(x), float(y)] for x, y in points
    ]
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)


def points_yaml_snippet(points: Sequence[Point]) -> str:
    """A pasteable YAML snippet for the calibration source_points."""
    lines = ["calibration:", "  source_points:"]
    for x, y in points:
        lines.append(f"    - [{x:.1f}, {y:.1f}]")
    return "\n".join(lines)
