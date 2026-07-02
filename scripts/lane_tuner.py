"""Lane-band tuning diagnostic.

Renders the de-warped frame with:
  * the road quad + corners labeled C0..C3,
  * faint reference lines every 10% across the road (a road-space ruler,
    0 = first quad edge, 100 = opposite edge),
  * the current lane-band boundary lines (B1..) drawn over the visible paint,
    each labeled with its position %, so band edges can be lined up against the
    real lane lines and adjusted via calibration.lane_model.band_widths.

Usage: python scripts/lane_tuner.py <src.jpg> <out.jpg> [config.yaml]
"""
import sys

import cv2
import numpy as np

from traffic_logger.analyze.lane_model import lane_band_edges
from traffic_logger.analyze.project import build_transform
from traffic_logger.analyze.undistort import build_undistorter
from traffic_logger.config import load_config

src = sys.argv[1]
out = sys.argv[2]
cfg_path = sys.argv[3] if len(sys.argv) > 3 else "config/config.camera.yaml"
cal = load_config(cfg_path).calibration

img = cv2.imread(src)
und = build_undistorter(cal)
if und is not None:
    img = und(img)
proj = build_transform(cal)
h, w = img.shape[:2]


def across_line(pos, color, thick, label=None, side="left"):
    """Draw a line at across-road position ``pos`` (0..1), spanning the road length.

    ``side`` places the label near the left or right end of the line so the
    ruler (right) and band-boundary (left) labels don't overlap.
    """
    a = proj.unproject(pos, 0.0)
    b = proj.unproject(pos, 1.0)
    pa = (int(round(a[0])), int(round(a[1])))
    pb = (int(round(b[0])), int(round(b[1])))
    cv2.line(img, pa, pb, color, thick)
    if label:
        left, right = (pa, pb) if pa[0] <= pb[0] else (pb, pa)
        org = (left[0] + 8, left[1] + 4) if side == "left" else (right[0] - 52, right[1] + 4)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Road-space ruler: faint reference lines every 10% (labels on the right).
for i in range(1, 10):
    across_line(i / 10.0, (150, 150, 150), 1, f"{i * 10}", side="right")

# Current band boundaries (the internal edges, far->near), labeled B1..Bn (left).
edges = lane_band_edges(cal.get("lane_model", {}))
for n, (_name, _start, end) in enumerate(edges[:-1], start=1):
    across_line(end, (0, 255, 255), 2, f"B{n}={int(round(end * 100))}", side="left")

# Quad outline + labeled corners.
pts = np.array(proj.source_points, dtype=np.int32)
cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, (255, 0, 0), 2)
for i, (x, y) in enumerate(proj.source_points):
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.putText(img, f"C{i}", (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imwrite(out, cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR))
print(out)
