"""Grab a 4K frame from the latest ring segment and draw the calibration road
quad on it, so the ground landmarks under each corner can be identified and
measured (e.g. on a satellite map) to convert the quad to real metres.

The quad (source_points) lives in DE-WARPED 704x480 sub space; StreamProjector
maps each corner to the raw 4K main pixel, so the drawn quad sits on the actual
(distorted) 4K frame the camera records.

    .venv\\Scripts\\python.exe scripts/quad_overlay_still.py [config.yaml] [out.jpg]
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from traffic_logger.config import load_config
from traffic_logger.events.overlay_render import StreamProjector
from traffic_logger.util.ffmpeg import ffmpeg_path

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.run.local.yaml"
out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "samples/quad_overlay_4k.jpg")

cfg = load_config(cfg_path)
cal = cfg.calibration
ucfg = cal.get("undistort") or {}

# Newest finalized 4K segment in the ring (skip the still-writing 'incoming' one).
ring = Path("data/ring")
segs = sorted(
    (p for p in ring.rglob("segment_*.mp4") if "incoming" not in p.parts),
    key=lambda p: p.stat().st_mtime,
)
if not segs:
    raise SystemExit("No finalized ring segments found under data/ring")
seg = segs[-1]
print(f"Using segment: {seg}")

# Extract a frame ~5s in.
tmp = out_path.with_name("_quad_src.jpg")
subprocess.run(
    [ffmpeg_path(), "-y", "-loglevel", "error", "-ss", "5", "-i", str(seg),
     "-frames:v", "1", str(tmp)],
    check=True,
)
frame = cv2.imread(str(tmp))
if frame is None:
    raise SystemExit(f"Could not read extracted frame {tmp}")
h, w = frame.shape[:2]
print(f"4K frame: {w}x{h}")

proj = StreamProjector(
    (704, 480), (w, h),
    k1=float(ucfg.get("k1", 0.0)), k2=float(ucfg.get("k2", 0.0)),
    roll_degrees=float(ucfg.get("roll_degrees", 0.0)),
    homography=cal.get("overlay_homography"),
)

# source_points order: far-left(TL), far-right(TR), near-right(BR), near-left(BL).
sp = cal.get("source_points")
labels = ["FL far-left", "FR far-right", "NR near-right", "NL near-left"]
pts = [proj.project(float(x), float(y)) for x, y in sp]

poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
cv2.polylines(frame, [poly], True, (0, 255, 255), 3)

# Edge midpoint captions: which dimension each edge measures.
def mid(a, b):
    return (int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2))

edge_notes = [
    (mid(pts[0], pts[1]), "FAR curb  (along-road LENGTH)"),
    (mid(pts[3], pts[2]), "NEAR curb  (along-road LENGTH)"),
    (mid(pts[0], pts[3]), "WIDTH (curb-to-curb)"),
    (mid(pts[1], pts[2]), "WIDTH (curb-to-curb)"),
]
for (mx, my), txt in edge_notes:
    cv2.putText(frame, txt, (mx - 120, my), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, txt, (mx - 120, my), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 255), 2, cv2.LINE_AA)

for (px, py), lab in zip(pts, labels):
    cv2.circle(frame, (px, py), 12, (0, 0, 255), -1)
    cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)
    cv2.putText(frame, lab, (px + 16, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, lab, (px + 16, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255), 2, cv2.LINE_AA)

out_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_path), frame)
tmp.unlink(missing_ok=True)
print(f"WROTE {out_path}  ({w}x{h})")
print("Corners (4K px):")
for lab, (px, py) in zip(labels, pts):
    print(f"  {lab:14s} ({px}, {py})")
