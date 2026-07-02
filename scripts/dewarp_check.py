"""Diagnose the speed-calibration non-uniformity: de-warp a frame and overlay the
road quad + a horizontal reference grid, to see whether the curbs come out
straight (de-warp OK -> quad-corner issue) or still bowed (k1 under-corrects ->
far-field compression -> far lane reads slow).

    .venv\\Scripts\\python.exe scripts/dewarp_check.py [config]
"""
import sys

import cv2
import numpy as np

from traffic_logger.analyze.undistort import build_undistorter
from traffic_logger.config import load_config
from traffic_logger.capture.segment_index import SegmentIndex

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/config.run.local.yaml")
with SegmentIndex(cfg.recording.get("segment_index_path")) as idx:
    seg = idx.get_all()[-2]  # a recent finalized segment
cap = cv2.VideoCapture(seg.path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("could not read frame")

small = cv2.resize(frame, (704, 480), interpolation=cv2.INTER_AREA)
und = build_undistorter(cfg.calibration)
dw = und(small) if und is not None else small

# Horizontal reference lines (perfectly straight). If the real curbs/lane paint
# run parallel to these after de-warp, the lens model is good.
for y in range(0, 480, 30):
    cv2.line(dw, (0, y), (704, y), (80, 80, 80), 1)

# The configured road quad (source_points, de-warped 704 space).
sp = cfg.calibration.get("source_points")
if sp:
    pts = np.array(sp, dtype=np.int32)
    cv2.polylines(dw, [pts.reshape(-1, 1, 2)], True, (0, 255, 255), 2)
    for i, (x, y) in enumerate(sp):
        cv2.circle(dw, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(dw, ["FL", "FR", "NR", "NL"][i], (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

big = cv2.resize(dw, (704 * 2, 480 * 2), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("samples/_dewarp_check.jpg", big)
print(f"k1={(cfg.calibration.get('undistort') or {}).get('k1')}  -> samples/_dewarp_check.jpg")
print(f"source_points={sp}")
