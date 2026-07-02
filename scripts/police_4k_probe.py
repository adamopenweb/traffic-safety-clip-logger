"""End-to-end smoke test for the 4K-confirm plumbing on live ring data.

Detects a real vehicle in the de-warped sub-stream, then exercises the exact
Police4KConfirmer path: pull the covering 4K frame from the ring, project the
sub bbox -> 4K, crop, and score. Saves the 4K crop so the projection can be
eyeballed, and prints the sub-vs-4K scores for the same vehicle.

    .venv\\Scripts\\python.exe scripts/police_4k_probe.py
"""
import subprocess
from pathlib import Path

import cv2

from traffic_logger.analyze.detector import build_detector
from traffic_logger.analyze.police_classifier import Police4KConfirmer, PoliceClassifier
from traffic_logger.analyze.undistort import build_undistorter
from traffic_logger.capture.segment_index import SegmentIndex
from traffic_logger.config import load_config
from traffic_logger.util.ffmpeg import ffmpeg_path

cfg = load_config("config/config.run.local.yaml")
det = build_detector(cfg)
undistort = build_undistorter(cfg.calibration)
clf = PoliceClassifier(device=str((cfg.events.get("police") or {}).get("device")
                                   or cfg.analysis.get("device", "cuda")))
index_path = cfg.recording.get("segment_index_path")

# Find a recent segment that actually has a vehicle in it.
with SegmentIndex(index_path) as idx:
    segs = idx.get_all()[-12:]
chosen = None
for seg in reversed(segs):
    mid = (seg.start_ts + seg.end_ts) / 2.0
    tmp = "samples/_4kprobe_sub.jpg"
    subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error", "-ss",
                    f"{(seg.end_ts-seg.start_ts)/2:.3f}", "-i", seg.path,
                    "-vf", "scale=704:480", "-frames:v", "1", tmp], check=True)
    frame = cv2.imread(tmp)
    if frame is None:
        continue
    dw = undistort(frame) if undistort is not None else frame
    dets = det.detect(dw)
    n = 0 if dets.xyxy is None else len(dets.xyxy)
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in dets.xyxy[i])
        if (x2 - x1) >= 50 and (y2 - y1) >= 40:
            chosen = (seg, mid, (x1, y1, x2, y2), dw[int(y1):int(y2), int(x1):int(x2)])
            break
    if chosen:
        break

if not chosen:
    raise SystemExit("No vehicle found in the recent ring segments to probe.")

seg, wall_ts, bbox, sub_crop = chosen
sub_score = clf.score(sub_crop)
print(f"segment {Path(seg.path).name}  wall_ts={wall_ts:.1f}  sub bbox={tuple(round(v) for v in bbox)}")
print(f"sub-crop score:  {sub_score:.2f}")

conf = Police4KConfirmer(cfg, clf, log_db=None, sub_size=(704, 480))
crop_4k = conf._crop_4k(wall_ts, bbox)
if crop_4k is None:
    print("4K crop: None (segment/projection miss)")
else:
    cv2.imwrite("samples/_4kprobe_crop.jpg", crop_4k)
    h, w = crop_4k.shape[:2]
    print(f"4K crop: {w}x{h}  score: {clf.score(crop_4k):.2f}  -> samples/_4kprobe_crop.jpg")
