"""Sanity-check the police CLIP classifier on real vehicle crops.

Runs the YOLO detector over frames sampled from event clips (or any video),
crops each detected vehicle, and prints the police-class probability. Use it to
see how ordinary traffic scores (should sit well below the threshold) and, once
a real cruiser is captured, to confirm it scores high -- then set
events.police.confidence_threshold accordingly.

    .venv\\Scripts\\python.exe scripts/police_check.py [clip_or_dir] [config.yaml]

With no clip, samples the newest few relative_speeding clips under data/events.
"""
import subprocess
import sys
from pathlib import Path

import cv2

from traffic_logger.analyze.detector import build_detector
from traffic_logger.analyze.police_classifier import PoliceClassifier
from traffic_logger.analyze.undistort import build_undistorter
from traffic_logger.config import load_config
from traffic_logger.util.ffmpeg import ffmpeg_path

arg = sys.argv[1] if len(sys.argv) > 1 else "data/events"
cfg_path = sys.argv[2] if len(sys.argv) > 2 else "config/config.run.local.yaml"
cfg = load_config(cfg_path)

# Resolve a list of clips to sample.
p = Path(arg)
if p.is_dir():
    clips = sorted(p.rglob("*relative_speeding*.mp4"),
                   key=lambda f: f.stat().st_mtime)
    clips = [c for c in clips if "_annotated" not in c.name][-4:]
else:
    clips = [p]
if not clips:
    raise SystemExit(f"No clips found under {arg}")

det = build_detector(cfg)
clf = PoliceClassifier(
    device=str((cfg.events.get("police") or {}).get("device")
               or cfg.analysis.get("device", "cuda")))
undistort = build_undistorter(cfg.calibration)  # match the live (de-warped) crops
threshold = float((cfg.events.get("police") or {}).get("confidence_threshold", 0.5))

print(f"Scoring {len(clips)} clip(s); threshold={threshold}\n")
for clip in clips:
    tmp = Path("samples/_police_probe.jpg")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # Grab a frame near the middle of the clip (the flagged car is on-screen).
    subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error", "-ss", "12",
                    "-i", str(clip), "-frames:v", "1", str(tmp)], check=True)
    frame = cv2.imread(str(tmp))
    if frame is None:
        print(f"{clip.name}: could not read frame"); continue
    if undistort is not None:
        frame = undistort(frame)
    dets = det.detect(frame)
    n = 0 if dets.xyxy is None else len(dets.xyxy)
    scores = []
    for i in range(n):
        x1, y1, x2, y2 = (int(v) for v in dets.xyxy[i])
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1) < 40 or (y2 - y1) < 40:
            continue
        scores.append(clf.score(crop))
    tag = ""
    if scores:
        hi = max(scores)
        tag = "  <-- ABOVE THRESHOLD" if hi >= threshold else ""
    pretty = ", ".join(f"{s:.2f}" for s in sorted(scores, reverse=True)) or "no vehicles"
    print(f"{clip.name}: [{pretty}]{tag}")
print("\nDone. Ordinary traffic should score low; tune the threshold once a real "
      "cruiser is captured.")
