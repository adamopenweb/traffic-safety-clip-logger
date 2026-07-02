"""Calibrate the police threshold on REAL sub-stream crops (production-equivalent).

Samples a captured sub-stream clip, de-warps each frame exactly as the live
analyzer does, detects vehicles, and scores every crop under the DEFAULT and a
CANDIDATE prompt set. Prints the per-set score distribution so the threshold can
be set above the daytime civilian max (with margin).

    .venv\\Scripts\\python.exe scripts/police_sub_calib.py [clip.mp4] [stride_frames]
"""
import sys

import cv2

from traffic_logger.analyze.detector import build_detector
from traffic_logger.analyze.police_classifier import PoliceClassifier
from traffic_logger.analyze.undistort import build_undistorter
from traffic_logger.config import load_config
from scripts.police_prompt_tune import CANDIDATE_CIVILIAN, CANDIDATE_POLICE  # reuse

clip = sys.argv[1] if len(sys.argv) > 1 else "data/_subprobe.mp4"
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 8

cfg = load_config("config/config.run.local.yaml")
det = build_detector(cfg)
undistort = build_undistorter(cfg.calibration)
device = str((cfg.events.get("police") or {}).get("device") or cfg.analysis.get("device", "cuda"))
default_clf = PoliceClassifier(device=device)
cand_clf = PoliceClassifier(device=device, police_prompts=CANDIDATE_POLICE,
                            civilian_prompts=CANDIDATE_CIVILIAN)

cap = cv2.VideoCapture(clip)
defaults, cands = [], []
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    i += 1
    if i % stride != 0:
        continue
    if undistort is not None:
        frame = undistort(frame)
    dets = det.detect(frame)
    n = 0 if dets.xyxy is None else len(dets.xyxy)
    for j in range(n):
        x1, y1, x2, y2 = (int(v) for v in dets.xyxy[j])
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1) < 40 or (y2 - y1) < 40:
            continue
        defaults.append(default_clf.score(crop))
        cands.append(cand_clf.score(crop))
cap.release()


def summary(name, xs):
    if not xs:
        print(f"{name}: no crops"); return
    xs = sorted(xs)
    n = len(xs)
    over5 = sum(1 for x in xs if x >= 0.5)
    over7 = sum(1 for x in xs if x >= 0.7)
    print(f"{name}: n={n} max={xs[-1]:.2f} p95={xs[int(0.95*(n-1))]:.2f} "
          f"median={xs[n//2]:.2f}  >=0.5:{over5}  >=0.7:{over7}")
    print("   top: " + ", ".join(f"{x:.2f}" for x in xs[-8:]))


print(f"Scored {len(defaults)} civilian sub-crops from {clip}\n")
summary("DEFAULT  ", defaults)
summary("CANDIDATE", cands)
