"""Compare police/civilian prompt sets on real vehicle crops.

Detects vehicles in each image, crops them, and scores the police-class
probability under the DEFAULT prompts vs a CANDIDATE set, side by side, so the
prompts can be tuned to push known civilians (e.g. a plain black SUV) below
threshold without a model change. Edit CANDIDATE_* below and re-run.

    .venv\\Scripts\\python.exe scripts/police_prompt_tune.py img1.jpg [img2.jpg ...]
"""
import sys
from pathlib import Path

import cv2

from traffic_logger.analyze.detector import build_detector
from traffic_logger.analyze.police_classifier import (
    DEFAULT_CIVILIAN_PROMPTS,
    DEFAULT_POLICE_PROMPTS,
    PoliceClassifier,
)
from traffic_logger.analyze.undistort import build_undistorter
from traffic_logger.config import load_config

# Candidate prompts: key on unambiguous police hardware, and explicitly cover
# dark/plain civilian SUVs+sedans so body-colour alone doesn't read as police.
CANDIDATE_POLICE = [
    "a police car with a roof-mounted emergency light bar",
    "a police SUV with POLICE text and reflective livery decals on the doors",
    "a law enforcement patrol vehicle with a push bumper and roof lights",
    "a police cruiser with blue and red emergency lights",
]
CANDIDATE_CIVILIAN = [
    "an ordinary civilian car with no markings",
    "a plain black SUV with no light bar",
    "a dark colored sedan or SUV",
    "a compact crossover SUV like a Toyota RAV4 or Honda CR-V",
    "a pickup truck, minivan, or delivery van",
    "a garbage truck or waste collection truck",
    "a large work truck, dump truck, or construction vehicle",
    "a city bus or a commercial box truck",
]

imgs = sys.argv[1:] or ["samples/_sight_b0.jpg"]
cfg = load_config("config/config.run.local.yaml")
det = build_detector(cfg)
undistort = build_undistorter(cfg.calibration)
device = str((cfg.events.get("police") or {}).get("device") or cfg.analysis.get("device", "cuda"))

default_clf = PoliceClassifier(device=device)  # DEFAULT_* prompts
cand_clf = PoliceClassifier(device=device, police_prompts=CANDIDATE_POLICE,
                            civilian_prompts=CANDIDATE_CIVILIAN)

print(f"{'image / crop':<34}  {'default':>8}  {'candidate':>9}")
for ip in imgs:
    frame = cv2.imread(str(ip))
    if frame is None:
        print(f"{Path(ip).name}: unreadable"); continue
    if undistort is not None:
        frame = undistort(frame)
    dets = det.detect(frame)
    n = 0 if dets.xyxy is None else len(dets.xyxy)
    for i in range(n):
        x1, y1, x2, y2 = (int(v) for v in dets.xyxy[i])
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1) < 40 or (y2 - y1) < 40:
            continue
        d = default_clf.score(crop)
        c = cand_clf.score(crop)
        print(f"{Path(ip).name + f' [{x2-x1}x{y2-y1}]':<34}  {d:>8.2f}  {c:>9.2f}")
print("\n(Lower is better for these civilian crops.)")
