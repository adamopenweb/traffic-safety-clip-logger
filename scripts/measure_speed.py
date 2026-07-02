"""Measure a specific vehicle's speed from the ring (drive-by calibration).

Replays the recorded 4K ring footage for a wall-clock window, downscaled to the
sub-stream's 704x480 so the de-warp + road-quad calibration apply exactly as they
do live, runs the same detect -> ByteTrack -> project -> speed pipeline, and
prints every track's computed km/h. Use it to read the km/h the system assigns to
a known drive-by car, then scale target_length_units by (true / reported).

    .venv\\Scripts\\python.exe scripts/measure_speed.py 16:17:50 16:19:10 [config]

Times are local (config timezone), today's date. Saves a labelled thumbnail per
track to samples/_spd_track<id>.jpg so the target car can be identified.
"""
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from traffic_logger.analyze.detector import build_detector
from traffic_logger.analyze.metrics import across_speed_factor, metric_scale, speed_kmh
from traffic_logger.analyze.tracker import VehicleTracker
from traffic_logger.analyze.project import build_transform
from traffic_logger.analyze.undistort import build_undistorter
from traffic_logger.capture.segment_index import SegmentIndex
from traffic_logger.config import load_config

start_s, end_s = sys.argv[1], sys.argv[2]
cfg_path = sys.argv[3] if len(sys.argv) > 3 else "config/config.run.local.yaml"
cfg = load_config(cfg_path)
tz = ZoneInfo(cfg.app.timezone)


def _to_unix(hms: str) -> float:
    today = datetime.now(tz).date()
    h, m, *s = hms.split(":")
    sec = int(s[0]) if s else 0
    return datetime(today.year, today.month, today.day, int(h), int(m), sec, tzinfo=tz).timestamp()


start_ts, end_ts = _to_unix(start_s), _to_unix(end_s)
scale = metric_scale(cfg.calibration)
# Optional scale override (diagnostics): args 4,5 = width_m length_m.
import os
if os.environ.get("SCALE_OVERRIDE"):
    w, l = os.environ["SCALE_OVERRIDE"].split(",")
    scale = (float(w), float(l))
print(f"window {start_s}..{end_s}  metric scale (W,L)m = {scale}")

undistort = build_undistorter(cfg.calibration)
detector = build_detector(cfg)
projector = build_transform(cfg.calibration)
tracker = VehicleTracker(cfg, projector=projector)
SW = 704, 480
speed_window = float(cfg.analysis.get("speed_window_seconds", 0.5))

with SegmentIndex(cfg.recording.get("segment_index_path")) as idx:
    segs = idx.get_overlapping(start_ts, end_ts)
print(f"{len(segs)} segment(s) cover the window")

# Keep one representative frame per track id for the thumbnail.
thumb = {}
for seg in segs:
    cap = cv2.VideoCapture(seg.path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        ts = seg.start_ts + i / fps
        if ts < start_ts or ts > end_ts:
            continue
        small = cv2.resize(frame, SW, interpolation=cv2.INTER_AREA)
        dw = undistort(small) if undistort is not None else small
        tracks = tracker.update(detector.detect(dw), ts)
        for t in tracks:
            b = t.latest_bbox
            if b is not None:
                thumb[t.track_id] = (dw.copy(), b)
    cap.release()


def kmh_samples(gh):
    out = []
    for i in range(2, len(gh) + 1):
        v = speed_kmh(gh[:i], speed_window, scale)
        if v is not None:
            out.append(v)
    return out


def e2e_kmh(gh):
    """Clean average speed: trimmed end-to-end ground displacement / time."""
    if len(gh) < 6 or scale is None:
        return None
    k = max(1, len(gh) // 10)            # trim ~10% off each end (edge jitter)
    t0, x0, y0 = gh[k]
    t1, x1, y1 = gh[-1 - k]
    dt = t1 - t0
    if dt <= 0:
        return None
    import math
    dist = math.hypot((x1 - x0) * scale[0], (y1 - y0) * scale[1])
    return dist / dt * 3.6


print(f"\n{'id':>4} {'t+s':>5} {'nobs':>4} {'dir':>13} {'med kmh':>8} {'max kmh':>8} "
      f"{'e2e kmh':>8} {'gx':>5} {'x0->x1':>11} {'botY':>5}")
rows = []
for t in tracker.store.active_tracks(min_observations=4):
    gh = t.ground_point_history()
    ks = kmh_samples(gh)
    if not ks:
        continue
    bc = t.bottom_center_history()
    x0, x1 = bc[0][1], bc[-1][1]
    by = sum(p[2] for p in bc) / len(bc)
    gx = sum(p[1] for p in gh) / len(gh)   # mean across-road ground position
    med = sorted(ks)[len(ks) // 2]
    fac = across_speed_factor(gh, cfg.calibration)  # across-road correction
    e2e = e2e_kmh(gh)
    rows.append((t.first_ts, t.track_id, len(t), t.direction(), med * fac, max(ks) * fac,
                 (e2e * fac if e2e is not None else None), gx, x0, x1, by))
rows.sort()
for first, tid, n, d, med, mx, e2e, gx, x0, x1, by in rows:
    e2es = f"{e2e:.1f}" if e2e is not None else "  -"
    print(f"{tid:>4} {first-start_ts:>5.1f} {n:>4} {str(d):>13} {med:>8.1f} {mx:>8.1f} "
          f"{e2es:>8} {gx:>5.2f} {x0:>5.0f}->{x1:<5.0f} {by:>5.0f}")
    img, b = thumb.get(tid, (None, None))
    if img is not None:
        im = img.copy()
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.putText(im, f"#{tid} {med:.0f}km/h", (int(b[0]), int(b[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(f"samples/_spd_track{tid}.jpg", im)
print("\nx increasing = left->right in frame; larger botY = nearer lane.")
