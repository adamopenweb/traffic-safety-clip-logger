"""Contact sheet of ring footage over a wall-clock window, to eyeball a vehicle.

Samples one frame every STEP seconds from the 4K ring across [start,end] (local
time, today), labels each with its time, and tiles them into one image so a
distinctive car can be spotted and its exact second read off.

    .venv\\Scripts\\python.exe scripts/montage.py 16:18:00 16:18:20 0.5
"""
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from traffic_logger.capture.segment_index import SegmentIndex
from traffic_logger.config import load_config

start_s, end_s = sys.argv[1], sys.argv[2]
step = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
cfg = load_config("config/config.run.local.yaml")
tz = ZoneInfo(cfg.app.timezone)


def to_unix(hms):
    d = datetime.now(tz).date()
    h, m, *s = hms.split(":")
    return datetime(d.year, d.month, d.day, int(h), int(m), int(s[0]) if s else 0, tzinfo=tz).timestamp()


start_ts, end_ts = to_unix(start_s), to_unix(end_s)
TW, TH = 360, 203  # tile size (16:9)
thumbs = []
with SegmentIndex(cfg.recording.get("segment_index_path")) as idx:
    segs = idx.get_overlapping(start_ts, end_ts)
next_t = start_ts
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
        if ts < next_t or ts > end_ts:
            continue
        t = cv2.resize(frame, (TW, TH), interpolation=cv2.INTER_AREA)
        lbl = datetime.fromtimestamp(ts, tz).strftime("%H:%M:%S.") + f"{int((ts%1)*10)}"
        cv2.rectangle(t, (0, 0), (118, 18), (0, 0, 0), -1)
        cv2.putText(t, lbl, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        thumbs.append(t)
        next_t += step
    cap.release()

if not thumbs:
    raise SystemExit("no frames in window")
cols = 6
rows = (len(thumbs) + cols - 1) // cols
sheet = np.zeros((rows * TH, cols * TW, 3), dtype=np.uint8)
for k, t in enumerate(thumbs):
    r, c = divmod(k, cols)
    sheet[r*TH:(r+1)*TH, c*TW:(c+1)*TW] = t
out = "samples/_montage.jpg"
cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"{len(thumbs)} frames -> {out}  ({cols*TW}x{rows*TH})")
