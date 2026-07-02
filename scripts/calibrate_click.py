"""Interactive click calibration on the DE-WARPED frame.

Click the road quad and the interior lane lines directly on the (de-warped +
roll-corrected) image; the tool computes source_points + band_widths in the same
space the analyzer uses and writes them to samples/calib_result.json (and prints
them) so they can be dropped into the config.

Run on a machine with a display (e.g. in your own shell so the window appears):
    .venv\\Scripts\\python.exe scripts/calibrate_click.py samples/lanecheck.jpg [config.yaml]

Workflow:
  1. Click the 4 road corners IN ORDER: far-left, far-right, near-right, near-left.
     (The quad draws once all 4 are placed.)
  2. Click on each interior lane line, far -> near (4 clicks for the 5 bands:
     bike_a | travel_a | center | travel_b | bike_b). Each click becomes a band
     boundary at the across-road position it projects to.
Keys:  u = undo last point   r = restart   w = write/print result   q = quit
"""
import json
import sys

import cv2
import numpy as np

from traffic_logger.analyze.project import PerspectiveTransform, _resolve_swap
from traffic_logger.analyze.undistort import build_undistorter
from traffic_logger.config import load_config

# Display is fit to MAX_DISP so a 4K frame doesn't overflow the screen (the old
# fixed 2x upscale turned 3840x2160 into 7680x4320). SCALE is derived from the
# image size below; '+'/'-' zoom live for click precision.
MAX_DISP_W, MAX_DISP_H = 1440, 810

src = sys.argv[1] if len(sys.argv) > 1 else "samples/lanecheck.jpg"
cfg_path = sys.argv[2] if len(sys.argv) > 2 else "config/config.camera.yaml"
cal = load_config(cfg_path).calibration

base = cv2.imread(src)
if base is None:
    raise SystemExit(f"Could not read {src}")
und = build_undistorter(cal)
if und is not None:
    base = und(base)            # click in the same space the pipeline uses
H, W = base.shape[:2]
# Fit to screen: down-scale for 4K, up-scale for the small substream frame.
SCALE = min(MAX_DISP_W / W, MAX_DISP_H / H)

corners = []   # up to 4 (de-warped px)
lanes = []     # lane-line clicks (de-warped px)
grid_on = True # metric 1 m grid overlay (toggle with 'g')


def order_corners(pts):
    """Sort 4 clicked points into [far-left, far-right, near-right, near-left]
    regardless of click order (TL=min x+y, BR=max x+y, TR=max x-y, BL=min x-y)."""
    p = np.array(pts, dtype=float)
    s = p.sum(axis=1)
    d = p[:, 0] - p[:, 1]
    return [tuple(p[np.argmin(s)]), tuple(p[np.argmax(d)]),
            tuple(p[np.argmax(s)]), tuple(p[np.argmin(d)])]


def projector():
    if len(corners) < 4:
        return None
    return PerspectiveTransform(
        order_corners(corners),
        target_width_units=float(cal.get("target_width_units", 1.0)),
        target_length_units=float(cal.get("target_length_units", 1.0)),
        swap_xy=_resolve_swap(cal.get("swap_axes", "auto"), corners),
    )


def on_mouse(event, x, y, _flags, _param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    px, py = x / SCALE, y / SCALE
    if len(corners) < 4:
        corners.append((px, py))
    else:
        lanes.append((px, py))


def boundaries():
    """Across-road positions (sorted) of the lane-line clicks."""
    proj = projector()
    if proj is None:
        return []
    vals = sorted(max(0.0, min(1.0, proj.project(px, py)[0])) for px, py in lanes)
    return vals


def band_widths():
    b = boundaries()
    if len(b) != 4:
        return None
    return [round(b[0], 4), round(b[1] - b[0], 4), round(b[2] - b[1], 4),
            round(b[3] - b[2], 4), round(1.0 - b[3], 4)]


def draw_grid(img, proj):
    """Project a 1 m ground grid back onto the image to sanity-check the metric scale.

    Across-road lines (parallel to traffic) run far-curb -> near-curb every 1 m;
    along-road lines (perpendicular to traffic) every 1 m, labelled in metres. In a
    correct calibration the along-road lines bunch closer together toward the far
    curb (real perspective), and known objects read true size: a travel lane spans
    ~3.5 m across, a typical car ~4.5 m along. If the spacing looks uniform (no
    perspective foreshortening) or a car spans the wrong number of cells, the quad
    or the target_*_units are off.
    """
    wu = float(cal.get("target_width_units", 1.0))    # across, curb-to-curb metres
    lu = float(cal.get("target_length_units", 1.0))   # along-road metres
    grid = (90, 200, 90)
    m = 0.0
    while m <= wu + 1e-6:                              # lines running ALONG the road
        an = min(1.0, m / wu)
        a = proj.unproject(an, 0.0); b = proj.unproject(an, 1.0)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), grid, 1, cv2.LINE_AA)
        m += 1.0
    m = 0.0
    while m <= lu + 1e-6:                              # lines running ACROSS the road
        ln = min(1.0, m / lu)
        a = proj.unproject(0.0, ln); b = proj.unproject(1.0, ln)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), grid, 1, cv2.LINE_AA)
        cv2.putText(img, f"{int(round(m))}m", (int(a[0]) + 3, int(a[1]) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 255, 120), 1, cv2.LINE_AA)
        m += 1.0


def render():
    img = base.copy()
    proj = projector()
    if proj is not None:
        if grid_on:
            draw_grid(img, proj)
        for pos in boundaries():
            a = proj.unproject(pos, 0.0)
            b = proj.unproject(pos, 1.0)
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 255), 1)
        pts = np.array(proj.source_points, dtype=np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, (255, 0, 0), 1)
    shown = proj.source_points if proj is not None else corners
    for i, (px, py) in enumerate(shown):
        cv2.circle(img, (int(px), int(py)), 4, (0, 0, 255), -1)
        cv2.putText(img, (f"C{i}" if proj is not None else str(i + 1)),
                    (int(px) + 5, int(py) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for px, py in lanes:
        cv2.circle(img, (int(px), int(py)), 3, (0, 255, 0), -1)

    big = cv2.resize(img, (int(W * SCALE), int(H * SCALE)), interpolation=cv2.INTER_LINEAR)
    if len(corners) < 4:
        msg = f"Click the 4 road corners, any order ({len(corners)}/4 placed)"
    else:
        msg = f"Lanes far->near ({len(lanes)}).  g=grid({'on' if grid_on else 'off'}) +/-=zoom u=undo r=reset w=write q=quit"
    cv2.rectangle(big, (0, 0), (big.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(big, msg, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return big


def write_result():
    out = {"source_points": [[round(x, 1), round(y, 1)] for x, y in order_corners(corners)]}
    bw = band_widths()
    if bw is not None:
        out["band_widths"] = bw
    out["lane_clicks"] = [[round(x, 1), round(y, 1)] for x, y in lanes]  # raw, for re-derive
    with open("samples/calib_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("WROTE samples/calib_result.json:")
    print(json.dumps(out, indent=2))


cv2.namedWindow("calibrate", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("calibrate", on_mouse)
while True:
    cv2.imshow("calibrate", render())
    key = cv2.waitKey(20) & 0xFF
    if key == ord("q"):
        break
    if key == ord("u"):
        if len(corners) == 4 and lanes:
            lanes.pop()
        elif corners:
            corners.pop()
    elif key == ord("g"):
        grid_on = not grid_on
    elif key in (ord("+"), ord("=")):
        SCALE = min(SCALE * 1.25, 4.0)
    elif key == ord("-"):
        SCALE = max(SCALE * 0.8, 0.1)
    elif key == ord("r"):
        corners.clear()
        lanes.clear()
    elif key == ord("w"):
        if len(corners) == 4:
            write_result()
        else:
            print("Place all 4 corners before writing.")
cv2.destroyAllWindows()
