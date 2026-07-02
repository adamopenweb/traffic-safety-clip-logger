"""Render track boxes + speed onto a 4K evidence clip (Approach B annotation).

The boxes were measured in the **de-warped 704x480 sub-stream**; the clip is the
**raw 4K main stream**. Mapping one to the other is two exact stages:

1. de-warped-sub px -> raw-sub px: re-apply the lens distortion. OpenCV's
   ``initUndistortRectifyMap`` already produces exactly this table -- for each
   de-warped (destination) pixel it gives the raw (source) pixel to sample -- so
   we bilinearly look corners up in it.
2. raw-sub px -> raw-4K px: the two streams share an optical centre and (per the
   camera's spec) field of view, so this is a pure scale by the resolution ratio.
   If a stream is actually cropped, a 3x3 ``overlay_homography`` from config
   overrides the scale (fit once from clicked correspondences).

The drawing pass decodes the clean clip, finds the overlay snapshot nearest each
frame's timestamp, projects every box to 4K, and pipes annotated frames to ffmpeg
(libx264) so the companion clip keeps full quality. Best-effort and off the
analysis thread -- a failure here never touches the clean evidence clip.
"""

from __future__ import annotations

import math
import subprocess
from bisect import bisect_left
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from ..util.ffmpeg import ffmpeg_path
from ..util.logging import get_logger
from .overlay_buffer import OverlayBox, OverlaySnapshot, nearest_snapshot

log = get_logger(__name__)

# BGR colours for the box roles.
_PRIMARY = (0, 0, 255)    # the flagged vehicle -> red
_PASSED = (0, 165, 255)   # the vehicle it passed -> orange
_OTHER = (0, 255, 0)      # other tracked traffic -> green


class StreamProjector:
    """Maps a de-warped sub-stream point to a raw main-stream (4K) pixel."""

    def __init__(
        self,
        sub_size: Tuple[int, int],
        main_size: Tuple[int, int],
        k1: float,
        k2: float = 0.0,
        roll_degrees: float = 0.0,
        homography: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        import cv2
        import numpy as np

        self.sub_w, self.sub_h = int(sub_size[0]), int(sub_size[1])
        self.main_w, self.main_h = int(main_size[0]), int(main_size[1])

        fx = fy = float(self.sub_w)
        cx, cy = self.sub_w / 2.0, self.sub_h / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.array([float(k1), float(k2), 0.0, 0.0, 0.0], dtype=np.float64)
        R = None
        if roll_degrees:
            a = math.radians(float(roll_degrees))
            c, s = math.cos(a), math.sin(a)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        # map_x[dy, dx], map_y[dy, dx] = raw-sub pixel feeding de-warped (dx, dy).
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            K, D, R, K, (self.sub_w, self.sub_h), cv2.CV_32FC1
        )
        self._H = np.array(homography, dtype=np.float64) if homography is not None else None
        self._sx = self.main_w / float(self.sub_w)
        self._sy = self.main_h / float(self.sub_h)

    def _sample(self, x: float, y: float) -> Tuple[float, float]:
        """Bilinearly read the re-distort map at de-warped point (x, y)."""
        w, h = self.sub_w, self.sub_h
        x = min(max(x, 0.0), w - 1.0)
        y = min(max(y, 0.0), h - 1.0)
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        fx, fy = x - x0, y - y0
        mx, my = self._map_x, self._map_y
        rx = (mx[y0, x0] * (1 - fx) * (1 - fy) + mx[y0, x1] * fx * (1 - fy)
              + mx[y1, x0] * (1 - fx) * fy + mx[y1, x1] * fx * fy)
        ry = (my[y0, x0] * (1 - fx) * (1 - fy) + my[y0, x1] * fx * (1 - fy)
              + my[y1, x0] * (1 - fx) * fy + my[y1, x1] * fx * fy)
        return float(rx), float(ry)

    def project(self, x: float, y: float) -> Tuple[int, int]:
        """De-warped sub point -> raw 4K pixel (int)."""
        rx, ry = self._sample(x, y)  # stage 1: raw-sub
        if self._H is not None:       # stage 2: raw-sub -> raw-4K (homography)
            d = self._H[2, 0] * rx + self._H[2, 1] * ry + self._H[2, 2]
            if d == 0:
                d = 1e-9
            X = (self._H[0, 0] * rx + self._H[0, 1] * ry + self._H[0, 2]) / d
            Y = (self._H[1, 0] * rx + self._H[1, 1] * ry + self._H[1, 2]) / d
            return int(round(X)), int(round(Y))
        return int(round(rx * self._sx)), int(round(ry * self._sy))  # stage 2: scale

    def project_bbox(self, bbox: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
        """The 4 bbox corners projected to 4K, in TL, TR, BR, BL order."""
        x1, y1, x2, y2 = bbox
        return [self.project(x1, y1), self.project(x2, y1),
                self.project(x2, y2), self.project(x1, y2)]


def _median(vals: Sequence[float]) -> Optional[float]:
    s = sorted(vals)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def aggregate_track_speeds(snapshots: Sequence[OverlaySnapshot]) -> dict:
    """One stable km/h per track: the MEDIAN of its per-frame speeds over the clip.

    The rolling per-frame speed jitters and peaks above a vehicle's true speed, so a
    viewer watching the label sees it jump around. The median collapses that to a
    single steady number (near the gate speed) that we can draw on every frame -- the
    speed jump is noise the viewer doesn't care about; they just want the one figure."""
    from collections import defaultdict

    vals = defaultdict(list)
    for s in snapshots:
        for b in s.boxes:
            if b.speed_kmh is not None:
                vals[b.track_id].append(b.speed_kmh)
    return {tid: _median(v) for tid, v in vals.items() if v}


def _box_label(box: OverlayBox, speed_kmh: Optional[float] = None) -> str:
    parts = [f"#{box.track_id}"]
    kmh = speed_kmh if speed_kmh is not None else box.speed_kmh
    if kmh is not None:
        parts.append(f"{kmh:.0f}km/h")
    elif box.speed_rel is not None:
        parts.append(f"v={box.speed_rel:.2f}")
    return " ".join(parts)


def _draw_box(frame, proj: StreamProjector, box: OverlayBox, color, thickness: int,
              font_scale: float, speed_kmh: Optional[float] = None) -> None:
    import cv2
    import numpy as np

    quad = np.array(proj.project_bbox(box.bbox), dtype=np.int32)
    cv2.polylines(frame, [quad], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    label = _box_label(box, speed_kmh)
    anchor = (int(quad[:, 0].min()), int(quad[:, 1].min()))
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx, ty = anchor[0], max(anchor[1] - 6, th + 6)
    cv2.rectangle(frame, (tx, ty - th - base), (tx + tw, ty + base), color, -1)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness, cv2.LINE_AA)


def _color_for(track_id: int, primary_id: Optional[int], passed_ids) -> Tuple[int, int, int]:
    if primary_id is not None and track_id == primary_id:
        return _PRIMARY
    if track_id in passed_ids:
        return _PASSED
    return _OTHER


def render_annotated_clip(
    clean_clip: str | Path,
    out_path: str | Path,
    snapshots: Sequence[OverlaySnapshot],
    sub_size: Tuple[int, int],
    *,
    start_ts: float,
    k1: float = 0.0,
    k2: float = 0.0,
    roll_degrees: float = 0.0,
    homography: Optional[Sequence[Sequence[float]]] = None,
    primary_id: Optional[int] = None,
    passed_ids: Optional[Sequence[int]] = None,
    primary_speed_kmh: Optional[float] = None,
    sync_offset: float = 0.0,
    snap_tolerance: float = 0.2,
    crf: Optional[int] = None,
) -> Optional[Path]:
    """Burn track boxes + speed onto a copy of ``clean_clip``.

    ``snapshots`` are the overlay frames covering the clip window (absolute wall
    time); ``start_ts`` is the clip's first-frame wall time, so frame *i* maps to
    ``start_ts + i / fps``. ``sync_offset`` is added to that lookup time to cancel
    the pipeline-latency difference between the analysis sub-stream and the
    recorded 4K main (boxes otherwise trail moving cars); a positive value
    advances the boxes forward to where the car is in the 4K frame. Returns the
    written path, or None if nothing was rendered (no snapshots / unreadable clip).
    """
    import cv2

    snaps = sorted(snapshots, key=lambda s: s.ts)
    if not snaps:
        log.warning("No overlay snapshots for %s; skipping annotated clip.", clean_clip)
        return None

    cap = cv2.VideoCapture(str(clean_clip))
    if not cap.isOpened():
        log.warning("Could not open clip %s for annotation.", clean_clip)
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    thickness = max(2, round(width / 640))
    font_scale = max(0.5, width / 1280.0)
    passed = set(int(p) for p in (passed_ids or ()))
    # One steady speed per car for the whole clip (median of its per-frame speeds),
    # so the label doesn't jump; the flagged car uses the exact gate speed (the value
    # in the filename/report) when supplied.
    track_kmh = aggregate_track_speeds(snaps)

    proj = StreamProjector(sub_size, (width, height), k1, k2, roll_degrees, homography)

    ff = ffmpeg_path() or "ffmpeg"
    out = Path(out_path)
    # +faststart moves the moov atom to the front so a web <video> player can start
    # streaming immediately instead of fetching the index from the end of the file
    # first (the "slow to load" symptom over the funnel, esp. for 4K ~20MB clips).
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart"]
    if crf is not None:
        enc += ["-crf", str(int(crf))]
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.4f}", "-i", "-",
        *enc, "-an", str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            snap = nearest_snapshot(snaps, start_ts + i / fps + sync_offset, snap_tolerance)
            if snap is not None:
                for box in snap.boxes:
                    spd = track_kmh.get(box.track_id)
                    if (primary_id is not None and box.track_id == primary_id
                            and primary_speed_kmh is not None):
                        spd = primary_speed_kmh
                    _draw_box(frame, proj, box,
                              _color_for(box.track_id, primary_id, passed),
                              thickness, font_scale, speed_kmh=spd)
            proc.stdin.write(frame.tobytes())
            i += 1
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg annotation failed (exit {proc.returncode}) for {out}")
    log.info("Wrote annotated clip %s (%d frames).", out, i)
    return out


# -- detection-based annotation: draw boxes from the CLIP'S OWN frames ----------
# The drift-prone approach above projects LIVE sub-stream boxes onto the 4K clip
# and needs a per-clip sync offset (the analysis + recording are separate camera
# connections whose relative latency wanders). This path instead re-detects on the
# clip's own frames -- same downscale+undistort prep as analysis, so boxes land in
# the same de-warped space StreamProjector maps to 4K -- so a box drawn on frame i
# is on the car in frame i BY CONSTRUCTION, no offset. The live track is used only
# to pick WHICH detected car is the flagged one (offset-tolerant: it's an identity
# match over a ~1s trajectory, not a per-pixel timing). Runs once per event off the
# hot path, so the per-frame detection cost (bounded to the car's on-screen window)
# never has to keep up with real time.

def track_center_path(
    snapshots: Sequence[OverlaySnapshot], track_id: Optional[int]
) -> List[Tuple[float, Tuple[float, float]]]:
    """A track's de-warped bbox-centre path over the clip: ``[(ts, (cx, cy)), ...]``."""
    out: List[Tuple[float, Tuple[float, float]]] = []
    if track_id is None:
        return out
    for s in snapshots:
        for b in s.boxes:
            if b.track_id == track_id:
                x1, y1, x2, y2 = b.bbox
                out.append((s.ts, ((x1 + x2) / 2.0, (y1 + y2) / 2.0)))
                break
    return out


def _interp_center(times: Sequence[float], xy: Sequence[Tuple[float, float]], t: float):
    """Linear-interpolate the centre at time ``t``; None outside the path's range."""
    n = len(times)
    if n == 0 or t < times[0] or t > times[-1]:
        return None
    j = bisect_left(times, t)
    if j <= 0:
        return xy[0]
    if j >= n:
        return xy[-1]
    t0, t1 = times[j - 1], times[j]
    if t1 == t0:
        return xy[j]
    f = (t - t0) / (t1 - t0)
    (x0, y0), (x1, y1) = xy[j - 1], xy[j]
    return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)


def _match_offset(
    detections: Sequence[Tuple[float, Sequence[Tuple[float, float]]]],
    times: Sequence[float],
    xy: Sequence[Tuple[float, float]],
    *,
    search_center: float,
    radius: float,
    step: float,
    gate_px: float,
    min_matched: int = 3,
) -> Tuple[float, Optional[float]]:
    """Time-shift aligning the live primary path to the clip detections (identity).

    For each candidate offset ``delta``, the live primary's interpolated centre at
    ``clip_ts + delta`` is compared to the nearest clip detection; the offset with
    the lowest gated median distance wins. Returns ``(offset, cost)``; falls back to
    ``search_center`` when too few frames match. This only resolves *identity* (which
    detection is the flagged car) -- the drawn box is the detection itself, so a small
    offset error never misplaces a box, at most mislabels in a dense multi-car frame.
    """
    if len(times) < 2:
        return search_center, None
    best_off, best_cost = search_center, None
    n_steps = int(round(2 * radius / step))
    for i in range(n_steps + 1):
        delta = search_center - radius + i * step
        dists = []
        for clip_ts, centers in detections:
            if not centers:
                continue
            exp = _interp_center(times, xy, clip_ts + delta)
            if exp is None:
                continue
            dists.append(min(min(math.hypot(exp[0] - cx, exp[1] - cy) for cx, cy in centers), gate_px))
        if len(dists) >= min_matched:
            cost = _median(dists)
            if cost is not None and (best_cost is None or cost < best_cost):
                best_off, best_cost = delta, cost
    return round(best_off, 3), best_cost


def _draw_projected_bbox(frame, proj: "StreamProjector", bbox, color, thickness: int,
                         font_scale: float, label: Optional[str]) -> None:
    """Project a de-warped bbox to 4K and draw it (+ optional label)."""
    import cv2
    import numpy as np

    quad = np.array(proj.project_bbox(bbox), dtype=np.int32)
    cv2.polylines(frame, [quad], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    if not label:
        return
    anchor = (int(quad[:, 0].min()), int(quad[:, 1].min()))
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx, ty = anchor[0], max(anchor[1] - 6, th + 6)
    cv2.rectangle(frame, (tx, ty - th - base), (tx + tw, ty + base), color, -1)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness, cv2.LINE_AA)


def render_annotated_clip_detected(
    clean_clip: str | Path,
    out_path: str | Path,
    *,
    detector,
    prep: Callable,
    sub_size: Tuple[int, int],
    start_ts: float,
    k1: float = 0.0,
    k2: float = 0.0,
    roll_degrees: float = 0.0,
    homography: Optional[Sequence[Sequence[float]]] = None,
    primary_path: Sequence[Tuple[float, Tuple[float, float]]],
    passed_paths: Optional[dict] = None,
    primary_speed_kmh: Optional[float] = None,
    primary_track_id: Optional[int] = None,
    search_center: float = 0.0,
    radius: float = 1.4,
    step: float = 0.05,
    pad_seconds: float = 0.4,
    crf: Optional[int] = None,
) -> Optional[Path]:
    """Annotate ``clean_clip`` by detecting on its own frames (no sync offset).

    ``prep`` is the analysis frame transform (downscale + de-warp) so detections
    land in the de-warped space ``sub_size`` describes; ``StreamProjector`` then
    maps each box to the raw-4K clip pixel it was detected at. ``primary_path`` /
    ``passed_paths`` (live de-warped centre tracks) only resolve which detection is
    the flagged / passed car. Detection is limited to the primary's on-screen window
    (+/- ``radius``+``pad_seconds``) so the cost stays bounded. Returns the written
    path, or None if it couldn't run.
    """
    import cv2

    pt = sorted(primary_path, key=lambda p: p[0])
    if len(pt) < 2:
        log.warning("Primary path too short to detect-annotate %s; skipping.", clean_clip)
        return None
    p_times = [p[0] for p in pt]
    p_xy = [p[1] for p in pt]

    cap = cv2.VideoCapture(str(clean_clip))
    if not cap.isOpened():
        log.warning("Could not open clip %s for detect-annotate.", clean_clip)
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    win_lo = p_times[0] - start_ts - radius - pad_seconds
    win_hi = p_times[-1] - start_ts + radius + pad_seconds
    f_lo = max(0, int(math.floor(win_lo * fps)))
    f_hi = int(math.ceil(win_hi * fps))

    # PASS 1 (sequential, no seek so frame indices are exact): detect in-window only.
    dets_by_frame: dict = {}
    detections: List[Tuple[float, List[Tuple[float, float]]]] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f_lo <= i <= f_hi:
            d = detector.detect(prep(frame))
            xyxy = getattr(d, "xyxy", None)
            boxes = [tuple(map(float, b)) for b in xyxy] if xyxy is not None else []
            dets_by_frame[i] = boxes
            detections.append((start_ts + i / fps,
                               [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]))
        i += 1
        if i > f_hi:  # window covered; stop decoding the post-window tail (pass 2 re-reads)
            break

    gate_px = 0.06 * float(sub_size[0])
    delta, cost = _match_offset(detections, p_times, p_xy, search_center=search_center,
                                radius=radius, step=step, gate_px=gate_px)
    log.info("Detect-annotate %s: identity offset %.2fs (cost %s over %d det-frames).",
             Path(clean_clip).name, delta,
             f"{cost:.0f}px" if cost is not None else "n/a", len(detections))

    # Prep the passed-car paths for per-frame identity lookup.
    passed_prep = {}
    for tid, pth in (passed_paths or {}).items():
        sp = sorted(pth, key=lambda p: p[0])
        if len(sp) >= 2:
            passed_prep[tid] = ([p[0] for p in sp], [p[1] for p in sp])

    proj = StreamProjector(sub_size, (width, height), k1, k2, roll_degrees, homography)
    thickness = max(2, round(width / 640))
    font_scale = max(0.5, width / 1280.0)
    assign_gate = 0.10 * float(sub_size[0])
    p_label = None
    if primary_speed_kmh is not None:
        p_label = (f"#{primary_track_id} {primary_speed_kmh:.0f}km/h"
                   if primary_track_id is not None else f"{primary_speed_kmh:.0f}km/h")

    ff = ffmpeg_path() or "ffmpeg"
    out = Path(out_path)
    # +faststart moves the moov atom to the front so a web <video> player can start
    # streaming immediately instead of fetching the index from the end of the file
    # first (the "slow to load" symptom over the funnel, esp. for 4K ~20MB clips).
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart"]
    if crf is not None:
        enc += ["-crf", str(int(crf))]
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
           "-r", f"{fps:.4f}", "-i", "-", *enc, "-an", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = dets_by_frame.get(i)
            if boxes:
                clip_ts = start_ts + i / fps
                exp = _interp_center(p_times, p_xy, clip_ts + delta)
                prim_idx = None
                if exp is not None:
                    best = None
                    for idx, b in enumerate(boxes):
                        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                        dd = math.hypot(cx - exp[0], cy - exp[1])
                        if dd <= assign_gate and (best is None or dd < best):
                            best, prim_idx = dd, idx
                passed_idx = {}
                for tid, (pts, pxy) in passed_prep.items():
                    e = _interp_center(pts, pxy, clip_ts + delta)
                    if e is None:
                        continue
                    best, bi = None, None
                    for idx, b in enumerate(boxes):
                        if idx == prim_idx:
                            continue
                        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                        dd = math.hypot(cx - e[0], cy - e[1])
                        if dd <= assign_gate and (best is None or dd < best):
                            best, bi = dd, idx
                    if bi is not None:
                        passed_idx[bi] = tid
                # Draw ONLY the flagged car (and any car it passed). Drawing every raw
                # detection boxed parked cars in driveways and false positives (road
                # signs) on every frame -- clutter that made the clips "look wrong". The
                # clip is about the violator, so the other detections add noise, not info.
                if prim_idx is not None:
                    _draw_projected_bbox(frame, proj, boxes[prim_idx], _PRIMARY,
                                         thickness, font_scale, p_label)
                for idx, _tid in passed_idx.items():
                    _draw_projected_bbox(frame, proj, boxes[idx], _PASSED,
                                         thickness, font_scale, None)
            proc.stdin.write(frame.tobytes())
            i += 1
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg detect-annotate failed (exit {proc.returncode}) for {out}")
    log.info("Wrote detect-annotated clip %s (%d frames).", out, i)
    return out


# -- per-clip auto-alignment: solve the sync offset from the 4K itself ---------
# The sub<->4K latency jitters between events, so no single static offset keeps
# boxes on cars. Instead we detect the flagged car in a few sampled 4K frames and
# pick the time-shift that best lines the overlay's primary track up with those
# detections -- absorbing the jitter per clip.

def _search_offset(
    samples: Sequence[Tuple[float, Sequence[Tuple[float, float]]]],
    primary_track: Sequence[Tuple[float, Tuple[float, float]]],
    *,
    start_ts: float,
    search_center: float,
    radius: float = 0.6,
    step: float = 0.05,
    tolerance: float = 0.2,
    gate_px: float = 230.0,
    min_matched: int = 3,
    use_median: bool = False,
) -> float:
    """Grid-search the offset minimizing primary-box vs 4K-detection distance.

    ``samples`` are ``(clip_time, [detection_center, ...])`` from the 4K clip;
    ``primary_track`` is the flagged track's projected 4K centers ``(ts, (x, y))``
    sorted by ts. For each candidate offset, the primary's center at
    ``start_ts + clip_time + offset`` is compared to the nearest detection; the
    offset with the lowest gated average distance wins. Returns ``search_center``
    if too few samples matched (no confident estimate). Pure -- unit tested.
    """
    times = [p[0] for p in primary_track]
    if len(times) < 2:
        return search_center
    best_off, best_cost = search_center, None
    n_steps = int(round(2 * radius / step))
    for i in range(n_steps + 1):
        delta = search_center - radius + i * step
        dists = []
        for clip_time, centers in samples:
            if not centers:
                continue
            look = start_ts + clip_time + delta
            j = bisect_left(times, look)
            cands = [k for k in (j - 1, j) if 0 <= k < len(times)]
            if not cands:
                continue
            k = min(cands, key=lambda k: abs(times[k] - look))
            if abs(times[k] - look) > tolerance:
                continue
            cx, cy = primary_track[k][1]
            d = min(math.hypot(cx - dx, cy - dy) for dx, dy in centers)
            dists.append(min(d, gate_px))
        if len(dists) >= min_matched:
            # Median (robust to a few distractor-latched frames) when requested;
            # mean otherwise to preserve the original pure-function semantics.
            cost = _median(dists) if use_median else (sum(dists) / len(dists))
            if cost is not None and (best_cost is None or cost < best_cost):
                best_off, best_cost = delta, cost
    return round(best_off, 3)


def estimate_sync_offset(
    clean_clip: str | Path,
    snapshots: Sequence[OverlaySnapshot],
    sub_size: Tuple[int, int],
    *,
    start_ts: float,
    primary_id: Optional[int],
    detector,
    k1: float = 0.0,
    k2: float = 0.0,
    roll_degrees: float = 0.0,
    homography=None,
    search_center: float = 0.0,
    radius: float = 0.6,
    step: float = 0.05,
    n_samples: int = 24,
    tolerance: float = 0.2,
) -> float:
    """Estimate this clip's sync offset by matching the primary car to YOLO.

    Samples ``n_samples`` 4K frames across the primary's on-screen window, runs
    ``detector`` on each, and returns the offset (near ``search_center``) that
    best aligns the projected primary track to those detections. Falls back to
    ``search_center`` when the primary is absent or no confident match is found.
    """
    import cv2

    if primary_id is None:
        return search_center
    snaps = sorted(snapshots, key=lambda s: s.ts)
    present = [s.ts for s in snaps if any(b.track_id == primary_id for b in s.boxes)]
    if len(present) < 2:
        return search_center
    pa, pb = present[0], present[-1]

    cap = cv2.VideoCapture(str(clean_clip))
    if not cap.isOpened():
        return search_center
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        proj = StreamProjector(sub_size, (width, height), k1, k2, roll_degrees, homography)

        primary_track: List[Tuple[float, Tuple[float, float]]] = []
        for s in snaps:
            for b in s.boxes:
                if b.track_id == primary_id:
                    corners = proj.project_bbox(b.bbox)
                    cx = sum(p[0] for p in corners) / 4.0
                    cy = sum(p[1] for p in corners) / 4.0
                    primary_track.append((s.ts, (cx, cy)))
                    break

        # Sample clip-times covering the car for any offset in the search window.
        lo = (pa - start_ts) - (search_center + radius) - 0.2
        hi = (pb - start_ts) - (search_center - radius) + 0.2
        lo = max(0.0, lo)
        samples: List[Tuple[float, List[Tuple[float, float]]]] = []
        for i in range(n_samples):
            t = lo + (hi - lo) * i / max(n_samples - 1, 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if not ok:
                continue
            dets = detector.detect(frame)
            xyxy = getattr(dets, "xyxy", None)
            centers = ([((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)
                        for x1, y1, x2, y2 in xyxy] if xyxy is not None else [])
            samples.append((t, centers))
    finally:
        cap.release()

    return _search_offset(
        samples, primary_track, start_ts=start_ts, search_center=search_center,
        radius=radius, step=step, tolerance=tolerance, gate_px=0.06 * width,
        use_median=True,
    )


# -- overlay sidecar: persist render inputs for offline re-render / tuning -----
def write_overlay_sidecar(
    path: str | Path,
    snapshots: Sequence[OverlaySnapshot],
    *,
    clean_clip: str,
    sub_size: Tuple[int, int],
    start_ts: float,
    k1: float,
    k2: float,
    roll_degrees: float,
    homography,
    primary_id: Optional[int],
    passed_ids: Sequence[int],
    sync_offset: float,
    primary_speed_kmh: Optional[float] = None,
) -> Path:
    """Dump everything :func:`reannotate_from_sidecar` needs to re-render a clip.

    Lets the sync offset (or colours/homography) be retuned offline and the clip
    re-rendered, without waiting for fresh live events.
    """
    import json

    from .overlay_buffer import serialize_snapshots

    payload = {
        "clean_clip": clean_clip,
        "sub_size": list(sub_size),
        "start_ts": start_ts,
        "k1": k1, "k2": k2, "roll_degrees": roll_degrees,
        "homography": [list(r) for r in homography] if homography is not None else None,
        "primary_id": primary_id,
        "passed_ids": list(passed_ids),
        "primary_speed_kmh": primary_speed_kmh,
        "sync_offset": sync_offset,
        "snapshots": serialize_snapshots(snapshots),
    }
    out = Path(path)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def reannotate_from_sidecar(
    sidecar: str | Path,
    out_path: str | Path,
    *,
    sync_offset: Optional[float] = None,
    clean_clip: Optional[str | Path] = None,
) -> Optional[Path]:
    """Re-render an annotated clip from a saved sidecar.

    ``sync_offset`` overrides the stored value (use to sweep the offset);
    ``clean_clip`` overrides the stored clip path (e.g. if files moved).
    """
    import json

    from .overlay_buffer import deserialize_snapshots

    data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    snaps = deserialize_snapshots(data["snapshots"])
    offset = data["sync_offset"] if sync_offset is None else float(sync_offset)
    return render_annotated_clip(
        clean_clip if clean_clip is not None else data["clean_clip"],
        out_path, snaps, tuple(data["sub_size"]),
        start_ts=float(data["start_ts"]),
        k1=float(data["k1"]), k2=float(data["k2"]), roll_degrees=float(data["roll_degrees"]),
        homography=data.get("homography"),
        primary_id=data.get("primary_id"), passed_ids=data.get("passed_ids") or (),
        primary_speed_kmh=data.get("primary_speed_kmh"),
        sync_offset=offset,
    )
