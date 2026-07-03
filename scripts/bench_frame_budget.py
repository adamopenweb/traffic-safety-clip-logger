"""Measure the live-analysis frame budget, stage by stage.

Times each stage of the analysis loop (downscale, de-warp, YOLO inference,
tracking/projection, rules + overlay bookkeeping) plus the off-loop decode
cost, and compares the loop total against the frame budget implied by
``analysis.inference_fps``. Medians over ``--reps`` runs, so it is fair to run
while the production analyzer is live (that contention is the deployment
reality anyway). The numbers published in ``docs/challenges.md`` ("Fast
near-lane cars kept fragmenting") came from this script.

By default it benches the newest ring segment (real footage, the real codec)
with the live config's calibration and model settings:

    .venv/Scripts/python.exe scripts/bench_frame_budget.py
    .venv/Scripts/python.exe scripts/bench_frame_budget.py --source clip.mp4 --reps 100

Requires the CV stack (the `analyze` extra) and a CUDA torch for GPU numbers.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def med_ms(fn, reps: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(xs)


def newest_ring_segment(ring_path: Path) -> Path | None:
    # Only finalized segments in dated folders -- incoming/ holds the segment
    # ffmpeg is still writing (unreadable: no moov atom yet).
    segs = sorted(p for p in ring_path.glob("*/segment_*.mp4")
                  if p.parent.name != "incoming")
    return segs[-1] if segs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/config.run.local.yaml")
    ap.add_argument("--source", default=None,
                    help="Video to bench against (default: newest ring segment).")
    ap.add_argument("--reps", type=int, default=60)
    args = ap.parse_args()

    import cv2

    from traffic_logger.config import load_config

    cfg = load_config(args.config)
    source = args.source
    if source is None:
        seg = newest_ring_segment(Path(cfg.recording.get("ring_path", "data/ring")))
        if seg is None:
            print("no ring segments found; pass --source <video>")
            return 1
        source = str(seg)

    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    if not ok:
        print(f"could not read a frame from {source}")
        return 1
    h, w = frame.shape[:2]
    fps_target = float(cfg.analysis.get("inference_fps", 30))
    budget = 1000.0 / fps_target
    print(f"source: {source} ({w}x{h}); budget {budget:.1f} ms @ {fps_target:g} fps\n")

    # 0) decode (runs on the grabber thread, off the analysis loop)
    def decode():
        ok, _ = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap.read()
    decode_ms = med_ms(decode, args.reps * 2)

    # 1) downscale to analysis width (no-op when the source is already small)
    analyze_w = int(cfg.analysis.get("analyze_max_width", 0) or 0)
    if analyze_w and analyze_w < w:
        tw, th = analyze_w, round(h * analyze_w / w)
        resize_ms = med_ms(
            lambda: cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR),
            args.reps)
        small = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        resize_ms, small = 0.0, frame

    # 2) de-warp at analysis size, plus the rejected full-res ordering
    from traffic_logger.analyze.undistort import build_undistorter
    und = build_undistorter(cfg.calibration)
    if und is not None:
        und(small)  # build the cached remap tables before timing
        dewarp_ms = med_ms(lambda: und(small), args.reps)
        und_full = build_undistorter(cfg.calibration)
        und_full(frame)
        dewarp_full_ms = med_ms(lambda: und_full(frame), max(10, args.reps // 2))
        prepped = und(small)
    else:
        dewarp_ms = dewarp_full_ms = 0.0
        prepped = small

    # 3) detector inference, warm
    from traffic_logger.analyze.detector import build_detector
    det = build_detector(cfg)
    infer_ms = med_ms(lambda: det.detect(prepped), args.reps, warmup=10)
    detections = det.detect(prepped)

    # 4) ByteTrack + track store + road projection/lane assignment
    from traffic_logger.analyze.project import build_transform
    from traffic_logger.analyze.tracker import VehicleTracker
    projector = build_transform(cfg.calibration)
    tracker = VehicleTracker(cfg, projector=projector)
    clock = [0.0]

    def track():
        clock[0] += 1.0 / fps_target
        tracker.update(detections, clock[0])
    track_ms = med_ms(track, args.reps)

    # 5) rules + overlay bookkeeping (needs calibration; skipped without it)
    rules_ms = 0.0
    if projector is not None:
        from traffic_logger.analyze.live import _overlay_boxes, _run_rules
        from traffic_logger.analyze.metrics import SpeedEstimator, metric_scale
        from traffic_logger.analyze.rules.relative_speeding import RelativeSpeedingRule
        from traffic_logger.events.manager import EventManager

        mpu = metric_scale(cfg.calibration)
        est = SpeedEstimator(3600.0)
        rule = RelativeSpeedingRule(cfg.events.get("relative_speeding", {}),
                                    meters_per_unit=mpu,
                                    calibration=cfg.calibration)
        mgr = EventManager()
        tracks = tracker.update(detections, clock[0])

        def rules():
            _run_rules(tracks, clock[0], est, rule, None, mgr, 0.5)
            _overlay_boxes(tracks, mpu, 0.5, cfg.calibration, static_span_min=0.12)
        rules_ms = med_ms(rules, args.reps)

    loop = resize_ms + dewarp_ms + infer_ms + track_ms + rules_ms
    rows = [
        ("decode (grabber thread, off-loop)", decode_ms),
        (f"downscale {w}->{analyze_w or w}px", resize_ms),
        ("de-warp remap @analysis size", dewarp_ms),
        ("de-warp remap @full res (rejected order)", dewarp_full_ms),
        ("detector inference", infer_ms),
        ("tracking + store + projection", track_ms),
        ("rules + overlay bookkeeping", rules_ms),
    ]
    for label, ms in rows:
        print(f"  {label:<42} {ms:6.1f} ms")
    print(f"  {'ANALYSIS-LOOP TOTAL':<42} {loop:6.1f} ms  "
          f"({loop / budget:.0%} of the {budget:.1f} ms budget)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
