"""Offline analyzer.

``run_offline`` is the real Milestone 2 pipeline: read a video file, run the
detector at the configured inference fps, track vehicles with ByteTrack, and
(optionally) write an annotated debug video with boxes + track ids. It returns
a structured summary.

``run_stub_pipeline`` (Milestone 0) remains as a dependency-free fallback so
``traffic-log test`` still works on a core-only box: the CLI calls
``run_offline`` and falls back to the stub if the CV stack is unavailable.

The real pipeline lazily imports cv2/supervision so importing this module never
requires the analyze extra.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import Config
from ..util.ffmpeg import ffmpeg_available
from ..util.logging import get_logger
from ..util.paths import data_dir, ensure_dir
from . import lane_model

log = get_logger(__name__)


class MissingCVDependencies(ImportError):
    """Raised when the offline analysis stack (cv2/supervision/yolo) is absent."""


def _frame_stride(source_fps: float, inference_fps: float) -> int:
    """How many source frames to skip between processed frames."""
    if inference_fps <= 0 or source_fps <= 0:
        return 1
    return max(1, round(source_fps / inference_fps))


def run_offline(
    source: str,
    config: Config,
    *,
    detector: Any = None,
    output_dir: Optional[str | Path] = None,
    max_frames: Optional[int] = None,
    export_events: bool = False,
    events_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run detection + tracking over a video file.

    Parameters
    ----------
    detector:
        Optional detector instance to inject (e.g. a ScriptedDetector for tests
        or demos). Defaults to the configured YOLO detector.
    output_dir:
        Where to write the annotated debug video. Defaults to ``data/debug``.
        A debug video is written when ``analysis.save_debug_video`` is true or
        ``output_dir`` is given explicitly.
    max_frames:
        Optional cap on the number of *processed* frames (for quick runs/tests).
    """
    try:
        import cv2
        import supervision as sv
    except ImportError as exc:
        raise MissingCVDependencies(
            "run_offline requires the 'analyze' extra (opencv-python + supervision)."
        ) from exc

    from .detector import MissingDetectorDependency, build_detector
    from .project import build_transform
    from .tracker import VehicleTracker
    from .undistort import build_undistorter

    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source video not found: {src}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    inference_fps = float(config.analysis.get("inference_fps", 12))
    stride = _frame_stride(source_fps, inference_fps)

    try:
        det = build_detector(config, detector=detector)
    except MissingDetectorDependency as exc:
        cap.release()
        raise MissingCVDependencies(str(exc)) from exc

    # Lens de-warp: corrects residual barrel distortion so straight curbs/lanes
    # stay straight (planar homography + 4-point road quad both depend on it).
    # Applied to frames before detection; source_points are in de-warped space.
    undistorter = build_undistorter(config.calibration)
    if undistorter is not None:
        log.info("Lens de-warp enabled (k1=%.3f, k2=%.3f).", undistorter.k1, undistorter.k2)

    projector = build_transform(config.calibration)
    if projector is None:
        log.warning(
            "No 4-point calibration in config; lane bands + speed disabled. "
            "Run `traffic-log calibrate` to enable lane-band/speed analysis."
        )
    tracker = VehicleTracker(config, projector=projector)

    # Speed estimation + relative-speeding rule require calibration (projected
    # ground points). Without it, tracking/lanes still run but speed is skipped.
    speed_estimator = None
    speeding_rule = None
    center_rule = None
    speed_window = float(config.analysis.get("speed_window_seconds", 0.5))
    meters_per_unit = None
    if projector is not None:
        from .metrics import SpeedEstimator, metric_scale, track_speed
        from .rules.center_lane_pass import CenterLanePassRule
        from .rules.relative_speeding import RelativeSpeedingRule

        meters_per_unit = metric_scale(config.calibration)
        if meters_per_unit:
            log.info(
                "Metric speed enabled: %.1f m across x %.1f m along -> km/h reported.",
                meters_per_unit[0], meters_per_unit[1],
            )
        cooldown = float(config.events.get("cooldown_seconds", 8))
        min_age = int(config.tracking.get("minimum_consecutive_frames", 3))
        rs_cfg = config.events.get("relative_speeding", {})
        window_s = float(rs_cfg.get("rolling_window_minutes", 60)) * 60.0
        speed_estimator = SpeedEstimator(window_s)
        if rs_cfg.get("enabled", True):
            speeding_rule = RelativeSpeedingRule(
                rs_cfg, aggressiveness=config.aggressiveness,
                cooldown_seconds=cooldown, min_track_age=min_age,
                meters_per_unit=meters_per_unit, speed_window=speed_window,
            )
        cl_cfg = config.events.get("center_lane_pass", {})
        if cl_cfg.get("enabled", True):
            center_rule = CenterLanePassRule(
                cl_cfg, aggressiveness=config.aggressiveness,
                cooldown_seconds=cooldown, min_track_age=min_age,
                speed_window=speed_window, meters_per_unit=meters_per_unit,
            )

    # Event manager: dedup/merge candidate events into final saved events.
    event_manager = None
    if projector is not None and (speeding_rule is not None or center_rule is not None):
        from .. events.manager import EventManager

        event_manager = EventManager(
            merge_window_seconds=float(config.events.get("merge_window_seconds", 12)),
            cooldown_seconds=float(config.events.get("cooldown_seconds", 8)),
        )

    want_debug = bool(config.analysis.get("save_debug_video", False)) or output_dir is not None
    writer = None
    debug_path: Optional[Path] = None
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    if want_debug:
        out_dir = ensure_dir(Path(output_dir) if output_dir else data_dir() / "debug")
        debug_path = out_dir / f"{src.stem}_debug.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(debug_path), fourcc, inference_fps, (width, height))

    log.info(
        "Offline analysis: %s (%dx%d @ %.1ffps, %d frames); inference @ %.0ffps stride=%d; detector=%s",
        src.name, width, height, source_fps, total_frames, inference_fps, stride,
        type(det).__name__,
    )

    started = time.time()
    frame_idx = 0
    processed = 0
    total_detections = 0
    seen_track_ids: set[int] = set()
    candidate_events: list = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                if undistorter is not None:
                    frame = undistorter(frame)
                ts = frame_idx / source_fps
                detections = det.detect(frame)
                total_detections += len(detections)
                tracks = tracker.update(detections, ts)
                for t in tracks:
                    seen_track_ids.add(t.track_id)

                # Speed estimation + relative-speeding rule (calibration only).
                if speed_estimator is not None:
                    for t in tracks:
                        spd = track_speed(t.ground_point_history(), speed_window)
                        direction = t.direction()
                        if spd is None or direction is None:
                            continue
                        speed_estimator.observe(direction, ts, spd, t.track_id)
                        if speeding_rule is not None:
                            for ev in speeding_rule.evaluate(t, spd, direction, speed_estimator, ts):
                                candidate_events.append(ev)
                                if event_manager is not None:
                                    event_manager.add(ev, ts)
                                log.info(
                                    "EVENT %s track=%d pct=%.3f speed=%.4f dur=%.2fs%s",
                                    ev.event_type, ev.primary_track_id,
                                    ev.evidence["percentile"], ev.evidence["speed"],
                                    ev.evidence["duration_seconds"],
                                    " (warmup)" if ev.evidence["warmup"] else "",
                                )

                    if center_rule is not None:
                        for ev in center_rule.evaluate(tracks, speed_estimator, ts):
                            candidate_events.append(ev)
                            if event_manager is not None:
                                event_manager.add(ev, ts)
                            log.info(
                                "EVENT %s track=%d dwell=%.2fs pct=%.3f overtake=%s passed=%s",
                                ev.event_type, ev.primary_track_id,
                                ev.evidence["center_lane_time_seconds"],
                                ev.evidence["speed_percentile"],
                                ev.evidence["overtake_detected"],
                                ev.evidence["passed_track_id"],
                            )

                if writer is not None:
                    annotated = _annotate(
                        frame, tracker, sv, box_annotator, label_annotator,
                        projector=projector,
                        lane_cfg=config.calibration.get("lane_model", {}),
                        draw_lanes=bool(config.debug.get("draw_lane_bands", True)),
                        meters_per_unit=meters_per_unit,
                        speed_window=speed_window,
                    )
                    writer.write(annotated)

                processed += 1
                if max_frames is not None and processed >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # Finalize and export event clips from the source video.
    exported: list = []
    if event_manager is not None:
        final_events = event_manager.flush_all()
        if export_events and final_events:
            duration = (total_frames / source_fps) if (total_frames and source_fps) else (frame_idx / source_fps)
            root = Path(events_dir) if events_dir else Path(config.events.get("output_path", "/data/events"))
            for fe in final_events:
                try:
                    artifacts = _export_event_from_source(fe, src, duration, config, root)
                    exported.append(artifacts)
                    log.info("Exported event %s -> %s", fe.event_id[:8], artifacts["clip"])
                except Exception:  # noqa: BLE001 - one bad export shouldn't abort the run
                    log.exception("Failed to export event %s", fe.event_id)

    elapsed = time.time() - started

    # Lane usage: count tracks by the set of bands they ever occupied.
    lane_counts: Dict[str, int] = {}
    peak_kmh = None
    if projector is not None:
        from .metrics import speed_kmh as _speed_kmh

        for track in tracker.store.active_tracks():
            bands = {b for _ts, b in track.lane_band_history() if b}
            for band in bands:
                lane_counts[band] = lane_counts.get(band, 0) + 1
            # Peak km/h across on-road tracks (metric mode only).
            if meters_per_unit is not None and bands:
                gh = track.ground_point_history()
                if len(gh) >= 2:
                    track_peak = max(
                        (_speed_kmh(gh[:k], speed_window, meters_per_unit) or 0)
                        for k in range(2, len(gh) + 1)
                    )
                    peak_kmh = max(peak_kmh or 0.0, track_peak)

    summary: Dict[str, Any] = {
        "stub": False,
        "source": str(src),
        "resolution": [width, height],
        "source_fps": round(source_fps, 2),
        "inference_fps": inference_fps,
        "stride": stride,
        "frames_total": total_frames,
        "frames_processed": processed,
        "detections": total_detections,
        "unique_tracks": len(seen_track_ids),
        "calibrated": projector is not None,
        "metric_kmh": meters_per_unit is not None,
        "peak_kmh": round(peak_kmh, 1) if peak_kmh else None,
        "lane_track_counts": lane_counts,
        "candidate_events": len(candidate_events),
        "events_exported": len(exported),
        "event_clips": [a["clip"] for a in exported],
        "events": [
            {
                "event_type": ev.event_type,
                "primary_track_id": ev.primary_track_id,
                "trigger_ts": round(ev.trigger_ts, 3),
                "score": ev.score,
                "percentile": ev.evidence.get("percentile"),
            }
            for ev in candidate_events
        ],
        "debug_video": str(debug_path) if debug_path else None,
        "elapsed_s": round(elapsed, 3),
    }
    log.info("Offline analysis summary: %s", json.dumps(summary))
    return summary


def _export_event_from_source(final_event, source, duration, config, events_root):
    """Export one final event's clip + thumbnail + metadata from the source.

    Used for offline analysis, where the event ``trigger_ts`` is an offset into
    the source video. Returns a dict of the written artifact paths.
    """
    from ..events.exporter import clip_window, export_from_source
    from ..events.metadata import build_metadata, event_stem, write_metadata
    from ..events.thumbnail import generate_thumbnail
    from ..util.time import format_date_dir, format_segment_stamp, iso_now, now_unix_ms

    ev_cfg = config.events
    pre = float(ev_cfg.get("pre_roll_seconds", 10))
    post = float(ev_cfg.get("post_roll_seconds", 20))
    window = clip_window(final_event.trigger_ts, pre, post)
    start = max(0.0, window.start)
    end = min(duration, window.end) if duration > 0 else window.end
    clip_duration = max(0.1, end - start)

    tz = config.app.timezone
    wall_ms = now_unix_ms()
    stamp = format_segment_stamp(wall_ms, tz)
    date = format_date_dir(wall_ms, tz)
    short = final_event.event_id[:8]
    out_dir = Path(events_root) / date / final_event.event_type
    stem = event_stem(stamp, final_event.event_type, short,
                      (c.evidence for c in final_event.candidates))

    clip_path = out_dir / f"{stem}.mp4"
    export_from_source(source, start, clip_duration, clip_path)

    thumb_path = out_dir / f"{stem}.jpg"
    thumb_offset = min(float(ev_cfg.get("thumbnail_time_offset_seconds", 15)), clip_duration * 0.5)
    generate_thumbnail(clip_path, thumb_path, thumb_offset)

    meta = build_metadata(
        final_event, clip_path, thumb_path, config=config,
        created_at=iso_now(tz), start_ts=start, trigger_ts=final_event.trigger_ts, end_ts=end,
    )
    meta_path = out_dir / f"{stem}.json"
    write_metadata(meta, meta_path)

    return {"clip": str(clip_path), "thumbnail": str(thumb_path), "metadata": str(meta_path)}


def _annotate(
    frame, tracker, sv, box_annotator, label_annotator,
    projector=None, lane_cfg=None, draw_lanes=True,
    meters_per_unit=None, speed_window=0.5,
):
    """Draw lane bands + boxes + track-id/lane/direction labels for one frame."""
    from .lane_model import draw_lane_overlay

    annotated = frame.copy()
    if projector is not None and draw_lanes:
        annotated = draw_lane_overlay(annotated, projector, lane_cfg or {})

    tracked = tracker.last_tracked
    if tracked is None or len(tracked) == 0:
        return annotated

    labels = []
    ids = getattr(tracked, "tracker_id", None)
    for i in range(len(tracked)):
        track_id = int(ids[i]) if ids is not None and ids[i] is not None else -1
        track = tracker.store.get(track_id)
        label = f"#{track_id}"
        if track:
            if track.latest_lane_band:
                label += f" {track.latest_lane_band}"
            direction = track.direction()
            if direction:
                label += f" {direction}"
            if projector is not None:
                from .metrics import speed_kmh, track_speed

                if meters_per_unit is not None:
                    kmh = speed_kmh(track.ground_point_history(), speed_window, meters_per_unit)
                    if kmh is not None:
                        label += f" {kmh:.0f}km/h"
                else:
                    spd = track_speed(track.ground_point_history())
                    if spd is not None:
                        label += f" v={spd:.2f}"
        labels.append(label)
    annotated = box_annotator.annotate(annotated, detections=tracked)
    annotated = label_annotator.annotate(annotated, detections=tracked, labels=labels)
    return annotated


# --------------------------------------------------------------------------
# Milestone 0 stub (dependency-free fallback)
# --------------------------------------------------------------------------
def run_stub_pipeline(source: str, config: Config) -> Dict[str, Any]:
    """Dependency-free stub used when the CV stack is unavailable.

    Validates the source, exercises the real lane-band math, and emits a
    structured summary with the same shape (events always empty).
    """
    src_path = Path(source)
    exists = src_path.exists()
    if exists:
        size = src_path.stat().st_size
        log.info("Analyzing source: %s (%d bytes)", src_path, size)
    else:
        size = 0
        log.warning(
            "Source not found: %s - running a dry stub (no frames to process)",
            src_path,
        )

    analysis = config.analysis
    inference_fps = int(analysis.get("inference_fps", 12))
    input_size = int(analysis.get("inference_input_size", 640))
    placeholder_duration_s = 60 if exists else 0
    frames_considered = inference_fps * placeholder_duration_s

    log.info(
        "Stub plan: inference_fps=%d input_size=%d -> ~%d frames would be processed",
        inference_fps, input_size, frames_considered,
    )
    log.info("ffmpeg available: %s", ffmpeg_available())

    bands = lane_model.normalize_lane_ratios(config.calibration.get("lane_model", {}))
    for name, width in bands:
        log.info("lane band %-16s width=%.3f", name, width)

    summary: Dict[str, Any] = {
        "stub": True,
        "source": str(src_path),
        "source_exists": exists,
        "source_bytes": size,
        "inference_fps": inference_fps,
        "inference_input_size": input_size,
        "frames_considered": frames_considered,
        "lanes": [name for name, _ in bands],
        "events": [],
        "aggressiveness": config.aggressiveness,
    }
    log.info("Stub analysis summary: %s", json.dumps(summary))
    return summary
