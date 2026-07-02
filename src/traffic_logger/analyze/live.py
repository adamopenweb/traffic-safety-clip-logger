"""Live analyzer.

Runs the same detection -> tracking -> rules -> event pipeline as the offline
analyzer, but over a *live* source (an RTSP URL from the camera/mini-PC, a
device, or any cv2-openable stream) in real time:

* samples at ``analysis.inference_fps`` using wall-clock pacing, dropping
  intermediate frames so it stays current rather than falling behind;
* reconnects automatically on read failure / disconnect (spec: "If live RTSP
  analysis disconnects, reconnect automatically");
* flushes the event manager periodically so events fire promptly (there is no
  end-of-stream), writing a metadata sidecar + thumbnail per event.

The consumer is camera-agnostic: a USB camera re-streamed over RTSP for testing
and a production IP camera differ only by the source URL. Clip extraction in the
live deployment comes from the ring buffer (the recorder + ``export-event``);
this analyzer emits the event records and thumbnails.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..config import Config
from ..util.logging import get_logger

log = get_logger(__name__)


def is_stream_source(source: str) -> bool:
    """True if ``source`` looks like a live stream/device rather than a file."""
    s = str(source)
    if s.isdigit():
        return True  # camera index
    return s.startswith((
        "rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://",
        "tcp://", "/dev/video",
    ))


def run_live(
    source: str,
    config: Config,
    *,
    detector: Any = None,
    max_seconds: Optional[float] = None,
    max_events: Optional[int] = None,
    on_event: Optional[Callable[[Any, float], None]] = None,
    overlay_buffer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Analyze a live source until stopped (or a test limit is reached).

    ``on_event(final_event, wall_trigger_ts)`` is invoked for each finalized
    event with the event's absolute (unix) trigger time. When given (the M7
    combined ``run``, which records the ring), it replaces the default
    metadata+thumbnail emission so the caller can cut a clip from the ring.

    ``overlay_buffer`` (an :class:`~...events.overlay_buffer.OverlayBuffer`), when
    given, receives one snapshot of every visible track's box + speed per
    processed frame, so the clip exporter can later burn boxes onto the 4K clip.
    """
    import os

    # Prefer RTSP-over-TCP for reliable analysis over Wi-Fi (UDP drops frames).
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("run_live requires opencv-python (the 'analyze' extra).") from exc

    from .detector import MissingDetectorDependency, build_detector
    from .metrics import SpeedEstimator, metric_scale, track_speed
    from .project import build_transform
    from .rules.center_lane_pass import CenterLanePassRule
    from .rules.relative_speeding import RelativeSpeedingRule
    from .tracker import VehicleTracker
    from .undistort import build_undistorter
    from .best_frame import BestFrameCache
    from ..events.manager import EventManager

    inference_fps = float(config.analysis.get("inference_fps", 12))
    min_interval = 1.0 / inference_fps if inference_fps > 0 else 0.0
    speed_window = float(config.analysis.get("speed_window_seconds", 0.5))
    # "stream" (default) = a live RTSP connection; "ring" = tail the recorded ring
    # (every frame, no delivery gaps; ~1 segment behind; annotations align exactly).
    frame_source_mode = str(config.analysis.get("frame_source", "stream")).lower()

    try:
        det = build_detector(config, detector=detector)
    except MissingDetectorDependency as exc:
        raise RuntimeError(str(exc)) from exc

    # Lens de-warp before detection (see analyze/undistort.py); de-warped frames
    # also feed event thumbnails so they match the calibrated geometry.
    undistorter = build_undistorter(config.calibration)
    if undistorter is not None:
        log.info("Lens de-warp enabled (k1=%.3f, k2=%.3f).", undistorter.k1, undistorter.k2)

    # Downscale the heavy 4K frame before the de-warp remap: a full-4K remap is
    # ~44 ms (caps the pipeline at ~14 fps -> dropped frames -> near-lane cars
    # fragment). Remapping a 1280-wide frame is ~5 ms, so the pipeline sustains
    # 30 fps. Detection is unaffected (YOLO resizes to imgsz anyway) and the
    # homography is in normalized coords, so the GPS-validated speed scale is
    # preserved -- only source_points scale to the analysis resolution.
    analyze_w = int(config.analysis.get("analyze_max_width", 0) or 0)
    src_w = int((config.camera.get("capture_resolution") or [0, 0])[0] or 0)
    ds_scale = (analyze_w / src_w) if (analyze_w and src_w and analyze_w < src_w) else 1.0
    if ds_scale != 1.0:
        log.info("Analysis downscaled to %dpx wide (x%.3f) for fast de-warp.", analyze_w, ds_scale)

    def _prep(frame):
        """Downscale (cheap) then de-warp the small frame (cheap) so the pipeline
        keeps up at full fps. No-op downscale when analyze_max_width is unset."""
        if ds_scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (analyze_w, max(1, round(h * analyze_w / w))),
                               interpolation=cv2.INTER_LINEAR)
        if undistorter is not None:
            frame = undistorter(frame)
        return frame

    cal_for_proj = config.calibration
    if ds_scale != 1.0:
        cal_for_proj = dict(config.calibration)
        cal_for_proj["source_points"] = [[x * ds_scale, y * ds_scale]
                                          for x, y in config.calibration["source_points"]]
    projector = build_transform(cal_for_proj)
    if projector is None:
        log.warning("No calibration; live analysis runs detection/tracking only.")
    tracker = VehicleTracker(config, projector=projector)
    meters_per_unit = metric_scale(config.calibration) if projector else None

    # Rules + event manager (only meaningful with calibration).
    speed_estimator = speeding_rule = center_rule = manager = None
    if projector is not None:
        cooldown = float(config.events.get("cooldown_seconds", 8))
        min_age = int(config.tracking.get("minimum_consecutive_frames", 3))
        rs_cfg = config.events.get("relative_speeding", {})
        speed_estimator = SpeedEstimator(float(rs_cfg.get("rolling_window_minutes", 60)) * 60.0)
        if rs_cfg.get("enabled", True):
            speeding_rule = RelativeSpeedingRule(
                rs_cfg, aggressiveness=config.aggressiveness, cooldown_seconds=cooldown,
                min_track_age=min_age, meters_per_unit=meters_per_unit, speed_window=speed_window,
                calibration=config.calibration)
        cl_cfg = config.events.get("center_lane_pass", {})
        if cl_cfg.get("enabled", True):
            center_rule = CenterLanePassRule(
                cl_cfg, aggressiveness=config.aggressiveness, cooldown_seconds=cooldown,
                min_track_age=min_age, speed_window=speed_window, meters_per_unit=meters_per_unit,
                calibration=config.calibration)
        manager = EventManager(
            merge_window_seconds=float(config.events.get("merge_window_seconds", 12)),
            cooldown_seconds=float(config.events.get("cooldown_seconds", 8)))

    # Optional police-vehicle recognition (CLIP zero-shot). Off unless
    # events.police.enabled; additive, so the rest of the pipeline is unchanged.
    police = None
    try:
        from .police_classifier import build_police_tagger, PoliceSession

        tagger = build_police_tagger(config)
        if tagger is not None:
            from ..events.police_log import PoliceLog

            pol = config.events.get("police") or {}
            db = PoliceLog(pol.get("db_path", "data/index/police_sightings.sqlite"))
            police = PoliceSession(config, tagger, db, inference_fps=inference_fps)
            log.info("Police recognition enabled (threshold %.2f).", police.threshold)
    except Exception:  # noqa: BLE001 - police is optional; never block analysis
        log.exception("Police recognition setup failed; continuing without it.")
        police = None

    started = time.monotonic()
    started_wall = time.time()  # map monotonic event ts -> absolute (ring) time
    # The absolute time that ts=0 maps to. started_wall for the stream path; the
    # first frame's capture time for the ring path (so events stamp true capture time).
    clock_origin_holder = [started_wall]
    processed = 0
    events_emitted = 0
    last_flush = 0.0
    last_frame = None
    last_seq = 0

    # Optional pass logging (the traffic denominator): record EVERY completed track,
    # not just gated violations, into the unified store. Additive + best-effort.
    passes = None
    try:
        pcfg = config.events.get("passes") or {}
        if pcfg.get("enabled", False) and projector is not None:
            from .pass_recorder import PassRecorder
            from ..events.store import TrafficStore

            rs = config.events.get("relative_speeding", {})
            store = TrafficStore(pcfg.get("db_path", "data/index/traffic.sqlite"))
            session_id = f"run-{int(started_wall)}"
            store.start_session(session_id, started_wall,
                                camera_id=config.camera.get("profile"))
            passes = PassRecorder(
                store, session_id, meters_per_unit=meters_per_unit,
                calibration=config.calibration,
                speeding_gate_kmh=rs.get("absolute_kmh_threshold"),
                finalize_after=int(pcfg.get("finalize_after_frames", 20)),
                min_age=int(pcfg.get("min_track_age", 5)),
                min_ground_span=float(pcfg.get("min_ground_span",
                                               config.events.get("annotate_static_span_min", 0.12))))
            log.info("Pass logging on (denominator) -> %s (session %s).",
                     store.db_path, session_id)
    except Exception:  # noqa: BLE001 - pass logging is optional; never block analysis
        log.exception("Pass-logging setup failed; continuing without it.")
        passes = None

    # Per-track best-frame capture: the sub frame where each vehicle's box was largest,
    # so the thumbnail can be cropped to the actual violator (no sub<->4K offset).
    best_frames = BestFrameCache()

    def _dispatch(fe) -> None:
        # Relabel the speed with the GPS-validated full-track value (and drop sub-gate
        # false triggers) BEFORE anything reads it -- the violation log, the
        # clip/screenshot gate, and the filename all derive from the event's speed, so
        # one correction here fixes the label AND the count. See _relabel_event_speed.
        if passes is not None and not _relabel_event_speed(fe, passes):
            return
        # Attach the violator's best sub frame for a pixel-accurate thumbnail crop.
        fe.thumb_source = best_frames.take(fe.primary_track_id)
        # on_event gets the absolute trigger time so the caller can cut a ring
        # clip; otherwise write a standalone metadata + thumbnail (analyze-only).
        if police is not None:
            police.note_event(fe)
        if on_event is not None:
            on_event(fe, clock_origin_holder[0] + fe.trigger_ts)
        else:
            _emit_event(fe, last_frame, config, meters_per_unit, speed_window, tracker)

    def _open():
        c = cv2.VideoCapture(source if not str(source).isdigit() else int(source), cv2.CAP_FFMPEG)
        try:
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only the freshest frame
        except Exception:  # noqa: BLE001
            pass
        return c

    # -- RING source: analyze the recorded ring instead of a 2nd live connection.
    # Reads EVERY recorded frame (no delivery gaps -> fast near-lane cars track
    # cleanly end-to-end) and stamps each with its true capture time, so events and
    # overlay snapshots align exactly with the clip the exporter cuts. ~1 segment
    # (~10 s) behind real time. Returns early; the stream path below is untouched.
    if frame_source_mode == "ring":
        from .ring_source import RingFrameSource
        ring_dir = config.recording.get("ring_path", "data/ring")
        source_fps = float(config.camera.get("capture_fps", 30) or 30)
        log.info("Live analysis from the RECORDED RING %s (every frame, ~1 segment behind).",
                 ring_dir)
        ring = RingFrameSource(ring_dir, default_fps=source_fps)
        origin = None
        try:
            for frame, capture_ts in ring.frames():
                if max_seconds is not None and time.monotonic() - started >= max_seconds:
                    break
                if max_events is not None and events_emitted >= max_events:
                    break
                if origin is None:
                    origin = capture_ts
                    clock_origin_holder[0] = origin
                ts = capture_ts - origin
                frame = _prep(frame)
                last_frame = frame
                processed += 1
                tracks = tracker.update(det.detect(frame), ts)
                if police is not None:
                    police.observe_frame(frame, tracks, processed, capture_ts)
                    police.finalize_departed(processed, tracker.store, meters_per_unit)
                if passes is not None:
                    passes.observe_frame(tracks, processed, capture_ts)
                    passes.finalize_departed(processed, tracker.store)
                _sweep_departed(tracker, ts, processed, speeding_rule, center_rule, passes)
                best_frames.observe(tracks, last_frame, processed)
                if overlay_buffer is not None:
                    if overlay_buffer.frame_size is None:
                        h, w = frame.shape[:2]
                        overlay_buffer.set_frame_size(w, h)
                    overlay_buffer.append(
                        capture_ts,
                        _overlay_boxes(tracks, meters_per_unit, speed_window, config.calibration,
                                       static_span_min=float(config.events.get(
                                           "annotate_static_span_min", 0.12))),
                    )
                if manager is not None:
                    _run_rules(tracks, ts, speed_estimator, speeding_rule, center_rule,
                               manager, speed_window)
                    if ts - last_flush >= 1.0:
                        last_flush = ts
                        for fe in manager.flush(ts):
                            _dispatch(fe)
                            events_emitted += 1
        except KeyboardInterrupt:
            log.info("Live analysis interrupted.")
        finally:
            ring.stop.set()
            if manager is not None:
                for fe in manager.flush_all():
                    _dispatch(fe)
                    events_emitted += 1
            if police is not None:
                police.close(tracker.store, meters_per_unit)
            if passes is not None:
                passes.close(tracker.store)
                log.info("Pass log: %d passes recorded, %d filtered (parked/flicker).",
                         passes.counted, passes.skipped)
                passes.store.close()
        elapsed = time.monotonic() - started
        summary = {
            "mode": "live-ring", "source": f"ring:{ring_dir}",
            "frames_processed": processed, "events_emitted": events_emitted,
            "calibrated": projector is not None, "metric_kmh": meters_per_unit is not None,
            "elapsed_s": round(elapsed, 1),
            "effective_fps": round(processed / elapsed, 1) if elapsed > 0 else 0.0,
        }
        log.info("Live analysis summary: %s", summary)
        return summary

    log.info("Live analysis starting on %s (inference @ %.0f fps).", source, inference_fps)
    # A synchronous cv2.read() over RTSP stalls this loop below the stream rate
    # even with GPU headroom; a background grabber keeps the decode drained so we
    # process the freshest frame at compute speed (see _FrameGrabber).
    grabber = _FrameGrabber(_open).start()
    try:
        while True:
            loop_start = time.monotonic()
            if max_seconds is not None and loop_start - started >= max_seconds:
                break
            if max_events is not None and events_emitted >= max_events:
                break

            seq, frame = grabber.read()
            if frame is None or seq == last_seq:
                time.sleep(0.002)  # no new frame yet (startup / between frames)
                continue
            last_seq = seq

            ts = loop_start - started
            frame = _prep(frame)
            last_frame = frame
            processed += 1

            tracks = tracker.update(det.detect(frame), ts)
            if police is not None:
                police.observe_frame(frame, tracks, processed, started_wall + ts)
                police.finalize_departed(processed, tracker.store, meters_per_unit)
            if passes is not None:
                passes.observe_frame(tracks, processed, started_wall + ts)
                passes.finalize_departed(processed, tracker.store)
            _sweep_departed(tracker, ts, processed, speeding_rule, center_rule, passes)
            best_frames.observe(tracks, last_frame, processed)
            if overlay_buffer is not None:
                if overlay_buffer.frame_size is None:
                    h, w = frame.shape[:2]
                    overlay_buffer.set_frame_size(w, h)
                overlay_buffer.append(
                    started_wall + ts,
                    _overlay_boxes(tracks, meters_per_unit, speed_window, config.calibration,
                                   static_span_min=float(config.events.get(
                                       "annotate_static_span_min", 0.12))),
                )
            if manager is not None:
                _run_rules(tracks, ts, speed_estimator, speeding_rule, center_rule,
                           manager, speed_window)
                # Flush ~once per second so events fire promptly.
                if loop_start - last_flush >= 1.0:
                    last_flush = loop_start
                    for fe in manager.flush(ts):
                        _dispatch(fe)
                        events_emitted += 1

            # Pace to the inference rate; the grabber keeps decoding meanwhile.
            elapsed = time.monotonic() - loop_start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
    except KeyboardInterrupt:
        log.info("Live analysis interrupted.")
    finally:
        grabber.stop()
        if manager is not None:
            for fe in manager.flush_all():
                _dispatch(fe)
                events_emitted += 1
        if police is not None:
            police.close(tracker.store, meters_per_unit)
        if passes is not None:
            passes.close(tracker.store)
            log.info("Pass log: %d passes recorded, %d filtered (parked/flicker).",
                     passes.counted, passes.skipped)
            passes.store.close()

    elapsed = time.monotonic() - started
    summary = {
        "mode": "live",
        "source": str(source),
        "frames_processed": processed,
        "events_emitted": events_emitted,
        "calibrated": projector is not None,
        "metric_kmh": meters_per_unit is not None,
        "elapsed_s": round(elapsed, 1),
        "effective_fps": round(processed / elapsed, 1) if elapsed > 0 else 0.0,
    }
    log.info("Live analysis summary: %s", summary)
    return summary


class _FrameGrabber:
    """Background RTSP reader that always holds the freshest decoded frame.

    A synchronous ``cv2.read()`` over RTSP couples network + decode latency into
    the analysis loop, capping throughput well below the stream rate even when
    the GPU has spare headroom. A dedicated daemon thread keeps the decode
    pipeline drained so the analyzer can pull the latest frame and run at compute
    speed. The frame is tagged with a monotonically increasing sequence number so
    the consumer processes each frame at most once (no duplicates) and never
    blocks on the network. Reconnects on read failure with capped backoff.
    """

    def __init__(self, opener):
        self._opener = opener
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "_FrameGrabber":
        self._thread = threading.Thread(target=self._run, name="rtsp-grabber", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        cap = None
        backoff = 1.0
        try:
            while not self._stop.is_set():
                if cap is None or not cap.isOpened():
                    cap = self._opener()
                    if cap is None or not cap.isOpened():
                        self._stop.wait(backoff)
                        backoff = min(backoff * 2, 15.0)
                        continue
                    backoff = 1.0
                ok, frame = cap.read()
                if not ok:
                    log.warning("Live stream read failed; reconnecting.")
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                    cap = None
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 15.0)
                    continue
                with self._lock:
                    self._frame = frame
                    self._seq += 1
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:  # noqa: BLE001
                    pass

    def read(self):
        """Return ``(seq, frame)`` of the latest frame; ``(0, None)`` before the first."""
        with self._lock:
            return self._seq, self._frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _overlay_boxes(tracks, meters_per_unit, speed_window, cal_cfg=None,
                   static_span_min=0.0, static_min_obs=6):
    """One :class:`OverlayBox` per visible *moving* track for the annotation buffer.

    Stationary vehicles (e.g. neighbours' cars parked across the street) are
    skipped once a track has enough history but has barely moved on the ground --
    they're not traffic, and a 0 km/h box just clutters the clip. New tracks (fewer
    than ``static_min_obs`` points) are always drawn so a real car isn't blanked
    during its first frames before its motion is established. ``static_span_min`` of
    0 disables the filter.
    """
    from .metrics import ground_span, speed_kmh_calibrated, track_speed
    from ..events.overlay_buffer import OverlayBox

    boxes = []
    for t in tracks:
        bbox = t.latest_bbox
        if bbox is None:
            continue
        gh = t.ground_point_history()
        if (static_span_min > 0 and len(gh) >= static_min_obs
                and ground_span(gh) < static_span_min):
            continue
        rel = track_speed(gh, speed_window)
        kmh = (speed_kmh_calibrated(gh, speed_window, meters_per_unit, cal_cfg)
               if meters_per_unit is not None else None)
        boxes.append(OverlayBox(
            track_id=t.track_id, bbox=bbox, speed_kmh=kmh, speed_rel=rel,
            lane=t.latest_lane_band, direction=t.direction(),
        ))
    return boxes


# Long-run memory hygiene: evict tracks (and their per-track rule/recorder state) that
# have been gone far longer than any consumer needs them. Runs every N frames, not per
# frame, so it costs ~nothing. See TrackStore.sweep for why 24/7 mode needs this.
_SWEEP_INTERVAL_FRAMES = 300     # ~25s at 12 fps -- sweep cadence
_TRACK_MAX_IDLE_SECONDS = 60.0   # >> pass/police finalize + confirmer windows


def _sweep_departed(tracker, ts, processed, speeding_rule, center_rule, passes):
    """Every ``_SWEEP_INTERVAL_FRAMES`` frames, evict long-departed tracks and release
    their per-track state in the rules + pass recorder, keeping a 24/7 run bounded."""
    if processed % _SWEEP_INTERVAL_FRAMES != 0:
        return
    evicted = tracker.store.sweep(ts, _TRACK_MAX_IDLE_SECONDS)
    if not evicted:
        return
    if speeding_rule is not None:
        speeding_rule.evict(evicted)
    if center_rule is not None:
        center_rule.evict(evicted)
    if passes is not None:
        passes.evict(evicted)


def _run_rules(tracks, ts, estimator, speeding_rule, center_rule, manager, speed_window):
    from .metrics import track_speed

    for t in tracks:
        spd = track_speed(t.ground_point_history(), speed_window)
        direction = t.direction()
        if spd is None or direction is None:
            continue
        estimator.observe(direction, ts, spd, t.track_id)
        if speeding_rule is not None:
            for ev in speeding_rule.evaluate(t, spd, direction, estimator, ts):
                manager.add(ev, ts)
    if center_rule is not None:
        for ev in center_rule.evaluate(tracks, estimator, ts):
            manager.add(ev, ts)


def _relabel_event_speed(fe, passes) -> bool:
    """Replace a speeding event's PARTIAL-track trigger speed with the validated
    full-track value, and say whether to keep the event.

    The absolute-speeding gate fires the moment ``steady_speed_kmh`` over the track
    *so far* (the car is still mid-crossing) first clears the threshold -- a short,
    noisy estimate that reads systematically ~7 km/h high. When the track completes,
    the PassRecorder computes the SAME metric over the whole crossing (the
    GPS-validated value the speed-test page checks) and stores it. By dispatch the pass
    is finalized, so we look it up and overwrite the event's speed. Returns False to
    DROP the event (validated speed never cleared the gate -> a partial-track false
    trigger), True to keep it (relabeled, or no pass / non-speeding -> left untouched).
    Best-effort: any failure keeps the original event."""
    cands = [c for c in (getattr(fe, "candidates", []) or [])
             if (getattr(c, "evidence", None) or {}).get("rule") == "absolute_speeding"]
    if not cands:
        return True  # not a speed-gated event (e.g. center-lane) -> leave as-is
    try:
        v = passes.store.get_pass_steady(passes.session_id, fe.primary_track_id)
    except Exception:  # noqa: BLE001 - never block dispatch on a store read
        return True
    if v is None:
        return True  # no finalized pass (track filtered / not done) -> keep trigger value
    if passes.gate is not None and v < float(passes.gate):
        return False  # below the gate by the validated metric -> drop the false trigger
    for c in cands:
        ev = c.evidence
        ev["speed_kmh"] = round(v, 1)
        if ev.get("threshold_kmh") is not None:
            ev["over_by_kmh"] = round(v - float(ev["threshold_kmh"]), 1)
    return True


def _emit_event(final_event, frame, config, meters_per_unit, speed_window, tracker):
    """Write a metadata sidecar + thumbnail for a live event."""
    import cv2

    from ..events.metadata import build_metadata, event_stem, write_metadata
    from ..util.paths import ensure_dir
    from ..util.time import format_date_dir, format_segment_stamp, iso_now, now_unix_ms

    tz = config.app.timezone
    ms = now_unix_ms()
    stamp = format_segment_stamp(ms, tz)
    short = final_event.event_id[:8]
    out_dir = ensure_dir(
        Path(config.events.get("output_path", "/data/events"))
        / format_date_dir(ms, tz) / final_event.event_type)
    stem = event_stem(stamp, final_event.event_type, short,
                      (c.evidence for c in final_event.candidates))

    thumb_path = out_dir / f"{stem}.jpg"
    src = getattr(final_event, "thumb_source", None)
    cropped = None
    if src is not None:
        from ..events.thumbnail import save_cropped_thumbnail

        cropped = save_cropped_thumbnail(src[0], src[1], thumb_path)
    if cropped is None and frame is not None:
        cv2.imwrite(str(thumb_path), frame)

    meta = build_metadata(
        final_event, clip_path="", thumbnail_path=thumb_path, config=config,
        created_at=iso_now(tz), start_ts=final_event.trigger_ts,
        trigger_ts=final_event.trigger_ts, end_ts=final_event.trigger_ts)
    write_metadata(meta, out_dir / f"{stem}.json")
    log.info(
        "LIVE EVENT %s track=%d score=%.2f -> %s",
        final_event.event_type, final_event.primary_track_id or -1,
        final_event.score, out_dir / f"{stem}.json")
