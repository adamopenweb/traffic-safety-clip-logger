"""CLI subcommand handlers.

Each handler takes parsed argparse ``args`` plus a loaded :class:`Config` and
returns a process exit code. Keeping handlers here (separate from argument
parsing in ``main.py``) makes them importable and unit-testable.

For Milestone 0 most handlers are stubs that log which later milestone owns the
real implementation and exit 0. The ``test`` handler runs the real stub
offline pipeline.
"""

from __future__ import annotations

import argparse

from .config import Config
from .util.logging import get_logger

log = get_logger(__name__)


def _stub(command: str, milestone: str) -> int:
    """Log a clear not-yet-implemented message and exit successfully."""
    log.warning("`%s` is a stub - real implementation planned for %s.", command, milestone)
    return 0


def handle_probe_camera(args: argparse.Namespace, config: Config) -> int:
    """Enumerate camera devices and print their supported formats."""
    from .capture.camera_probe import probe_cameras, select_format

    cameras = probe_cameras(config)
    if not cameras:
        log.error("No camera devices found.")
        return 1

    preference = config.camera.get("pixel_format_preference", [])
    resolution = config.camera.get("capture_resolution")
    fps = config.camera.get("capture_fps")

    for cam in cameras:
        print(f"Device: {cam.device}  (backend: {cam.backend})")
        for fmt in cam.formats:
            sizes = ", ".join(
                f"{s['width']}x{s['height']}@[{','.join(str(x) for x in s['fps'])}]"
                for s in fmt.sizes
            )
            label = f"{fmt.pixel_format}"
            if fmt.description:
                label += f" ({fmt.description})"
            print(f"  {label}: {sizes}")
        chosen = select_format(cam.formats, preference, resolution, fps)
        if chosen:
            print(f"  -> selected format for {resolution}@{fps}: {chosen}")
        else:
            print("  -> no configured-preference format matches; check config")
    return 0


def handle_capture(args: argparse.Namespace, config: Config) -> int:
    """Record the camera stream to H.264 ring-buffer segments (blocks)."""
    from .capture.recorder import Recorder
    from .util.ffmpeg import ffmpeg_available

    if not config.recording.get("enabled", True):
        log.error("recording.enabled is false in config; nothing to capture.")
        return 1
    if not ffmpeg_available():
        log.error("ffmpeg not found on PATH; cannot record.")
        return 1

    recorder = Recorder(config)
    log.info(
        "Recording %s -> %s (%ds segments, %d GB ring cap)",
        config.camera.get("source"),
        recorder.ring_root,
        recorder.segment_seconds,
        int(recorder.ring_max_bytes / (1024 ** 3)),
    )
    return recorder.run()


def handle_analyze(args: argparse.Namespace, config: Config) -> int:
    """Analyze a live RTSP/camera stream (or a file) in real time."""
    from pathlib import Path

    from .analyze.live import is_stream_source, run_live
    from .analyze.offline import run_offline
    from .util.ffmpeg import ffmpeg_available

    source = args.source or config.analysis.get("source")
    if not source or source == "file_or_stream":
        log.error(
            "No analysis source. Pass --source <rtsp-url|device|file> "
            "or set analysis.source in the config."
        )
        return 1
    if not ffmpeg_available():
        log.error("ffmpeg/opencv required for analysis (the 'analyze' extra).")
        return 1

    try:
        if is_stream_source(source):
            run_live(source, config, max_seconds=getattr(args, "max_seconds", None))
        elif Path(source).exists():
            log.info("Source is a file; running offline analysis.")
            run_offline(source, config, export_events=True)
        else:
            log.error("Source not found and not a recognized stream URL: %s", source)
            return 1
        return 0
    except (RuntimeError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1


def handle_run(args: argparse.Namespace, config: Config) -> int:
    """Combined single-box run: record the ring + analyze live together.

    The recorder streams the camera's main (4K) stream to the ring on a
    background thread; the live analyzer runs on the sub-stream; each finalized
    event is handed to a clip exporter that cuts a 30s evidence clip from the
    ring once the post-roll has been recorded. Blocks until interrupted.
    """
    import threading

    from .analyze.live import run_live
    from .capture.recorder import Recorder
    from .events.ring_clip_exporter import RingClipExporter
    from .util.ffmpeg import ffmpeg_available

    if not ffmpeg_available():
        log.error("ffmpeg/opencv required for run (the 'analyze' extra).")
        return 1

    analysis_source = config.analysis.get("source")
    if not analysis_source or analysis_source in ("file", "file_or_stream"):
        analysis_source = config.camera.get("source")
    if not analysis_source:
        log.error("No analysis source. Set analysis.source (the RTSP sub-stream) in the config.")
        return 1

    record = bool(config.recording.get("enabled", True))
    recorder = exporter = rec_thread = None
    on_event = None
    overlay_buffer = None
    speed_log = None

    if record:
        recorder = Recorder(config)
        if bool(config.events.get("annotate_clips", True)):
            from .events.overlay_buffer import OverlayBuffer

            ev = config.events
            # Buffer must outlast the clip window PLUS the worst-case render
            # backlog: events are captured eagerly when ready, but under a burst
            # the serial 4K renders queue up, so the buffer holds generous
            # headroom so a delayed capture still finds its snapshots.
            capacity = (float(ev.get("pre_roll_seconds", 10))
                        + float(ev.get("post_roll_seconds", 20))
                        + float(ev.get("annotate_buffer_headroom_seconds", 180)))
            overlay_buffer = OverlayBuffer(capacity_seconds=capacity)
        exporter = RingClipExporter(config, overlay_buffer=overlay_buffer).start()
        on_event = exporter.enqueue
        # Two-tier speeding: log EVERY violation over the gate (text), but only cut
        # a clip for the excessive ones (clip_kmh_threshold). Keeps disk/render
        # cost on the egregious cases while still tracking every speeder.
        from .analyze.metrics import metric_scale
        from .events.speed_log import SpeedLog, SpeedRecord, event_speed_and_direction

        rs_cfg = config.events.get("relative_speeding", {})
        clip_threshold = rs_cfg.get("clip_kmh_threshold")
        shot_threshold = rs_cfg.get("screenshot_kmh_threshold")
        if metric_scale(config.calibration) is not None and rs_cfg.get("absolute_kmh_threshold") is not None:
            speed_log = SpeedLog(config.events.get("speed_log_path", "data/index/speed_log.sqlite"))
            log.info("Speed log on (text every %.0f+; image %s+; full clip %s+).",
                     float(rs_cfg["absolute_kmh_threshold"]),
                     f"{float(shot_threshold):.0f}" if shot_threshold is not None else "off",
                     f"{float(clip_threshold):.0f}" if clip_threshold is not None else "all")

            # Three-tier evidence by speed: text log (every violation) -> a still image
            # of the car (mid-tier, no video) -> full clip + annotation (egregious).
            # Keeps disk/render cost on the most egregious cases while still capturing what
            # each mid-tier vehicle was, and avoids the unplayable HEVC short clips.
            def on_event(fe, wall_ts, _enqueue=exporter.enqueue,
                         _still=exporter.enqueue_screenshot):
                sd = event_speed_and_direction(fe)
                if sd is not None:                       # an absolute-speeding event
                    speed, direction, vtype = sd
                    clip = clip_threshold is None or speed >= float(clip_threshold)
                    speed_log.add(SpeedRecord(ts=wall_ts, speed_kmh=speed,
                                              direction=direction, clipped=clip,
                                              vehicle_type=vtype))
                    if clip:
                        _enqueue(fe, wall_ts)             # full video
                    elif shot_threshold is not None and speed >= float(shot_threshold):
                        _still(fe, wall_ts)               # image only (no video)
                    return                                # else: text-only
                _enqueue(fe, wall_ts)                     # non-speeding events keep clips

        rec_thread = threading.Thread(target=recorder.run, name="recorder", daemon=True)
        rec_thread.start()
        log.info(
            "Recording %s -> ring %s; analyzing %s. Live events -> 30s ring clips.",
            config.recording.get("source") or config.camera.get("source"),
            recorder.ring_root, analysis_source,
        )
    else:
        log.warning(
            "recording.enabled is false: analyze-only "
            "(events get a metadata sidecar + thumbnail, no ring clip)."
        )

    try:
        summary = run_live(
            analysis_source, config,
            on_event=on_event,
            overlay_buffer=overlay_buffer,
            max_seconds=getattr(args, "max_seconds", None),
        )
        log.info("Run finished: %s", summary)
        return 0
    except (RuntimeError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    finally:
        # Stop the exporter first so it flushes any already-ready clips off the
        # ring the recorder just wrote, then stop the recorder.
        if exporter is not None:
            exporter.stop()
        if recorder is not None:
            recorder.request_stop()
            if rec_thread is not None:
                rec_thread.join(timeout=10)
        if speed_log is not None:
            speed_log.close()


def handle_calibrate(args: argparse.Namespace, config: Config) -> int:
    """Interactive 4-point calibration -> lane-band preview image."""
    try:
        import cv2
    except ImportError:
        log.error("calibrate requires the 'analyze' extra (opencv-python).")
        return 1

    from pathlib import Path

    from .analyze import calibrate as cal
    from .util.paths import data_dir, ensure_dir

    try:
        frame = cal.load_frame(args.source)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1

    # De-warp first so corners land in the same space the analyzer operates in.
    from .analyze.undistort import build_undistorter

    undistorter = build_undistorter(config.calibration)
    if undistorter is not None:
        frame = undistorter(frame)
        log.info("Applied lens de-warp (k1=%.3f) before calibration.", undistorter.k1)

    # Resolve the four corners: explicit --points, else config, else interactive.
    if args.points:
        points = cal.parse_points(args.points)
    elif len(config.calibration.get("source_points") or []) == 4:
        points = [tuple(p) for p in config.calibration["source_points"]]
        log.info("Using existing source_points from config.")
    else:
        log.info("No points provided; opening interactive window (click 4 corners).")
        try:
            points = cal.collect_points_interactive(frame)
        except Exception as exc:  # noqa: BLE001 - headless / aborted
            log.error("Interactive calibration failed (%s). Pass --points instead.", exc)
            return 1

    preview, projector = cal.render_preview(frame, points, config.calibration)

    out = Path(args.output) if args.output else ensure_dir(data_dir() / "calibration") / "calibration_preview.jpg"
    ensure_dir(out.parent)
    cv2.imwrite(str(out), preview)

    log.info("Calibration corners (ordered TL,TR,BR,BL): %s", projector.source_points)
    log.info("Preview written to %s", out)
    print(cal.points_yaml_snippet(points))

    if getattr(args, "write", False):
        cal.write_source_points(args.config, points)
        log.info("Wrote source_points into %s (review: comments are not preserved).", args.config)
    return 0


def handle_test(args: argparse.Namespace, config: Config) -> int:
    """Run offline detection + tracking against a saved video file.

    Uses the real pipeline (YOLO + ByteTrack + debug video) when the CV stack
    is installed; falls back to the dependency-free stub otherwise so the
    command works on a core-only box.
    """
    from .analyze.offline import (
        MissingCVDependencies,
        run_offline,
        run_stub_pipeline,
    )

    from pathlib import Path

    if not Path(args.source).exists():
        # Keep the M0 acceptance behavior: a missing source is a non-fatal dry run.
        log.warning("Source not found; running dependency-free stub.")
        run_stub_pipeline(args.source, config)
        return 0

    try:
        summary = run_offline(args.source, config, export_events=True)
        log.info(
            "Done: %d candidate event(s), %d clip(s) exported.",
            summary.get("candidate_events", 0), summary.get("events_exported", 0),
        )
        return 0
    except MissingCVDependencies as exc:
        log.warning("CV stack unavailable (%s); running stub instead.", exc)
        run_stub_pipeline(args.source, config)
        return 0


def handle_export_event(args: argparse.Namespace, config: Config) -> int:
    """Export a clip for an explicit [start_ts, end_ts] window from the ring."""
    import uuid
    from pathlib import Path

    from .capture.segment_index import SegmentIndex
    from .events.exporter import export_from_segments
    from .events.metadata import write_metadata
    from .events.thumbnail import generate_thumbnail
    from .util.ffmpeg import ffmpeg_available
    from .util.paths import ensure_dir
    from .util.time import format_date_dir, format_segment_stamp

    if not ffmpeg_available():
        log.error("ffmpeg not found on PATH; cannot export.")
        return 1

    start_ts, end_ts = float(args.start_ts), float(args.end_ts)
    if end_ts <= start_ts:
        log.error("--end-ts must be greater than --start-ts.")
        return 1

    index_path = config.recording.get("segment_index_path", "/data/index/segments.sqlite")
    with SegmentIndex(index_path) as index:
        segments = index.get_overlapping(start_ts, end_ts)
    if not segments:
        log.error("No ring segments cover [%s, %s]; nothing to export.", start_ts, end_ts)
        return 1

    ms = int(start_ts * 1000)
    stamp = format_segment_stamp(ms, config.app.timezone)
    short = uuid.uuid4().hex[:8]
    out_dir = ensure_dir(
        Path(config.events.get("output_path", "/data/events"))
        / format_date_dir(ms, config.app.timezone) / "manual"
    )
    stem = f"{stamp}_manual_{short}"
    clip_path = out_dir / f"{stem}.mp4"
    export_from_segments(segments, start_ts, end_ts, clip_path)

    thumb_path = out_dir / f"{stem}.jpg"
    generate_thumbnail(clip_path, thumb_path, min(15.0, (end_ts - start_ts) / 2))

    write_metadata(
        {
            "event_id": short, "event_type": "manual", "event_types": ["manual"],
            "start_ts": start_ts, "trigger_ts": start_ts, "end_ts": end_ts,
            "clip_path": str(clip_path), "thumbnail_path": str(thumb_path),
            "segments": [s.path for s in segments],
        },
        out_dir / f"{stem}.json",
    )
    log.info("Exported manual clip -> %s", clip_path)
    return 0


def handle_health(args: argparse.Namespace, config: Config) -> int:
    """Report capture health: 0 if a fresh segment exists, 1 otherwise."""
    from .capture.health import max_segment_age, recording_health
    from .capture.segment_index import SegmentIndex
    from .util.time import now_unix

    index_path = config.recording.get("segment_index_path", "/data/index/segments.sqlite")
    segment_seconds = float(config.recording.get("segment_seconds", 10))
    with SegmentIndex(index_path) as index:
        latest = index.latest_end_ts()
    healthy, reason = recording_health(latest, now_unix(), max_segment_age(segment_seconds))
    if healthy:
        log.info("healthy: %s", reason)
        return 0
    log.error("unhealthy: %s", reason)
    return 1


def handle_police_report(args: argparse.Namespace, config: Config) -> int:
    """Summarize police-vehicle sightings logged by the live analyzer."""
    from pathlib import Path

    from .events.police_log import PoliceLog
    from .util.time import format_segment_stamp, now_unix

    pol = config.events.get("police") or {}
    db_path = pol.get("db_path", "data/index/police_sightings.sqlite")
    if not Path(db_path).exists():
        log.error("No police-sighting log at %s "
                  "(is events.police.enabled set, and has the run produced any?).", db_path)
        return 1

    hours = float(args.hours)
    now = now_unix()
    start = now - hours * 3600.0
    tz = config.app.timezone
    with PoliceLog(db_path) as plog:
        sightings = plog.in_window(start, now, police_only=True)

    speeding = [s for s in sightings if s.was_speeding]
    show = speeding if getattr(args, "speeding_only", False) else sightings
    print(f"Police sightings in the last {hours:g}h: {len(sightings)} "
          f"({len(speeding)} also speeding)")
    if not show:
        return 0
    print(f"{'time':<17}  {'dir':<14}  {'conf':>5}  {'km/h':>5}  speeding")
    for s in show:
        stamp = format_segment_stamp(int(s.ts * 1000), tz)
        kmh = f"{s.max_speed_kmh:.0f}" if s.max_speed_kmh is not None else "-"
        print(f"{stamp:<17}  {str(s.direction or '-'):<14}  "
              f"{s.confidence:>5.2f}  {kmh:>5}  {'YES' if s.was_speeding else ''}")
    return 0


def _kill_tree(proc) -> None:
    """Hard-stop a run subprocess and its children (ffmpeg) on Windows."""
    import subprocess
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, timeout=20)
    except Exception:  # noqa: BLE001 - fall back to a direct kill
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        pass


def _next_daylight_start(now, lat, lon, tz, buf):
    from datetime import timedelta

    from .util.sun import daylight_window
    for off in range(0, 4):
        win = daylight_window((now + timedelta(days=off)).date(), lat, lon, tz, buffer_minutes=buf)
        if win and win[0] > now:
            return win[0]
    return None


def _run_child_spec(config_path: str):
    """Argv + env for the supervised ``run`` child.

    Launched **unbuffered** (``-u`` plus ``PYTHONUNBUFFERED`` in the env) so that a
    crash's final error/traceback line flushes to the log immediately instead of
    dying in an unflushed block buffer on exit. (The child inherits the supervisor's
    stderr, which is redirected to ``supervise.log``; without unbuffering, the steady
    INFO stream flushes by volume but the last line before an ``exit 1`` is lost --
    which is exactly why past crashes left no traceback.) Putting PYTHONUNBUFFERED in
    the env -- not just ``-u`` -- means the exporter's render worker (a multiprocessing
    child) inherits unbuffered output too."""
    import os
    import sys

    argv = [sys.executable, "-u", "-m", "traffic_logger.main",
            "run", "--config", config_path]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return argv, env


def handle_supervise(args: argparse.Namespace, config: Config) -> int:
    """Run the analyzer only during daylight, and keep it alive while it should be.

    The camera is blind once fully dark, so there's no point recording/analyzing
    black frames. This computes the civil-twilight window for Hamilton's lat/long
    each day and starts ``run`` at dawn, stops it at dusk, and restarts it if it
    dies during the day -- which also covers the earlier resilience gap.
    """
    import subprocess
    import time
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from .util.sun import daylight_window

    sched = (getattr(config, "raw", {}) or {}).get("schedule") or {}
    if not sched.get("enabled", False):
        log.error("schedule.enabled is false; nothing to supervise.")
        return 1
    lat, lon = float(sched["latitude"]), float(sched["longitude"])
    buf = float(sched.get("buffer_minutes", 20))
    interval = float(sched.get("check_interval_seconds", 60))
    # all_day: keep run alive 24/7 (record + analyze through the night too) instead of
    # gating to the daylight window. Used once the camera has a usable night exposure.
    all_day = bool(sched.get("all_day", False))
    tz = ZoneInfo(config.app.timezone)

    proc = None
    if all_day:
        log.info("Supervisor up (24/7 all-day mode): %s, check %.0fs; run stays alive day and night.",
                 config.app.timezone, interval)
    else:
        log.info("Daylight supervisor up: (%.3f, %.3f) %s, civil twilight +/- %.0fmin, check %.0fs.",
                 lat, lon, config.app.timezone, buf, interval)
        _now = datetime.now(tz)
        _win = daylight_window(_now.date(), lat, lon, tz, buffer_minutes=buf)
        if _win and _win[0] <= _now <= _win[1]:
            log.info("Currently daylight (window %s..%s); starting run now.",
                     _win[0].strftime("%H:%M"), _win[1].strftime("%H:%M"))
        else:
            _nxt = _next_daylight_start(_now, lat, lon, tz, buf)
            log.info("Currently dark; holding. Next run start %s.",
                     _nxt.strftime("%Y-%m-%d %H:%M") if _nxt else "unknown")
    try:
        while True:
            now = datetime.now(tz)
            win = daylight_window(now.date(), lat, lon, tz, buffer_minutes=buf)
            active = True if all_day else (win is not None and win[0] <= now <= win[1])
            alive = proc is not None and proc.poll() is None
            if active and not alive:
                if proc is not None:
                    log.warning("Run exited (code %s); restarting.", proc.returncode)
                argv, child_env = _run_child_spec(args.config)
                proc = subprocess.Popen(argv, env=child_env)
                until = "24/7" if all_day or win is None else win[1].strftime("%H:%M")
                log.info("Started run (pid %s), active until %s.", proc.pid, until)
            elif not active and alive:
                log.info("Past dusk: stopping run.")
                _kill_tree(proc)
                nxt = _next_daylight_start(now, lat, lon, tz, buf)
                log.info("Run stopped; next start %s.",
                         nxt.strftime("%Y-%m-%d %H:%M") if nxt else "unknown")
                proc = None
            time.sleep(interval)
    except KeyboardInterrupt:
        log.info("Supervisor interrupted.")
    finally:
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)
    return 0


def handle_speed_report(args: argparse.Namespace, config: Config) -> int:
    """Summarize absolute-speed-gate violations for a community-safety case."""
    import json
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo

    from .events.speed_log import SpeedLog
    from .events.speed_report import aggregate, violation_from_record
    from .util.time import now_unix

    db_path = config.events.get("speed_log_path", "data/index/speed_log.sqlite")
    if not Path(db_path).exists():
        log.error("No speed log at %s (has the run logged any 55+ violations yet?).", db_path)
        return 1
    tz = config.app.timezone
    limit = float(args.limit)
    now = now_unix()
    start = now - float(args.days) * 86400.0
    with SpeedLog(db_path) as slog:
        records = slog.in_window(start, now)
    violations = [violation_from_record(r, limit) for r in records]

    st = aggregate(violations, tz, top=int(args.top))
    zone = ZoneInfo(tz)

    def bar(n, mx):
        return "#" * int(round(20 * n / mx)) if mx else ""

    print(f"\nSpeeding report — last {args.days:g} day(s)  "
          f"(posted limit {limit:g} km/h, gate {limit + 5:g}+)")
    print("=" * 62)
    if st.count == 0:
        print("No absolute-gate violations recorded in this window yet.")
        return 0
    clipped = sum(1 for v in violations if v.clipped)
    print(f"Violations: {st.count}"
          + (f"  over ~{st.span_days:.1f} days  (~{st.per_day:g}/day)" if st.span_days >= 1 else ""))
    print(f"Clips kept (excessive): {clipped}   text-only (logged, no clip): {st.count - clipped}")
    print(f"Speed: max {st.max_kmh:g}  mean {st.mean_kmh:g}  median {st.median_kmh:g} km/h")

    print("\nSpeed distribution (km/h):")
    mx = max((c for _, c in st.by_speed_bin), default=0)
    for label, c in st.by_speed_bin:
        print(f"  {label:>6}: {c:>4}  {bar(c, mx)}")

    print("\nViolations by hour of day:")
    mh = max((c for _, c in st.by_hour), default=0)
    for h, c in st.by_hour:
        print(f"  {h:02d}:00  {c:>4}  {bar(c, mh)}")

    print("\nBy direction:")
    for d, c in st.by_direction:
        print(f"  {d:<14} {c}")

    print("\nBy vehicle type:")
    for t, c in st.by_vehicle_type:
        print(f"  {t:<12} {c}")

    print(f"\nTop {len(st.worst)} fastest:")
    for v in st.worst:
        stamp = datetime.fromtimestamp(v.ts, zone).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {stamp}  {v.speed_kmh:>5.1f} km/h  (+{v.over_limit_kmh:g} over)  "
              f"{str(v.vehicle_type or '-'):<11} {str(v.direction or '-'):<14} "
              f"{'clip' if v.clipped else '-'}")

    if getattr(args, "csv", None):
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            fh.write("time_local,speed_kmh,over_limit_kmh,vehicle_type,direction,clip_kept\n")
            for v in sorted(violations, key=lambda x: x.ts):
                stamp = datetime.fromtimestamp(v.ts, zone).strftime("%Y-%m-%d %H:%M:%S")
                fh.write(f"{stamp},{v.speed_kmh},{v.over_limit_kmh},"
                         f"{v.vehicle_type or ''},{v.direction or ''},{int(v.clipped)}\n")
        print(f"\nCSV written: {out}  ({len(violations)} rows)")
    return 0


def handle_prune_ring(args: argparse.Namespace, config: Config) -> int:
    """Prune the ring buffer down to its configured size cap (one pass)."""
    from .capture.ring_pruner import prune_ring
    from .capture.segment_index import SegmentIndex

    rec = config.recording
    index_path = rec.get("segment_index_path", "/data/index/segments.sqlite")
    max_bytes = int(float(rec.get("ring_max_gb", 200)) * (1024 ** 3))

    with SegmentIndex(index_path) as index:
        before = index.total_bytes()
        result = prune_ring(index, max_bytes)

    gb = 1024 ** 3
    log.info(
        "Ring prune: %d segment(s) deleted, freed %.2f GB; ring %.2f GB -> %.2f GB (cap %.0f GB)",
        len(result.deleted_paths),
        result.freed_bytes / gb,
        before / gb,
        result.remaining_bytes / gb,
        max_bytes / gb,
    )
    return 0


def handle_serve(args: argparse.Namespace, config: Config) -> int:
    """Run the web dashboard, or (``--new-link``) rotate its secret unlock link.

    The dashboard is gated by a 404-everything ASGI middleware unlocked by a secret
    URL (see ``web/auth.py``). It only reads the analyzer's stores, so it is safe to
    start/stop independently of the capture+analysis process."""
    if getattr(args, "new_link", False):
        return _regenerate_web_link(config)

    from .web.app import WebSettings, create_app

    try:
        settings = WebSettings.from_config(config)
    except Exception as exc:  # noqa: BLE001 - report a clean message, not a traceback
        log.error("Invalid web config in %s: %s", config.source_path, exc)
        return 2

    missing = [name for name, val in (("web.access_token", settings.access_token),
                                      ("web.session_secret", settings.session_secret))
               if not val]
    if missing:
        log.error("Missing %s. Generate an unlock link first:\n"
                  "  traffic-log serve --new-link --config %s",
                  " and ".join(missing), config.source_path)
        return 2

    try:
        import uvicorn
    except ImportError:
        log.error("uvicorn is not installed. Install the web extra: pip install -e .[web]")
        return 2

    app = create_app(settings)
    log.info("Traffic Watch on http://%s:%d  (unlock path: /%s/<token>)",
             settings.host, settings.port, settings.unlock_prefix)
    log.info("Reading events=%s  speed_log=%s", settings.events_dir, settings.speed_log_path)
    log.info("Expose with:  tailscale funnel %d", settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
    return 0


def _regenerate_web_link(config: Config) -> int:
    """Mint a fresh access token (rotating/revoking old links), write it into the
    config's ``web:`` block, and print the unlock path to bookmark."""
    import re
    import secrets
    from pathlib import Path

    from .config import load_config
    from .web import auth
    from .web.app import WebSettings
    from .web.auth import new_token

    cfg = config.source_path
    if cfg is None or not Path(cfg).exists():
        log.error("Config file not found; cannot store the unlock link.")
        return 2
    text = Path(cfg).read_text(encoding="utf-8")
    if "access_token:" not in text:
        log.error("No `web:` block with `access_token:` in %s. Add the web section "
                  "first (see DEPLOY.md).", cfg)
        return 2

    token = new_token()
    # Guard the regex surgery: only rewrite when there's EXACTLY one access_token:
    # key, so a second one (e.g. another section growing the same field) can't cause
    # us to rotate the wrong line and silently leave the real token stale.
    token_pat = re.compile(r"(?m)^(\s*access_token:).*$")
    n_token = len(token_pat.findall(text))
    if n_token != 1:
        log.error("Expected exactly one `access_token:` in %s, found %d; refusing to "
                  "rewrite (fix the config by hand).", cfg, n_token)
        return 2
    text = token_pat.sub(lambda m: f'{m.group(1)} "{token}"', text, count=1)

    def _fill_secret(m: "re.Match") -> str:
        # Rotate the cookie-SIGNING secret too, not just the unlock token. Otherwise every
        # previously issued session cookie (signed with the old secret) stays valid for its
        # full 30-day life -- so a compromised device keeps access and "old links revoked"
        # is a lie. Rotating it invalidates all existing sessions; devices re-unlock with
        # the new link. (Takes effect once the running server is restarted to reload it.)
        return f'{m.group(1)} "{secrets.token_hex(32)}"'

    if re.search(r"(?m)^(\s*session_secret:)(.*)$", text):
        text = re.sub(r"(?m)^(\s*session_secret:)(.*)$", _fill_secret, text, count=1)
    Path(cfg).write_text(text, encoding="utf-8")

    settings = WebSettings.from_config(load_config(cfg))
    days = auth.COOKIE_MAX_AGE // 86400
    print("\nNew unlock link generated (old links AND existing sessions are now revoked).")
    print("Open this once on each device, prepending your Tailscale funnel host:\n")
    print(f"    https://<your-funnel-host>{settings.unlock_path()}\n")
    print(f"Local test URL:  http://localhost:{settings.port}{settings.unlock_path()}")
    print(f"It sets a signed cookie good for {days} days; everything else returns 404.")
    print("Restart the running `serve` process for the rotation to take effect.")
    return 0
