"""Build the DEMO dataset for privacy-safe dashboard screenshots.

The offline pipeline (``traffic-log test`` on licensed stock footage, see
``config/config.demo.local.yaml``) produces real events with video-offset
timestamps and writes no pass/speed stores (those are live-loop only). This
script finishes the job so `serve` renders a fully-populated dashboard:

1. **Re-stamp events**: each exported event gets a plausible wall-clock trigger
   spread over the past N days (the newest few land today so the Now page has
   content), files are moved into the matching date folders and renamed with
   the project's own stem builder, and the sidecar timestamps are shifted.
2. **Synthesize the passes table** (the traffic denominator) with a realistic
   hour-of-day volume curve and speed distribution, written through the real
   :class:`TrafficStore` API with validity derived by the real
   :func:`speed_validity` invariants -- so the demo rows are exactly the shape
   production rows have.
3. **Synthesize the speed log** rows for every synthetic violation.

Deterministic under ``--seed``. Never touches the real deployment stores; all
paths default into ``data/demo/``.

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/make_demo_data.py
    .venv/Scripts/python.exe scripts/make_demo_data.py --days 7 --per-day 1600
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_logger.analyze.pass_validity import PassGeometry, speed_validity  # noqa: E402
from traffic_logger.events.metadata import event_stem  # noqa: E402
from traffic_logger.events.speed_log import SpeedLog, SpeedRecord  # noqa: E402
from traffic_logger.events.store import PassRecord, TrafficStore  # noqa: E402

# Relative hourly traffic volume (residential arterial): overnight trickle,
# AM/PM commute peaks, steady daytime.
HOUR_WEIGHTS = [0.10, 0.06, 0.05, 0.05, 0.10, 0.30, 0.70, 1.30, 1.60, 1.20,
                1.00, 1.00, 1.05, 1.05, 1.10, 1.30, 1.55, 1.65, 1.35, 1.00,
                0.75, 0.55, 0.35, 0.20]
VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle"]
VEHICLE_WEIGHTS = [0.86, 0.09, 0.02, 0.03]
DIRECTIONS = ["left_to_right", "right_to_left"]


# ---------------------------------------------------------------- events ----
def restamp_events(events_dir: Path, tz: ZoneInfo, now: float, days: int,
                   rng: random.Random) -> int:
    """Shift each exported event to a plausible wall-clock time and refile it."""
    sidecars = sorted(p for p in events_dir.glob("*/*/*.json")
                      if not p.name.endswith("_overlay.json"))
    if not sidecars:
        print("no event sidecars found; run the demo pipeline first")
        return 0

    # Newest few events land today (recent hours) so the Now page has rows;
    # the rest spread over the past `days`, daytime-weighted.
    triggers = []
    for i in range(len(sidecars)):
        if i < 3:
            t = now - rng.uniform(0.4, 6.0) * 3600
        else:
            day_off = rng.randint(1, max(1, days - 1))
            hour = rng.choices(range(24), weights=HOUR_WEIGHTS)[0]
            base = datetime.fromtimestamp(now, tz).replace(
                hour=hour, minute=rng.randint(0, 59), second=rng.randint(0, 59),
                microsecond=0) - timedelta(days=day_off)
            t = base.timestamp()
        triggers.append(t)
    triggers.sort(reverse=True)

    moved = 0
    for sidecar, new_trigger in zip(sidecars, triggers):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        delta = new_trigger - float(meta.get("trigger_ts", 0.0))
        for key in ("start_ts", "trigger_ts", "end_ts"):
            if meta.get(key) is not None:
                meta[key] = round(float(meta[key]) + delta, 3)
        local = datetime.fromtimestamp(new_trigger, tz)
        meta["created_at"] = local.isoformat(timespec="seconds")

        evidences = [t.get("evidence") or {}
                     for t in (meta.get("evidence", {}) or {}).get("triggers", [])]
        stamp = local.strftime("%Y%m%d_%H%M%S")
        short = (meta.get("event_id") or "demo0000")[:8]
        etype = meta.get("event_type", "event")
        stem = event_stem(stamp, etype, short, evidences)

        new_dir = events_dir / local.strftime("%Y-%m-%d") / etype
        new_dir.mkdir(parents=True, exist_ok=True)
        old_stem = sidecar.stem
        for ext in (".mp4", ".jpg"):
            src = sidecar.parent / f"{old_stem}{ext}"
            if src.exists():
                shutil.move(str(src), str(new_dir / f"{stem}{ext}"))
        meta["clip_path"] = (new_dir / f"{stem}.mp4").as_posix()
        meta["thumbnail_path"] = (new_dir / f"{stem}.jpg").as_posix()
        (new_dir / f"{stem}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        if (new_dir / f"{stem}.json") != sidecar:
            sidecar.unlink()
        moved += 1

    # Sweep now-empty old date/type folders.
    for d in sorted(events_dir.glob("*/*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    for d in sorted(events_dir.glob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return moved


# ---------------------------------------------------------------- passes ----
def _event_passes(events_dir: Path, tz: ZoneInfo) -> list[dict]:
    """One pass per re-stamped event so the stats agree with the visible clips
    (otherwise the Top Speeds clip can read faster than the stats' max)."""
    out = []
    for sidecar in events_dir.glob("*/*/*.json"):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        speed = direction = vtype = None
        for t in (meta.get("evidence", {}) or {}).get("triggers", []):
            ev = t.get("evidence") or {}
            if ev.get("speed_kmh") is not None:
                speed = max(speed or 0, float(ev["speed_kmh"]))
                direction = direction or ev.get("direction")
                vtype = vtype or ev.get("vehicle_type")
        if speed:
            out.append({"ts": float(meta["trigger_ts"]), "speed": speed,
                        "direction": direction, "vehicle_type": vtype or "car"})
    return out


def synthesize_passes(db_path: Path, speed_log_path: Path, tz: ZoneInfo,
                      now: float, days: int, per_day: int,
                      rng: random.Random, events_dir: Path) -> tuple[int, int]:
    store = TrafficStore(db_path)
    slog = SpeedLog(speed_log_path)
    total = sum(HOUR_WEIGHTS)
    written = violations = 0
    try:
        # The visible event clips become real passes first, so every dashboard
        # number that references "the fastest car" agrees with a watchable clip.
        for i, ep in enumerate(_event_passes(events_dir, tz)):
            day = datetime.fromtimestamp(ep["ts"], tz).strftime("%Y%m%d")
            session = f"demo-{day}"
            store.start_session(session, ep["ts"], camera_id="demo_stock")
            track_seconds = rng.uniform(1.6, 2.4)
            store.upsert_pass(PassRecord(
                session_id=session, track_id=90000 + i,
                first_ts=ep["ts"] - track_seconds, last_ts=ep["ts"],
                direction=ep["direction"], vehicle_type=ep["vehicle_type"],
                max_speed_kmh=round(ep["speed"] * 1.04, 1),
                steady_speed_kmh=ep["speed"], was_speeding=ep["speed"] >= 55.0,
                steady_speed_raw=ep["speed"], ground_span=round(rng.uniform(0.95, 1.2), 3),
                n_points=int(track_seconds * 20), steady_valid=True,
                steady_invalid_reason=None))
            written += 1
            if ep["speed"] >= 55.0:
                violations += 1
                slog.add(SpeedRecord(ts=ep["ts"], speed_kmh=ep["speed"],
                                     direction=ep["direction"], clipped=True,
                                     vehicle_type=ep["vehicle_type"]))
        for day_off in range(days - 1, -1, -1):
            day_local = datetime.fromtimestamp(now, tz).replace(
                hour=0, minute=0, second=0, microsecond=0) - timedelta(days=day_off)
            session = f"demo-{day_local.strftime('%Y%m%d')}"
            store.start_session(session, day_local.timestamp(), camera_id="demo_stock")
            track_id = 0
            for hour in range(24):
                n = round(per_day * HOUR_WEIGHTS[hour] / total * rng.uniform(0.85, 1.15))
                for _ in range(n):
                    ts = (day_local + timedelta(hours=hour,
                                                seconds=rng.uniform(0, 3599))).timestamp()
                    if ts > now:
                        continue
                    track_id += 1
                    track_seconds = rng.uniform(1.3, 3.4)
                    if rng.random() < 0.04:
                        speed = None            # unmeasured pass (flicker/occlusion)
                    else:
                        vt_slow = rng.random() < 0.10
                        speed = rng.gauss(38.0 if vt_slow else 45.0,
                                          5.0 if vt_slow else 7.5)
                        speed = max(16.0, min(93.0, speed))
                    span = rng.uniform(0.85, 1.25)
                    n_pts = int(track_seconds * rng.uniform(16, 22))
                    valid, reason = speed_validity(PassGeometry(
                        steady_kmh=speed, track_seconds=track_seconds,
                        ground_span=span, n_points=n_pts))
                    vtype = rng.choices(VEHICLE_TYPES, weights=VEHICLE_WEIGHTS)[0]
                    direction = rng.choice(DIRECTIONS)
                    speeding = bool(speed is not None and valid and speed >= 55.0)
                    store.upsert_pass(PassRecord(
                        session_id=session, track_id=track_id,
                        first_ts=ts - track_seconds, last_ts=ts,
                        direction=direction, vehicle_type=vtype,
                        max_speed_kmh=round(speed * rng.uniform(1.02, 1.10), 1) if speed else None,
                        steady_speed_kmh=round(speed, 1) if speed else None,
                        was_speeding=speeding,
                        steady_speed_raw=round(speed, 1) if speed else None,
                        ground_span=round(span, 3), n_points=n_pts,
                        steady_valid=valid, steady_invalid_reason=reason))
                    written += 1
                    if speeding:
                        violations += 1
                        slog.add(SpeedRecord(ts=ts, speed_kmh=round(speed, 1),
                                             direction=direction,
                                             clipped=speed >= 70.0,
                                             vehicle_type=vtype))
    finally:
        store.close()
        slog.close()
    return written, violations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events-dir", default="data/demo/events")
    ap.add_argument("--db", default="data/demo/index/traffic.sqlite")
    ap.add_argument("--speed-log", default="data/demo/index/speed_log.sqlite")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--per-day", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tz", default="America/Toronto")
    args = ap.parse_args()

    tz = ZoneInfo(args.tz)
    now = time.time()
    rng = random.Random(args.seed)

    db, slog = Path(args.db), Path(args.speed_log)
    for p in (db, slog):
        if p.exists():
            p.unlink()                      # rebuild deterministically each run

    moved = restamp_events(Path(args.events_dir), tz, now, args.days, rng)
    written, violations = synthesize_passes(db, slog, tz, now, args.days,
                                            args.per_day, rng,
                                            Path(args.events_dir))
    print(f"events re-stamped: {moved}")
    print(f"passes written:    {written} over {args.days} days "
          f"({violations} violations >=55; {violations / max(written,1):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
