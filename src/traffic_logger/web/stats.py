"""Speeding statistics for the dashboard, computed from the validated passes log.

A "violation" is a completed vehicle pass whose GPS-validated full-track steady
speed cleared the gate. The numbers come from the ``passes`` table (via
:func:`read_passes` + :func:`passes_to_violations`) rather than the live event log:
the event log records the *trigger* speed, measured on a still-partial track, which
reads ~7 km/h high. Sourcing violations from passes keeps the dashboard on the same
GPS-validated metric as the speed-test page, and on the same table as the traffic
denominator (``volume_summary``).

Access is split so the maths is trivially unit-testable:

* :func:`read_passes` / :func:`first_pass_ts` are the only DB-touching functions --
  windowed SELECTs.
* Everything else is pure: it takes a list of :class:`Violation` (built by
  :func:`passes_to_violations`) and returns plain JSON-able dicts. Day/hour bucketing
  is done in Python against a ``ZoneInfo`` tz (not in SQL) so "per day" means local
  calendar days, not UTC. The dataset is small, so pulling a window into memory is cheap.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from ..util.logging import get_logger

log = get_logger(__name__)

# sqlite3.OperationalError covers BOTH "table/DB not created yet" (a legitimate empty
# result on a fresh install) AND transient failures like "database is locked" under
# writer contention. Treating them alike makes a locked read render as ZERO stats with
# no trace. So classify: a benign-missing error -> empty quietly; anything else (lock,
# corruption) -> log a warning so the degraded read is visible, then still return empty
# (never 500 the dashboard over a transient lock).
_BENIGN_DB_MISSING = ("no such table", "unable to open database", "no such column")


def _benign_missing(exc: Exception) -> bool:
    return any(s in str(exc).lower() for s in _BENIGN_DB_MISSING)


def _connect_ro(db_path: str) -> sqlite3.Connection:
    """Read-only connection with an explicit busy timeout, so a read waits out a
    writer's lock (up to 5s) instead of failing instantly."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)

# Speed buckets for the distribution chart (lower-inclusive, km/h). The open-ended
# top bucket catches the rare extreme speeders.
_BUCKETS = [(55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 999)]


@dataclass(frozen=True)
class Violation:
    ts: float
    speed_kmh: float
    direction: Optional[str]
    vehicle_type: Optional[str]
    clipped: bool


# --- pure aggregations ------------------------------------------------------

def passes_to_violations(passes: List["Pass"], over_limit_kmh: float,
                         clip_threshold: float = 70.0) -> List[Violation]:
    """Derive 'violations' from the validated passes log instead of the event log.

    The Stats page historically counted ``speed_events`` -- rows logged at the live
    *trigger* speed, which reads ~7 km/h high (it measures a noisy PARTIAL track). The
    passes log carries each car's GPS-validated full-track steady speed, so a violation
    here = a completed car whose validated speed cleared the gate. This keeps the
    dashboard's numerator on the same metric as its denominator (``volume_summary``)
    and the speed-test page. ``clipped`` approximates "kept a clip" (steady >=
    clip_threshold). Pure -- the DB read stays in :func:`read_passes`."""
    out: List[Violation] = []
    for p in passes:
        s = p.steady_kmh
        if s is None or s < over_limit_kmh:
            continue
        out.append(Violation(ts=p.ts, speed_kmh=float(s), direction=p.direction,
                             vehicle_type=p.vehicle_type, clipped=(s >= clip_threshold)))
    return out


def _local_date(ts: float, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d")


def _round(x: Optional[float], n: int = 1) -> Optional[float]:
    return None if x is None else round(x, n)


def summarize(viols: List[Violation], *, speed_limit: float = 50.0,
              fast_threshold: float = 70.0) -> Dict:
    """Headline numbers for a set of violations: count, top/avg speed, how many
    cleared the clip threshold, and the share over the posted limit."""
    n = len(viols)
    if n == 0:
        return {"count": 0, "max_kmh": None, "avg_kmh": None,
                "over_fast": 0, "over_limit_pct": None, "clipped": 0,
                "fast_threshold": fast_threshold, "speed_limit": speed_limit}
    speeds = [v.speed_kmh for v in viols]
    over_limit = sum(1 for s in speeds if s > speed_limit)
    return {
        "count": n,
        "max_kmh": _round(max(speeds)),
        "avg_kmh": _round(sum(speeds) / n),
        "over_fast": sum(1 for s in speeds if s >= fast_threshold),
        "over_limit_pct": _round(100.0 * over_limit / n),
        "clipped": sum(1 for v in viols if v.clipped),
        "fast_threshold": fast_threshold,
        "speed_limit": speed_limit,
    }


def daily_series(viols: List[Violation], tz: ZoneInfo, *, days: int,
                 now_ts: float, fast_threshold: float = 70.0) -> List[Dict]:
    """Per-local-day rollup for the last ``days`` days (gap-filled with zeros so the
    chart has a continuous x-axis). Newest day last."""
    today = datetime.fromtimestamp(now_ts, tz).date()
    wanted = [(today - timedelta(days=i)).strftime("%Y-%m-%d")
              for i in range(days - 1, -1, -1)]
    by_day: Dict[str, List[Violation]] = {d: [] for d in wanted}
    for v in viols:
        d = _local_date(v.ts, tz)
        if d in by_day:
            by_day[d].append(v)
    out = []
    for d in wanted:
        items = by_day[d]
        speeds = [v.speed_kmh for v in items]
        out.append({
            "date": d,
            "count": len(items),
            "over_fast": sum(1 for s in speeds if s >= fast_threshold),
            "max_kmh": _round(max(speeds)) if speeds else None,
            "avg_kmh": _round(sum(speeds) / len(speeds)) if speeds else None,
        })
    return out


def hourly_histogram(viols: List[Violation], tz: ZoneInfo) -> List[Dict]:
    """Count + average speed by hour-of-day (0..23) across the supplied window --
    answers "when do people speed?"."""
    counts = [0] * 24
    sums = [0.0] * 24
    for v in viols:
        h = datetime.fromtimestamp(v.ts, tz).hour
        counts[h] += 1
        sums[h] += v.speed_kmh
    return [
        {"hour": h, "count": counts[h],
         "avg_kmh": _round(sums[h] / counts[h]) if counts[h] else None}
        for h in range(24)
    ]


def speed_distribution(viols: List[Violation]) -> List[Dict]:
    """Histogram of violations across fixed km/h buckets."""
    out = []
    for lo, hi in _BUCKETS:
        c = sum(1 for v in viols if lo <= v.speed_kmh < hi)
        label = f"{lo}+" if hi >= 999 else f"{lo}-{hi - 1}"
        out.append({"bucket": label, "min_kmh": lo, "count": c})
    return out


def vehicle_breakdown(viols: List[Violation]) -> List[Dict]:
    """Violation counts per vehicle type, most common first."""
    counts: Dict[str, int] = {}
    for v in viols:
        key = v.vehicle_type or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return [{"vehicle_type": k, "count": c}
            for k, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]


# --- traffic volume / denominator (the `passes` table) ----------------------

@dataclass(frozen=True)
class Pass:
    ts: float
    steady_kmh: Optional[float]
    direction: Optional[str]
    vehicle_type: Optional[str]


def read_passes(db_path: str, *, since_ts: Optional[float] = None,
                until_ts: Optional[float] = None) -> List[Pass]:
    """Load completed vehicle passes (one row per drive-by) in the window. Missing
    DB / table yields an empty list, so the dashboard works before the unified store
    exists. Keyed on ``last_ts`` (when the car left).

    Speed comes from the **validated view**: the raw measurement ``steady_speed_raw``
    is surfaced only when ``steady_valid`` is set, else the speed is None (the car still
    counts as traffic -- only its untrustworthy speed is dropped). Validity is derived
    once at write time from the scene-physics invariants in
    :mod:`~traffic_logger.analyze.pass_validity` (and backfilled over history by
    ``scripts/revalidate_passes.py``), so this read path carries no plausibility logic of
    its own -- it just trusts the flag. Policy thresholds (posted limit, fast/Hall cutoffs)
    stay in the pure aggregations that consume these passes."""
    clauses, params = [], []
    if since_ts is not None:
        clauses.append("last_ts >= ?")
        params.append(since_ts)
    if until_ts is not None:
        clauses.append("last_ts <= ?")
        params.append(until_ts)
    if not db_path:
        return []
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = ("SELECT last_ts, steady_speed_raw, steady_valid, direction, vehicle_type "
           "FROM passes" + where + " ORDER BY last_ts ASC")
    try:
        conn = _connect_ro(db_path)
    except sqlite3.OperationalError as exc:
        if not _benign_missing(exc):
            log.warning("pass stats read degraded (%s): %s", db_path, exc)
        return []
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        if not _benign_missing(exc):
            log.warning("pass stats query degraded (%s): %s", db_path, exc)
        return []
    finally:
        conn.close()
    out: List[Pass] = []
    for r in rows:
        # Trust the write-time validity flag: surface the raw speed only when valid.
        steady = r["steady_speed_raw"] if r["steady_valid"] == 1 else None
        out.append(Pass(ts=float(r["last_ts"]), steady_kmh=steady,
                        direction=r["direction"], vehicle_type=r["vehicle_type"]))
    return out


def first_pass_ts(db_path: str) -> Optional[float]:
    """Earliest recorded pass (``min(last_ts)``), or None if the store is empty/missing.

    The dashboard clamps multi-day stats to this so the violation numerator and the
    car denominator always cover the same period -- pass logging started long after the
    violation log, so an unclamped window would divide N days of violations by only the
    days we've been counting cars."""
    if not db_path:
        return None
    try:
        conn = _connect_ro(db_path)
    except sqlite3.OperationalError as exc:
        if not _benign_missing(exc):
            log.warning("first_pass_ts read degraded (%s): %s", db_path, exc)
        return None
    try:
        row = conn.execute("SELECT MIN(last_ts) FROM passes").fetchone()
    except sqlite3.OperationalError as exc:
        if not _benign_missing(exc):
            log.warning("first_pass_ts query degraded (%s): %s", db_path, exc)
        return None
    finally:
        conn.close()
    return float(row[0]) if row and row[0] is not None else None


def volume_summary(passes: List[Pass], *, over_threshold: float = 55.0) -> Dict:
    """The denominator view: how many cars, and what share were speeding.

    ``over_threshold`` is the *buffered* limit (default 55 in a 50 zone) -- real
    enforcement allows a few km/h of tolerance, and it matches the system's own gate
    (``speed_log`` records >=55), so a car at 51 isn't branded a speeder. ``measured``
    is the subset with a steady speed (brief/odd tracks have none); the over % is taken
    over ``measured`` since that's what we can actually judge."""
    total = len(passes)
    measured = [p for p in passes if p.steady_kmh is not None]
    over = [p for p in measured if p.steady_kmh >= over_threshold]
    return {
        "total": total,
        "measured": len(measured),
        "over_limit": len(over),
        "over_limit_pct": _round(100.0 * len(over) / len(measured)) if measured else None,
        "avg_kmh": _round(sum(p.steady_kmh for p in measured) / len(measured)) if measured else None,
        "over_kmh": over_threshold,
    }
