"""Re-derive each pass's ``steady_valid`` flag from the shared invariant module.

This is the re-runnable heart of the validated-view design: because the raw measurement
(and, for new rows, the geometry) is preserved, "backfill" is just *re-run the validator* --
idempotent, cheap, repeatable. When the pipeline surprises you with a new artifact class,
you add ONE predicate to ``analyze/pass_validity.py`` and re-run this; no new read-time
config knob, no era gate.

It calls the SAME :func:`speed_validity` the live writer (``PassRecorder``) calls, so the
two paths can't drift. For rows written before the geometry columns existed, ``ground_span``
/ ``n_points`` are NULL and the module falls back to the implied-distance proxy; their raw
speed is taken from the legacy ``steady_speed_kmh`` column (and backfilled into
``steady_speed_raw`` so every row exposes one uniform raw value).

Dry-run by default -- pass ``--write`` to persist. Safe to run against a live DB (WAL +
busy timeout), though a quiet moment avoids writer contention.

Usage:
    python scripts/revalidate_passes.py [db_path] [--write]
    (db_path default: data/index/traffic.sqlite)
"""

from __future__ import annotations

import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from traffic_logger.analyze.pass_validity import (  # noqa: E402
    PassGeometry, ValidityThresholds, speed_validity)

THRESHOLDS = ValidityThresholds()


def revalidate(db_path: str, write: bool) -> dict:
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        rows = conn.execute(
            "SELECT id, first_ts, last_ts, steady_speed_kmh, steady_speed_raw, "
            "ground_span, n_points FROM passes").fetchall()
        reasons: Counter = Counter()
        updates = []
        for r in rows:
            raw = r["steady_speed_raw"]
            if raw is None:
                raw = r["steady_speed_kmh"]      # legacy rows: the guarded column is our best raw
            track_seconds = float(r["last_ts"]) - float(r["first_ts"])
            valid, reason = speed_validity(
                PassGeometry(steady_kmh=raw, track_seconds=track_seconds,
                             ground_span=r["ground_span"], n_points=r["n_points"]),
                THRESHOLDS)
            reasons[reason or "valid"] += 1
            updates.append((raw, 1 if valid else 0, reason, r["id"]))
        if write:
            conn.executemany(
                "UPDATE passes SET steady_speed_raw = COALESCE(steady_speed_raw, ?), "
                "steady_valid = ?, steady_invalid_reason = ? WHERE id = ?", updates)
            conn.commit()
        return {"total": len(rows), "reasons": dict(reasons)}
    finally:
        conn.close()


def check_drift(db_path: str) -> list:
    """Rows currently flagged valid that FAIL the invariants -- the regression signal.

    Post-migration this must be empty; a non-zero count means the live pipeline grew a
    new artifact class that the write-time validity missed (find out from a counter, not
    from a 152 km/h truck on the Top Speeds page). Add the predicate + re-run this script."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, first_ts, last_ts, steady_speed_raw, ground_span, n_points "
            "FROM passes WHERE steady_valid = 1").fetchall()
    finally:
        conn.close()
    bad = []
    for r in rows:
        valid, reason = speed_validity(
            PassGeometry(steady_kmh=r["steady_speed_raw"],
                         track_seconds=float(r["last_ts"]) - float(r["first_ts"]),
                         ground_span=r["ground_span"], n_points=r["n_points"]),
            THRESHOLDS)
        if not valid:
            bad.append((r["id"], reason))
    return bad


def main() -> int:
    args = [a for a in sys.argv[1:] if a not in ("--write", "--check")]
    write = "--write" in sys.argv
    db = args[0] if args else "data/index/traffic.sqlite"
    if not Path(db).exists():
        print(f"no such DB: {db}")
        return 2
    if "--check" in sys.argv:
        bad = check_drift(db)
        print(f"drift check {db}: {len(bad)} valid rows fail invariants"
              + (" (OK)" if not bad else ""))
        for pid, reason in bad[:20]:
            print(f"  pass id={pid} -> {reason}")
        return 1 if bad else 0
    res = revalidate(db, write)
    print(f"{'WROTE' if write else 'DRY-RUN'} {db}: {res['total']} passes")
    for reason, n in sorted(res["reasons"].items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<22} {n}")
    if not write:
        print("\n(dry run -- re-run with --write to persist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
