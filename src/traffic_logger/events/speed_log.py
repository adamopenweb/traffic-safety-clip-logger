"""Lightweight speed log (SQLite).

Records one row per flagged speeding vehicle -- the cheap text record kept for
*every* violation over the gate threshold -- while video clips are reserved for
the egregious ones (a separate, higher clip threshold). So we "track every
speeder, keep the footage only when it's excessive". The ``speed-report`` CLI
reads this log; ``events/speed_report.py`` does the aggregation.

Dependency-free (stdlib sqlite3) so it imports and unit-tests without the CV
stack. The live run writes it; a report read can run concurrently.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..util.paths import ensure_dir


@dataclass(frozen=True)
class SpeedRecord:
    ts: float                    # wall time the vehicle passed (trigger_ts)
    speed_kmh: float
    direction: Optional[str]
    clipped: bool                # whether a video clip was also saved
    vehicle_type: Optional[str] = None   # YOLO class: car/truck/bus/motorcycle


def event_speed_and_direction(final_event) -> Optional[tuple]:
    """(max speed_kmh, direction, vehicle_type) from an absolute-speeding event.

    Reads the event's candidate evidence; returns None for non-speeding events
    (e.g. center-lane) so the caller leaves their clip behaviour unchanged.
    """
    best = None
    for c in getattr(final_event, "candidates", []) or []:
        ev = getattr(c, "evidence", None) or {}
        if ev.get("rule") == "absolute_speeding" and ev.get("speed_kmh") is not None:
            spd = float(ev["speed_kmh"])
            if best is None or spd > best[0]:
                best = (spd, ev.get("direction"), ev.get("vehicle_type"))
    return best


_SCHEMA = """
CREATE TABLE IF NOT EXISTS speed_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    speed_kmh    REAL NOT NULL,
    direction    TEXT,
    clipped      INTEGER NOT NULL,
    vehicle_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_speed_ts ON speed_events(ts);
"""


class SpeedLog:
    """SQLite-backed log of every speeding violation over the gate threshold."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            ensure_dir(Path(self.db_path).parent)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # Migrate a pre-existing log that lacks the vehicle_type column.
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(speed_events)")}
            if "vehicle_type" not in cols:
                self._conn.execute("ALTER TABLE speed_events ADD COLUMN vehicle_type TEXT")
            self._conn.commit()

    def add(self, rec: SpeedRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO speed_events (ts, speed_kmh, direction, clipped, vehicle_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (rec.ts, rec.speed_kmh, rec.direction, int(rec.clipped), rec.vehicle_type),
            )
            self._conn.commit()

    def in_window(self, start_ts: float, end_ts: float) -> List[SpeedRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM speed_events WHERE ts >= ? AND ts <= ? ORDER BY ts ASC",
                (start_ts, end_ts),
            ).fetchall()
        return [SpeedRecord(ts=r["ts"], speed_kmh=r["speed_kmh"], direction=r["direction"],
                            clipped=bool(r["clipped"]), vehicle_type=r["vehicle_type"])
                for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SpeedLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
