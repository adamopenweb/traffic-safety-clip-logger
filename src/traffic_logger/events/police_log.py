"""Police-vehicle sighting log (SQLite) + pure sighting helpers.

Every completed vehicle track that the live analyzer classifies (see
``analyze/police_classifier.py``) becomes one row here: when it was seen, which
way it travelled, how fast, the aggregated police confidence, and whether it was
also flagged speeding. The ``police-report`` CLI command queries this table over
a time window to answer "how many police drove by, and how many were speeding?".

Dependency-free (stdlib ``sqlite3`` only) so it imports and unit-tests without
the CV/torch stack. The CLIP model lives in ``analyze/police_classifier.py``;
this module only persists and aggregates its results.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from ..util.paths import ensure_dir


@dataclass(frozen=True)
class Sighting:
    """One completed vehicle drive-by with its police classification."""

    ts: float                       # wall time the track was last seen (finalized)
    first_ts: float                 # wall time first seen
    track_id: int
    direction: Optional[str]
    confidence: float               # aggregated police probability (0..1)
    is_police: bool
    max_speed_rel: Optional[float]  # peak relative speed (normalized units/s)
    max_speed_kmh: Optional[float]  # peak km/h (only when calibrated to meters)
    was_speeding: bool              # track also triggered a relative_speeding event


def aggregate_score(scores: Sequence[float]) -> float:
    """A track's police confidence: the **mean** of its per-sample scores.

    Mean (not max) so a single fluke-high frame can't flag a vehicle -- on
    low-res sub-stream crops CLIP occasionally spikes on a plain dark SUV, but
    only a genuinely marked unit scores high across *most* of its frames. Empty
    -> 0.0 (never classified -> not police).
    """
    return sum(scores) / len(scores) if scores else 0.0


def decide_police(
    scores: Sequence[float], threshold: float, min_samples: int = 2
) -> tuple[bool, float]:
    """(is_police, confidence) from a track's sample scores.

    Police requires BOTH agreement across frames (at least ``min_samples``
    scored) AND a mean confidence over ``threshold`` -- a fast vehicle caught in
    a single frame is left unclassified rather than flagged on one reading.
    """
    confidence = aggregate_score(scores)
    is_police = len(scores) >= min_samples and confidence >= threshold
    return is_police, confidence


_SCHEMA = """
CREATE TABLE IF NOT EXISTS police_sightings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    first_ts      REAL NOT NULL,
    track_id      INTEGER NOT NULL,
    direction     TEXT,
    confidence    REAL NOT NULL,
    is_police     INTEGER NOT NULL,
    max_speed_rel REAL,
    max_speed_kmh REAL,
    was_speeding  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_police_ts ON police_sightings(ts);
"""


class PoliceLog:
    """SQLite-backed police-sighting log.

    Pass ``":memory:"`` for an ephemeral log (tests) or a path for the on-disk
    log (parent dirs created automatically). Thread-safe: the live analyzer's
    finalize path and an occasional report read can share one connection.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            ensure_dir(Path(self.db_path).parent)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def add(self, s: Sighting) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO police_sightings "
                "(ts, first_ts, track_id, direction, confidence, is_police, "
                " max_speed_rel, max_speed_kmh, was_speeding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (s.ts, s.first_ts, s.track_id, s.direction, s.confidence,
                 int(s.is_police), s.max_speed_rel, s.max_speed_kmh,
                 int(s.was_speeding)),
            )
            self._conn.commit()

    def in_window(self, start_ts: float, end_ts: float,
                  police_only: bool = False) -> List[Sighting]:
        """Sightings with ts in [start_ts, end_ts], oldest first."""
        q = ("SELECT * FROM police_sightings WHERE ts >= ? AND ts <= ?")
        if police_only:
            q += " AND is_police = 1"
        q += " ORDER BY ts ASC"
        with self._lock:
            rows = self._conn.execute(q, (start_ts, end_ts)).fetchall()
        return [_row_to_sighting(r) for r in rows]

    def count(self, start_ts: float, end_ts: float,
              police_only: bool = False) -> int:
        return len(self.in_window(start_ts, end_ts, police_only))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "PoliceLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _row_to_sighting(r: sqlite3.Row) -> Sighting:
    return Sighting(
        ts=r["ts"], first_ts=r["first_ts"], track_id=r["track_id"],
        direction=r["direction"], confidence=r["confidence"],
        is_police=bool(r["is_police"]), max_speed_rel=r["max_speed_rel"],
        max_speed_kmh=r["max_speed_kmh"], was_speeding=bool(r["was_speeding"]),
    )
