"""Unified traffic store (SQLite) -- Phase 0 of the unified-data-store plan.

One relational store built around a stable **pass** (a single vehicle drive-by)
instead of the three disconnected logs we have today (per-event JSON sidecars,
``speed_log.sqlite``, ``police_sightings.sqlite``). A pass is identified by
``(session_id, track_id)`` -- ``track_id`` alone resets every run, so the session
id is what makes a drive-by addressable.

The crux is :meth:`TrafficStore.upsert_pass`: the inline speeding writer (main
thread) and the async police writer (worker thread) both touch the *same* pass row
via ``INSERT ... ON CONFLICT(session_id, track_id) DO UPDATE``. The merge is
order-independent -- whichever writer fires first creates the row, the other fills
in its columns -- so neither needs to know about the other or wait for it. NULL
columns from one writer never clobber values another writer already set.

This module is PHASE 0: additive only, nothing is wired into the live loop yet.
It is dependency-free (stdlib ``sqlite3``) so it imports and unit-tests without the
CV/torch stack, and thread-safe (one locked connection, ``check_same_thread=False``)
like ``police_log.py`` -- the police worker and the main loop will share it.

``plate`` / ``vehicle_id`` columns and the ``vehicles`` table are reserved now (all
nullable) so future license-plate / cross-session identity work is a write path, not
a migration. The ``speed_events`` / ``police_sightings`` views emulate the legacy log
schemas so Phase 2 can repoint the reports at this store without changing their SQL.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..util.paths import ensure_dir


# --- records ----------------------------------------------------------------

@dataclass(frozen=True)
class PassRecord:
    """One vehicle drive-by (a completed track). Most fields are optional because
    different writers know different things: the speeding writer sets speed +
    ``was_speeding``; the police writer sets ``is_police`` + confidence. ``upsert_pass``
    merges them onto one row without either clobbering the other's columns."""

    session_id: str
    track_id: int
    first_ts: float
    last_ts: float
    direction: Optional[str] = None
    vehicle_type: Optional[str] = None         # car/truck/bus/motorcycle
    max_speed_kmh: Optional[float] = None
    steady_speed_kmh: Optional[float] = None   # source-guarded (None on span reject) -- legacy
    was_speeding: Optional[bool] = None
    is_police: Optional[bool] = None
    police_confidence: Optional[float] = None
    plate: Optional[str] = None                # future (ALPR), nullable
    vehicle_id: Optional[int] = None           # future cross-session identity, nullable
    # Phase 3 (validated-view): the raw measured speed (un-span-gated, preserved for
    # forensics/re-tuning), the geometry the invariants need, and the derived validity.
    steady_speed_raw: Optional[float] = None
    ground_span: Optional[float] = None
    n_points: Optional[int] = None
    steady_valid: Optional[bool] = None
    steady_invalid_reason: Optional[str] = None


@dataclass(frozen=True)
class EventRecord:
    """One unsafe-driving event (a clip), referencing the pass that triggered it."""

    event_id: str
    session_id: str
    event_type: str
    event_types: List[str]
    trigger_ts: float
    score: float
    primary_track_id: Optional[int]
    clipped: bool
    clip_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_final_event(
        cls,
        fe,
        session_id: str,
        *,
        clipped: bool,
        clip_path: Optional[str] = None,
        thumbnail_path: Optional[str] = None,
    ) -> "EventRecord":
        """Build from a :class:`~..events.manager.FinalEvent`, flattening its
        candidates into the same evidence-trigger list the JSON sidecar carries."""
        evidence = [
            {
                "event_type": c.event_type,
                "trigger_ts": round(c.trigger_ts, 3),
                "score": c.score,
                "evidence": c.evidence,
            }
            for c in getattr(fe, "candidates", []) or []
        ]
        return cls(
            event_id=fe.event_id,
            session_id=session_id,
            event_type=fe.event_type,
            event_types=list(fe.event_types),
            trigger_ts=fe.trigger_ts,
            score=fe.score,
            primary_track_id=fe.primary_track_id,
            clipped=clipped,
            clip_path=clip_path,
            thumbnail_path=thumbnail_path,
            evidence=evidence,
        )


# --- schema -----------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    started_at  REAL NOT NULL,
    camera_id   TEXT,
    config_hash TEXT
);

CREATE TABLE IF NOT EXISTS passes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL,
    track_id          INTEGER NOT NULL,
    first_ts          REAL NOT NULL,
    last_ts           REAL NOT NULL,
    direction         TEXT,
    vehicle_type      TEXT,
    max_speed_kmh     REAL,
    steady_speed_kmh  REAL,
    was_speeding      INTEGER,
    is_police         INTEGER,
    police_confidence REAL,
    plate             TEXT,
    vehicle_id        INTEGER,
    steady_speed_raw      REAL,
    ground_span           REAL,
    n_points              INTEGER,
    steady_valid          INTEGER,
    steady_invalid_reason TEXT,
    UNIQUE(session_id, track_id)
);
CREATE INDEX IF NOT EXISTS idx_passes_last_ts ON passes(last_ts);
CREATE INDEX IF NOT EXISTS idx_passes_vehicle_type ON passes(vehicle_type);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL UNIQUE,
    session_id      TEXT NOT NULL,
    primary_pass_id INTEGER REFERENCES passes(id),
    event_type      TEXT NOT NULL,
    event_types     TEXT,
    trigger_ts      REAL NOT NULL,
    score           REAL,
    clipped         INTEGER NOT NULL,
    clip_path       TEXT,
    thumbnail_path  TEXT,
    evidence_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_trigger_ts ON events(trigger_ts);

CREATE TABLE IF NOT EXISTS vehicles (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    plate      TEXT UNIQUE,
    first_seen REAL,
    last_seen  REAL
);

-- Legacy-schema compatibility views (Phase 2 repoints the reports at these).
CREATE VIEW IF NOT EXISTS police_sightings AS
    SELECT last_ts AS ts, first_ts, track_id, direction,
           police_confidence AS confidence, is_police,
           NULL AS max_speed_rel, max_speed_kmh,
           COALESCE(was_speeding, 0) AS was_speeding
    FROM passes WHERE is_police = 1;

CREATE VIEW IF NOT EXISTS speed_events AS
    SELECT e.trigger_ts AS ts, p.max_speed_kmh AS speed_kmh, p.direction AS direction,
           e.clipped AS clipped, p.vehicle_type AS vehicle_type
    FROM events e JOIN passes p ON e.primary_pass_id = p.id
    WHERE p.was_speeding = 1;
"""


def _b(x: Optional[bool]) -> Optional[int]:
    return None if x is None else int(bool(x))


# Columns merged on conflict, with the merge rule for each.
#   MIN  -> earliest wins (first_ts)
#   MAX  -> latest wins (last_ts)
#   MAXNN-> larger non-NULL wins; once set/true it stays (speeds, was_speeding, is_police)
#   COAL -> excluded wins when present, else keep existing
def _maxnn(col: str) -> str:
    # MAX of the two values, ignoring NULLs (so a NULL from one writer can't erase
    # a value the other writer set).
    return (f"{col} = MAX(COALESCE(excluded.{col}, passes.{col}), "
            f"COALESCE(passes.{col}, excluded.{col}))")


_MERGE_SET = ",\n    ".join([
    "first_ts = MIN(passes.first_ts, excluded.first_ts)",
    "last_ts = MAX(passes.last_ts, excluded.last_ts)",
    "direction = COALESCE(excluded.direction, passes.direction)",
    "vehicle_type = COALESCE(excluded.vehicle_type, passes.vehicle_type)",
    _maxnn("max_speed_kmh"),
    _maxnn("steady_speed_kmh"),
    _maxnn("was_speeding"),
    _maxnn("is_police"),
    "police_confidence = COALESCE(excluded.police_confidence, passes.police_confidence)",
    "plate = COALESCE(excluded.plate, passes.plate)",
    "vehicle_id = COALESCE(excluded.vehicle_id, passes.vehicle_id)",
    # Speed measurement + validity: only the speeding writer sets these, so excluded wins
    # when present (the police writer passes NULLs, which must not clobber them).
    "steady_speed_raw = COALESCE(excluded.steady_speed_raw, passes.steady_speed_raw)",
    "ground_span = COALESCE(excluded.ground_span, passes.ground_span)",
    "n_points = COALESCE(excluded.n_points, passes.n_points)",
    "steady_valid = COALESCE(excluded.steady_valid, passes.steady_valid)",
    "steady_invalid_reason = COALESCE(excluded.steady_invalid_reason, passes.steady_invalid_reason)",
])

_UPSERT_PASS = f"""
INSERT INTO passes (session_id, track_id, first_ts, last_ts, direction, vehicle_type,
                    max_speed_kmh, steady_speed_kmh, was_speeding, is_police,
                    police_confidence, plate, vehicle_id,
                    steady_speed_raw, ground_span, n_points, steady_valid,
                    steady_invalid_reason)
VALUES (:session_id, :track_id, :first_ts, :last_ts, :direction, :vehicle_type,
        :max_speed_kmh, :steady_speed_kmh, :was_speeding, :is_police,
        :police_confidence, :plate, :vehicle_id,
        :steady_speed_raw, :ground_span, :n_points, :steady_valid,
        :steady_invalid_reason)
ON CONFLICT(session_id, track_id) DO UPDATE SET
    {_MERGE_SET}
"""


# --- store ------------------------------------------------------------------

class TrafficStore:
    """SQLite-backed unified store. ``":memory:"`` for tests, a path on disk
    otherwise (parent dirs created). Thread-safe via one locked connection."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            ensure_dir(Path(self.db_path).parent)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    # Columns added after the passes table shipped; CREATE TABLE IF NOT EXISTS won't add
    # them to an existing DB, so ALTER any that are missing (idempotent, additive).
    _ADDED_COLUMNS = (
        ("steady_speed_raw", "REAL"),
        ("ground_span", "REAL"),
        ("n_points", "INTEGER"),
        ("steady_valid", "INTEGER"),
        ("steady_invalid_reason", "TEXT"),
    )

    def _migrate(self) -> None:
        have = {row["name"] for row in self._conn.execute("PRAGMA table_info(passes)")}
        for col, decl in self._ADDED_COLUMNS:
            if col not in have:
                self._conn.execute(f"ALTER TABLE passes ADD COLUMN {col} {decl}")

    # -- writers --

    def start_session(self, session_id: str, started_at: float,
                      camera_id: Optional[str] = None,
                      config_hash: Optional[str] = None) -> None:
        """Register a run. Idempotent -- re-registering the same id is a no-op."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, started_at, camera_id, "
                "config_hash) VALUES (?, ?, ?, ?)",
                (session_id, started_at, camera_id, config_hash),
            )
            self._conn.commit()

    def upsert_pass(self, rec: PassRecord) -> int:
        """Create or merge the pass row for ``(session_id, track_id)``; return its id.

        Order-independent: NULL fields from this writer keep whatever another writer
        already set; speeds/flags take the larger non-NULL; first/last ts widen the
        window. This is what lets the speeding and police writers share one row."""
        params = {
            "session_id": rec.session_id, "track_id": rec.track_id,
            "first_ts": rec.first_ts, "last_ts": rec.last_ts,
            "direction": rec.direction, "vehicle_type": rec.vehicle_type,
            "max_speed_kmh": rec.max_speed_kmh, "steady_speed_kmh": rec.steady_speed_kmh,
            "was_speeding": _b(rec.was_speeding), "is_police": _b(rec.is_police),
            "police_confidence": rec.police_confidence, "plate": rec.plate,
            "vehicle_id": rec.vehicle_id,
            "steady_speed_raw": rec.steady_speed_raw, "ground_span": rec.ground_span,
            "n_points": rec.n_points, "steady_valid": _b(rec.steady_valid),
            "steady_invalid_reason": rec.steady_invalid_reason,
        }
        with self._lock:
            self._conn.execute(_UPSERT_PASS, params)
            row = self._conn.execute(
                "SELECT id FROM passes WHERE session_id = ? AND track_id = ?",
                (rec.session_id, rec.track_id),
            ).fetchone()
            self._conn.commit()
        return int(row["id"])

    def add_event(self, rec: EventRecord) -> Optional[int]:
        """Insert an event, linking it to its primary pass when one exists. Re-adding
        the same ``event_id`` is ignored. Returns the row id (None if ignored)."""
        with self._lock:
            pass_id = None
            if rec.primary_track_id is not None:
                row = self._conn.execute(
                    "SELECT id FROM passes WHERE session_id = ? AND track_id = ?",
                    (rec.session_id, rec.primary_track_id),
                ).fetchone()
                pass_id = int(row["id"]) if row else None
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO events (event_id, session_id, primary_pass_id, "
                "event_type, event_types, trigger_ts, score, clipped, clip_path, "
                "thumbnail_path, evidence_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rec.event_id, rec.session_id, pass_id, rec.event_type,
                 json.dumps(rec.event_types), rec.trigger_ts, rec.score,
                 int(rec.clipped), rec.clip_path, rec.thumbnail_path,
                 json.dumps(rec.evidence)),
            )
            self._conn.commit()
            return cur.lastrowid if cur.rowcount else None

    # -- readers (for tests + Phase 2 reports) --

    def get_pass(self, session_id: str, track_id: int) -> Optional[PassRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM passes WHERE session_id = ? AND track_id = ?",
                (session_id, track_id),
            ).fetchone()
        return _row_to_pass(row) if row else None

    def passes_in_window(self, start_ts: float, end_ts: float) -> List[PassRecord]:
        """Passes whose last_ts falls in [start_ts, end_ts], oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM passes WHERE last_ts >= ? AND last_ts <= ? "
                "ORDER BY last_ts ASC", (start_ts, end_ts),
            ).fetchall()
        return [_row_to_pass(r) for r in rows]

    def get_pass_steady(self, session_id: str, track_id: int) -> Optional[float]:
        """The GPS-validated full-track steady km/h recorded for a completed pass,
        or None if the pass isn't finalized yet / the track was filtered out. Used to
        relabel live speeding events (whose trigger measures a noisy PARTIAL track that
        reads systematically high) with the same metric the speed-test page validated."""
        with self._lock:
            row = self._conn.execute(
                "SELECT steady_speed_raw FROM passes "
                "WHERE session_id=? AND track_id=? AND steady_valid=1",
                (session_id, int(track_id)),
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["event_types"] = json.loads(d["event_types"]) if d["event_types"] else []
        d["evidence"] = json.loads(d["evidence_json"]) if d["evidence_json"] else []
        return d

    def query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Read-only escape hatch for the compat views / reports.

        Enforces SELECT/WITH so it can't be used to mutate the store by accident --
        it's a query helper, not a write path (use the typed ``upsert_*`` methods for
        writes). ``sqlite3.execute`` already rejects multiple statements, so a single
        leading-keyword check is sufficient."""
        head = sql.lstrip().lstrip("(").lstrip()[:6].upper()
        if not (head.startswith("SELECT") or head.startswith("WITH")):
            raise ValueError("TrafficStore.query is read-only (SELECT/WITH only)")
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "TrafficStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _row_to_pass(r: sqlite3.Row) -> PassRecord:
    def _ob(x):  # int-or-None -> bool-or-None
        return None if x is None else bool(x)
    return PassRecord(
        session_id=r["session_id"], track_id=r["track_id"],
        first_ts=r["first_ts"], last_ts=r["last_ts"], direction=r["direction"],
        vehicle_type=r["vehicle_type"], max_speed_kmh=r["max_speed_kmh"],
        steady_speed_kmh=r["steady_speed_kmh"], was_speeding=_ob(r["was_speeding"]),
        is_police=_ob(r["is_police"]), police_confidence=r["police_confidence"],
        plate=r["plate"], vehicle_id=r["vehicle_id"],
        steady_speed_raw=r["steady_speed_raw"], ground_span=r["ground_span"],
        n_points=r["n_points"], steady_valid=_ob(r["steady_valid"]),
        steady_invalid_reason=r["steady_invalid_reason"],
    )
