"""Segment index (SQLite).

Records one row per recorded ring-buffer segment so the ring pruner and (later)
the event exporter can query what footage exists and when. Schema matches the
spec's required fields: path, start/end timestamp, duration, file size, codec,
resolution, fps.

The index only ever contains *completed* segments; the segment currently being
written lives in the recorder's incoming directory and is added once finalized.
That keeps the pruner from ever touching the active segment.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..util.paths import ensure_dir


@dataclass(frozen=True)
class SegmentRecord:
    """One recorded segment.

    Attribute names ``path`` / ``start_ts`` / ``size_bytes`` are intentionally
    compatible with :func:`ring_pruner.select_segments_to_delete`.
    """

    path: str
    start_ts: float
    end_ts: float
    duration: float
    size_bytes: int
    codec: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    path       TEXT PRIMARY KEY,
    start_ts   REAL NOT NULL,
    end_ts     REAL NOT NULL,
    duration   REAL NOT NULL,
    size_bytes INTEGER NOT NULL,
    codec      TEXT,
    width      INTEGER,
    height     INTEGER,
    fps        REAL
);
CREATE INDEX IF NOT EXISTS idx_segments_start_ts ON segments(start_ts);
"""

_COLUMNS = (
    "path", "start_ts", "end_ts", "duration",
    "size_bytes", "codec", "width", "height", "fps",
)


class SegmentIndex:
    """SQLite-backed segment index.

    Pass ``":memory:"`` for an ephemeral index (tests) or a file path for the
    persistent on-disk index (parent directories are created automatically).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            ensure_dir(Path(self.db_path).parent)
        # check_same_thread=False so the recorder's indexer loop and an
        # occasional prune can share one connection; guarded by a lock.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- writes ------------------------------------------------------------
    def add_segment(self, record: SegmentRecord) -> None:
        """Insert (or replace) a segment row, keyed by path."""
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO segments ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
                (
                    record.path, record.start_ts, record.end_ts, record.duration,
                    record.size_bytes, record.codec, record.width,
                    record.height, record.fps,
                ),
            )
            self._conn.commit()

    def delete_segment(self, path: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM segments WHERE path = ?", (path,))
            self._conn.commit()

    # -- reads -------------------------------------------------------------
    def get_segment(self, path: str) -> Optional[SegmentRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM segments WHERE path = ?", (path,)
            ).fetchone()
        return _row_to_record(row) if row else None

    def get_all(self) -> List[SegmentRecord]:
        """All segments, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM segments ORDER BY start_ts ASC"
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_overlapping(self, start_ts: float, end_ts: float) -> List[SegmentRecord]:
        """Segments overlapping the [start_ts, end_ts] window (for clip export)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM segments WHERE start_ts <= ? AND end_ts >= ? "
                "ORDER BY start_ts ASC",
                (end_ts, start_ts),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def has(self, path: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM segments WHERE path = ? LIMIT 1", (path,)
            ).fetchone()
        return row is not None

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM segments"
            ).fetchone()[0])

    def latest_end_ts(self) -> Optional[float]:
        """End timestamp of the most recent segment, or None if empty."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(end_ts) FROM segments"
            ).fetchone()
        return row[0] if row and row[0] is not None else None

    def total_bytes(self) -> int:
        with self._lock:
            value = self._conn.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM segments"
            ).fetchone()[0]
        return int(value)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SegmentIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _row_to_record(row: sqlite3.Row) -> SegmentRecord:
    return SegmentRecord(
        path=row["path"],
        start_ts=row["start_ts"],
        end_ts=row["end_ts"],
        duration=row["duration"],
        size_bytes=row["size_bytes"],
        codec=row["codec"] or "",
        width=row["width"] or 0,
        height=row["height"] or 0,
        fps=row["fps"] or 0.0,
    )
