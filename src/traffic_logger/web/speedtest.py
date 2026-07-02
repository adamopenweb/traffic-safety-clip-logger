"""Speed-test helper: validate measured km/h against GPS drive-bys.

The analyzer already records *every* vehicle pass -- including the slow ones below
the 55 km/h clip gate -- into the unified store's ``passes`` table (see
:class:`~..analyze.pass_recorder.PassRecorder`). So a calibration drive-by at, say,
37 km/h is already captured; we don't need a special "test mode" or any analyzer
change. This module just:

* :func:`recent_passes` -- the last few minutes of passes, for the pick-list. The
  user drives by, reads the live PC clock the page shows, and taps the matching row.
* :class:`SpeedTestLog` -- a file-backed log of labelled passes. Picking a row and
  entering the true GPS speed snapshots ``(ts, measured, direction, vehicle_type,
  true_speed, error_pct)`` into one self-contained record, so the log stands alone
  even if the store is pruned. Mirrors the atomic-write pattern of ``Exclusions``.

Read-only against the analyzer; the page is served behind the existing auth gate.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def recent_passes(db_path: str, *, since_ts: float, limit: int = 60) -> List[Dict[str, Any]]:
    """Recent passes (newest first) for the pick-list: ts, measured km/h, dir, type."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.OperationalError:
        return []
    try:
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT session_id, track_id, last_ts, steady_speed_raw, direction, vehicle_type "
                "FROM passes WHERE last_ts >= ? AND steady_valid = 1 "
                "ORDER BY last_ts DESC LIMIT ?",
                (since_ts, int(limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # table not created yet
    finally:
        con.close()
    return [
        {
            "key": f"{r['session_id']}:{r['track_id']}",
            "ts": r["last_ts"],
            "measured": round(r["steady_speed_raw"], 1),
            "direction": r["direction"],
            "vehicle_type": r["vehicle_type"],
        }
        for r in rows
    ]


def _error_pct(measured: float, true_speed: float) -> Optional[float]:
    if not true_speed:
        return None
    return round((measured - true_speed) / true_speed * 100.0, 1)


class SpeedTestLog:
    """Thread-safe, file-backed log of labelled speed-test passes (newest first)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rows: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._rows, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._rows)

    def add(self, *, key: str, ts: float, measured: float, true_speed: float,
            direction: Optional[str], vehicle_type: Optional[str],
            note: str = "") -> Dict[str, Any]:
        """Label a pass with its true GPS speed. Re-labelling the same pass replaces it."""
        rec = {
            "id": key,
            "ts": float(ts),
            "measured": round(float(measured), 1),
            "true_speed": round(float(true_speed), 1),
            "error_pct": _error_pct(float(measured), float(true_speed)),
            "direction": direction,
            "vehicle_type": vehicle_type,
            "note": note,
            "created_ts": time.time(),
        }
        with self._lock:
            self._rows = [r for r in self._rows if r.get("id") != key]
            self._rows.insert(0, rec)
            self._save()
        return rec

    def remove(self, key: str) -> bool:
        with self._lock:
            before = len(self._rows)
            self._rows = [r for r in self._rows if r.get("id") != key]
            if len(self._rows) == before:
                return False
            self._save()
            return True
