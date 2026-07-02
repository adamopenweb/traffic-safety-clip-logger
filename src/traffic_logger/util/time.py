"""Time helpers.

Centralizes timestamp formatting so segment filenames, event filenames, and
metadata all agree on representation. Unix milliseconds are the canonical
internal timestamp (matches the segment filename format in the spec:
``segment_<start_unix_ms>.mp4``).
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timezone
from typing import Optional

try:  # Python 3.9+ stdlib; available on the 3.11+ target.
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - defensive only
    ZoneInfo = None  # type: ignore[assignment]


def now_unix() -> float:
    """Current time as Unix seconds (float)."""
    return _time.time()


def now_unix_ms() -> int:
    """Current time as Unix milliseconds (int)."""
    return int(_time.time() * 1000)


def unix_ms_to_dt(unix_ms: int, tz: Optional[str] = None) -> datetime:
    """Convert Unix milliseconds to a timezone-aware datetime.

    Defaults to UTC; pass an IANA tz name (e.g. ``"America/Toronto"``) to
    localize. Falls back to UTC if zoneinfo data is unavailable.
    """
    dt = datetime.fromtimestamp(unix_ms / 1000.0, tz=timezone.utc)
    return _localize(dt, tz)


def format_segment_stamp(unix_ms: Optional[int] = None, tz: Optional[str] = None) -> str:
    """Format a timestamp as ``YYYYMMDD_HHMMSS`` (used in event filenames)."""
    if unix_ms is None:
        unix_ms = now_unix_ms()
    return unix_ms_to_dt(unix_ms, tz).strftime("%Y%m%d_%H%M%S")


def format_date_dir(unix_ms: Optional[int] = None, tz: Optional[str] = None) -> str:
    """Format the per-day folder name ``YYYY-MM-DD`` used under ring/events."""
    if unix_ms is None:
        unix_ms = now_unix_ms()
    return unix_ms_to_dt(unix_ms, tz).strftime("%Y-%m-%d")


def iso_now(tz: Optional[str] = None) -> str:
    """ISO-8601 timestamp for `created_at` metadata, localized if tz given."""
    dt = datetime.now(timezone.utc)
    return _localize(dt, tz).isoformat()


def _localize(dt: datetime, tz: Optional[str]) -> datetime:
    if not tz or ZoneInfo is None:
        return dt
    try:
        return dt.astimezone(ZoneInfo(tz))
    except Exception:
        # Unknown tz name or missing tzdata — keep UTC rather than crash.
        return dt
