"""Capture health check.

A simple liveness signal for the unattended recording appliance: recording is
healthy when a fresh segment has been written to the ring within the expected
window. Used by ``traffic-log health`` and the compose healthcheck so a stuck
recorder is restarted (spec Milestone 7 "health checks").
"""

from __future__ import annotations

from typing import Optional, Tuple


def max_segment_age(segment_seconds: float) -> float:
    """Allowed staleness before recording is considered unhealthy.

    Three segment durations plus a 30s slack, floored at 60s — generous enough
    to tolerate one slow/again-being-written segment without false alarms.
    """
    return max(60.0, segment_seconds * 3 + 30.0)


def recording_health(
    latest_end_ts: Optional[float], now: float, max_age_seconds: float
) -> Tuple[bool, str]:
    """Decide whether recording looks alive.

    ``latest_end_ts`` is the end timestamp of the newest indexed segment (None
    if the index is empty). Returns (healthy, reason).
    """
    if latest_end_ts is None:
        return (False, "no segments recorded yet")
    age = now - latest_end_ts
    if age > max_age_seconds:
        return (False, f"newest segment is {age:.0f}s old (> {max_age_seconds:.0f}s)")
    return (True, f"newest segment {age:.0f}s old")
