"""Speeding-report aggregation (advocacy / evidence summaries).

Reads the absolute-speed-gate events the analyzer wrote (one metadata JSON per
flagged driver) and rolls them up into the numbers that make a community-safety
case: how many violations, how fast, when, and the top recorded speeds. Pure logic
here (parse a metadata dict -> Violation; aggregate -> Stats) so it is unit-
testable without touching the filesystem; the CLI handler does the I/O + print.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Violation:
    ts: float                      # trigger_ts (unix) -- when the driver passed
    speed_kmh: float
    over_limit_kmh: float          # speed - posted limit
    direction: Optional[str]
    clipped: bool                  # whether a video clip was kept for this one
    vehicle_type: Optional[str] = None   # car / truck / bus / motorcycle


def violation_from_record(rec, limit_kmh: float) -> Violation:
    """Build a Violation from a SpeedLog :class:`SpeedRecord` (the primary source)."""
    return Violation(
        ts=float(rec.ts), speed_kmh=float(rec.speed_kmh),
        over_limit_kmh=round(float(rec.speed_kmh) - limit_kmh, 1),
        direction=rec.direction, clipped=bool(rec.clipped),
        vehicle_type=getattr(rec, "vehicle_type", None),
    )


def violation_from_metadata(meta: dict, limit_kmh: float) -> Optional[Violation]:
    """Extract a violation from a clipped event's metadata dict (fallback source).

    Returns None for events that weren't produced by the absolute gate (e.g.
    legacy relative-percentile events).
    """
    triggers = (meta.get("evidence") or {}).get("triggers") or []
    best = None
    for t in triggers:
        ev = t.get("evidence") or {}
        if ev.get("rule") == "absolute_speeding" and ev.get("speed_kmh") is not None:
            if best is None or float(ev["speed_kmh"]) > float(best["speed_kmh"]):
                best = ev
    if best is None:
        return None
    speed = float(best["speed_kmh"])
    return Violation(
        ts=float(meta.get("trigger_ts", 0.0)),
        speed_kmh=speed,
        over_limit_kmh=round(speed - limit_kmh, 1),
        direction=best.get("direction"),
        clipped=bool(meta.get("clip_path")),
        vehicle_type=best.get("vehicle_type"),
    )


@dataclass
class Stats:
    count: int = 0
    span_days: float = 0.0
    per_day: float = 0.0
    max_kmh: float = 0.0
    mean_kmh: float = 0.0
    median_kmh: float = 0.0
    by_speed_bin: List = field(default_factory=list)     # (label, count)
    by_hour: List = field(default_factory=list)          # (hour, count) 0..23
    by_direction: List = field(default_factory=list)     # (direction, count)
    by_vehicle_type: List = field(default_factory=list)  # (type, count)
    worst: List[Violation] = field(default_factory=list)


# Speed bins (lower-inclusive) for the distribution, e.g. 55-59, 60-64, ...
_DEFAULT_BIN_EDGES = (55, 60, 65, 70, 80)


def aggregate(violations: Sequence[Violation], timezone: str, *,
              top: int = 10, bin_edges: Sequence[int] = _DEFAULT_BIN_EDGES) -> Stats:
    """Roll a set of violations into report :class:`Stats` (local-time grouped)."""
    vs = sorted(violations, key=lambda v: v.ts)
    st = Stats(count=len(vs))
    if not vs:
        return st
    tz = ZoneInfo(timezone)
    speeds = sorted(v.speed_kmh for v in vs)
    st.max_kmh = speeds[-1]
    st.mean_kmh = round(sum(speeds) / len(speeds), 1)
    st.median_kmh = speeds[len(speeds) // 2]
    st.span_days = max((vs[-1].ts - vs[0].ts) / 86400.0, 0.0)
    st.per_day = round(st.count / st.span_days, 1) if st.span_days >= 1.0 else float(st.count)

    edges = list(bin_edges)
    labels = [f"{edges[i]}-{edges[i + 1] - 1}" for i in range(len(edges) - 1)] + [f"{edges[-1]}+"]
    bin_counts = [0] * len(labels)
    for v in vs:
        idx = len(edges) - 1
        for i in range(len(edges) - 1):
            if v.speed_kmh < edges[i + 1]:
                idx = i
                break
        bin_counts[idx] += 1
    st.by_speed_bin = list(zip(labels, bin_counts))

    hours = Counter(datetime.fromtimestamp(v.ts, tz).hour for v in vs)
    st.by_hour = [(h, hours.get(h, 0)) for h in range(24) if hours.get(h, 0)]
    dirs = Counter(v.direction or "unknown" for v in vs)
    st.by_direction = sorted(dirs.items(), key=lambda kv: -kv[1])
    types = Counter(v.vehicle_type or "unknown" for v in vs)
    st.by_vehicle_type = sorted(types.items(), key=lambda kv: -kv[1])
    st.worst = sorted(vs, key=lambda v: -v.speed_kmh)[:top]
    return st
