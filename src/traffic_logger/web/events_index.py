"""Index of saved event clips for the dashboard.

The on-disk layout ``data/events/<date>/<event_type>/<stem>.{json,jpg,mp4}`` is the
source of truth -- no separate DB to drift. Each event has a JSON sidecar with the
authoritative facts (event_id, trigger_ts, speed/direction/vehicle from the rule
evidence); sibling files give the thumbnail and the clean/annotated videos.

:class:`EventsIndex` scans that tree, caches the result for ``ttl`` seconds (the
folder is small and bounded by the pruner, so a full re-scan is cheap), and answers
three things the UI needs: a filtered list, the "latest fast clips" for the Now page,
and event_id -> file-path resolution for the media routes (which is also what keeps
playback from being a path-traversal vector -- callers pass an id, never a path).

The video resolver prefers ``<stem>_annotated.mp4`` over the clean ``<stem>.mp4`` --
the annotated clip is the one worth showing when it exists.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MediaPaths:
    clip: Optional[Path] = None        # clean <stem>.mp4
    annotated: Optional[Path] = None   # <stem>_annotated.mp4
    thumb: Optional[Path] = None       # <stem>.jpg

    def best_video(self, prefer_annotated: bool = True) -> Optional[Path]:
        if prefer_annotated and self.annotated:
            return self.annotated
        return self.clip or self.annotated


def speed_dir_type(meta: dict) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Pull (max speed_kmh, direction, vehicle_type) out of a sidecar's rule
    evidence -- mirrors how the filename descriptor is built, so list cards and
    filenames agree. Tolerant of older/partial sidecars."""
    speed = direction = vtype = None
    triggers = (meta.get("evidence", {}) or {}).get("triggers", []) or []
    for t in triggers:
        ev = (t or {}).get("evidence", {}) or {}
        s = ev.get("speed_kmh")
        if s is not None:
            speed = float(s) if speed is None else max(speed, float(s))
        if direction is None and ev.get("direction"):
            direction = ev["direction"]
        if vtype is None and ev.get("vehicle_type"):
            vtype = ev["vehicle_type"]
    # fall back to per-track summaries for direction
    if direction is None:
        for tr in meta.get("tracks", []) or []:
            if tr.get("direction"):
                direction = tr["direction"]
                break
    return speed, direction, vtype


def _summarize(meta: dict, stem: str, date: str, paths: MediaPaths,
               tz: ZoneInfo) -> dict:
    """Pure: build the JSON summary the API returns for one event."""
    speed, direction, vtype = speed_dir_type(meta)
    ts = float(meta.get("trigger_ts") or 0.0)
    annotated = paths.annotated is not None
    has_video = paths.best_video() is not None
    eid = meta.get("event_id") or stem
    return {
        "id": eid,
        "stem": stem,
        "date": date,
        "event_type": meta.get("event_type", "event"),
        "trigger_ts": ts,
        "iso": datetime.fromtimestamp(ts, tz).isoformat() if ts else None,
        "speed_kmh": round(speed, 1) if speed is not None else None,
        "direction": direction,
        "vehicle_type": vtype,
        "media": meta.get("media", "clip"),
        "has_video": has_video,
        "annotated": annotated,
        "clip_url": f"/media/clip/{eid}" if has_video else None,
        "thumb_url": f"/media/thumb/{eid}" if paths.thumb else None,
    }


def scan_events(events_dir: Path, tz: ZoneInfo) -> Tuple[List[dict], Dict[str, MediaPaths]]:
    """Walk the events tree once. Returns (summaries newest-first, id->paths).

    Reads ``<stem>.json`` sidecars (skipping ``_overlay.json``) and probes for the
    sibling clip/annotated/thumb files."""
    summaries: List[dict] = []
    paths_by_id: Dict[str, MediaPaths] = {}
    if not events_dir.exists():
        return summaries, paths_by_id
    for sidecar in events_dir.glob("*/*/*.json"):
        if sidecar.name.endswith("_overlay.json"):
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        stem = sidecar.stem
        d = sidecar.parent
        clip = d / f"{stem}.mp4"
        annotated = d / f"{stem}_annotated.mp4"
        thumb = d / f"{stem}.jpg"
        paths = MediaPaths(
            clip=clip if clip.exists() else None,
            annotated=annotated if annotated.exists() else None,
            thumb=thumb if thumb.exists() else None,
        )
        # date folder is the grandparent (events/<date>/<type>/file)
        date = d.parent.name
        summary = _summarize(meta, stem, date, paths, tz)
        summaries.append(summary)
        paths_by_id[summary["id"]] = paths
    summaries.sort(key=lambda s: s["trigger_ts"], reverse=True)
    return summaries, paths_by_id


class EventsIndex:
    """TTL-cached view over the events folder. Thread-safe (sync routes run in a
    threadpool, so concurrent reads are possible)."""

    def __init__(self, events_dir: Path, tz: ZoneInfo, ttl: float = 20.0) -> None:
        self.events_dir = Path(events_dir)
        self.tz = tz
        self.ttl = ttl
        self._lock = threading.Lock()
        self._cached_at = 0.0
        self._summaries: List[dict] = []
        self._paths: Dict[str, MediaPaths] = {}

    def _ensure_fresh(self, *, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            if now - self._cached_at <= self.ttl and self._cached_at:
                return
            self._summaries, self._paths = scan_events(self.events_dir, self.tz)
            self._cached_at = now

    def all(self) -> List[dict]:
        self._ensure_fresh()
        return self._summaries

    def get_paths(self, event_id: str) -> Optional[MediaPaths]:
        self._ensure_fresh()
        return self._paths.get(event_id)

    def query(self, *, days: Optional[int] = None, min_speed: Optional[float] = None,
              event_type: Optional[str] = None, since_ts: Optional[float] = None,
              until_ts: Optional[float] = None, limit: int = 100, offset: int = 0) -> List[dict]:
        items = self.all()
        if since_ts is not None:
            items = [e for e in items if e["trigger_ts"] >= since_ts]
        if until_ts is not None:
            items = [e for e in items if e["trigger_ts"] < until_ts]
        if min_speed is not None:
            items = [e for e in items
                     if e["speed_kmh"] is not None and e["speed_kmh"] >= min_speed]
        if event_type:
            items = [e for e in items if e["event_type"] == event_type]
        return items[offset:offset + limit]

    def latest_fast(self, threshold: float, limit: int = 12) -> List[dict]:
        """Newest events that have a playable video and cleared ``threshold`` km/h --
        the Now page's headline reel."""
        out = []
        for e in self.all():
            if e["has_video"] and e["speed_kmh"] is not None and e["speed_kmh"] >= threshold:
                out.append(e)
                if len(out) >= limit:
                    break
        return out

    def top_speeders(self, threshold: float, limit: int = 50) -> List[dict]:
        """Events with a playable video at/above ``threshold`` km/h, fastest first --
        the Top Speeds page. All-time (no window): the fastest passes on record."""
        out = [e for e in self.all()
               if e["has_video"] and e["speed_kmh"] is not None and e["speed_kmh"] >= threshold]
        out.sort(key=lambda e: e["speed_kmh"], reverse=True)
        return out[:limit]
