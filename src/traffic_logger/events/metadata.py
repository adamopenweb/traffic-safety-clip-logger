"""Event metadata sidecar.

Builds and writes the per-event JSON sidecar described in the spec's "Metadata
Schema": identity, timestamps, clip/thumbnail paths, score, primary track,
per-track summaries, evidence (all triggers), and a config snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _track_summaries(candidates) -> list:
    """Best-effort per-track summaries assembled from candidate evidence."""
    tracks: Dict[int, Dict[str, Any]] = {}
    for cand in candidates:
        ev = cand.evidence or {}
        tid = ev.get("track_id", ev.get("candidate_track_id", cand.primary_track_id))
        if tid is None:
            continue
        entry = tracks.setdefault(tid, {"track_id": tid})
        if ev.get("direction"):
            entry["direction"] = ev["direction"]
        if ev.get("lane_sequence"):
            entry["lane_band_sequence"] = ev["lane_sequence"]
        speed: Dict[str, Any] = {}
        if "speed" in ev:
            speed["value"] = ev["speed"]
        pct = ev.get("percentile", ev.get("speed_percentile"))
        if pct is not None:
            speed["percentile"] = pct
        if speed:
            speed["mode"] = "relative"
            speed["units"] = "normalized_units_per_second"
            entry["speed"] = speed
    return list(tracks.values())


_DIR_ABBR = {"left_to_right": "LtR", "right_to_left": "RtL"}


def descriptor_tokens(evidences) -> list:
    """Human-scannable filename tokens from an event's rule evidence.

    Returns the bits that make a clip identifiable in a folder listing without
    opening it: the steady gate speed (the GPS-validated through-speed, not the
    jumpy real-time figure the annotations draw), the YOLO vehicle type, and an
    abbreviated direction. Each token is omitted when not known, so a non-speeding
    event (e.g. center-lane) simply gets fewer tokens. ``evidences`` is any iterable
    of per-candidate evidence dicts."""
    speed = direction = vtype = None
    for ev in evidences:
        ev = ev or {}
        if ev.get("rule") == "absolute_speeding" and ev.get("speed_kmh") is not None:
            s = float(ev["speed_kmh"])
            speed = s if speed is None else max(speed, s)
        if direction is None and ev.get("direction"):
            direction = ev["direction"]
        if vtype is None and ev.get("vehicle_type"):
            vtype = ev["vehicle_type"]
    tokens = []
    if speed is not None:
        tokens.append(f"{round(speed)}kmh")
    if vtype and vtype != "unknown":
        tokens.append(str(vtype))
    if direction in _DIR_ABBR:
        tokens.append(_DIR_ABBR[direction])
    return tokens


def event_stem(stamp: str, event_type: str, short: str, evidences) -> str:
    """Filename stem: ``<stamp>_<speed/type/dir tokens>_<event_type>_<short_id>``.

    Speed/type/direction sit right after the timestamp so a folder sorted by name
    reads chronologically with the violation facts up front; ``event_type`` + the
    short id stay at the tail (the type is also the folder, kept here for portability
    when a clip is moved out). Falls back to ``<stamp>_<event_type>_<short>`` when no
    descriptive tokens are available."""
    tokens = descriptor_tokens(evidences)
    return "_".join([stamp, *tokens, event_type, short])


def build_metadata(
    event,
    clip_path: str | Path,
    thumbnail_path: str | Path,
    *,
    config,
    created_at: str,
    start_ts: float,
    trigger_ts: float,
    end_ts: float,
    media_kind: str = "clip",
    truncated_pre: float = 0.0,
    truncated_post: float = 0.0,
) -> Dict[str, Any]:
    """Assemble the event metadata document.

    ``media_kind`` records what evidence was kept: ``"clip"`` (full video, the
    default) or ``"screenshot"`` (a single 4K still for the mid-tier speeders we
    identify but don't keep video for). It makes the two queryable apart.

    ``truncated_pre`` / ``truncated_post`` are seconds of pre/post-roll a recording gap
    cost this clip (0 for the normal case). Recorded so a shortened clip is honest about
    the missing padding rather than silently off.
    """
    events_cfg = config.events
    doc = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "event_types": list(event.event_types),
        "media": media_kind,
        "created_at": created_at,
        "start_ts": round(start_ts, 3),
        "trigger_ts": round(trigger_ts, 3),
        "end_ts": round(end_ts, 3),
        "clip_path": str(clip_path),
        "thumbnail_path": str(thumbnail_path),
        "score": round(event.score, 4),
        "primary_track_id": event.primary_track_id,
        "tracks": _track_summaries(event.candidates),
        "evidence": {
            "triggers": [
                {
                    "event_type": c.event_type,
                    "trigger_ts": round(c.trigger_ts, 3),
                    "score": c.score,
                    "evidence": c.evidence,
                }
                for c in event.candidates
            ],
        },
        "config_snapshot": {
            "aggressiveness": config.aggressiveness,
            "clip_total_seconds": events_cfg.get("clip_total_seconds", 30),
            "pre_roll_seconds": events_cfg.get("pre_roll_seconds", 10),
            "post_roll_seconds": events_cfg.get("post_roll_seconds", 20),
        },
    }
    # Only present when a recording gap actually shortened the clip -- keeps the common
    # (full-length) clip's metadata clean while making the rare truncated one honest.
    if truncated_pre or truncated_post:
        doc["gap_truncated"] = {"pre_seconds": round(truncated_pre, 3),
                                "post_seconds": round(truncated_post, 3)}
    return doc


def write_metadata(metadata: Dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return out
