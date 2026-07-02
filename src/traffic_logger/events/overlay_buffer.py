"""Time-indexed buffer of per-frame track overlays (Approach B annotation).

The live analyzer knows where every vehicle is *as it processes each frame*, but
the evidence clip is cut from the ring ~post-roll seconds later, by which point
the tracker's bounded per-track history has scrolled past. To burn boxes + speed
onto the 4K clip we therefore have to capture the overlay geometry **live** and
replay it at export time.

This module is the hand-off: the analysis loop pushes one :class:`OverlaySnapshot`
per processed frame (every visible track's de-warped bbox + speed/lane/direction,
stamped with absolute wall time), and the clip exporter slices the window it needs.
It is pure data + a lock — no CV stack — so the buffer and snapshot selection are
unit-tested directly. The pixel mapping and drawing live in ``overlay_render``.
"""

from __future__ import annotations

import threading
from bisect import bisect_left
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

from collections import deque

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 in de-warped sub px


@dataclass(frozen=True)
class OverlayBox:
    """One tracked vehicle in one frame, in de-warped sub-stream pixel space."""

    track_id: int
    bbox: BBox
    speed_kmh: Optional[float] = None
    speed_rel: Optional[float] = None
    lane: Optional[str] = None
    direction: Optional[str] = None


@dataclass(frozen=True)
class OverlaySnapshot:
    """All boxes visible in a single processed frame, at absolute wall time ``ts``."""

    ts: float
    boxes: Tuple[OverlayBox, ...]


def nearest_snapshot(
    snapshots: Sequence[OverlaySnapshot], ts: float, tolerance: float
) -> Optional[OverlaySnapshot]:
    """The snapshot closest in time to ``ts``, or None if none within ``tolerance``.

    ``snapshots`` must be sorted ascending by ``.ts`` (as :meth:`OverlayBuffer.slice`
    returns them). Used per clip frame to find the overlay geometry recorded
    nearest that frame's timestamp.
    """
    if not snapshots:
        return None
    times = [s.ts for s in snapshots]
    i = bisect_left(times, ts)
    best: Optional[OverlaySnapshot] = None
    best_dt = tolerance
    for j in (i - 1, i):
        if 0 <= j < len(snapshots):
            dt = abs(snapshots[j].ts - ts)
            if dt <= best_dt:
                best, best_dt = snapshots[j], dt
    return best


class OverlayBuffer:
    """Thread-safe rolling buffer of recent :class:`OverlaySnapshot`s.

    Producer: the analysis loop (one append per processed frame). Consumer: the
    ring clip exporter daemon (one slice per event). ``capacity_seconds`` must
    span pre-roll + post-roll + the exporter's margin so a clip window is still
    fully covered when it is finally cut.
    """

    def __init__(self, capacity_seconds: float, frame_size: Optional[Tuple[int, int]] = None) -> None:
        self.capacity_seconds = float(capacity_seconds)
        self._frame_size = frame_size
        self._snaps: Deque[OverlaySnapshot] = deque()
        self._lock = threading.Lock()

    @property
    def frame_size(self) -> Optional[Tuple[int, int]]:
        """(width, height) of the de-warped analysis frames the boxes live in."""
        with self._lock:
            return self._frame_size

    def set_frame_size(self, width: int, height: int) -> None:
        """Record the analysis frame size once (first frame); later calls no-op."""
        with self._lock:
            if self._frame_size is None:
                self._frame_size = (int(width), int(height))

    def append(self, ts: float, boxes: Sequence[OverlayBox]) -> None:
        """Add one frame's snapshot and evict anything older than the capacity."""
        with self._lock:
            self._snaps.append(OverlaySnapshot(ts=float(ts), boxes=tuple(boxes)))
            cutoff = ts - self.capacity_seconds
            while self._snaps and self._snaps[0].ts < cutoff:
                self._snaps.popleft()

    def slice(self, start_ts: float, end_ts: float) -> List[OverlaySnapshot]:
        """Snapshots with ``start_ts <= ts <= end_ts``, oldest-first (a copy)."""
        with self._lock:
            return [s for s in self._snaps if start_ts <= s.ts <= end_ts]

    def __len__(self) -> int:
        with self._lock:
            return len(self._snaps)


# -- serialization (overlay sidecar, for offline re-render / offset tuning) ----
def box_to_dict(box: OverlayBox) -> dict:
    return {
        "track_id": box.track_id,
        "bbox": list(box.bbox),
        "speed_kmh": box.speed_kmh,
        "speed_rel": box.speed_rel,
        "lane": box.lane,
        "direction": box.direction,
    }


def box_from_dict(d: dict) -> OverlayBox:
    return OverlayBox(
        track_id=int(d["track_id"]),
        bbox=tuple(d["bbox"]),  # type: ignore[arg-type]
        speed_kmh=d.get("speed_kmh"),
        speed_rel=d.get("speed_rel"),
        lane=d.get("lane"),
        direction=d.get("direction"),
    )


def serialize_snapshots(snapshots: Sequence[OverlaySnapshot]) -> List[dict]:
    """Snapshots -> JSON-ready dicts (oldest-first as given)."""
    return [{"ts": s.ts, "boxes": [box_to_dict(b) for b in s.boxes]} for s in snapshots]


def deserialize_snapshots(rows: Sequence[dict]) -> List[OverlaySnapshot]:
    """Inverse of :func:`serialize_snapshots`."""
    return [
        OverlaySnapshot(ts=float(r["ts"]), boxes=tuple(box_from_dict(b) for b in r["boxes"]))
        for r in rows
    ]
