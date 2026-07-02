"""Per-track 'best frame' capture for violator thumbnails.

The event thumbnail should show the *violating vehicle*, but the clip is cut from
the 4K ring after the fact and a fixed-time grab usually misses a fast car. Detection
runs on the sub-stream, though -- so on a sub frame we know a vehicle's box *exactly*
(the box IS the detection on that frame; no sub->4K sync offset to solve).

:class:`BestFrameCache` rides along the live loop and remembers, for each track, the
sub frame where that track's bounding box was **largest** -- i.e. the moment the car
is closest / best framed. When the event finally fires (up to the merge window later)
the exporter ``take``s that frame + box and crops a pixel-accurate thumbnail.

Bounded memory: one frame per recently-seen track, evicted ``idle_evict_frames``
after a track was last seen (long enough to outlast the merge-window dispatch delay).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

BBox = Tuple[float, float, float, float]


class BestFrameCache:
    def __init__(self, idle_evict_frames: int = 510) -> None:  # ~17s @ 30fps
        self.idle_evict = int(idle_evict_frames)
        self._best: Dict[int, Tuple[float, object, BBox]] = {}  # tid -> (area, frame, bbox)
        self._seen: Dict[int, int] = {}                          # tid -> last frame index

    def observe(self, tracks, frame, frame_idx: int) -> None:
        """Record this frame as a track's best if its box is the largest seen so far."""
        for t in tracks:
            bb = t.latest_bbox
            if bb is None:
                continue
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            cur = self._best.get(t.track_id)
            if cur is None or area > cur[0]:
                # copy: the live frame is reused/overwritten next iteration
                self._best[t.track_id] = (area, frame.copy(), tuple(float(v) for v in bb))
            self._seen[t.track_id] = frame_idx
        if frame_idx % 30 == 0:  # cheap periodic eviction of departed tracks
            stale = [tid for tid, seen in self._seen.items()
                     if frame_idx - seen > self.idle_evict]
            for tid in stale:
                self._best.pop(tid, None)
                self._seen.pop(tid, None)

    def take(self, track_id: Optional[int]) -> Optional[Tuple[object, BBox]]:
        """Pop and return ``(frame, bbox)`` for a track, or None if not held."""
        if track_id is None:
            return None
        self._seen.pop(track_id, None)
        entry = self._best.pop(track_id, None)
        return (entry[1], entry[2]) if entry else None
