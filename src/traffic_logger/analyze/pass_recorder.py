"""Record every completed vehicle pass -- the traffic denominator.

The rules only persist cars that trip a gate (>=55 km/h), so the stores have a
numerator but no denominator: we can't say "X% of cars sped" without knowing how
many cars there were. Yet the tracker already assigns *every* passing vehicle a
track id; we simply never wrote the non-speeders down.

:class:`PassRecorder` closes that gap. It mirrors the police departed-track
lifecycle (``observe_frame`` / ``finalize_departed`` / ``close``): it remembers when
each track was last seen, and once a track has been gone for ``finalize_after``
frames it finalizes it into one :class:`~..events.store.PassRecord` row -- one per
real drive-by -- in the unified store's ``passes`` table.

To make the count trustworthy it drops what isn't a clean through-pass:
* tracks with too few observations (``min_age``) -- detection flickers / spurious;
* stationary tracks (ground span below ``min_ground_span``) -- parked cars across the
  street, the same filter the annotation overlay uses;
* tracks that never established a travel direction.

It carries each pass's steady (GPS-validated end-to-end) km/h, so the denominator
*and* the speed distribution of all traffic come from one place. Caveat: ByteTrack
can split one vehicle into two ids under occlusion, so the count is a solid estimate
(a few % high), not a perfect census.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

from .metrics import ground_span, steady_speed_kmh
from .pass_validity import PassGeometry, ValidityThresholds, speed_validity
from ..events.store import PassRecord, TrafficStore
from ..util.logging import get_logger

log = get_logger(__name__)


class PassRecorder:
    def __init__(self, store: TrafficStore, session_id: str, *,
                 meters_per_unit, calibration: dict,
                 speeding_gate_kmh: Optional[float] = None,
                 finalize_after: int = 20, min_age: int = 5,
                 min_ground_span: float = 0.12) -> None:
        self.store = store
        self.session_id = session_id
        self.meters_per_unit = meters_per_unit
        self.cal = calibration or {}
        self.gate = speeding_gate_kmh
        self.finalize_after = finalize_after
        self.min_age = min_age
        self.min_ground_span = min_ground_span
        # Scene-physics bounds for the derived validity flag (same call revalidation makes).
        self.validity = ValidityThresholds()
        self._last_seen: Dict[int, int] = {}        # track id -> last frame index
        self._first_wall: Dict[int, float] = {}     # track id -> wall ts first seen
        self._last_wall: Dict[int, float] = {}       # track id -> wall ts last seen
        self._done: Set[int] = set()                 # already finalized (de-dupe)
        self.counted = 0
        self.skipped = 0

    def observe_frame(self, tracks, processed: int, wall_now: float) -> None:
        """Note that each visible track was seen at frame ``processed`` / ``wall_now``."""
        for t in tracks:
            tid = t.track_id
            self._last_seen[tid] = processed
            self._last_wall[tid] = wall_now
            self._first_wall.setdefault(tid, wall_now)

    def finalize_departed(self, processed: int, track_store) -> None:
        """Finalize every track not seen for more than ``finalize_after`` frames."""
        gone = [tid for tid, seen in self._last_seen.items()
                if processed - seen > self.finalize_after]
        for tid in gone:
            self._finalize(tid, track_store)

    def close(self, track_store) -> None:
        """Finalize all still-active tracks at end of run."""
        for tid in list(self._last_seen.keys()):
            self._finalize(tid, track_store)

    def evict(self, track_ids) -> None:
        """Drop the finalize-dedupe marker for long-departed tracks so ``_done`` doesn't
        grow one int per vehicle forever. Only called for ids the TrackStore sweep
        evicted (idle far longer than ``finalize_after``), so they're already finalized
        and ByteTrack won't reissue the id -- discarding can't cause a re-finalize."""
        for tid in track_ids:
            self._done.discard(tid)

    # -- internals --
    def _finalize(self, tid: int, track_store) -> None:
        first_wall = self._first_wall.get(tid)
        last_wall = self._last_wall.get(tid, first_wall)
        self._forget(tid)
        if tid in self._done:
            return
        self._done.add(tid)

        track = track_store.get(tid) if track_store is not None else None
        if track is None or track.age < self.min_age:
            self.skipped += 1
            return
        gh = track.ground_point_history()
        if self.min_ground_span > 0 and ground_span(gh) < self.min_ground_span:
            self.skipped += 1   # parked / stationary -- not traffic
            return
        direction = track.direction()
        if direction is None:
            self.skipped += 1   # never established travel -- not a clean pass
            return

        # The legacy steady_speed_kmh column keeps the source-guarded value (None on a span
        # reject) so existing readers are unchanged; steady_speed_raw preserves the raw
        # measurement (un-span-gated) for forensics/re-tuning. Validity is DERIVED from the
        # stored geometry via the shared invariant module -- the identical call the
        # revalidation script makes over history, so the two paths can never drift.
        steady = raw = None
        if self.meters_per_unit is not None:
            steady = steady_speed_kmh(gh, self.meters_per_unit, self.cal)
            raw = steady_speed_kmh(gh, self.meters_per_unit, self.cal, max_span=float("inf"))
        gspan = ground_span(gh)
        n_pts = len(gh)
        track_seconds = ((last_wall - first_wall)
                         if (first_wall is not None and last_wall is not None) else 0.0)
        raw_r = round(raw, 1) if raw is not None else None
        valid, reason = speed_validity(
            PassGeometry(steady_kmh=raw_r, track_seconds=track_seconds,
                         ground_span=gspan, n_points=n_pts),
            self.validity)
        was_speeding = ((raw_r >= self.gate)
                        if (valid and raw_r is not None and self.gate is not None) else None)
        self.store.upsert_pass(PassRecord(
            session_id=self.session_id, track_id=tid,
            first_ts=first_wall if first_wall is not None else 0.0,
            last_ts=last_wall if last_wall is not None else 0.0,
            direction=direction, vehicle_type=track.dominant_class(),
            steady_speed_kmh=round(steady, 1) if steady is not None else None,
            steady_speed_raw=raw_r, ground_span=round(gspan, 4), n_points=n_pts,
            steady_valid=valid, steady_invalid_reason=reason, was_speeding=was_speeding))
        self.counted += 1

    def _forget(self, tid: int) -> None:
        for d in (self._last_seen, self._first_wall, self._last_wall):
            d.pop(tid, None)
