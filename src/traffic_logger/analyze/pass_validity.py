"""Physical-plausibility invariants for a pass's steady speed (pure, no CV deps).

The dashboard used to carry ~80 lines of read-time filtering that masked implausible
stored speeds (a 0.9s track that read 152 km/h, a 1.25s "100 km/h truck" whose implied
ground span was ~2.3x the road). That was data-repair masquerading as read logic, and it
kept accreting a new clause per artifact class. This module is the principled version:
ONE set of *validity invariants* -- true for every pipeline version and every consumer --
computed once at write time (:class:`~..analyze.pass_recorder.PassRecorder`) and re-runnable
over history (``scripts/revalidate_passes.py``).

Split of concerns:
* **Validity invariants** (here): scene physics -- a track must last long enough to be a
  crossing, can't traverse more than ~the road, can't exceed a clean full-FOV speed.
* **Policy thresholds** (NOT here): the posted limit, "fast"/Top-Speeds cutoffs. Those
  are user-tunable judgment calls and stay read-time config forever.

The predicates run on :class:`PassGeometry` -- a handful of scalars -- not on the raw
ground history, precisely so the SAME function runs at write time (built from the track)
and in revalidation (read from stored columns). If those two paths computed validity from
different inputs they would drift, which is the failure this design exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PassGeometry:
    """The scalars a validity check needs, decoupled from the ground history.

    ``ground_span`` (normalized road-plane diagonal extent) and ``n_points`` are stored at
    write time. For rows written before those columns existed they're ``None``, and the
    implied-distance proxy (``steady_kmh``/3.6 * ``track_seconds``) stands in -- the read-time
    guard that discovered these artifacts in the first place.
    """

    steady_kmh: Optional[float]
    track_seconds: float
    ground_span: Optional[float] = None
    n_points: Optional[int] = None


@dataclass(frozen=True)
class ValidityThresholds:
    """Scene-physics bounds. Defaults match the guards they replace (source ``max_span``
    1.6, the 0.4s fragment floor, the ~120-130 km/h clean-crossing ceiling, the ~2x-road
    implied-distance proxy = 30 m for the 14.94 m road)."""

    min_track_seconds: float = 0.4
    max_kmh: float = 130.0
    max_ground_span: float = 1.6
    max_implied_distance_m: float = 30.0
    min_points: int = 6


def speed_validity(geo: PassGeometry, thr: ValidityThresholds = ValidityThresholds()
                   ) -> Tuple[bool, Optional[str]]:
    """Is this pass's steady speed physically plausible? Returns ``(is_valid, reason)``.

    ``reason`` is a stable slug (for the stored ``steady_invalid_reason`` + the
    "valid rows that fail invariants" monitoring query) or ``None`` when valid. A pass
    with no measured speed is ``(False, "no_speed")`` -- the car still counts as traffic;
    only the speed is untrustworthy.
    """
    s = geo.steady_kmh
    if s is None:
        return False, "no_speed"
    if geo.track_seconds < thr.min_track_seconds:
        return False, "fragment_track"          # too brief to be a full FOV crossing
    if geo.n_points is not None and geo.n_points < thr.min_points:
        return False, "too_few_points"          # not enough samples for a robust estimate
    if s > thr.max_kmh:
        return False, "over_max_kmh"             # beyond any clean full-FOV crossing
    if geo.ground_span is not None:
        if geo.ground_span > thr.max_ground_span:
            return False, "span_exceeds_road"    # traversed >~the road -> multi-frame jump
    else:
        # Pre-column history: no stored span, so fall back to the implied-distance proxy.
        implied_m = (s / 3.6) * geo.track_seconds
        if implied_m > thr.max_implied_distance_m:
            return False, "implied_distance"
    return True, None
