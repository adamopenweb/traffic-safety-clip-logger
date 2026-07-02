"""Center-lane passing rule (Milestone 5).

The most important custom rule: detect vehicles using the shared center turning
lane as a passing lane. Two patterns (spec "Event Rule: Center Lane Pass"):

* Pattern A - fast center-lane traversal: a moving vehicle dwells in the center
  turn lane for a minimum time while its speed percentile is above the
  center-lane threshold.
* Pattern B - overtake through the center lane (stronger): a candidate that
  starts behind another same-direction vehicle, moves through the center lane,
  and ends ahead of it within the overtake window.

The rule consumes :class:`Track` histories (lane bands + bottom-center path)
plus the per-direction :class:`SpeedEstimator`, and emits candidate events. The
geometric helpers are pure and unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ...util.aggressiveness import resolve_thresholds
from ..metrics import speed_kmh_calibrated, track_speed
from ..tracker import LEFT_TO_RIGHT, RIGHT_TO_LEFT
from .base import CandidateEvent, Rule

CENTER_LANE = "center_turn_lane"
TRAVEL_LANES = ("travel_lane_a", "travel_lane_b")
# A confirmed overtake is a strong signal; floor its score here.
OVERTAKE_SCORE_FLOOR = 0.85
# Candidate must spend at least this fraction of the overtake window in center.
DEFAULT_CENTER_FRACTION = 0.5


# --------------------------------------------------------------------------
# Pure geometric helpers (tested)
# --------------------------------------------------------------------------
def compress_lane_sequence(bands: List[Optional[str]]) -> List[str]:
    """Collapse consecutive identical lane bands into a sequence.

    None (off-road / pre-calibration) is rendered as ``"off_road"``.
    """
    out: List[str] = []
    for band in bands:
        label = band if band is not None else "off_road"
        if not out or out[-1] != label:
            out.append(label)
    return out


def center_dwell_seconds(
    lane_history: List[Tuple[float, Optional[str]]], center: str = CENTER_LANE
) -> float:
    """Contiguous time the track has been in the center lane, ending now.

    Returns 0 when the latest observation is not in the center lane.
    """
    if not lane_history or lane_history[-1][1] != center:
        return 0.0
    last_ts = lane_history[-1][0]
    run_start_ts = last_ts
    for ts, band in reversed(lane_history):
        if band != center:
            break
        run_start_ts = ts
    return last_ts - run_start_ts


def detect_overtake(
    cand_progress: List[Tuple[float, float]],
    other_progress: List[Tuple[float, float]],
    dir_sign: int,
    window_seconds: float,
    ts: float,
    min_advance: float = 0.0,
) -> Tuple[bool, Optional[float], Optional[float], int]:
    """Did the candidate flip from behind to ahead of ``other`` in the window?

    Each progress history is ``(ts, value)`` along the road's length axis
    (normalized y), oriented by ``dir_sign`` (+1 when the direction means the
    value increases). Positions are compared at the earliest and latest shared
    timestamps in the window. Returns (detected, rel_before, rel_after,
    common_sample_count); a positive relative position means "ahead".

    ``min_advance`` is the minimum *relative gain* (rel_after - rel_before) the
    candidate must make on ``other`` to count as a real pass. Without it, two
    side-by-side vehicles (rel ~ 0) flip sign on noise-level jitter and every slow
    car dipping into the center turn lane was mislabelled an overtake (the
    false-positive bug). A genuine overtake gains a clear chunk of road length.
    """
    lo = ts - window_seconds
    cand = {round(t, 6): v for (t, v) in cand_progress if t >= lo}
    other = {round(t, 6): v for (t, v) in other_progress if t >= lo}
    common = sorted(set(cand) & set(other))
    if len(common) < 2:
        return (False, None, None, len(common))
    t0, t1 = common[0], common[-1]
    rel_before = (cand[t0] - other[t0]) * dir_sign
    rel_after = (cand[t1] - other[t1]) * dir_sign
    detected = rel_before < 0 and rel_after > 0 and (rel_after - rel_before) >= min_advance
    return (detected, rel_before, rel_after, len(common))


@dataclass
class _TrackState:
    last_emit_ts: Optional[float] = None


@dataclass
class _OvertakeResult:
    detected: bool = False
    passed_track_id: Optional[int] = None
    rel_before: Optional[float] = None
    rel_after: Optional[float] = None


class CenterLanePassRule(Rule):
    event_type = "center_lane_pass"

    def __init__(
        self,
        config: Dict,
        aggressiveness: float = 0.3,
        cooldown_seconds: float = 8.0,
        min_track_age: int = 3,
        speed_window: float = 0.5,
        meters_per_unit: Optional[Tuple[float, float]] = None,
        calibration: Optional[Dict] = None,
    ) -> None:
        super().__init__(config, enabled=bool(config.get("enabled", True)))
        resolved = resolve_thresholds(config, aggressiveness)
        self.center_min_time = float(resolved.get("center_lane_min_time_seconds", 0.6))
        self.speed_pct_threshold = float(resolved.get("speed_percentile_threshold", 0.85))
        self.detect_overtake = bool(config.get("detect_overtake", True))
        self.overtake_window = float(config.get("overtake_window_seconds", 6))
        # Minimum relative road-length gain for a flip to count as a real pass
        # (normalized along-road units). Excludes noise-level side-by-side jitter
        # that was flagging slow center-turn-lane traffic as overtakes.
        self.overtake_min_advance = float(config.get("overtake_min_advance", 0.12))
        # Optional travel-direction filter. The center turn lane is shared, so a
        # car approaching a nearby turn has a legitimate reason to sit in it when
        # travelling one way -- only the *other* direction (no turn ahead, high
        # speed) is a credible passing maneuver. ``directions`` is a list of the
        # opaque camera-relative labels ("left_to_right"/"right_to_left"); None
        # (key absent) means evaluate both directions.
        dirs = config.get("directions")
        self.allowed_directions = tuple(dirs) if dirs else None
        self.cooldown_seconds = float(cooldown_seconds)
        self.min_track_age = int(min_track_age)
        self.speed_window = float(speed_window)
        self.meters_per_unit = meters_per_unit
        self.calibration = calibration or {}
        self._state: Dict[int, _TrackState] = {}

    def evict(self, track_ids) -> None:
        """Release per-track state for departed tracks so a long-lived run doesn't keep
        one ``_TrackState`` per vehicle forever. Driven by the loop's TrackStore sweep."""
        for tid in track_ids:
            self._state.pop(tid, None)

    def evaluate(self, tracks: List, estimator, ts: float) -> List[CandidateEvent]:
        if not self.enabled:
            return []
        events: List[CandidateEvent] = []
        for candidate in tracks:
            ev = self._evaluate_candidate(candidate, tracks, estimator, ts)
            if ev is not None:
                events.append(ev)
        return events

    def _evaluate_candidate(self, cand, tracks, estimator, ts) -> Optional[CandidateEvent]:
        if cand.age < self.min_track_age:
            return None
        direction = cand.direction()
        if direction is None:
            return None
        if self.allowed_directions is not None and direction not in self.allowed_directions:
            return None

        dwell = center_dwell_seconds(cand.lane_band_history())
        if dwell < self.center_min_time:
            return None

        speed = track_speed(cand.ground_point_history(), self.speed_window)
        percentile = estimator.percentile(direction, speed) if speed is not None else 0.0
        fast = percentile >= self.speed_pct_threshold

        overtake = _OvertakeResult()
        if self.detect_overtake:
            overtake = self._find_overtake(cand, tracks, direction, ts)

        # A center-lane pass = two same-direction vehicles co-occurring: one in
        # the center lane (this candidate, moving at pass speed) and one in a
        # travel lane (the car being passed). Requiring that companion is what
        # the short along-road FOV can actually see, and it rejects a lone center
        # track -- typically a far-lane car misclassified into the center band.
        # An overtake (position flip) already implies a same-direction companion.
        companion_id = self._same_dir_travel_companion(cand, tracks, direction)
        if not overtake.detected and not (fast and companion_id is not None):
            return None

        # Cooldown: one event per track per cooldown window.
        state = self._state.setdefault(cand.track_id, _TrackState())
        if state.last_emit_ts is not None and ts - state.last_emit_ts < self.cooldown_seconds:
            return None
        state.last_emit_ts = ts

        score = percentile
        if overtake.detected:
            score = max(score, OVERTAKE_SCORE_FLOOR)
        score = round(min(1.0, score), 4)

        passed_id = overtake.passed_track_id if overtake.detected else companion_id
        evidence = {
            "rule": "center_lane_overtake" if overtake.detected else "center_lane_pass",
            "candidate_track_id": cand.track_id,
            "passed_track_id": passed_id,
            "direction": direction,
            "lane_sequence": compress_lane_sequence([b for _t, b in cand.lane_band_history()]),
            "center_lane_time_seconds": round(dwell, 3),
            "speed_percentile": round(percentile, 4),
            "speed_kmh": self._kmh(cand),
            "overtake_detected": overtake.detected,
            "relative_position_before": (
                round(overtake.rel_before, 2) if overtake.rel_before is not None else None
            ),
            "relative_position_after": (
                round(overtake.rel_after, 2) if overtake.rel_after is not None else None
            ),
        }
        return CandidateEvent(
            event_type=self.event_type,
            trigger_ts=ts,
            primary_track_id=cand.track_id,
            score=score,
            evidence=evidence,
            track_ids=[cand.track_id] + ([passed_id] if passed_id is not None else []),
        )

    @staticmethod
    def _progress(track) -> List[Tuple[float, float]]:
        """Along-road progress history: (ts, normalized-y)."""
        return [(ts, gy) for ts, _gx, gy in track.ground_point_history()]

    def _kmh(self, track) -> Optional[float]:
        if self.meters_per_unit is None:
            return None
        v = speed_kmh_calibrated(track.ground_point_history(), self.speed_window,
                                 self.meters_per_unit, self.calibration)
        return round(v, 1) if v is not None else None

    def _same_dir_travel_companion(self, cand, tracks, direction) -> Optional[int]:
        """A co-occurring same-direction track currently in a travel lane (the
        vehicle being passed). Returns its track id, or None."""
        for other in tracks:
            if other.track_id == cand.track_id:
                continue
            if other.latest_lane_band not in TRAVEL_LANES:
                continue
            if other.direction() != direction:
                continue
            return other.track_id
        return None

    def _find_overtake(self, cand, tracks, direction, ts) -> _OvertakeResult:
        dir_sign = 1 if direction == LEFT_TO_RIGHT else -1
        if self._center_fraction(cand, ts) < DEFAULT_CENTER_FRACTION:
            return _OvertakeResult()
        cand_progress = self._progress(cand)
        for other in tracks:
            if other.track_id == cand.track_id:
                continue
            if other.direction() != direction:
                continue
            detected, rel_before, rel_after, _n = detect_overtake(
                cand_progress, self._progress(other), dir_sign, self.overtake_window, ts,
                min_advance=self.overtake_min_advance,
            )
            if detected:
                return _OvertakeResult(
                    detected=True,
                    passed_track_id=other.track_id,
                    rel_before=rel_before,
                    rel_after=rel_after,
                )
        return _OvertakeResult()

    def _center_fraction(self, track, ts) -> float:
        lo = ts - self.overtake_window
        bands = [b for t, b in track.lane_band_history() if t >= lo]
        if not bands:
            return 0.0
        return sum(1 for b in bands if b == CENTER_LANE) / len(bands)
