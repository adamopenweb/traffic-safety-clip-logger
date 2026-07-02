"""Speeding rule (Milestone 4).

Two modes:

* **Absolute** (preferred once calibrated to real metres): a track triggers when
  its true km/h stays over ``absolute_kmh_threshold`` for a short duration. This
  is the "actual speed gate" -- a posted-limit-relative threshold, independent of
  how the rest of traffic is moving.
* **Relative** (fallback, uncalibrated deployments): triggers when the track's
  speed percentile against the rolling per-direction baseline stays over an
  aggressiveness-tuned threshold. Used only when no absolute threshold is set or
  the calibration isn't in metric mode.

Either way a cooldown prevents repeat triggers for the same track and evidence is
attached for the event sidecar. The rule is pure logic: it consumes a
:class:`Track` history plus the current speed/direction and a
:class:`SpeedEstimator`, and returns candidate events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ...util.aggressiveness import resolve_thresholds
from ..metrics import speed_kmh_calibrated, steady_speed_kmh
from .base import CandidateEvent, Rule

# Below this many windowed samples the distribution is too thin to judge at all.
WARMUP_MIN_SAMPLES = 5
# Confidence multiplier while the baseline hasn't reached min_tracks_for_baseline.
WARMUP_SCORE_FACTOR = 0.6


@dataclass
class _TrackState:
    exceed_start: Optional[float] = None
    last_emit_ts: Optional[float] = None


class RelativeSpeedingRule(Rule):
    """Per-track relative-speeding detector."""

    event_type = "relative_speeding"

    def __init__(
        self,
        config: Dict,
        aggressiveness: float = 0.3,
        cooldown_seconds: float = 8.0,
        min_track_age: int = 3,
        meters_per_unit: Optional[Tuple[float, float]] = None,
        speed_window: float = 0.5,
        calibration: Optional[Dict] = None,
    ) -> None:
        super().__init__(config, enabled=bool(config.get("enabled", True)))
        resolved = resolve_thresholds(config, aggressiveness)
        self.percentile_threshold = float(resolved.get("percentile_threshold", 0.95))
        self.min_duration = float(resolved.get("min_duration_seconds", 0.6))
        self.min_tracks_for_baseline = int(config.get("min_tracks_for_baseline", 20))
        self.cooldown_seconds = float(cooldown_seconds)
        self.min_track_age = int(min_track_age)
        # Optional (across_m, along_m) per unit; when set, evidence reports km/h.
        self.meters_per_unit = meters_per_unit
        self.speed_window = float(speed_window)
        self.calibration = calibration or {}
        # Absolute speed gate: once calibrated to real metres, flag on true km/h
        # over a fixed threshold instead of the relative percentile. ``None`` keeps
        # the legacy percentile behaviour (for uncalibrated deployments).
        thr = config.get("absolute_kmh_threshold")
        self.absolute_kmh_threshold = float(thr) if thr is not None else None
        # A genuine speeder is briefly in frame, so the sustain here is short.
        self.absolute_min_duration = float(config.get("absolute_min_duration_seconds", 0.3))
        self._state: Dict[int, _TrackState] = {}

    def evict(self, track_ids) -> None:
        """Release per-track state for departed tracks (keyed by track id) so a
        long-lived run doesn't accumulate one ``_TrackState`` per vehicle forever.
        Driven by the loop's TrackStore sweep -- ids gone long enough to be evicted
        are past any in-progress detection, so dropping their state is safe."""
        for tid in track_ids:
            self._state.pop(tid, None)

    def _kmh(self, track) -> Optional[float]:
        """Track speed in km/h (across-road corrected), or None if not metric."""
        if self.meters_per_unit is None:
            return None
        v = speed_kmh_calibrated(track.ground_point_history(), self.speed_window,
                                 self.meters_per_unit, self.calibration)
        return round(v, 1) if v is not None else None

    def _steady_kmh(self, track) -> Optional[float]:
        """Jitter-robust steady km/h for the gate (not the peaky rolling window)."""
        if self.meters_per_unit is None:
            return None
        v = steady_speed_kmh(track.ground_point_history(), self.meters_per_unit, self.calibration)
        return round(v, 1) if v is not None else None

    def _evaluate_absolute(self, track, direction: str, ts: float) -> List[CandidateEvent]:
        """Fire when a track's steady true km/h stays over ``absolute_kmh_threshold``."""
        kmh = self._steady_kmh(track)
        if kmh is None:
            return []
        state = self._state.setdefault(track.track_id, _TrackState())
        if kmh < self.absolute_kmh_threshold:
            state.exceed_start = None
            return []
        if state.exceed_start is None:
            state.exceed_start = ts
        duration = ts - state.exceed_start
        if duration < self.absolute_min_duration:
            return []
        if state.last_emit_ts is not None and ts - state.last_emit_ts < self.cooldown_seconds:
            return []
        state.last_emit_ts = ts
        over = kmh - self.absolute_kmh_threshold
        # 0.5 at the threshold, rising to 1.0 at ~1.5x it.
        score = max(0.0, min(1.0, 0.5 + over / max(self.absolute_kmh_threshold, 1.0)))
        evidence = {
            "rule": "absolute_speeding",
            "track_id": track.track_id,
            "direction": direction,
            "speed_kmh": kmh,
            "threshold_kmh": round(self.absolute_kmh_threshold, 1),
            "over_by_kmh": round(over, 1),
            "duration_seconds": round(duration, 3),
            "vehicle_type": track.dominant_class(),
        }
        return [
            CandidateEvent(
                event_type=self.event_type, trigger_ts=ts,
                primary_track_id=track.track_id, score=round(score, 4),
                evidence=evidence, track_ids=[track.track_id],
            )
        ]

    def evaluate(
        self, track, speed: Optional[float], direction: Optional[str], estimator, ts: float
    ) -> List[CandidateEvent]:
        if not self.enabled or speed is None or direction is None:
            return []
        if track.age < self.min_track_age:
            return []

        # Absolute km/h gate (preferred once calibrated): trigger on sustained
        # true speed over the threshold, ignoring the relative baseline.
        if self.absolute_kmh_threshold is not None and self.meters_per_unit is not None:
            return self._evaluate_absolute(track, direction, ts)

        sample_count = estimator.count(direction)
        if sample_count < WARMUP_MIN_SAMPLES:
            return []  # not enough data in this direction to judge anything yet

        percentile = estimator.percentile(direction, speed)
        state = self._state.setdefault(track.track_id, _TrackState())

        if percentile < self.percentile_threshold:
            state.exceed_start = None
            return []

        # Track how long the over-threshold condition has persisted.
        if state.exceed_start is None:
            state.exceed_start = ts
        duration = ts - state.exceed_start
        if duration < self.min_duration:
            return []

        # Suppress repeat triggers for the same track within the cooldown.
        if state.last_emit_ts is not None and ts - state.last_emit_ts < self.cooldown_seconds:
            return []
        state.last_emit_ts = ts

        warmup = estimator.distinct_tracks(direction) < self.min_tracks_for_baseline
        score = percentile * (WARMUP_SCORE_FACTOR if warmup else 1.0)
        median = estimator.median(direction)

        evidence = {
            "rule": "relative_speeding",
            "track_id": track.track_id,
            "direction": direction,
            "speed": round(speed, 5),
            "speed_kmh": self._kmh(track),
            "percentile": round(percentile, 4),
            "rolling_median": round(median, 5) if median is not None else None,
            "threshold_used": round(self.percentile_threshold, 4),
            "duration_seconds": round(duration, 3),
            "warmup": warmup,
            "baseline_tracks": estimator.distinct_tracks(direction),
            "baseline_samples": sample_count,
        }
        return [
            CandidateEvent(
                event_type=self.event_type,
                trigger_ts=ts,
                primary_track_id=track.track_id,
                score=round(score, 4),
                evidence=evidence,
                track_ids=[track.track_id],
            )
        ]
