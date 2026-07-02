"""Multi-object tracking and per-track history.

Two layers:

* :class:`Track` / :class:`TrackStore` are **pure** project logic that maintain
  per-track history (timestamped bboxes, bottom-center road-contact points, a
  camera-relative direction estimate, age/confidence). They have no heavy
  dependencies and are unit tested directly with synthetic observations.
* :class:`VehicleTracker` wraps ``supervision.ByteTrack`` (lazy-imported), feeds
  its tracked detections into a :class:`TrackStore`, and is what the offline
  analyzer drives. Rules (M4/M5) consume the resulting :class:`Track` histories.

The bottom-center of each bbox is the default road-contact point (spec:
"Use bottom-center of vehicle bbox as the default road-contact point").
Projection to the ground plane, lane bands, and speed are layered on in
Milestones 3 and 4.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Tuple

from .lane_model import assign_lane_band as _assign_lane_band

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2

# Camera-relative direction labels (spec: no real east/west for MVP).
LEFT_TO_RIGHT = "left_to_right"
RIGHT_TO_LEFT = "right_to_left"

# Default per-track history cap: ~10s at 30fps. Bounds memory for long runs.
DEFAULT_MAX_HISTORY = 300
# Minimum net image-x travel (px) before a direction is asserted (uncalibrated).
DEFAULT_DIRECTION_MIN_DX = 8.0
# Minimum net along-road travel (normalized units) before asserting direction.
DEFAULT_DIRECTION_MIN_DY = 0.05


def bottom_center(bbox: BBox) -> Tuple[float, float]:
    """Road-contact point: horizontal center of the bbox at its bottom edge."""
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


@dataclass(frozen=True)
class Observation:
    ts: float
    bbox: BBox
    confidence: float = 1.0
    # Filled in once calibration exists (Milestone 3): projected ground-plane
    # point (normalized) and the lane band it falls in.
    ground_point: Optional[Tuple[float, float]] = None
    lane_band: Optional[str] = None
    # YOLO/COCO vehicle class for this detection (car/truck/bus/motorcycle).
    vehicle_class: Optional[str] = None

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return bottom_center(self.bbox)


class Track:
    """History of a single tracked object."""

    def __init__(self, track_id: int, max_history: int = DEFAULT_MAX_HISTORY) -> None:
        self.track_id = track_id
        self.observations: Deque[Observation] = deque(maxlen=max_history)

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)

    # -- basic accessors ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.observations)

    @property
    def age(self) -> int:
        """Number of observations recorded for this track."""
        return len(self.observations)

    @property
    def first_ts(self) -> Optional[float]:
        return self.observations[0].ts if self.observations else None

    @property
    def last_ts(self) -> Optional[float]:
        return self.observations[-1].ts if self.observations else None

    @property
    def latest_bbox(self) -> Optional[BBox]:
        return self.observations[-1].bbox if self.observations else None

    @property
    def latest_confidence(self) -> float:
        return self.observations[-1].confidence if self.observations else 0.0

    @property
    def latest_bottom_center(self) -> Optional[Tuple[float, float]]:
        return self.observations[-1].bottom_center if self.observations else None

    def bottom_center_history(self) -> List[Tuple[float, float, float]]:
        """(ts, x, y) of the bottom-center point across all observations."""
        out = []
        for obs in self.observations:
            x, y = obs.bottom_center
            out.append((obs.ts, x, y))
        return out

    def ground_point_history(self) -> List[Tuple[float, float, float]]:
        """(ts, gx, gy) of the projected ground point (skips unprojected obs)."""
        return [
            (obs.ts, obs.ground_point[0], obs.ground_point[1])
            for obs in self.observations
            if obs.ground_point is not None
        ]

    def lane_band_history(self) -> List[Tuple[float, Optional[str]]]:
        """(ts, lane_band) across observations (band may be None off-road)."""
        return [(obs.ts, obs.lane_band) for obs in self.observations]

    @property
    def latest_lane_band(self) -> Optional[str]:
        return self.observations[-1].lane_band if self.observations else None

    def dominant_class(self) -> Optional[str]:
        """Most common vehicle class across this track's observations (a robust
        single label vs per-frame YOLO flicker between e.g. car/truck)."""
        from collections import Counter

        classes = [o.vehicle_class for o in self.observations if o.vehicle_class]
        if not classes:
            return None
        return Counter(classes).most_common(1)[0][0]

    def direction(
        self,
        min_dx: float = DEFAULT_DIRECTION_MIN_DX,
        min_dy: float = DEFAULT_DIRECTION_MIN_DY,
    ) -> Optional[str]:
        """Camera-relative travel direction.

        When calibration is present (observations carry projected ground
        points), direction is the sign of net travel along the road's *length*
        axis (normalized y) — so a vehicle staying within one lane still has a
        well-defined direction. Without calibration it falls back to net image
        bottom-center x travel. ``left_to_right`` / ``right_to_left`` are opaque
        camera-relative labels used to separate the two traffic streams.
        Returns None when the track hasn't moved far enough to decide.
        """
        obs = self.observations
        if len(obs) < 2:
            return None
        if obs[0].ground_point is not None and obs[-1].ground_point is not None:
            dy = obs[-1].ground_point[1] - obs[0].ground_point[1]
            if abs(dy) < min_dy:
                return None
            return LEFT_TO_RIGHT if dy > 0 else RIGHT_TO_LEFT
        dx = obs[-1].bottom_center[0] - obs[0].bottom_center[0]
        if abs(dx) < min_dx:
            return None
        return LEFT_TO_RIGHT if dx > 0 else RIGHT_TO_LEFT


class TrackStore:
    """Holds all tracks keyed by tracker id and records observations."""

    def __init__(self, max_history: int = DEFAULT_MAX_HISTORY) -> None:
        self._tracks: dict[int, Track] = {}
        self._max_history = max_history

    def record(self, track_id: int, observation: Observation) -> None:
        """Append a fully-formed observation to a track (creating it if new)."""
        track = self._tracks.get(track_id)
        if track is None:
            track = Track(track_id, max_history=self._max_history)
            self._tracks[track_id] = track
        track.add(observation)

    def update(
        self,
        observations: Iterable[Tuple[int, BBox, float]],
        ts: float,
    ) -> None:
        """Record one frame's (track_id, bbox, confidence) observations at ``ts``."""
        for track_id, bbox, confidence in observations:
            self.record(track_id, Observation(ts=ts, bbox=bbox, confidence=confidence))

    def get(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    def active_tracks(self, min_observations: int = 1) -> List[Track]:
        """Tracks with at least ``min_observations``, ordered by track id."""
        return [
            t for _, t in sorted(self._tracks.items())
            if t.age >= min_observations
        ]

    def sweep(self, now_ts: float, max_idle_seconds: float) -> List[int]:
        """Evict tracks whose last observation is older than ``max_idle_seconds``;
        return the evicted ids.

        Without this every vehicle ever tracked stays resident -- each :class:`Track`
        keeps up to ``max_history`` observations -- so a long-lived 24/7 run grows
        unbounded (the daylight supervisor used to mask it by killing the child nightly;
        all-day mode has no such backstop). Every consumer references a track only
        briefly after it departs (the pass/police finalizers fire within ~a second of the
        gone-threshold, the police confirmer within ~a segment), so a threshold well
        beyond those windows drops only tracks nothing will read again. The caller passes
        the returned ids to the rules' / recorder's ``evict`` so their per-track state
        (keyed by the same ids) is released in lockstep. ``<= 0`` disables the sweep."""
        if max_idle_seconds <= 0:
            return []
        stale = [tid for tid, t in self._tracks.items()
                 if t.observations and now_ts - t.observations[-1].ts > max_idle_seconds]
        for tid in stale:
            del self._tracks[tid]
        return stale

    def __len__(self) -> int:
        return len(self._tracks)


class VehicleTracker:
    """ByteTrack-backed tracker producing :class:`Track` histories.

    Lazily imports supervision so the module can be imported without the CV
    stack installed (e.g. on a core-only box). Construct only where the
    ``analyze`` extra is available (Docker/WSL, or the dev box once installed).
    """

    def __init__(self, config, projector=None) -> None:
        import supervision as sv  # lazy: requires the analyze extra

        self._sv = sv
        # Optional perspective transform; when present, each observation gets a
        # projected ground point and lane band (Milestone 3).
        self.projector = projector
        self.lane_model_cfg = config.calibration.get("lane_model", {})
        # Lane classification uses the bbox bottom-center (tire contact), which an
        # angled camera places on the NEAR-curb side of the vehicle. contact_bias
        # shifts the classified across-position back toward the far curb so a car
        # lands in its true lane while the bands stay drawn on the paint.
        self._contact_bias = float(self.lane_model_cfg.get("contact_bias", 0.0))
        tracking = config.tracking
        analysis = config.analysis
        # ByteTrack is deprecated-but-functional in supervision 0.28; it is the
        # tracker the spec calls for. Silence the construction-time warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bytetrack = sv.ByteTrack(
                track_activation_threshold=float(tracking.get("track_activation_threshold", 0.25)),
                lost_track_buffer=int(tracking.get("lost_track_buffer", 30)),
                minimum_matching_threshold=float(tracking.get("minimum_matching_threshold", 0.8)),
                frame_rate=int(analysis.get("inference_fps", 12)),
                minimum_consecutive_frames=int(tracking.get("minimum_consecutive_frames", 3)),
            )
        self.store = TrackStore()
        # The most recent tracked sv.Detections (with tracker_id), for drawing.
        self.last_tracked = None

    def update(self, detections, ts: float) -> List[Track]:
        """Advance the tracker by one frame; return the tracks observed *this*
        frame (ordered by id).

        Only tracks seen this frame are returned — NOT every track ever recorded.
        A departed track's history freezes at its last observation (the deque
        only evicts on append), so re-evaluating stale tracks would make the
        speed/event rules re-fire a long-gone vehicle's last speed every cooldown
        forever. The full set lives in ``self.store`` for end-of-run summaries.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tracked = self.bytetrack.update_with_detections(detections)
        self.last_tracked = tracked

        seen_ids: List[int] = []
        ids = getattr(tracked, "tracker_id", None)
        names = tracked.data.get("class_name") if getattr(tracked, "data", None) else None
        if ids is not None:
            for i, track_id in enumerate(ids):
                if track_id is None:
                    continue
                x1, y1, x2, y2 = tracked.xyxy[i]
                bbox = (float(x1), float(y1), float(x2), float(y2))
                conf = 1.0
                if tracked.confidence is not None:
                    conf = float(tracked.confidence[i])
                vclass = str(names[i]) if names is not None and i < len(names) else None

                ground_point = None
                lane_band = None
                if self.projector is not None:
                    bx, by = bottom_center(bbox)
                    ground_point = self.projector.project(bx, by)
                    lane_band = _assign_lane_band(
                        ground_point[0] - self._contact_bias, self.lane_model_cfg
                    )

                self.store.record(
                    int(track_id),
                    Observation(
                        ts=ts, bbox=bbox, confidence=conf,
                        ground_point=ground_point, lane_band=lane_band,
                        vehicle_class=vclass,
                    ),
                )
                seen_ids.append(int(track_id))
        return [self.store.get(tid) for tid in sorted(set(seen_ids))]
