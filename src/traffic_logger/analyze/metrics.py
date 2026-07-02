"""Relative speed estimation and per-direction rolling baselines.

Speed is measured in *normalized units per second* in the calibrated road plane
(no km/h for the MVP). For each track we smooth the projected ground-point path
over a short window; per direction we keep a time-windowed pool of speed samples
and expose percentiles (median, 85/90/95/97th) plus the percentile rank of an
arbitrary speed. The relative-speeding rule (M4) compares a track's speed to the
baseline for its direction.

All pure Python + math — unit-testable without the CV stack.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

# (ts, gx, gy) projected ground-point samples.
GroundHistory = List[Tuple[float, float, float]]


KMH_PER_MS = 3.6


def ground_span(ground_history: GroundHistory) -> float:
    """Diagonal extent a track covers on the ground over its whole history.

    The bounding-box of its (across, along) ground positions, as a single
    magnitude. ~0 for a vehicle that never moves (a parked car across the street);
    approaches the road's normalized extent for one that traverses the frame. Used
    to keep stationary vehicles out of the annotation overlay. <2 points -> 0.0.
    """
    if len(ground_history) < 2:
        return 0.0
    xs = [p[1] for p in ground_history]
    ys = [p[2] for p in ground_history]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def track_speed(
    ground_history: GroundHistory,
    window_seconds: float = 0.5,
    scale: Tuple[float, float] = (1.0, 1.0),
) -> Optional[float]:
    """Smoothed speed from a track's ground-point path.

    Uses the endpoints of the most recent ``window_seconds`` of samples (falling
    back to the last two points), which smooths per-frame jitter. Returns None
    when there isn't enough history or time elapsed.

    ``scale`` multiplies the (across, along) displacement before taking its
    magnitude. With the default (1, 1) the result is normalized units/sec
    (relative speed). Pass the real meters-per-unit spans to get **m/s**.
    """
    if len(ground_history) < 2:
        return None
    t_last = ground_history[-1][0]
    window = [p for p in ground_history if t_last - p[0] <= window_seconds]
    if len(window) < 2:
        window = ground_history[-2:]
    t0, x0, y0 = window[0]
    t1, x1, y1 = window[-1]
    dt = t1 - t0
    if dt <= 0:
        return None
    dist = math.hypot((x1 - x0) * scale[0], (y1 - y0) * scale[1])
    return dist / dt


def speed_kmh(
    ground_history: GroundHistory,
    window_seconds: float,
    scale: Tuple[float, float],
) -> Optional[float]:
    """Track speed in km/h, given the real meters-per-unit ``scale``."""
    ms = track_speed(ground_history, window_seconds, scale=scale)
    return ms * KMH_PER_MS if ms is not None else None


def across_speed_factor(ground_history: GroundHistory, calibration_cfg: dict) -> float:
    """Per-track km/h correction for across-road perspective non-uniformity.

    The road-quad homography isn't perfectly metric, so a real along-road metre
    maps to fewer ground units in the far lane than the near lane -- measured
    speed reads progressively low with distance. GPS drive-bys pinned the ratio
    (measured/true) at two across-road positions (``gx``); we interpolate the
    ratio at a track's mean ``gx`` and divide it out. Returns 1.0 when the
    correction is disabled or unconfigured. Pure: unit-tested without the CV stack.
    """
    cfg = (calibration_cfg or {}).get("speed_across_correction") or {}
    if not cfg.get("enabled") or not ground_history:
        return 1.0
    try:
        ng, nf = float(cfg["near_gx"]), float(cfg["near_factor"])
        fg, ff = float(cfg["far_gx"]), float(cfg["far_factor"])
    except (KeyError, TypeError, ValueError):
        return 1.0
    if ng == fg:
        return 1.0
    gx = sum(p[1] for p in ground_history) / len(ground_history)
    lo, hi = (fg, ng) if fg < ng else (ng, fg)
    g = min(max(gx, lo), hi)  # clamp into the calibrated range (no wild extrapolation)
    factor = ff + (nf - ff) * (g - fg) / (ng - fg)
    return 1.0 / factor if factor > 0 else 1.0


def speed_kmh_calibrated(
    ground_history: GroundHistory,
    window_seconds: float,
    scale: Tuple[float, float],
    calibration_cfg: dict,
) -> Optional[float]:
    """``speed_kmh`` with the across-road perspective correction applied."""
    v = speed_kmh(ground_history, window_seconds, scale)
    if v is None:
        return None
    return v * across_speed_factor(ground_history, calibration_cfg)


def steady_speed_kmh(
    ground_history: GroundHistory,
    scale: Tuple[float, float],
    calibration_cfg: dict,
    trim_frac: float = 0.1,
    jump_ratio: float = 1.5,
    max_span: float = 1.6,
) -> Optional[float]:
    """Robust 'steady' km/h: trimmed end-to-end displacement over the track / time.

    The short rolling-window :func:`speed_kmh` jitters and *peaks* well above a
    vehicle's true speed (tracking/projection noise), so triggering a speed gate
    on it flags steady ~50 km/h cars as 56+. This averages over the whole observed
    track (trimming the noisy entry/exit frames), matching the GPS-validated e2e
    metric. Returns None until the track has enough history to judge.

    Guards against an **endpoint detection jump**: a bad box that leaps on the first or
    last retained frame inflates the end-to-end displacement, so the car looks far
    faster than it was (a 0.6s track once read 111 km/h this way). We cross-check the
    e2e against the MEDIAN per-frame speed, which a single jump can't move; if the e2e
    runs more than ``jump_ratio`` above it, an endpoint jumped, so we use the robust
    median instead -- recovering the car's true speed rather than the artifact (and not
    losing a genuinely fast car: clean/fast tracks have even steps, so e2e ~= median and
    the GPS-validated value is untouched). A *mid*-track spike never reaches the
    endpoints, so the e2e already ignores it and the two agree. Pure.
    """
    if scale is None or len(ground_history) < 6:
        return None
    k = max(1, int(len(ground_history) * trim_frac))
    retained = ground_history[k:len(ground_history) - k]
    if len(retained) < 2:
        return None
    t0, x0, y0 = retained[0]
    t1, x1, y1 = retained[-1]
    dt = t1 - t0
    if dt <= 0:
        return None
    # Physical impossibility guard: the ground plane is the calibrated road region
    # (unit square), so a real crossing spans ~1 length-unit. A track that appears to
    # traverse more than ``max_span`` of it left the field of view -> a multi-frame
    # detection jump the median test can miss (a 0.9s track read 152 km/h == ~2x the
    # road). Reject rather than report an impossible speed.
    if math.hypot(x1 - x0, y1 - y0) > max_span:
        return None
    dist = math.hypot((x1 - x0) * scale[0], (y1 - y0) * scale[1])
    v = dist / dt * KMH_PER_MS
    if len(retained) >= 4:
        step_speeds = []
        for i in range(len(retained) - 1):
            sdt = retained[i + 1][0] - retained[i][0]
            if sdt > 0:
                sd = math.hypot((retained[i + 1][1] - retained[i][1]) * scale[0],
                                (retained[i + 1][2] - retained[i][2]) * scale[1])
                step_speeds.append(sd / sdt * KMH_PER_MS)
        if step_speeds:
            ss = sorted(step_speeds)
            m = len(ss)
            med_step = ss[m // 2] if m % 2 else (ss[m // 2 - 1] + ss[m // 2]) / 2.0
            if med_step > 0 and v > jump_ratio * med_step:
                v = med_step  # endpoint jump inflated the e2e -> use the robust median
    return v * across_speed_factor(ground_history, calibration_cfg)


def metric_scale(calibration_cfg: dict) -> Optional[Tuple[float, float]]:
    """(across_m, along_m) per normalized unit, or None if not in metric mode.

    Active when ``calibration.units == "meters"``; then ``target_width_units``
    and ``target_length_units`` are the real distances (meters) the calibration
    quad spans across and along the road. Because the normalized plane is the
    unit square, 1 normalized unit == that full span, so meters-per-unit equals
    the span itself.
    """
    if str(calibration_cfg.get("units", "relative")).lower() != "meters":
        return None
    width = float(calibration_cfg.get("target_width_units", 1.0))
    length = float(calibration_cfg.get("target_length_units", 1.0))
    if width <= 0 or length <= 0:
        return None
    return (width, length)


class DirectionSpeedBaseline:
    """Time-windowed pool of speed samples for a single direction."""

    def __init__(self, window_seconds: float) -> None:
        self.window_seconds = float(window_seconds)
        # (ts, speed, track_id)
        self._samples: Deque[Tuple[float, float, Optional[int]]] = deque()

    def add(self, ts: float, speed: float, track_id: Optional[int] = None) -> None:
        self._samples.append((ts, speed, track_id))
        self._prune(ts)

    def _prune(self, now: float) -> None:
        w = self.window_seconds
        while self._samples and now - self._samples[0][0] > w:
            self._samples.popleft()

    @property
    def count(self) -> int:
        """Number of speed samples currently in the window."""
        return len(self._samples)

    @property
    def distinct_tracks(self) -> int:
        """Number of distinct tracks contributing samples in the window."""
        return len({tid for _ts, _s, tid in self._samples if tid is not None})

    def _sorted_speeds(self) -> List[float]:
        return sorted(s for _ts, s, _tid in self._samples)

    def percentile_of(self, speed: float) -> float:
        """Fraction of windowed samples <= ``speed`` (0..1)."""
        speeds = [s for _ts, s, _tid in self._samples]
        if not speeds:
            return 0.0
        return sum(1 for s in speeds if s <= speed) / len(speeds)

    def quantile(self, q: float) -> Optional[float]:
        """Nearest-rank quantile value, or None if there are no samples."""
        speeds = self._sorted_speeds()
        if not speeds:
            return None
        idx = int(math.ceil(q * len(speeds))) - 1
        idx = min(len(speeds) - 1, max(0, idx))
        return speeds[idx]

    def stats(self) -> Dict[str, Optional[float]]:
        return {
            "median": self.quantile(0.50),
            "p85": self.quantile(0.85),
            "p90": self.quantile(0.90),
            "p95": self.quantile(0.95),
            "p97": self.quantile(0.97),
        }


class SpeedEstimator:
    """Holds a :class:`DirectionSpeedBaseline` per camera-relative direction."""

    def __init__(self, window_seconds: float) -> None:
        self.window_seconds = float(window_seconds)
        self._baselines: Dict[str, DirectionSpeedBaseline] = {}

    def _baseline(self, direction: str) -> DirectionSpeedBaseline:
        b = self._baselines.get(direction)
        if b is None:
            b = DirectionSpeedBaseline(self.window_seconds)
            self._baselines[direction] = b
        return b

    def observe(
        self, direction: str, ts: float, speed: float, track_id: Optional[int] = None
    ) -> None:
        self._baseline(direction).add(ts, speed, track_id)

    def percentile(self, direction: str, speed: float) -> float:
        return self._baseline(direction).percentile_of(speed)

    def count(self, direction: str) -> int:
        return self._baseline(direction).count

    def distinct_tracks(self, direction: str) -> int:
        return self._baseline(direction).distinct_tracks

    def median(self, direction: str) -> Optional[float]:
        return self._baseline(direction).quantile(0.5)

    def stats(self, direction: str) -> Dict[str, Optional[float]]:
        return self._baseline(direction).stats()
