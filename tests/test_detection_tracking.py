"""Detector + ByteTrack integration tests (real supervision, no YOLO/torch)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HAS_CV = (
    importlib.util.find_spec("supervision") is not None
    and importlib.util.find_spec("numpy") is not None
)
_HAS_YOLO = importlib.util.find_spec("ultralytics") is not None

pytestmark = pytest.mark.skipif(not _HAS_CV, reason="supervision/numpy not installed")

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _config():
    from traffic_logger.config import load_config

    return load_config(CONFIG_DIR / "config.dev.yaml")


def test_scripted_detector_emits_detections():
    from traffic_logger.analyze.detector import ScriptedDetector

    det = ScriptedDetector([
        [(10, 10, 50, 50, 0.9, 2), (60, 60, 90, 90, 0.8, 7)],
        [],  # empty frame
    ])
    first = det.detect(None)
    assert len(first) == 2
    assert list(first.class_id) == [2, 7]
    second = det.detect(None)
    assert len(second) == 0
    # Exhausted script -> empty detections.
    assert len(det.detect(None)) == 0


def test_build_detector_unknown_type_raises():
    from traffic_logger.analyze.detector import build_detector

    cfg = _config()
    cfg.raw["models"]["detector_type"] = "nonsense-net"
    with pytest.raises(ValueError):
        build_detector(cfg)


def test_build_detector_injection_bypasses_yolo():
    from traffic_logger.analyze.detector import ScriptedDetector, build_detector

    scripted = ScriptedDetector([[]])
    assert build_detector(_config(), detector=scripted) is scripted


@pytest.mark.skipif(_HAS_YOLO, reason="ultralytics present; this checks the missing-dep path")
def test_yolo_detector_missing_dependency():
    from traffic_logger.analyze.detector import (
        MissingDetectorDependency,
        build_detector,
    )

    with pytest.raises(MissingDetectorDependency):
        build_detector(_config())


def test_bytetrack_assigns_and_persists_track_id():
    import supervision as sv

    from traffic_logger.analyze.tracker import LEFT_TO_RIGHT, VehicleTracker
    import numpy as np

    tracker = VehicleTracker(_config())

    # One vehicle drifting slowly right (high frame-to-frame overlap so
    # ByteTrack matches and confirms it).
    confirmed_ids = set()
    for i in range(12):
        x = 10 + i * 6
        det = sv.Detections(
            xyxy=np.array([[x, 50, x + 40, 100]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([2]),
        )
        tracks = tracker.update(det, ts=i / 12.0)
        for t in tracks:
            confirmed_ids.add(t.track_id)

    # A single stable track id should dominate (IDs persist across frames).
    assert len(confirmed_ids) == 1
    track_id = next(iter(confirmed_ids))
    track = tracker.store.get(track_id)
    assert track.age >= 5
    assert track.direction() == LEFT_TO_RIGHT


def test_update_returns_only_current_frame_tracks():
    """A departed track must not keep being returned (else stale speed re-fires).

    Drive a vehicle for several frames, then feed empty frames. Once it's gone,
    update() must stop returning it even though the store still remembers it.
    """
    import numpy as np
    import supervision as sv

    from traffic_logger.analyze.tracker import VehicleTracker

    tracker = VehicleTracker(_config())

    seen_id = None
    for i in range(8):
        x = 10 + i * 6
        det = sv.Detections(
            xyxy=np.array([[x, 50, x + 40, 100]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([2]),
        )
        tracks = tracker.update(det, ts=i / 12.0)
        if tracks:
            seen_id = tracks[0].track_id
    assert seen_id is not None

    # Vehicle leaves: empty detections for longer than the lost-track buffer.
    empty = sv.Detections.empty()
    last_returned = None
    for j in range(8, 60):
        last_returned = tracker.update(empty, ts=j / 12.0)
    assert last_returned == []  # nothing visible now
    # ...but the store still remembers it for the end-of-run summary.
    assert tracker.store.get(seen_id) is not None


def test_contact_bias_shifts_lane_classification():
    """contact_bias offsets the classified across-position toward the far curb."""
    import numpy as np
    import supervision as sv

    from traffic_logger.analyze.project import PerspectiveTransform
    from traffic_logger.analyze.tracker import VehicleTracker

    # Calibrate to the image rectangle so across = x/200 (no swap).
    projector = PerspectiveTransform([(0, 0), (200, 0), (200, 200), (0, 200)], swap_xy=False)

    def latest_band(bias):
        cfg = _config()
        cfg.raw["calibration"]["lane_model"]["contact_bias"] = bias
        tracker = VehicleTracker(cfg, projector=projector)
        for i in range(6):
            cx = 100 + i  # bottom-center across ~0.5 (center band)
            det = sv.Detections(
                xyxy=np.array([[cx - 10, 150, cx + 10, 160]], dtype=float),
                confidence=np.array([0.9]), class_id=np.array([2]),
            )
            tracker.update(det, ts=i / 12.0)
        return tracker.store.active_tracks()[-1].latest_lane_band

    assert latest_band(0.0) == "center_turn_lane"        # across 0.5 -> center
    assert latest_band(0.25) == "travel_lane_a"          # shifted to ~0.25 -> far travel


def test_tracker_assigns_lane_bands_over_time():
    import numpy as np
    import supervision as sv

    from traffic_logger.analyze.project import PerspectiveTransform
    from traffic_logger.analyze.tracker import VehicleTracker

    # Calibrate to the visible image rectangle so normalized x = fraction across.
    projector = PerspectiveTransform([(0, 0), (200, 0), (200, 200), (0, 200)])
    tracker = VehicleTracker(_config(), projector=projector)

    # A vehicle sitting in the center of the road (nx ~ 0.5 -> center_turn_lane),
    # nudging slightly so ByteTrack confirms it.
    last_track_id = None
    for i in range(8):
        cx = 100 + i  # ~middle of the 200px-wide calibrated road
        det = sv.Detections(
            xyxy=np.array([[cx - 15, 80, cx + 15, 120]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([2]),
        )
        tracks = tracker.update(det, ts=i / 12.0)
        if tracks:
            last_track_id = tracks[0].track_id

    track = tracker.store.get(last_track_id)
    bands = [b for _ts, b in track.lane_band_history()]
    assert track.latest_lane_band == "center_turn_lane"
    assert all(b == "center_turn_lane" for b in bands)
    # Ground points were recorded for speed estimation (M4).
    assert len(track.ground_point_history()) == track.age
