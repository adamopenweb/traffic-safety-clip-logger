"""End-to-end offline analyzer test.

Generates a synthetic video with OpenCV, drives ``run_offline`` with a
ScriptedDetector (no YOLO/torch needed), and checks detection counts, track-id
persistence, and that a playable annotated debug video is produced.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HAS_CV = (
    importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("supervision") is not None
    and importlib.util.find_spec("numpy") is not None
)
pytestmark = pytest.mark.skipif(not _HAS_CV, reason="cv2/supervision not installed")

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

W, H, FPS, N = 320, 240, 12, 24


def _make_video(path: Path) -> None:
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))
    for _ in range(N):
        writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    writer.release()


def _scripted_track():
    from traffic_logger.analyze.detector import ScriptedDetector

    # One box drifting right by 6 px/frame -> ByteTrack keeps a single id.
    frames = []
    for i in range(N):
        x = 10 + i * 6
        frames.append([(x, 80, x + 40, 140, 0.9, 2)])
    return ScriptedDetector(frames)


def test_run_offline_detects_tracks_and_writes_debug_video(tmp_path):
    import cv2

    from traffic_logger.analyze.offline import run_offline
    from traffic_logger.config import load_config

    video = tmp_path / "synthetic.mp4"
    _make_video(video)

    cfg = load_config(CONFIG_DIR / "config.dev.yaml")  # inference_fps 12 -> stride 1
    out_dir = tmp_path / "debug"

    summary = run_offline(str(video), cfg, detector=_scripted_track(), output_dir=out_dir)

    assert summary["stub"] is False
    assert summary["resolution"] == [W, H]
    assert summary["stride"] == 1
    assert summary["frames_processed"] == N
    assert summary["detections"] == N          # one box per processed frame
    assert summary["unique_tracks"] >= 1        # ids persist into >=1 track

    # A playable debug video was written.
    debug_path = Path(summary["debug_video"])
    assert debug_path.exists() and debug_path.stat().st_size > 0
    cap = cv2.VideoCapture(str(debug_path))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None
    assert frame.shape[0] == H and frame.shape[1] == W


def test_run_offline_missing_source_raises(tmp_path):
    from traffic_logger.analyze.offline import run_offline
    from traffic_logger.config import load_config

    cfg = load_config(CONFIG_DIR / "config.dev.yaml")
    with pytest.raises(FileNotFoundError):
        run_offline(str(tmp_path / "nope.mp4"), cfg, detector=_scripted_track())


def _speeding_scenario():
    """Four slow vehicles then one obviously-fast one, all driving along the road.

    Vehicles travel along the road's length axis (image y) within a single lane
    (fixed image x), which is how speed/direction are measured once calibrated.
    Returns (ScriptedDetector, n_frames); each vehicle is a separate track.
    """
    from traffic_logger.analyze.detector import ScriptedDetector

    frames = []
    cx = 50  # fixed lane column
    # (dy per frame, num frames); 4 slow vehicles then one ~3x faster. The fast
    # one is kept on-screen long enough (18 frames) to satisfy min_duration on
    # its own — speeding is judged only while a track is actually visible, not by
    # re-evaluating a departed track's frozen last speed.
    plan = [(3, 16)] * 4 + [(9, 18)]
    for dy, nf in plan:
        y = 20
        for _ in range(nf):
            frames.append([(cx - 15, y, cx + 15, y + 20, 0.9, 2)])  # bottom-center (cx, y+20)
            y += dy
        frames.extend([[]] * 3)  # gap so the next vehicle gets a new id
    return ScriptedDetector(frames), len(frames)


def test_run_offline_emits_relative_speeding_candidate(tmp_path):
    from traffic_logger.analyze.offline import run_offline
    from traffic_logger.config import load_config

    detector, n = _speeding_scenario()

    video = tmp_path / "speeding.mp4"
    SPEED_FPS = 10
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), SPEED_FPS, (200, 200))
    for _ in range(n):
        writer.write(np.zeros((200, 200, 3), dtype=np.uint8))
    writer.release()

    cfg = load_config(CONFIG_DIR / "config.dev.yaml")
    cfg.raw["analysis"]["inference_fps"] = SPEED_FPS                    # stride 1
    cfg.raw["calibration"]["source_points"] = [[0, 0], [200, 0], [200, 200], [0, 200]]
    cfg.raw["events"]["aggressiveness"] = 1.0                           # sensitive
    cfg.raw["events"]["relative_speeding"]["min_tracks_for_baseline"] = 3

    summary = run_offline(str(video), cfg, detector=detector, output_dir=tmp_path / "dbg")

    assert summary["calibrated"] is True
    assert summary["unique_tracks"] >= 2
    # Speed estimation produced relative-speeding candidate events...
    assert summary["candidate_events"] >= 1
    # ...and the obviously-fast vehicle sits at the very top of the distribution
    # (top few percent). It need not be exactly 1.0: speeding is judged only
    # while a track is visible, so the baseline is no longer diluted by stale
    # re-evaluations of departed slow vehicles.
    assert max(e["percentile"] for e in summary["events"]) >= 0.95


def test_run_offline_emits_center_lane_pass_candidate(tmp_path):
    from traffic_logger.analyze.detector import ScriptedDetector
    from traffic_logger.analyze.offline import run_offline
    from traffic_logger.config import load_config
    import cv2
    import numpy as np

    # One vehicle driving fast straight down the center turning lane (cx=100 of
    # a 200-wide road -> normalized x 0.5 -> center_turn_lane), travelling along
    # the road (image y increasing).
    cx, N_FRAMES, FPS = 100, 16, 10
    frames = []
    y = 20
    for _ in range(N_FRAMES):
        frames.append([
            (cx - 15, y, cx + 15, y + 20, 0.9, 2),   # center-lane car (nx 0.5)
            (35, y, 65, y + 20, 0.9, 2),             # same-direction travel-lane car (nx 0.25)
        ])
        y += 12

    video = tmp_path / "center.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (200, 200))
    for _ in range(N_FRAMES):
        writer.write(np.zeros((200, 200, 3), dtype=np.uint8))
    writer.release()

    cfg = load_config(CONFIG_DIR / "config.dev.yaml")
    cfg.raw["analysis"]["inference_fps"] = FPS
    cfg.raw["calibration"]["source_points"] = [[0, 0], [200, 0], [200, 200], [0, 200]]
    cfg.raw["events"]["aggressiveness"] = 1.0  # sensitive

    summary = run_offline(str(video), cfg, detector=ScriptedDetector(frames), output_dir=tmp_path / "dbg")

    assert summary["calibrated"] is True
    types = {e["event_type"] for e in summary["events"]}
    assert "center_lane_pass" in types


def test_run_offline_exports_event_clip_with_metadata_and_thumbnail(tmp_path):
    """End-to-end M6: a center-lane event exports a playable mp4 + json + jpg."""
    import json

    import cv2
    import numpy as np

    from traffic_logger.analyze.detector import ScriptedDetector
    from traffic_logger.analyze.offline import run_offline
    from traffic_logger.config import load_config

    cx, N_FRAMES, FPS = 100, 30, 10
    frames = []
    y = 20
    for _ in range(N_FRAMES):
        frames.append([
            (cx - 15, y, cx + 15, y + 40, 0.95, 2),  # center-lane car (nx 0.5)
            (35, y, 65, y + 40, 0.95, 2),            # same-direction travel-lane car (nx 0.25)
        ])
        y += 4  # gentle along-road motion so ByteTrack keeps one id

    video = tmp_path / "center.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (200, 220))
    for _ in range(N_FRAMES):
        writer.write(np.zeros((220, 200, 3), dtype=np.uint8))
    writer.release()

    cfg = load_config(CONFIG_DIR / "config.dev.yaml")
    cfg.raw["analysis"]["inference_fps"] = FPS
    cfg.raw["calibration"]["source_points"] = [[0, 0], [200, 0], [200, 220], [0, 220]]
    cfg.raw["events"]["aggressiveness"] = 1.0

    events_dir = tmp_path / "events"
    summary = run_offline(
        str(video), cfg, detector=ScriptedDetector(frames),
        export_events=True, events_dir=events_dir,
    )

    assert summary["events_exported"] >= 1
    clip = Path(summary["event_clips"][0])
    assert clip.exists() and clip.suffix == ".mp4" and clip.stat().st_size > 0
    # Playable clip.
    cap = cv2.VideoCapture(str(clip))
    ok, _frame = cap.read()
    cap.release()
    assert ok

    # Sidecar json + thumbnail jpg next to the clip, and metadata is consistent.
    meta_path = clip.with_suffix(".json")
    thumb_path = clip.with_suffix(".jpg")
    assert meta_path.exists() and thumb_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["clip_path"] == str(clip)
    assert "center_lane_pass" in meta["event_types"]
    assert meta["primary_track_id"] is not None
    # Folder layout: events/<date>/<event_type>/<file>
    assert clip.parent.name == meta["event_type"]
