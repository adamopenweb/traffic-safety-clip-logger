"""Mid-tier events save a still image only (no video clip)."""

from __future__ import annotations

from pathlib import Path

from traffic_logger.config import load_config
from traffic_logger.events.ring_clip_exporter import RingClipExporter

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class _FakeEvent:
    def __init__(self, eid="abcd1234", etype="relative_speeding", track_id=1):
        self.event_id = eid
        self.event_type = etype
        self.primary_track_id = track_id
        self.track_ids = [track_id]


def _exporter(overlay_buffer=None, annotate=False):
    cfg = load_config(CONFIG_DIR / "config.dev.yaml")
    cfg.raw["events"]["annotate_clips"] = annotate
    return RingClipExporter(cfg, margin=3.0, overlay_buffer=overlay_buffer)


def test_screenshot_is_ready_immediately_and_routes_to_still(monkeypatch):
    exp = _exporter()
    calls = []
    monkeypatch.setattr(exp, "_export_screenshot", lambda fe, wt: calls.append((fe, wt)))
    monkeypatch.setattr(exp, "_export", lambda fe, wt, snapshots=None: calls.append(("CLIP", wt)))

    now = 1_000.0
    ev = _FakeEvent()
    exp.enqueue_screenshot(ev, now)   # mid-tier image: no post-roll wait
    exp._drain(now)                   # same instant -> fires (ready_at == trigger)
    assert calls == [(ev, now)]       # routed to the still path, not a clip
    assert exp.exported == 1


def test_screenshot_skips_overlay_capture(monkeypatch):
    from traffic_logger.events.overlay_buffer import OverlayBuffer

    buf = OverlayBuffer(capacity_seconds=100.0)
    buf.set_frame_size(704, 480)
    exp = _exporter(overlay_buffer=buf, annotate=True)
    grabbed = []
    monkeypatch.setattr(exp, "_grab_overlay", lambda wt: grabbed.append(wt) or [])
    monkeypatch.setattr(exp, "_export_screenshot", lambda fe, wt: None)

    exp.enqueue_screenshot(_FakeEvent(), 100.0)
    exp._drain(200.0)
    assert grabbed == []   # no overlay slice for a still -- there's no clip to annotate
