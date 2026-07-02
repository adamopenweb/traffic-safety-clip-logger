"""Tests for the live-event -> ring clip scheduling (Milestone 7).

The export of the actual MP4 (export_from_segments) is covered elsewhere; here
we verify the timing logic: an event is only exported once its post-roll has
been recorded, un-ready events stay queued, and the overlay slice for the
annotation is captured eagerly (before the slow serial render) so a burst
backlog can't evict it from the rolling buffer.
"""

from __future__ import annotations

from pathlib import Path

from traffic_logger.config import load_config
from traffic_logger.events.overlay_buffer import OverlayBox, OverlayBuffer
from traffic_logger.events.ring_clip_exporter import RingClipExporter, _median, smooth_offset

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

_KW = dict(center=1.0, radius=0.8, step=0.05, fallback=1.0)
_HIST = [1.0, 1.1, 1.2, 1.05]   # 4 good solves; median 1.075


def test_smooth_offset_trusts_the_per_clip_solve():
    # A sane non-railed solve is used AS-IS (per-clip latency is real) and recorded --
    # NOT flattened to the run's median, which would push boxes off the car.
    chosen, record = smooth_offset(0.75, _HIST, **_KW)
    assert record is True and chosen == 0.75


def test_smooth_offset_trusts_a_non_railed_outlier_too():
    # Even a non-railed solve FAR from the consensus is trusted -- the latency genuinely
    # jitters, and visual checks show the raw solve aligns while the median doesn't.
    # (0.3 is inside the [0.2,1.8] window, just far from the 1.075 median.)
    chosen, record = smooth_offset(0.3, _HIST, **_KW)
    assert record is True and chosen == 0.3


def test_smooth_offset_railed_not_recorded_uses_median():
    # 1.8 == center+radius -> railed at the search edge: unreliable, not recorded.
    chosen, record = smooth_offset(1.8, _HIST, **_KW)
    assert record is False and chosen == _median(_HIST)


def test_smooth_offset_failed_solve_uses_median_then_fallback():
    assert smooth_offset(None, _HIST, **_KW) == (_median(_HIST), False)
    assert smooth_offset(None, [1.0], **dict(_KW, fallback=0.9)) == (0.9, False)  # <min_history


def test_smooth_offset_no_history_uses_solve():
    assert smooth_offset(0.5, [1.0, 1.1], **_KW) == (0.5, True)    # non-railed
    assert smooth_offset(1.8, [1.0, 1.1], **_KW) == (1.8, False)   # railed, no median


def test_smooth_offset_uses_median_fallback_only_when_railed():
    """The median exists only as a fallback for railed/failed solves. A non-railed solve
    is always trusted, so a regime shift tracks instantly (no warm-up). Uses the widened
    window (centre 0.9, radius 1.1) so a low evening offset doesn't rail."""
    from collections import deque

    wide = dict(center=0.9, radius=1.1, step=0.05, fallback=0.9)
    hist = deque([0.9, 0.95, 0.9, 1.0], maxlen=9)  # afternoon consensus
    chosen, record = smooth_offset(0.1, list(hist), **wide)  # evening solve, non-railed
    assert chosen == 0.1 and record is True            # trusted immediately, not the median
    # a railed solve still falls back to the recent median
    chosen2, record2 = smooth_offset(2.0, list(hist), **wide)  # 2.0 == centre+radius
    assert record2 is False and chosen2 == _median(list(hist))


class _FakeEvent:
    def __init__(self, eid="abcd1234", etype="relative_speeding", track_id=1):
        self.event_id = eid
        self.event_type = etype
        self.primary_track_id = track_id
        self.track_ids = [track_id]


def _exporter(margin=3.0, overlay_buffer=None, annotate=False):
    cfg = load_config(CONFIG_DIR / "config.dev.yaml")
    cfg.raw["events"]["pre_roll_seconds"] = 10
    cfg.raw["events"]["post_roll_seconds"] = 20
    cfg.raw["events"]["annotate_clips"] = annotate
    cfg.raw["events"]["annotate_sync_offset_seconds"] = 0.45
    cfg.raw["events"]["annotate_save_overlay"] = False
    return RingClipExporter(cfg, margin=margin, overlay_buffer=overlay_buffer)


def test_export_waits_for_postroll_then_fires(monkeypatch):
    exp = _exporter()
    calls = []
    monkeypatch.setattr(exp, "_export", lambda fe, wt, snapshots=None: calls.append((fe, wt)))

    now = 1_000_000.0
    ev = _FakeEvent()
    exp.enqueue(ev, wall_trigger_ts=now)  # ready_at = now + post(20) + margin(3) = now+23

    exp._drain(now + 10)  # post-roll not recorded yet
    assert calls == []
    exp._drain(now + 24)  # now past ready time
    assert len(calls) == 1
    assert calls[0][0] is ev and calls[0][1] == now
    assert exp.exported == 1


def test_drain_keeps_unready_events(monkeypatch):
    exp = _exporter()
    monkeypatch.setattr(exp, "_export", lambda fe, wt, snapshots=None: None)
    now = 5_000.0
    exp.enqueue(_FakeEvent("e1"), now - 100)  # long past -> ready
    exp.enqueue(_FakeEvent("e2"), now)        # ready in ~23s -> not yet
    exp._drain(now)
    assert len(exp._pending) == 1
    assert exp._pending[0].final_event.event_id == "e2"


def test_export_failure_does_not_break_thread(monkeypatch):
    exp = _exporter()

    def _boom(fe, wt, snapshots=None):
        raise RuntimeError("ring read failed")

    monkeypatch.setattr(exp, "_export", _boom)
    # A failing export is swallowed; the event is still consumed from the queue.
    exp.enqueue(_FakeEvent(), 0.0)
    exp._drain(1e12)
    assert exp._pending == []
    assert exp.exported == 0  # failed, so not counted


def test_background_thread_drains_due_events(monkeypatch):
    import time

    exp = _exporter()
    monkeypatch.setattr(exp, "_export", lambda fe, wt, snapshots=None: None)
    exp.start()
    try:
        exp.enqueue(_FakeEvent("old"), time.time() - 100)  # already past-due
        deadline = time.time() + 3.0
        while time.time() < deadline and exp.exported < 1:
            time.sleep(0.05)
        assert exp.exported >= 1
    finally:
        exp.stop()


def test_eager_capture_survives_slow_serial_render(monkeypatch):
    # Buffer densely covering the windows of three bursty events at t=100/110/120.
    buf = OverlayBuffer(capacity_seconds=1000.0)
    buf.set_frame_size(704, 480)
    t = 85.0
    while t <= 145.0:
        buf.append(t, [OverlayBox(track_id=1, bbox=(0.0, 0.0, 10.0, 10.0))])
        t += 0.5

    exp = _exporter(overlay_buffer=buf, annotate=True)
    for i, trig in enumerate((100.0, 110.0, 120.0)):
        exp.enqueue(_FakeEvent(f"e{i}", track_id=i), trig)

    # Simulate a SLOW render: the first export clears the buffer (as a burst
    # backlog would let snapshots scroll out). Record what each export received.
    captured = []

    def fake_export(fe, wall_trigger, snapshots=None):
        captured.append((wall_trigger, snapshots))
        buf._snaps.clear()  # the rolling buffer "scrolls away" mid-render

    monkeypatch.setattr(exp, "_export", fake_export)
    exp._drain(now=1e9)  # all three are ready

    # Each event still got a non-empty overlay slice despite the buffer being
    # cleared during the first (slow) render -- phase 1 captured every ready item
    # before any render ran.
    assert [wt for wt, _ in captured] == [100.0, 110.0, 120.0]
    assert all(snaps for _wt, snaps in captured)
    assert all(len(snaps) > 10 for _wt, snaps in captured)


def test_capture_skipped_when_annotation_disabled(monkeypatch):
    buf = OverlayBuffer(capacity_seconds=1000.0)
    buf.set_frame_size(704, 480)
    buf.append(100.0, [OverlayBox(track_id=1, bbox=(0.0, 0.0, 1.0, 1.0))])

    exp = _exporter(overlay_buffer=buf, annotate=False)
    exp.enqueue(_FakeEvent("e0", track_id=1), 100.0)
    seen = []
    monkeypatch.setattr(exp, "_export", lambda fe, wt, snapshots=None: seen.append(snapshots))
    exp._drain(now=1e9)
    assert seen == [None]  # annotate off -> no eager capture
