"""Tests for the event clip exporter."""

from __future__ import annotations

from pathlib import Path

from traffic_logger.events import exporter


class _Seg:
    def __init__(self, path, start_ts, end_ts=None):
        self.path = path
        self.start_ts = start_ts
        self.end_ts = start_ts + 10.0 if end_ts is None else end_ts


def _run(*spans):
    """SegmentRecord-likes from (start, end) spans."""
    return [_Seg(f"s{i}.mp4", a, b) for i, (a, b) in enumerate(spans)]


def test_clamp_no_gap_keeps_full_window():
    segs = _run((100.0, 110.0), (110.0, 120.0), (120.0, 130.0))
    r = exporter.clamp_to_trigger_run(segs, abs_start=105.0, abs_end=125.0, trigger_ts=115.0)
    assert r is not None
    assert (r.start_ts, r.end_ts) == (105.0, 125.0)
    assert r.truncated_pre == 0.0 and r.truncated_post == 0.0
    assert len(r.segments) == 3


def test_clamp_gap_before_trigger_truncates_pre():
    # gap between 110 and 118 (>0.5s); trigger 120 is in the LATER run.
    segs = _run((100.0, 110.0), (118.0, 128.0))
    r = exporter.clamp_to_trigger_run(segs, abs_start=110.0, abs_end=126.0, trigger_ts=120.0)
    assert r is not None
    assert r.start_ts == 118.0 and r.end_ts == 126.0          # clamped to the later run's start
    assert r.truncated_pre == 8.0 and r.truncated_post == 0.0
    assert [s.start_ts for s in r.segments] == [118.0]


def test_clamp_gap_after_trigger_truncates_post():
    # trigger 105 is in the EARLIER run; a gap follows at 110->119.
    segs = _run((100.0, 110.0), (119.0, 129.0))
    r = exporter.clamp_to_trigger_run(segs, abs_start=102.0, abs_end=125.0, trigger_ts=105.0)
    assert r is not None
    assert r.start_ts == 102.0 and r.end_ts == 110.0          # clamped to the earlier run's end
    assert r.truncated_pre == 0.0 and r.truncated_post == 15.0


def test_clamp_trigger_in_gap_returns_none():
    segs = _run((100.0, 110.0), (119.0, 129.0))
    assert exporter.clamp_to_trigger_run(segs, 102.0, 125.0, trigger_ts=114.0) is None


def test_clamp_empty_returns_none():
    assert exporter.clamp_to_trigger_run([], 0.0, 10.0, trigger_ts=5.0) is None


def test_concat_list_uses_absolute_paths(tmp_path, monkeypatch):
    """The concat demuxer resolves relative entries against the LIST file's dir
    (a temp dir), so segment paths must be written absolute — else a relative
    ring path (e.g. data/ring/...) isn't found at export time."""
    segs = [_Seg("data/ring/a.mp4", 100.0), _Seg("data/ring/b.mp4", 110.0)]
    captured = {}

    def fake_run(cmd):
        list_path = cmd[cmd.index("-i") + 1]
        captured["content"] = Path(list_path).read_text(encoding="utf-8")

    monkeypatch.setattr(exporter, "_run_ffmpeg", fake_run)
    exporter.export_from_segments(segs, 105.0, 115.0, tmp_path / "out.mp4")

    lines = [ln for ln in captured["content"].splitlines() if ln.startswith("file ")]
    assert len(lines) == 2
    for ln in lines:
        p = ln[len("file '"):-1]  # strip: file '...'
        assert Path(p).is_absolute(), f"concat entry not absolute: {p}"


def test_copy_mode_streamcopies_else_reencodes(tmp_path, monkeypatch):
    """copy=True stream-copies (-c copy, no re-encode); default re-encodes H.264."""
    segs = [_Seg("data/ring/a.mp4", 100.0)]
    cmds = {}
    monkeypatch.setattr(exporter, "_run_ffmpeg", lambda cmd: cmds.update(last=cmd))

    exporter.export_from_segments(segs, 105.0, 109.0, tmp_path / "c.mp4", copy=True)
    assert "copy" in cmds["last"] and "libx264" not in cmds["last"]
    assert cmds["last"][cmds["last"].index("-c") + 1] == "copy"

    exporter.export_from_segments(segs, 105.0, 109.0, tmp_path / "e.mp4")
    assert "libx264" in cmds["last"] and "copy" not in cmds["last"]
    assert "-crf" not in cmds["last"]                 # no crf -> x264 default


def test_crf_added_when_set_and_ignored_on_copy(tmp_path, monkeypatch):
    segs = [_Seg("data/ring/a.mp4", 100.0)]
    cmds = {}
    monkeypatch.setattr(exporter, "_run_ffmpeg", lambda cmd: cmds.update(last=cmd))

    exporter.export_from_segments(segs, 105.0, 109.0, tmp_path / "e.mp4", crf=30)
    assert cmds["last"][cmds["last"].index("-crf") + 1] == "30"

    # copy can't take a crf -> it must not appear even if passed
    exporter.export_from_segments(segs, 105.0, 109.0, tmp_path / "c.mp4", copy=True, crf=30)
    assert "-crf" not in cmds["last"]


def test_clip_window_geometry():
    w = exporter.clip_window(trigger_ts=1000.0, pre_roll=10.0, post_roll=20.0)
    assert (w.start, w.trigger, w.end) == (990.0, 1000.0, 1020.0)
    assert w.duration == 30.0
