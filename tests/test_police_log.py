"""Tests for police sighting persistence + the session finalize logic.

The CLIP model itself (analyze/police_classifier.PoliceClassifier) needs torch +
weights and is validated separately; here we cover the dependency-free pieces:
the SQLite log, the pure score aggregation/threshold, and the PoliceSession's
sample -> depart -> finalize bookkeeping (driven with a fake tagger + tracks).
"""

from __future__ import annotations

import pytest

from traffic_logger.events.police_log import (
    PoliceLog,
    Sighting,
    aggregate_score,
    decide_police,
)


def _sighting(ts, *, is_police=True, speeding=False, conf=0.9):
    return Sighting(ts=ts, first_ts=ts - 1.0, track_id=int(ts), direction="left_to_right",
                    confidence=conf, is_police=is_police, max_speed_rel=1.0,
                    max_speed_kmh=None, was_speeding=speeding)


def test_aggregate_and_decide():
    # aggregate_score is now the MEAN across a track's samples.
    assert aggregate_score([]) == 0.0
    assert aggregate_score([0.1, 0.8, 0.3]) == pytest.approx(0.4)
    assert decide_police([], 0.5) == (False, 0.0)
    # Mean below threshold -> not police (confidence is the mean).
    assert decide_police([0.4, 0.49], 0.5) == (False, pytest.approx(0.445))
    # Mean over threshold with >=2 samples -> police.
    assert decide_police([0.4, 0.7], 0.5) == (True, pytest.approx(0.55))
    # A single high frame is NOT enough (needs min_samples agreement).
    assert decide_police([0.9], 0.5) == (False, 0.9)
    assert decide_police([0.9, 0.9], 0.5, min_samples=2) == (True, pytest.approx(0.9))


def test_log_window_and_filters():
    with PoliceLog(":memory:") as log:
        log.add(_sighting(1000))
        log.add(_sighting(2000, speeding=True))
        log.add(_sighting(3000))
        assert log.count(0, 5000) == 3
        assert log.count(1500, 5000) == 2
        speeding = [s for s in log.in_window(0, 5000) if s.was_speeding]
        assert len(speeding) == 1 and speeding[0].ts == 2000
        # police_only filter excludes non-police rows.
        log.add(_sighting(4000, is_police=False))
        assert log.count(0, 5000, police_only=True) == 3


# --- PoliceSession finalize logic (fake tagger + tracks, no torch) -----------

class _FakeTagger:
    def __init__(self, results):
        self._results = results          # track_id -> list[score]
        self.submitted = []
        self.stopped = False

    def submit(self, tid, crop):
        self.submitted.append((tid, crop.shape))

    def pop(self, tid):
        return self._results.get(tid, [])

    def stop(self):
        self.stopped = True


class _FakeTrack:
    def __init__(self, tid, *, age, bbox, gh, direction="left_to_right"):
        self.track_id = tid
        self.age = age
        self.latest_bbox = bbox
        self._gh = gh
        self._dir = direction

    def direction(self):
        return self._dir

    def ground_point_history(self):
        return self._gh


class _FakeStore:
    def __init__(self, tracks):
        self._t = {t.track_id: t for t in tracks}

    def get(self, tid):
        return self._t.get(tid)


class _Cfg:
    def __init__(self, police):
        self.events = {"police": police}
        self.analysis = {"speed_window_seconds": 0.5}
        self.calibration = {}
        self.recording = {"segment_index_path": ":memory:"}


class _FakeConfirmer:
    def __init__(self):
        self.jobs = []
        self.stopped = False

    def submit(self, job):
        self.jobs.append(job)

    def stop(self):
        self.stopped = True


class _FakeClassifier:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def score(self, crop):
        self.calls += 1
        return self.value


def _session(results, **police):
    np = pytest.importorskip("numpy")
    from traffic_logger.analyze.police_classifier import PoliceSession

    police.setdefault("confidence_threshold", 0.5)
    police.setdefault("min_track_age", 1)
    police.setdefault("sample_count", 2)
    police.setdefault("sample_every_frames", 1)
    cfg = _Cfg(police)
    tagger = _FakeTagger(results)
    log = PoliceLog(":memory:")
    sess = PoliceSession(cfg, tagger, log, inference_fps=1.0)
    return sess, tagger, log, np


def test_session_finalizes_positive_sighting():
    gh = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5)]
    track = _FakeTrack(7, age=5, bbox=(0, 0, 100, 100), gh=gh)
    sess, tagger, log, np = _session({7: [0.6, 0.92]})
    frame = np.zeros((480, 704, 3), dtype=np.uint8)
    store = _FakeStore([track])
    # Two observed frames -> two samples submitted.
    sess.observe_frame(frame, [track], processed=0, wall_now=1000.0)
    sess.observe_frame(frame, [track], processed=1, wall_now=1000.5)
    assert len(tagger.submitted) == 2
    # Track departs (gone well beyond finalize_after) -> one police sighting.
    sess.finalize_departed(processed=200, store=store, meters_per_unit=None)
    rows = log.in_window(0, 1e12, police_only=True)
    assert len(rows) == 1
    assert rows[0].track_id == 7
    assert rows[0].confidence == pytest.approx(0.76)  # mean of [0.6, 0.92]
    assert rows[0].direction == "left_to_right"
    assert rows[0].max_speed_rel == pytest.approx(1.0)  # 0.5 units / 0.5 s
    assert rows[0].was_speeding is False


def test_session_skips_below_threshold():
    track = _FakeTrack(3, age=5, bbox=(0, 0, 100, 100), gh=[(0.0, 0.0, 0.0)])
    sess, tagger, log, np = _session({3: [0.2, 0.3]})
    frame = np.zeros((480, 704, 3), dtype=np.uint8)
    sess.observe_frame(frame, [track], processed=0, wall_now=1000.0)
    sess.finalize_departed(processed=200, store=_FakeStore([track]), meters_per_unit=None)
    assert log.count(0, 1e12) == 0


def test_session_marks_speeding_from_event():
    class _FE:
        event_types = ["relative_speeding"]
        primary_track_id = 9
        track_ids = [9]

    track = _FakeTrack(9, age=5, bbox=(0, 0, 100, 100), gh=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.2)])
    sess, tagger, log, np = _session({9: [0.8, 0.85]})
    frame = np.zeros((480, 704, 3), dtype=np.uint8)
    sess.observe_frame(frame, [track], processed=0, wall_now=1000.0)
    sess.note_event(_FE())
    sess.finalize_departed(processed=200, store=_FakeStore([track]), meters_per_unit=None)
    rows = log.in_window(0, 1e12, police_only=True)
    assert len(rows) == 1 and rows[0].was_speeding is True


def test_session_skips_small_crops():
    track = _FakeTrack(5, age=5, bbox=(0, 0, 20, 20), gh=[(0.0, 0.0, 0.0)])  # < min_crop_px
    sess, tagger, log, np = _session({5: [0.99]}, min_crop_px=40)
    frame = np.zeros((480, 704, 3), dtype=np.uint8)
    sess.observe_frame(frame, [track], processed=0, wall_now=1000.0)
    assert tagger.submitted == []


# --- Cascade: sub pre-filter -> 4K confirm -----------------------------------

def test_session_escalation_gate():
    np = pytest.importorskip("numpy")
    frame = np.zeros((480, 704, 3), dtype=np.uint8)
    store = _FakeStore([])

    # Promising sub score (mean >= escalate 0.35) -> escalated to the confirmer.
    hot = _FakeTrack(11, age=5, bbox=(0, 0, 100, 100), gh=[(0.0, 0.0, 0.0)])
    sess, tagger, log, _ = _session({11: [0.5, 0.6]}, confirm_4k=True, escalate_threshold=0.35)
    sess.sub_size = (704, 480)
    sess.confirmer = _FakeConfirmer()
    sess.observe_frame(frame, [hot], processed=0, wall_now=1000.0)
    sess.finalize_departed(processed=200, store=store, meters_per_unit=None)
    assert len(sess.confirmer.jobs) == 1
    assert sess.confirmer.jobs[0].track_id == 11
    assert log.count(0, 1e12) == 0  # nothing logged until the 4K stage confirms

    # Cold sub score (mean < gate) -> rejected cheaply, never escalated.
    cold = _FakeTrack(12, age=5, bbox=(0, 0, 100, 100), gh=[(0.0, 0.0, 0.0)])
    sess2, _, log2, _ = _session({12: [0.1, 0.2]}, confirm_4k=True, escalate_threshold=0.35)
    sess2.sub_size = (704, 480)
    sess2.confirmer = _FakeConfirmer()
    sess2.observe_frame(frame, [cold], processed=0, wall_now=1000.0)
    sess2.finalize_departed(processed=200, store=store, meters_per_unit=None)
    assert sess2.confirmer.jobs == []
    assert log2.count(0, 1e12) == 0


def _confirmer(value):
    from traffic_logger.analyze.police_classifier import Police4KConfirmer

    cfg = _Cfg({"confidence_threshold": 0.55})
    log = PoliceLog(":memory:")
    conf = Police4KConfirmer(cfg, _FakeClassifier(value), log, (704, 480))
    return conf, log


def test_4k_confirmer_logs_on_confirm():
    np = pytest.importorskip("numpy")
    from traffic_logger.analyze.police_classifier import Confirm4KJob

    conf, log = _confirmer(0.8)
    conf._crop_4k = lambda wt, bbox: np.zeros((60, 60, 3), dtype=np.uint8)  # stub the ring pull
    job = Confirm4KJob(track_id=21, direction="left_to_right", ts=1000.0, first_ts=999.0,
                       max_speed_rel=1.0, max_speed_kmh=None, was_speeding=True, sub_conf=0.5,
                       candidates=[(1000.0, (0, 0, 50, 50)), (1000.5, (0, 0, 50, 50))])
    conf._process(job)
    rows = log.in_window(0, 1e12, police_only=True)
    assert len(rows) == 1
    assert rows[0].track_id == 21 and rows[0].was_speeding is True
    assert rows[0].confidence == pytest.approx(0.8)  # 4K score replaces the sub score
    assert conf.confirmed == 1


def test_4k_confirmer_defers_until_segment_indexed():
    from traffic_logger.analyze.police_classifier import Confirm4KJob

    conf, _ = _confirmer(0.8)
    assert conf.ready_delay == pytest.approx(16.0)  # default segment_seconds 10 + 6
    job = Confirm4KJob(track_id=1, direction=None, ts=0, first_ts=0, max_speed_rel=None,
                       max_speed_kmh=None, was_speeding=False, sub_conf=0.6,
                       candidates=[(1000.0, (0, 0, 9, 9)), (1001.0, (0, 0, 9, 9))])
    # Not runnable until the latest candidate frame's segment is indexed.
    assert conf._ready_at(job) == pytest.approx(1017.0)


def test_4k_confirmer_rejects_low_4k_score():
    np = pytest.importorskip("numpy")
    from traffic_logger.analyze.police_classifier import Confirm4KJob

    conf, log = _confirmer(0.1)  # 4K disagrees with the sub pre-filter
    conf._crop_4k = lambda wt, bbox: np.zeros((60, 60, 3), dtype=np.uint8)
    job = Confirm4KJob(track_id=22, direction=None, ts=1000.0, first_ts=999.0,
                       max_speed_rel=None, max_speed_kmh=None, was_speeding=False, sub_conf=0.6,
                       candidates=[(1000.0, (0, 0, 50, 50))])
    conf._process(job)
    assert log.count(0, 1e12) == 0 and conf.rejected == 1
