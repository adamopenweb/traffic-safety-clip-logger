"""PassRecorder: one row per real drive-by, filtering parked cars and flickers."""

from __future__ import annotations

from traffic_logger.analyze.pass_recorder import PassRecorder
from traffic_logger.analyze.tracker import Observation, TrackStore
from traffic_logger.events.store import TrafficStore

SCALE = (11.46, 14.94)  # metres per normalized unit (across, along)


def _add(store, tid, points, vclass="car"):
    """points: list of (ts, gx, gy)."""
    for ts, gx, gy in points:
        store.record(tid, Observation(
            ts=ts, bbox=(gx * 100, gy * 100, gx * 100 + 40, gy * 100 + 20),
            ground_point=(gx, gy), vehicle_class=vclass))


def _moving():
    # 10 obs travelling along the road (gy 0.18 -> 0.85): real pass, ~mid speed.
    return [(i * 0.1, 0.5, 0.18 + i * 0.074) for i in range(10)]


def _build():
    ts_store = TrackStore()
    _add(ts_store, 1, _moving(), vclass="car")
    _add(ts_store, 2, [(i * 0.1, 0.2, 0.9) for i in range(8)], vclass="truck")  # parked
    _add(ts_store, 3, [(0.0, 0.6, 0.2), (0.1, 0.6, 0.27)])                       # flicker (age<5)
    return ts_store


def _recorder(db, **kw):
    opts = dict(meters_per_unit=SCALE, calibration={}, speeding_gate_kmh=35,
                finalize_after=3, min_age=5, min_ground_span=0.12)
    opts.update(kw)
    return PassRecorder(db, "s1", **opts)


def test_records_only_real_passes():
    ts_store = _build()
    db = TrafficStore(":memory:")
    db.start_session("s1", 1000.0)
    pr = _recorder(db)
    for f in range(10):
        pr.observe_frame([ts_store.get(1), ts_store.get(2), ts_store.get(3)],
                         f, 1000.0 + f * 0.1)
    pr.close(ts_store)

    assert pr.counted == 1 and pr.skipped == 2
    p1 = db.get_pass("s1", 1)
    assert p1 is not None
    assert p1.direction == "left_to_right" and p1.vehicle_type == "car"
    assert p1.steady_speed_kmh is not None and 25 < p1.steady_speed_kmh < 75
    assert p1.was_speeding is True          # ~40 km/h > gate 35
    assert db.get_pass("s1", 2) is None     # parked -> filtered
    assert db.get_pass("s1", 3) is None     # flicker -> filtered


def test_was_speeding_false_under_gate():
    ts_store = _build()
    db = TrafficStore(":memory:")
    db.start_session("s1", 1000.0)
    pr = _recorder(db, speeding_gate_kmh=200)   # nobody clears 200
    for f in range(10):
        pr.observe_frame([ts_store.get(1)], f, 1000.0 + f * 0.1)
    pr.close(ts_store)
    assert db.get_pass("s1", 1).was_speeding is False


def test_finalize_departed_waits_for_the_gap():
    ts_store = _build()
    db = TrafficStore(":memory:")
    db.start_session("s1", 1000.0)
    pr = _recorder(db)
    for f in range(10):
        pr.observe_frame([ts_store.get(1)], f, 1000.0 + f * 0.1)
    pr.finalize_departed(11, ts_store)          # 11 - 9 = 2 <= finalize_after(3): not yet
    assert pr.counted == 0
    pr.finalize_departed(13, ts_store)          # 13 - 9 = 4 > 3: now finalized
    assert pr.counted == 1


def test_no_double_count_on_repeat_finalize():
    ts_store = _build()
    db = TrafficStore(":memory:")
    db.start_session("s1", 1000.0)
    pr = _recorder(db)
    for f in range(10):
        pr.observe_frame([ts_store.get(1)], f, 1000.0 + f * 0.1)
    pr.close(ts_store)
    pr.close(ts_store)                          # idempotent
    assert pr.counted == 1
