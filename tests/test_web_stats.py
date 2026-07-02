"""Phase A: speed-log stats aggregation (DB read + pure rollups)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from traffic_logger.web import stats

TZ = ZoneInfo("America/Toronto")


def _ts(y, mo, d, h=12, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=TZ).timestamp()


@pytest.fixture
def viols():
    """Violations across a few local days/hours -- the list the aggregations consume
    (the dashboard builds it from passes via stats.passes_to_violations)."""
    # Violation(ts, speed_kmh, direction, vehicle_type, clipped)
    return [
        stats.Violation(_ts(2026, 6, 18, 8), 58.0, "left_to_right", "car", False),
        stats.Violation(_ts(2026, 6, 18, 8), 72.0, "left_to_right", "car", True),
        stats.Violation(_ts(2026, 6, 18, 17), 81.0, "right_to_left", "truck", True),
        stats.Violation(_ts(2026, 6, 20, 9), 66.0, "left_to_right", "car", False),
        stats.Violation(_ts(2026, 6, 20, 9), 90.0, "right_to_left", "motorcycle", True),
    ]


def test_passes_to_violations_filters_and_maps():
    passes = [
        stats.Pass(_ts(2026, 6, 20, 9), 52.0, "left_to_right", "car"),   # under gate -> out
        stats.Pass(_ts(2026, 6, 20, 9), 58.0, "left_to_right", "car"),   # in, not clipped
        stats.Pass(_ts(2026, 6, 20, 9), 80.0, "right_to_left", "truck"),  # in, clipped
        stats.Pass(_ts(2026, 6, 20, 9), None, "left_to_right", "car"),   # no speed -> out
    ]
    v = stats.passes_to_violations(passes, over_limit_kmh=55, clip_threshold=70)
    assert [x.speed_kmh for x in v] == [58.0, 80.0]
    assert [x.clipped for x in v] == [False, True]


def test_summarize(viols):
    s = stats.summarize(viols, speed_limit=50, fast_threshold=70)
    assert s["count"] == 5
    assert s["max_kmh"] == 90.0
    assert s["over_fast"] == 3        # 72, 81, 90
    assert s["clipped"] == 3
    assert s["over_limit_pct"] == 100.0  # all 5 are >50


def test_summarize_empty():
    s = stats.summarize([])
    assert s["count"] == 0 and s["max_kmh"] is None and s["over_fast"] == 0


def test_daily_series_gap_filled(viols):
    series = stats.daily_series(viols, TZ, days=3, now_ts=_ts(2026, 6, 20, 23),
                                fast_threshold=70)
    assert [d["date"] for d in series] == ["2026-06-18", "2026-06-19", "2026-06-20"]
    assert series[0]["count"] == 3 and series[0]["over_fast"] == 2  # 58/72/81
    assert series[1]["count"] == 0 and series[1]["max_kmh"] is None  # gap day
    assert series[2]["count"] == 2 and series[2]["max_kmh"] == 90.0


def test_hourly_histogram(viols):
    h = stats.hourly_histogram(viols, TZ)
    assert len(h) == 24
    assert h[8]["count"] == 2     # two 08:00 violations on the 18th
    assert h[9]["count"] == 2     # two 09:00 on the 20th
    assert h[17]["count"] == 1
    assert h[0]["count"] == 0 and h[0]["avg_kmh"] is None


def test_speed_distribution(viols):
    dist = {d["bucket"]: d["count"] for d in
            stats.speed_distribution(viols)}
    assert dist["55-59"] == 1   # 58
    assert dist["65-69"] == 1   # 66
    assert dist["70-74"] == 1   # 72
    assert dist["80-84"] == 1   # 81
    assert dist["85+"] == 1     # 90


def test_read_passes_and_volume(tmp_path):
    from traffic_logger.events.store import PassRecord, TrafficStore
    path = str(tmp_path / "traffic.sqlite")
    s = TrafficStore(path)
    s.start_session("x", 1000.0)
    # 52 is over the posted 50 but under the buffered 55 -> NOT counted as speeding.
    rows = [(1, 45.0), (2, 52.0), (3, 58.0), (4, 80.0), (5, None)]
    for tid, kmh in rows:
        s.upsert_pass(PassRecord(session_id="x", track_id=tid,
                                 first_ts=_ts(2026, 6, 20, 9), last_ts=_ts(2026, 6, 20, 9),
                                 direction="left_to_right", vehicle_type="car",
                                 steady_speed_raw=kmh, steady_valid=kmh is not None))
    s.close()

    passes = stats.read_passes(path, since_ts=_ts(2026, 6, 20, 0))
    assert len(passes) == 5
    vol = stats.volume_summary(passes, over_threshold=55)
    assert vol["total"] == 5
    assert vol["measured"] == 4            # the None-speed one excluded
    assert vol["over_limit"] == 2          # 58, 80 (>=55); 52 buffered out
    assert vol["over_kmh"] == 55
    assert vol["over_limit_pct"] == round(100 * 2 / 4, 1)


def test_read_passes_missing_or_empty_is_empty(tmp_path):
    assert stats.read_passes("") == []
    assert stats.read_passes(str(tmp_path / "nope.sqlite")) == []
    assert stats.volume_summary([])["total"] == 0


def test_read_passes_surfaces_speed_only_when_valid(tmp_path):
    from traffic_logger.events.store import PassRecord, TrafficStore
    path = str(tmp_path / "traffic.sqlite")
    s = TrafficStore(path)
    s.start_session("x", 1000.0)
    base = _ts(2026, 6, 20, 9)
    # Plausibility is decided at write time (steady_valid). read_passes just trusts the
    # flag: an invalid pass keeps its raw speed in the DB but reads as no-speed, so it
    # still counts as traffic yet never skews the speed stats.
    s.upsert_pass(PassRecord(session_id="x", track_id=1, first_ts=base, last_ts=base + 0.7,
                             direction="left_to_right", vehicle_type="car",
                             steady_speed_raw=75.0, steady_valid=True))
    s.upsert_pass(PassRecord(session_id="x", track_id=2, first_ts=base, last_ts=base + 1.25,
                             direction="left_to_right", vehicle_type="truck",
                             steady_speed_raw=100.0, steady_valid=False,
                             steady_invalid_reason="implied_distance"))
    s.close()
    p = stats.read_passes(path, since_ts=base - 1)
    vals = [x.steady_kmh for x in p]
    assert len(p) == 2 and 75.0 in vals and None in vals   # invalid -> no speed, still a row
    assert stats.volume_summary(p)["total"] == 2           # denominator intact
    assert stats.summarize(stats.passes_to_violations(p, 55, 70))["max_kmh"] == 75.0


def test_first_pass_ts(tmp_path):
    from traffic_logger.events.store import PassRecord, TrafficStore
    assert stats.first_pass_ts("") is None
    assert stats.first_pass_ts(str(tmp_path / "none.sqlite")) is None
    path = str(tmp_path / "traffic.sqlite")
    s = TrafficStore(path)
    s.start_session("x", 1.0)
    s.upsert_pass(PassRecord(session_id="x", track_id=1, first_ts=100.0, last_ts=110.0))
    s.upsert_pass(PassRecord(session_id="x", track_id=2, first_ts=200.0, last_ts=205.0))
    s.close()
    assert stats.first_pass_ts(path) == 110.0   # MIN(last_ts)


def test_stats_endpoint_clamps_numerator_to_pass_floor(tmp_path):
    import time

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from traffic_logger.events.store import PassRecord, TrafficStore
    from traffic_logger.web.app import WebSettings, create_app

    now = time.time()
    sl = tmp_path / "speed.sqlite"
    conn = sqlite3.connect(sl)
    conn.execute("CREATE TABLE speed_events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "ts REAL NOT NULL, speed_kmh REAL NOT NULL, direction TEXT, "
                 "clipped INTEGER NOT NULL, vehicle_type TEXT)")
    conn.executemany("INSERT INTO speed_events (ts, speed_kmh, clipped) VALUES (?,?,0)",
                     [(now - 90000, 60.0), (now - 3600, 70.0)])  # ~25h ago + 1h ago
    conn.commit()
    conn.close()

    tdb = str(tmp_path / "traffic.sqlite")
    ts = TrafficStore(tdb)
    ts.start_session("x", now - 7200)
    ts.upsert_pass(PassRecord(session_id="x", track_id=1, first_ts=now - 7100.7,
                              last_ts=now - 7100, steady_speed_raw=60.0,
                              steady_valid=True))  # floor ~2h ago, 0.7s track
    ts.close()

    settings = WebSettings(
        events_dir=tmp_path / "events", speed_log_path=str(sl), traffic_db_path=tdb,
        timezone="America/Toronto", access_token="t", session_secret="s",
        cookie_secure=False)
    c = TestClient(create_app(settings))
    c.get("/k/t")
    d = c.get("/api/stats?days=30").json()
    assert d["clamped"] is True
    # the 25h-ago violation predates the pass floor -> excluded; only the 1h-ago one
    assert d["summary"]["count"] == 1
    assert d["volume"]["total"] == 1


def test_vehicle_breakdown(viols):
    vb = stats.vehicle_breakdown(viols)
    assert vb[0] == {"vehicle_type": "car", "count": 3}  # most common first
    types = {x["vehicle_type"] for x in vb}
    assert types == {"car", "truck", "motorcycle"}
