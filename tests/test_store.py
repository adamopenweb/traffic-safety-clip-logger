"""Tests for the unified traffic store (Phase 0).

Focus: the order-independent ``(session_id, track_id)`` upsert merge -- the property
that lets the inline speeding writer and the async police writer share one pass row
without coordinating -- plus event->pass linking and the legacy compatibility views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from traffic_logger.events.store import EventRecord, PassRecord, TrafficStore


def _store() -> TrafficStore:
    return TrafficStore(":memory:")


# --- pass upsert / merge ----------------------------------------------------

def test_upsert_creates_then_returns_stable_id():
    with _store() as s:
        i1 = s.upsert_pass(PassRecord("sess", 5, first_ts=100.0, last_ts=101.0))
        i2 = s.upsert_pass(PassRecord("sess", 5, first_ts=99.0, last_ts=103.0,
                                      vehicle_type="car"))
        assert i1 == i2  # same (session, track) -> same row
        p = s.get_pass("sess", 5)
        assert p.first_ts == 99.0 and p.last_ts == 103.0   # window widened
        assert p.vehicle_type == "car"


def test_track_id_namespaced_by_session():
    with _store() as s:
        a = s.upsert_pass(PassRecord("sessA", 1, first_ts=1.0, last_ts=2.0))
        b = s.upsert_pass(PassRecord("sessB", 1, first_ts=1.0, last_ts=2.0))
        assert a != b  # track_id 1 in two runs are two distinct passes


def test_merge_is_order_independent_speeding_then_police():
    """Speeding writer first, police writer second -> one merged row."""
    with _store() as s:
        s.upsert_pass(PassRecord("s", 7, first_ts=10.0, last_ts=12.0,
                                 vehicle_type="truck", max_speed_kmh=64.0,
                                 was_speeding=True))
        s.upsert_pass(PassRecord("s", 7, first_ts=10.0, last_ts=12.5,
                                 is_police=True, police_confidence=0.81))
        p = s.get_pass("s", 7)
        assert p.was_speeding is True and p.max_speed_kmh == 64.0
        assert p.is_police is True and p.police_confidence == 0.81
        assert p.vehicle_type == "truck"


def test_merge_is_order_independent_police_then_speeding():
    """Reverse arrival order yields the identical merged row."""
    with _store() as s:
        s.upsert_pass(PassRecord("s", 7, first_ts=10.0, last_ts=12.5,
                                 is_police=True, police_confidence=0.81))
        s.upsert_pass(PassRecord("s", 7, first_ts=10.0, last_ts=12.0,
                                 vehicle_type="truck", max_speed_kmh=64.0,
                                 was_speeding=True))
        p = s.get_pass("s", 7)
        assert p.was_speeding is True and p.max_speed_kmh == 64.0
        assert p.is_police is True and p.police_confidence == 0.81
        assert p.vehicle_type == "truck"


def test_null_does_not_clobber_existing_value():
    with _store() as s:
        s.upsert_pass(PassRecord("s", 1, first_ts=1.0, last_ts=2.0,
                                 vehicle_type="car", max_speed_kmh=58.0))
        # A later writer that knows nothing about type/speed must not erase them.
        s.upsert_pass(PassRecord("s", 1, first_ts=1.0, last_ts=2.0, direction="ltr"))
        p = s.get_pass("s", 1)
        assert p.vehicle_type == "car" and p.max_speed_kmh == 58.0
        assert p.direction == "ltr"


def test_validity_columns_round_trip_and_merge():
    with _store() as s:
        # speeding writer sets raw speed + geometry + derived validity
        s.upsert_pass(PassRecord("s", 1, first_ts=1.0, last_ts=2.0,
                                 steady_speed_raw=72.0, ground_span=1.05, n_points=18,
                                 steady_valid=True, steady_invalid_reason=None))
        # a later police-only writer (all validity fields None) must not erase them
        s.upsert_pass(PassRecord("s", 1, first_ts=1.0, last_ts=2.0, is_police=True))
        p = s.get_pass("s", 1)
        assert p.steady_speed_raw == 72.0 and p.ground_span == 1.05
        assert p.n_points == 18 and p.steady_valid is True
        assert p.is_police is True


def test_migration_adds_validity_columns_to_legacy_db(tmp_path):
    import sqlite3
    # a DB with the pre-validity passes schema (no steady_* / ground_span / n_points)
    path = str(tmp_path / "legacy.sqlite")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE passes (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "session_id TEXT NOT NULL, track_id INTEGER NOT NULL, first_ts REAL NOT NULL, "
                 "last_ts REAL NOT NULL, direction TEXT, vehicle_type TEXT, max_speed_kmh REAL, "
                 "steady_speed_kmh REAL, was_speeding INTEGER, is_police INTEGER, "
                 "police_confidence REAL, plate TEXT, vehicle_id INTEGER, "
                 "UNIQUE(session_id, track_id))")
    conn.execute("INSERT INTO passes (session_id, track_id, first_ts, last_ts, steady_speed_kmh) "
                 "VALUES ('s', 1, 1.0, 2.0, 99.0)")
    conn.commit()
    conn.close()
    # opening via the store must ALTER in the new columns without touching existing rows
    with TrafficStore(path) as s:
        s.upsert_pass(PassRecord("s", 2, first_ts=1.0, last_ts=2.0, steady_valid=True))
        assert s.get_pass("s", 1).steady_speed_kmh == 99.0     # legacy row intact
        assert s.get_pass("s", 2).steady_valid is True         # new column writable


def test_speed_takes_larger_nonnull():
    with _store() as s:
        s.upsert_pass(PassRecord("s", 1, first_ts=1.0, last_ts=2.0, max_speed_kmh=50.0))
        s.upsert_pass(PassRecord("s", 1, first_ts=1.0, last_ts=2.0, max_speed_kmh=70.0))
        s.upsert_pass(PassRecord("s", 1, first_ts=1.0, last_ts=2.0, max_speed_kmh=60.0))
        assert s.get_pass("s", 1).max_speed_kmh == 70.0  # peak preserved


def test_flag_once_true_stays_true():
    with _store() as s:
        s.upsert_pass(PassRecord("s", 1, first_ts=1.0, last_ts=2.0, was_speeding=True))
        s.upsert_pass(PassRecord("s", 1, first_ts=1.0, last_ts=2.0, was_speeding=None))
        assert s.get_pass("s", 1).was_speeding is True


def test_passes_in_window():
    with _store() as s:
        s.upsert_pass(PassRecord("s", 1, first_ts=10.0, last_ts=10.0))
        s.upsert_pass(PassRecord("s", 2, first_ts=20.0, last_ts=20.0))
        s.upsert_pass(PassRecord("s", 3, first_ts=30.0, last_ts=30.0))
        got = s.passes_in_window(15.0, 25.0)
        assert [p.track_id for p in got] == [2]


# --- events -----------------------------------------------------------------

@dataclass
class _Cand:
    event_type: str
    trigger_ts: float
    score: float
    evidence: dict


@dataclass
class _FE:
    event_id: str
    event_type: str
    event_types: List[str]
    trigger_ts: float
    score: float
    primary_track_id: int
    candidates: List[_Cand]


def _fe(event_id="e1", track_id=7):
    return _FE(
        event_id=event_id, event_type="relative_speeding",
        event_types=["relative_speeding", "absolute_speeding"],
        trigger_ts=12.0, score=0.9, primary_track_id=track_id,
        candidates=[_Cand("absolute_speeding", 12.0, 0.9,
                          {"rule": "absolute_speeding", "speed_kmh": 64.0})],
    )


def test_event_links_to_pass():
    with _store() as s:
        pass_id = s.upsert_pass(PassRecord("sess", 7, first_ts=10.0, last_ts=13.0))
        rec = EventRecord.from_final_event(_fe(), "sess", clipped=True,
                                           clip_path="a.mp4")
        s.add_event(rec)
        ev = s.get_event("e1")
        assert ev["primary_pass_id"] == pass_id  # resolved from (session, track_id)
        assert ev["clipped"] == 1 and ev["clip_path"] == "a.mp4"
        assert ev["event_types"] == ["relative_speeding", "absolute_speeding"]
        assert ev["evidence"][0]["evidence"]["speed_kmh"] == 64.0


def test_event_without_pass_links_null():
    with _store() as s:
        rec = EventRecord.from_final_event(_fe(track_id=99), "sess", clipped=False)
        s.add_event(rec)
        assert s.get_event("e1")["primary_pass_id"] is None


def test_duplicate_event_ignored():
    with _store() as s:
        s.add_event(EventRecord.from_final_event(_fe(), "sess", clipped=True))
        second = s.add_event(EventRecord.from_final_event(_fe(), "sess", clipped=True))
        assert second is None  # idempotent on event_id


# --- compatibility views ----------------------------------------------------

def test_police_sightings_view():
    with _store() as s:
        s.upsert_pass(PassRecord("s", 1, first_ts=1.0, last_ts=2.0, is_police=True,
                                 police_confidence=0.7, direction="ltr",
                                 max_speed_kmh=55.0, was_speeding=True))
        s.upsert_pass(PassRecord("s", 2, first_ts=1.0, last_ts=2.0, is_police=False))
        rows = s.query("SELECT * FROM police_sightings")
        assert len(rows) == 1  # only the police one
        r = rows[0]
        assert r["confidence"] == 0.7 and r["was_speeding"] == 1
        assert r["ts"] == 2.0 and r["direction"] == "ltr"


def test_speed_events_view():
    with _store() as s:
        s.upsert_pass(PassRecord("s", 7, first_ts=10.0, last_ts=13.0,
                                 vehicle_type="truck", max_speed_kmh=64.0,
                                 was_speeding=True, direction="ltr"))
        s.add_event(EventRecord.from_final_event(_fe(), "s", clipped=True))
        rows = s.query("SELECT * FROM speed_events")
        assert len(rows) == 1
        assert rows[0]["speed_kmh"] == 64.0 and rows[0]["vehicle_type"] == "truck"
        assert rows[0]["clipped"] == 1
