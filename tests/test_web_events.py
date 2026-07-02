"""Phase B: events index (scan/parse/filter) + media routes with range support."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from traffic_logger.web import events_index as ei

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from traffic_logger.web.app import WebSettings, create_app  # noqa: E402

TZ = ZoneInfo("America/Toronto")


def _ts(y, mo, d, h=12):
    return datetime(y, mo, d, h, tzinfo=TZ).timestamp()


def _meta(event_id, etype, ts, speed, direction, vtype):
    return {
        "event_id": event_id, "event_type": etype, "trigger_ts": ts,
        "evidence": {"triggers": [{"evidence": {
            "rule": "absolute_speeding", "speed_kmh": speed,
            "direction": direction, "vehicle_type": vtype}}]},
    }


def _write_event(events_dir, date, etype, stem, meta, *,
                 clip=b"CLEAN", annotated=None, thumb=b"JPG"):
    d = events_dir / date / etype
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
    if clip is not None:
        (d / f"{stem}.mp4").write_bytes(clip)
    if annotated is not None:
        (d / f"{stem}_annotated.mp4").write_bytes(annotated)
    if thumb is not None:
        (d / f"{stem}.jpg").write_bytes(thumb)
    # an overlay sidecar that must be ignored by the scan
    (d / f"{stem}_overlay.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def events_dir(tmp_path):
    root = tmp_path / "events"
    _write_event(root, "2026-06-18", "relative_speeding",
                 "20260618_080000_72kmh_car_LtR_relative_speeding_aaaaaaaa",
                 _meta("aaaaaaaa-1", "relative_speeding", _ts(2026, 6, 18, 8),
                       72.0, "left_to_right", "car"),
                 annotated=b"ANNOTATEDFRAMES")
    _write_event(root, "2026-06-20", "relative_speeding",
                 "20260620_090000_90kmh_motorcycle_RtL_relative_speeding_bbbbbbbb",
                 _meta("bbbbbbbb-2", "relative_speeding", _ts(2026, 6, 20, 9),
                       90.0, "right_to_left", "motorcycle"))  # clean only, no annotated
    _write_event(root, "2026-06-20", "center_lane_pass",
                 "20260620_100000_center_lane_pass_cccccccc",
                 _meta("cccccccc-3", "center_lane_pass", _ts(2026, 6, 20, 10),
                       None, "left_to_right", "car"),
                 clip=None, annotated=None)  # record only, no video
    return root


# -- pure parsing ------------------------------------------------------------

def test_speed_dir_type_from_evidence():
    m = _meta("x", "relative_speeding", 1.0, 66.0, "right_to_left", "truck")
    assert ei.speed_dir_type(m) == (66.0, "right_to_left", "truck")


def test_scan_newest_first_and_fields(events_dir):
    summaries, paths = ei.scan_events(events_dir, TZ)
    assert [s["id"] for s in summaries] == ["cccccccc-3", "bbbbbbbb-2", "aaaaaaaa-1"]
    fast = next(s for s in summaries if s["id"] == "aaaaaaaa-1")
    assert fast["speed_kmh"] == 72.0 and fast["vehicle_type"] == "car"
    assert fast["annotated"] is True and fast["has_video"] is True
    novideo = next(s for s in summaries if s["id"] == "cccccccc-3")
    assert novideo["has_video"] is False and novideo["clip_url"] is None


def test_best_video_prefers_annotated(events_dir):
    _, paths = ei.scan_events(events_dir, TZ)
    a = paths["aaaaaaaa-1"]
    assert a.best_video().read_bytes() == b"ANNOTATEDFRAMES"
    assert a.best_video(prefer_annotated=False).read_bytes() == b"CLEAN"
    # event with no annotated falls back to the clean clip
    assert paths["bbbbbbbb-2"].best_video().read_bytes() == b"CLEAN"


def test_index_query_and_latest_fast(events_dir):
    idx = ei.EventsIndex(events_dir, TZ, ttl=0)
    assert len(idx.all()) == 3
    assert [e["id"] for e in idx.query(min_speed=80)] == ["bbbbbbbb-2"]
    assert [e["id"] for e in idx.query(event_type="center_lane_pass")] == ["cccccccc-3"]
    assert idx.query(since_ts=_ts(2026, 6, 20, 0)) and \
        all(e["trigger_ts"] >= _ts(2026, 6, 20, 0) for e in idx.query(since_ts=_ts(2026, 6, 20, 0)))
    # latest_fast: video present AND >= threshold (center-lane has no video, excluded)
    fast = idx.latest_fast(threshold=70)
    assert [e["id"] for e in fast] == ["bbbbbbbb-2", "aaaaaaaa-1"]


def test_top_speeders_hall(events_dir):
    idx = ei.EventsIndex(events_dir, TZ, ttl=0)
    # threshold 85 -> only the 90 km/h car (the 72 is below; center-lane has no video)
    assert [e["id"] for e in idx.top_speeders(threshold=85)] == ["bbbbbbbb-2"]
    # fastest-first ordering
    assert [e["id"] for e in idx.top_speeders(threshold=70)] == ["bbbbbbbb-2", "aaaaaaaa-1"]


# -- API + media routes ------------------------------------------------------

_TOKEN = "events-token-xyz"


@pytest.fixture
def client(events_dir, tmp_path):
    settings = WebSettings(
        events_dir=events_dir,
        speed_log_path=str(tmp_path / "missing.sqlite"),
        timezone="America/Toronto",
        access_token=_TOKEN, session_secret="s",
        exclusions_path=str(tmp_path / "hall_excluded.json"),
        cookie_secure=False, fast_threshold=70.0)
    c = TestClient(create_app(settings))
    return c


def _login(c):
    assert c.get(f"/k/{_TOKEN}", follow_redirects=False).status_code == 302


def test_media_and_events_require_auth(client):
    assert client.get("/api/events").status_code == 404
    assert client.get("/api/now").status_code == 404
    assert client.get("/media/clip/aaaaaaaa-1").status_code == 404


def test_events_listing_and_now(client):
    _login(client)
    listing = client.get("/api/events").json()
    assert listing["count"] == 3
    now = client.get("/api/now").json()
    assert [e["id"] for e in now["events"]] == ["bbbbbbbb-2", "aaaaaaaa-1"]
    assert now["fast_threshold"] == 70.0


def test_hall_endpoint(client):
    _login(client)
    d = client.get("/api/hall").json()
    assert d["threshold"] == 85.0
    assert [e["id"] for e in d["events"]] == ["bbbbbbbb-2"]   # only the 90 km/h car
    assert d["hidden"] == []


def test_hall_manual_exclude_and_restore(client):
    _login(client)
    # exclude the lone Hall entry -> it moves from events to hidden
    r = client.post("/api/hall/exclude", json={"event_id": "bbbbbbbb-2"})
    assert r.status_code == 200 and r.json()["changed"] is True
    d = client.get("/api/hall").json()
    assert d["events"] == []
    assert [e["id"] for e in d["hidden"]] == ["bbbbbbbb-2"]
    assert d["hidden"][0]["excluded"] is True
    # re-excluding is idempotent (changed=False)
    assert client.post("/api/hall/exclude", json={"event_id": "bbbbbbbb-2"}).json()["changed"] is False
    # unknown id is rejected, not silently stored
    assert client.post("/api/hall/exclude", json={"event_id": "nope"}).status_code == 404
    # restore puts it back in the Hall
    r = client.post("/api/hall/restore", json={"event_id": "bbbbbbbb-2"})
    assert r.status_code == 200 and r.json()["changed"] is True
    d = client.get("/api/hall").json()
    assert [e["id"] for e in d["events"]] == ["bbbbbbbb-2"] and d["hidden"] == []


def test_clip_prefers_annotated_and_range(client):
    _login(client)
    # default prefers annotated
    r = client.get("/media/clip/aaaaaaaa-1")
    assert r.status_code == 200 and r.content == b"ANNOTATEDFRAMES"
    # explicit clean
    r = client.get("/media/clip/aaaaaaaa-1?annotated=false")
    assert r.content == b"CLEAN"
    # range request -> 206 partial
    r = client.get("/media/clip/aaaaaaaa-1", headers={"Range": "bytes=0-4"})
    assert r.status_code == 206
    assert r.content == b"ANNOT"
    assert r.headers["content-range"].startswith("bytes 0-4/")


def test_clip_404_for_record_only_and_unknown(client):
    _login(client)
    assert client.get("/media/clip/cccccccc-3").status_code == 404  # no video
    assert client.get("/media/clip/does-not-exist").status_code == 404
    assert client.get("/media/thumb/aaaaaaaa-1").status_code == 200


def test_clip_download_disposition(client):
    _login(client)
    inline = client.get("/media/clip/aaaaaaaa-1")
    assert "inline" in inline.headers.get("content-disposition", "")
    dl = client.get("/media/clip/aaaaaaaa-1?download=1")
    cd = dl.headers.get("content-disposition", "")
    assert "attachment" in cd and ".mp4" in cd
