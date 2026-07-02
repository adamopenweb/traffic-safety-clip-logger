"""Tests for the daylight-window sun calculation (Hamilton, ON)."""

from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

from traffic_logger.util.sun import daylight_window, sun_event

TZ = ZoneInfo("America/Toronto")
LAT, LON = 43.26, -79.87


def _hm(dt):
    return dt.hour * 60 + dt.minute


def test_civil_window_summer():
    start, end = daylight_window(date(2026, 6, 19), LAT, LON, TZ, buffer_minutes=0)
    # Hamilton civil dawn ~05:03, civil dusk ~21:38.
    assert abs(_hm(start) - (5 * 60 + 3)) <= 10
    assert abs(_hm(end) - (21 * 60 + 38)) <= 10
    assert start.date() == date(2026, 6, 19) and end.date() == date(2026, 6, 19)


def test_civil_window_winter_is_much_shorter():
    s_summer, e_summer = daylight_window(date(2026, 6, 19), LAT, LON, TZ, buffer_minutes=0)
    s_winter, e_winter = daylight_window(date(2026, 12, 21), LAT, LON, TZ, buffer_minutes=0)
    # Winter window ~07:15..17:19 -- starts later, ends earlier than summer.
    assert abs(_hm(s_winter) - (7 * 60 + 15)) <= 12
    assert abs(_hm(e_winter) - (17 * 60 + 19)) <= 12
    assert (_hm(e_winter) - _hm(s_winter)) < (_hm(e_summer) - _hm(s_summer))


def test_buffer_widens_window_symmetrically():
    s0, e0 = daylight_window(date(2026, 6, 19), LAT, LON, TZ, buffer_minutes=0)
    s1, e1 = daylight_window(date(2026, 6, 19), LAT, LON, TZ, buffer_minutes=30)
    assert _hm(s0) - _hm(s1) == 30
    assert _hm(e1) - _hm(e0) == 30


def test_sun_event_returns_local_date():
    dusk = sun_event(date(2026, 6, 19), LAT, LON, TZ, rising=False)
    assert dusk.date() == date(2026, 6, 19)   # not off-by-one from the UTC wrap
    assert dusk.tzinfo is not None
