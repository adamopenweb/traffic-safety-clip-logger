"""Sunrise / civil-twilight times for the daylight run schedule.

The camera is effectively blind once it's fully dark (a near-black frame -- the
built-in IR can't light the road across the street, and brightening via long
exposure would motion-blur fast cars). So the run is scheduled to the daylight
window: from civil dawn to civil dusk (the sun within 6 deg of the horizon, when
the sky still carries usable light), plus a small buffer.

Pure standard "sunrise equation" (Almanac for Computers) -- accurate to a couple
minutes, no dependencies, unit-tested against known Hamilton ON times.
"""

from __future__ import annotations

import math
from datetime import date as _date
from datetime import datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

# Zenith angle (deg from vertical) at the event. 90.833 = geometric sunrise/set;
# 96 = civil twilight (sun 6 deg below the horizon).
CIVIL_ZENITH = 96.0
OFFICIAL_ZENITH = 90.8333

_UTC = ZoneInfo("UTC")


def _event_ut_hours(d: _date, lat: float, lon: float, zenith: float, rising: bool) -> Optional[float]:
    """UTC time-of-day (hours) of the sun event on date ``d``, or None if it does
    not occur (polar day/night). Standard Almanac-for-Computers algorithm."""
    rad, deg = math.radians, math.degrees
    n = d.timetuple().tm_yday
    lng_hour = lon / 15.0
    t = n + ((6 - lng_hour) / 24.0 if rising else (18 - lng_hour) / 24.0)
    m = (0.9856 * t) - 3.289
    L = (m + 1.916 * math.sin(rad(m)) + 0.020 * math.sin(rad(2 * m)) + 282.634) % 360.0
    ra = deg(math.atan(0.91764 * math.tan(rad(L)))) % 360.0
    # Put RA in the same quadrant as L, then to hours.
    ra = (ra + (math.floor(L / 90) * 90) - (math.floor(ra / 90) * 90)) / 15.0
    sin_dec = 0.39782 * math.sin(rad(L))
    cos_dec = math.cos(math.asin(sin_dec))
    cos_h = (math.cos(rad(zenith)) - sin_dec * math.sin(rad(lat))) / (cos_dec * math.cos(rad(lat)))
    if cos_h > 1 or cos_h < -1:
        return None
    h = (360 - deg(math.acos(cos_h))) if rising else deg(math.acos(cos_h))
    h /= 15.0
    ut = (h + ra - (0.06571 * t) - 6.622 - lng_hour) % 24.0
    return ut


def sun_event(d: _date, lat: float, lon: float, tz: ZoneInfo, *,
              rising: bool, zenith: float = CIVIL_ZENITH) -> Optional[datetime]:
    """Local datetime of the sun event (rising/setting) on local date ``d``."""
    ut = _event_ut_hours(d, lat, lon, zenith, rising)
    if ut is None:
        return None
    hh = int(ut)
    mm = int((ut - hh) * 60)
    ss = int(round((((ut - hh) * 60) - mm) * 60)) % 60
    # The UTC instant may fall on d-1/d/d+1; pick the one that is local date d.
    for off in (0, 1, -1):
        cd = d + timedelta(days=off)
        local = datetime(cd.year, cd.month, cd.day, hh, mm, ss, tzinfo=_UTC).astimezone(tz)
        if local.date() == d:
            return local
    return datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=_UTC).astimezone(tz)


def daylight_window(d: _date, lat: float, lon: float, tz: ZoneInfo, *,
                    buffer_minutes: float = 20.0,
                    zenith: float = CIVIL_ZENITH) -> Optional[Tuple[datetime, datetime]]:
    """(start, end) local datetimes the camera can usefully see on date ``d``.

    Civil dawn minus buffer to civil dusk plus buffer. None at polar latitudes
    where the sun never reaches the zenith (caller decides what to do).
    """
    dawn = sun_event(d, lat, lon, tz, rising=True, zenith=zenith)
    dusk = sun_event(d, lat, lon, tz, rising=False, zenith=zenith)
    if dawn is None or dusk is None:
        return None
    buf = timedelta(minutes=buffer_minutes)
    return (dawn - buf, dusk + buf)
