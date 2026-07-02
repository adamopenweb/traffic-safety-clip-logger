"""FastAPI application factory for the ``traffic-log serve`` dashboard.

:func:`create_app` builds a small JSON API + static SPA over the stores the
analyzer already writes, wrapped in :class:`AuthGate` -- a pure-ASGI middleware
that 404s every unauthenticated request and unlocks via a secret link (see
:mod:`.auth`). Because the gate sits outside routing and passes authenticated
requests through *untouched*, ranged ``FileResponse`` video streaming keeps working.

It takes a small :class:`WebSettings` value object (not the whole
:class:`~traffic_logger.config.Config`) so tests spin up against a temp DB / events
dir without a full config file.
"""

from __future__ import annotations

import mimetypes
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from ..util.paths import project_root
from . import auth, stats, thumbs
from .events_index import EventsIndex
from .exclusions import Exclusions
from .speedtest import SpeedTestLog, recent_passes

_STATIC_DIR = Path(__file__).parent / "static"

# StaticFiles guesses content types via mimetypes, which doesn't know .webmanifest.
mimetypes.add_type("application/manifest+json", ".webmanifest")


@dataclass
class WebSettings:
    """Everything the web app needs, resolved from config (or built by hand in tests)."""

    events_dir: Path
    speed_log_path: str
    timezone: str
    access_token: str            # the unlock secret (the credential)
    session_secret: str          # signs the auth cookie
    unlock_prefix: str = "k"     # URL is /<prefix>/<access_token>
    traffic_db_path: str = ""    # unified store (passes table) for the denominator
    exclusions_path: str = ""    # manual Top-Speeds excludes (JSON id list)
    speedtest_log_path: str = "" # labelled GPS drive-by passes (JSON)
    cookie_secure: bool = True
    speed_limit: float = 50.0
    over_limit_kmh: float = 55.0   # buffered "speeding" threshold for the denominator %
    fast_threshold: float = 70.0   # the "Now" page / clip threshold
    hall_kmh: float = 85.0         # "Top Speeds" page threshold (egregious speeders)
    # (Speed plausibility is no longer a read-time concern: passes carry a write-time
    # steady_valid flag from the shared invariant module, and read_passes trusts it.)
    cache_ttl: float = 20.0
    thumb_width: int = 480         # grid thumbnails are downscaled to this width
    host: str = "0.0.0.0"
    port: int = 8090

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @classmethod
    def from_config(cls, config) -> "WebSettings":
        web = config.web or {}
        rs = config.events.get("relative_speeding", {}) or {}
        passes_cfg = config.events.get("passes", {}) or {}
        return cls(
            events_dir=_resolve(config.events.get("output_path", "data/events")),
            speed_log_path=str(_resolve(
                config.events.get("speed_log_path", "data/index/speed_log.sqlite"))),
            traffic_db_path=str(_resolve(
                passes_cfg.get("db_path", "data/index/traffic.sqlite"))),
            exclusions_path=str(_resolve(
                web.get("exclusions_path", "data/index/hall_excluded.json"))),
            speedtest_log_path=str(_resolve(
                web.get("speedtest_log_path", "data/index/speedtest_log.json"))),
            timezone=config.app.timezone,
            access_token=str(web.get("access_token", "")),
            session_secret=str(web.get("session_secret", "")),
            unlock_prefix=str(web.get("unlock_prefix", "k")).strip("/") or "k",
            cookie_secure=bool(web.get("cookie_secure", True)),
            speed_limit=float(web.get("speed_limit", 50.0)),
            over_limit_kmh=float(web.get("over_limit_kmh",
                                         rs.get("absolute_kmh_threshold", 55.0))),
            fast_threshold=float(web.get("fast_threshold",
                                         rs.get("clip_kmh_threshold", 70.0))),
            # top_speeds_kmh is the current key; hall_of_shame_kmh accepted for
            # configs written before the page was renamed.
            hall_kmh=float(web.get("top_speeds_kmh", web.get("hall_of_shame_kmh", 85.0))),
            cache_ttl=float(web.get("cache_ttl", 20.0)),
            thumb_width=int(web.get("thumb_width", 480)),
            host=str(web.get("host", "0.0.0.0")),
            port=int(web.get("port", 8090)),
        )

    def unlock_path(self) -> str:
        return f"/{self.unlock_prefix}/{self.access_token}"


def _resolve(p: str | Path) -> Path:
    """Resolve a possibly-relative config path against the project root (the
    analyzer writes these relative to the repo, so the web app must read the same)."""
    path = Path(p)
    return path if path.is_absolute() else (project_root() / path)


class AuthGate:
    """Pure-ASGI gate: 404 everything unless the request carries a valid cookie or
    hits the exact secret unlock path. Runs outside routing and, on success, calls
    the wrapped app unchanged so streaming/range responses are untouched."""

    def __init__(self, app, settings: WebSettings) -> None:
        self.app = app
        self.settings = settings
        self.signer = auth.make_signer(settings.session_secret)
        self._prefix = f"/{settings.unlock_prefix}/"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)

        cookie = request.cookies.get(auth.COOKIE_NAME, "")
        if auth.cookie_valid(self.signer, cookie):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path.startswith(self._prefix):
            token = path[len(self._prefix):]
            if auth.token_matches(token, self.settings.access_token):
                resp = RedirectResponse("/", status_code=302)
                resp.set_cookie(
                    auth.COOKIE_NAME, auth.issue_cookie(self.signer),
                    max_age=auth.COOKIE_MAX_AGE, httponly=True,
                    secure=self.settings.cookie_secure, samesite="lax", path="/")
                await resp(scope, receive, send)
                return

        # Indistinguishable 404 for probers: real path, fake path, asset, API -- all same.
        await PlainTextResponse("Not Found", status_code=404)(scope, receive, send)


def _window_start(settings: WebSettings, days: Optional[int], now_ts: float) -> Optional[float]:
    """Unix ts for the start of a stats window: local midnight ``days-1`` days ago
    (so ``days=1`` is "today"). ``None`` (all-time) when ``days`` is falsy."""
    if not days:
        return None
    midnight = datetime.fromtimestamp(now_ts, settings.tz).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp() - (days - 1) * 86400


def create_app(settings: WebSettings) -> FastAPI:
    app = FastAPI(title="Traffic Safety Dashboard", docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.settings = settings
    app.state.events = EventsIndex(settings.events_dir, settings.tz, ttl=settings.cache_ttl)
    app.state.thumb_cache = settings.events_dir.parent / "cache" / "thumbs"
    app.state.exclusions = Exclusions(
        settings.exclusions_path or (settings.events_dir.parent / "index" / "hall_excluded.json"))
    app.state.speedtest_log = SpeedTestLog(
        settings.speedtest_log_path or (settings.events_dir.parent / "index" / "speedtest_log.json"))

    # -- session ----------------------------------------------------------
    @app.post("/api/logout")
    def logout():
        resp = PlainTextResponse('{"ok":true}', media_type="application/json")
        resp.delete_cookie(auth.COOKIE_NAME, path="/")
        return resp

    # -- stats ------------------------------------------------------------
    @app.get("/api/stats/today")
    def stats_today():
        s = app.state.settings
        now_ts = time.time()
        # Violations come from the GPS-validated passes log (not the event log, whose
        # trigger speed reads ~7 km/h high), so the count/speeds match the speed-test.
        passes = stats.read_passes(s.traffic_db_path, since_ts=_window_start(s, 1, now_ts))
        viols = stats.passes_to_violations(passes, s.over_limit_kmh, s.fast_threshold)
        return {
            "summary": stats.summarize(viols, speed_limit=s.speed_limit,
                                       fast_threshold=s.fast_threshold),
            "hourly": stats.hourly_histogram(viols, s.tz),
        }

    @app.get("/api/stats")
    def stats_bundle(days: int = 30):
        s = app.state.settings
        days = max(1, min(int(days), 365))
        now_ts = time.time()
        start = _window_start(s, days, now_ts)
        # Clamp to when car-counting began so numerator (violations) and denominator
        # (cars) cover the same period; the chart spans only the covered days.
        floor = stats.first_pass_ts(s.traffic_db_path)
        eff_start = max(start, floor) if floor else start
        eff_days = days
        if floor and floor > start:
            eff_days = min(days, int((now_ts - floor) // 86400) + 1)
        passes = stats.read_passes(s.traffic_db_path, since_ts=eff_start)
        viols = stats.passes_to_violations(passes, s.over_limit_kmh, s.fast_threshold)
        return {
            "days": days,
            "coverage_since": datetime.fromtimestamp(eff_start, s.tz).strftime("%Y-%m-%d"),
            "clamped": bool(floor and floor > start),
            "summary": stats.summarize(viols, speed_limit=s.speed_limit,
                                       fast_threshold=s.fast_threshold),
            "volume": stats.volume_summary(passes, over_threshold=s.over_limit_kmh),
            "daily": stats.daily_series(viols, s.tz, days=eff_days, now_ts=now_ts,
                                        fast_threshold=s.fast_threshold),
            "hourly": stats.hourly_histogram(viols, s.tz),
            "distribution": stats.speed_distribution(viols),
            "vehicles": stats.vehicle_breakdown(viols),
        }

    # -- events + media ---------------------------------------------------
    @app.get("/api/events")
    def list_events(days: int = 0, min_speed: Optional[float] = None,
                    type: Optional[str] = None, limit: int = 100, offset: int = 0,
                    since: Optional[float] = None, until: Optional[float] = None):
        # `since`/`until` (epoch) target a specific day/range and override `days`
        # (a rolling window from now); `offset` paginates so Browse can reach older
        # records even when recent volume exceeds one page.
        s = app.state.settings
        since_ts = since if since is not None else (
            _window_start(s, days, time.time()) if days else None)
        items = app.state.events.query(
            min_speed=min_speed, event_type=type, since_ts=since_ts, until_ts=until,
            limit=max(1, min(int(limit), 500)), offset=max(0, int(offset)))
        return {"events": items, "count": len(items)}

    @app.get("/api/now")
    def now(limit: int = 12):
        s = app.state.settings
        now_ts = time.time()
        start = _window_start(s, 1, now_ts)
        passes = stats.read_passes(s.traffic_db_path, since_ts=start)
        viols = stats.passes_to_violations(passes, s.over_limit_kmh, s.fast_threshold)
        return {
            "events": app.state.events.latest_fast(s.fast_threshold,
                                                   limit=max(1, min(int(limit), 50))),
            "summary": stats.summarize(viols, speed_limit=s.speed_limit,
                                       fast_threshold=s.fast_threshold),
            "volume": stats.volume_summary(passes, over_threshold=s.over_limit_kmh),
            "fast_threshold": s.fast_threshold,
        }

    @app.get("/api/hall")
    def hall(limit: int = 50):
        s = app.state.settings
        excluded = app.state.exclusions.all()
        top = app.state.events.top_speeders(s.hall_kmh, limit=max(1, min(int(limit), 200)))
        active = [e for e in top if e["id"] not in excluded]
        # Manually hidden entries (e.g. police on a call) -- kept so they can be
        # restored from the UI, but never listed among the top speeds.
        hidden = [{**e, "excluded": True} for e in top if e["id"] in excluded]
        return {"threshold": s.hall_kmh, "events": active, "hidden": hidden}

    @app.post("/api/hall/exclude")
    def hall_exclude(event_id: str = Body(..., embed=True)):
        # Validate the id exists so a typo can't silently bury nothing.
        if app.state.events.get_paths(event_id) is None:
            raise HTTPException(status_code=404, detail="unknown event")
        added = app.state.exclusions.add(event_id)
        return {"ok": True, "excluded": True, "changed": added}

    @app.post("/api/hall/restore")
    def hall_restore(event_id: str = Body(..., embed=True)):
        removed = app.state.exclusions.remove(event_id)
        return {"ok": True, "excluded": False, "changed": removed}

    # -- speed test (GPS drive-by validation) -----------------------------
    @app.get("/api/time")
    def server_time():
        """Live PC clock so a drive-by can be matched to the system's timestamps."""
        now = time.time()
        return {"ts": now,
                "hms": datetime.fromtimestamp(now, settings.tz).strftime("%H:%M:%S"),
                "iso": datetime.fromtimestamp(now, settings.tz).isoformat(timespec="seconds")}

    @app.get("/api/speedtest/passes")
    def speedtest_passes(minutes: int = 10):
        """Recent measured passes (incl. sub-55) to pick your drive-by from."""
        s = app.state.settings
        minutes = max(1, min(int(minutes), 120))
        since = time.time() - minutes * 60
        return {"minutes": minutes,
                "passes": recent_passes(s.traffic_db_path, since_ts=since)}

    @app.get("/api/speedtest/log")
    def speedtest_log():
        return {"tests": app.state.speedtest_log.all()}

    @app.post("/api/speedtest/label")
    def speedtest_label(
        key: str = Body(..., embed=True),
        ts: float = Body(..., embed=True),
        measured: float = Body(..., embed=True),
        true_speed: float = Body(..., embed=True),
        direction: Optional[str] = Body(None, embed=True),
        vehicle_type: Optional[str] = Body(None, embed=True),
        note: str = Body("", embed=True),
    ):
        if true_speed <= 0:
            raise HTTPException(status_code=400, detail="true_speed must be > 0")
        rec = app.state.speedtest_log.add(
            key=key, ts=ts, measured=measured, true_speed=true_speed,
            direction=direction, vehicle_type=vehicle_type, note=note)
        return {"ok": True, "test": rec}

    @app.post("/api/speedtest/unlabel")
    def speedtest_unlabel(key: str = Body(..., embed=True)):
        return {"ok": True, "changed": app.state.speedtest_log.remove(key)}

    @app.get("/api/volume")
    def volume(days: int = 30):
        s = app.state.settings
        days = max(1, min(int(days), 365))
        start = _window_start(s, days, time.time())
        passes = stats.read_passes(s.traffic_db_path, since_ts=start)
        return {"days": days,
                "volume": stats.volume_summary(passes, over_threshold=s.over_limit_kmh)}

    @app.get("/media/clip/{event_id}")
    def media_clip(event_id: str, annotated: bool = True, download: bool = False):
        paths = app.state.events.get_paths(event_id)
        video = paths.best_video(prefer_annotated=annotated) if paths else None
        if not video or not video.exists():
            raise HTTPException(status_code=404, detail="clip not found")
        # download=1 -> attachment (Save Video on the phone); otherwise inline so
        # the <video> element streams it. FileResponse handles Range (206) either way.
        disposition = "attachment" if download else "inline"
        return FileResponse(video, media_type="video/mp4", filename=video.name,
                            content_disposition_type=disposition)

    @app.get("/media/thumb/{event_id}")
    def media_thumb(event_id: str):
        paths = app.state.events.get_paths(event_id)
        if not paths or not paths.thumb or not paths.thumb.exists():
            raise HTTPException(status_code=404, detail="thumbnail not found")
        # Serve a small cached copy when we can downscale it; else the original.
        small = thumbs.downscaled_thumb(paths.thumb, app.state.thumb_cache,
                                        max_w=settings.thumb_width)
        return FileResponse(small or paths.thumb, media_type="image/jpeg")

    # -- static UI --------------------------------------------------------
    # The SPA shell (index.html) and its JS/CSS are served with Cache-Control:
    # no-cache so an installed PWA -- which can't be hard-refreshed -- always
    # revalidates and picks up updates (ETag makes the revalidation a cheap 304
    # when unchanged). Icons/manifest stay on the cached /static mount. Serving
    # the JS/CSS off /app.js + /styles.css (not /static) also means this deploy's
    # new URLs sidestep any already-cached /static/app.js.
    def _shell(name: str, media: str):
        resp = FileResponse(_STATIC_DIR / name, media_type=media)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

        @app.get("/")
        def index():
            return _shell("index.html", "text/html")

        @app.get("/app.js")
        def app_js():
            return _shell("app.js", "application/javascript")

        @app.get("/styles.css")
        def styles_css():
            return _shell("styles.css", "text/css")

    # AuthGate wraps the whole app: it runs before routing, so unauthenticated
    # requests never reach any route above -- they get an identical 404.
    return AuthGate(app, settings)
