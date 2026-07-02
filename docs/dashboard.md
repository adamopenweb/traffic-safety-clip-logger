# Web Dashboard (`traffic-log serve`)

A responsive phone/desktop web app for reviewing the traffic system: a **Now**
page (latest clips ≥ the clip threshold + today's headline stats), a **Browse**
grid (filter by speed/type/date), a **Stats** page (per-day, by-hour, speed
distribution, by vehicle type), a **Top Speeds** page (fastest recorded passes, with
a manual hide/restore control), and a **Speed test** page (label GPS drive-bys to
validate calibration). Video playback prefers the annotated clip when one
exists, with a toggle to the clean original.

It is a **separate, decoupled process** from `run`/`supervise`: it only *reads* the
stores the analyzer writes (`data/index/speed_log.sqlite` for stats, `data/events/`
for clips), so starting or restarting it never disturbs recording or analysis.

## Security model: invisible behind a secret link

The app is meant to be exposed to the public internet via **Tailscale Funnel**, so
it assumes scanner bots will probe it. Auth is a **404-everything gate** at the ASGI
middleware layer: every unauthenticated request (`/`, `/static/app.js`, `/api/*`,
`/.env`, `/wp-login.php`) returns an identical `404 Not Found`. A prober can't tell
an app is even there, or learn anything about the filesystem.

You get in by visiting one unguessable **unlock link** once per device:

```
https://<your-funnel-host>/k/<access_token>
```

The `access_token` is a 256-bit secret (stronger than a typed password) and *is* the
credential. Visiting it sets a signed, HttpOnly, 30-day cookie; after that the
device just works. Rotating the token (`serve --new-link`) revokes every old
bookmark *and* every existing session (it rotates the cookie-signing secret too).
The "Sign out" button clears the cookie (the app then 404s until you open
your link again).

## One-time setup

```powershell
# 1. Install the web extra into the venv (additive; safe while the analyzer runs)
.\.venv\Scripts\python.exe -m pip install -e ".[web]"

# 2. Generate your unlock link (writes access_token into the gitignored config,
#    fills session_secret if blank, and prints the link).
.\.venv\Scripts\python.exe -m traffic_logger.main serve --new-link `
    --config config/config.run.local.yaml
```

That prints something like (token shown is illustrative; yours is unique and secret):

```
    https://<your-funnel-host>/k/<your-access-token>
```

The `web:` config block (in the gitignored `config/config.run.local.yaml`):

```yaml
web:
  host: "0.0.0.0"
  port: 8090
  unlock_prefix: "k"    # URL path before the token: /k/<token>
  access_token: "..."   # set by --new-link
  session_secret: "..." # signs the auth cookie (keep secret)
  cookie_secure: true   # funnel is HTTPS; keep true in production
  fast_threshold: 70    # "Now" page shows clips at/above this km/h
  speed_limit: 50       # posted limit, for the over-limit stat
  cache_ttl: 20         # seconds to cache the events-folder scan
```

> The access token, session secret, and the camera RTSP password all live in
> `config.run.local.yaml`, which is **gitignored** — never commit it.

## Run it

```powershell
.\.venv\Scripts\python.exe -m traffic_logger.main serve `
    --config config/config.run.local.yaml
```

Serves on `http://0.0.0.0:8090`. Test locally with the printed
`http://localhost:8090/k/<token>` URL.

## Expose it with Tailscale Funnel

Funnel proxies a public HTTPS endpoint to the local port. From an elevated shell
(run it yourself — it needs interactive Tailscale auth/consent the first time):

```powershell
tailscale funnel 8090
```

Funnel routes public HTTPS (ports 443/8443/10000) to your node; TLS is
terminated **on your machine**, so Tailscale's relays only ever see encrypted
bytes and can never observe request paths (or the unlock token). It will print
your public host
(`https://<machine>.<tailnet>.ts.net`). Your bookmark is that host + `/k/<token>`.

To run Funnel in the background as a service: `tailscale funnel --bg 8090`
(`tailscale funnel status` to check, `tailscale funnel reset` to stop).

## Daily use

1. On each device (phone, desktop), open the unlock link **once** and bookmark the
   site. The cookie keeps you in for 30 days.
2. If a bookmark stops working (cookie expired), open the unlock link again.
3. To revoke all access (lost phone, sharing the link by mistake), run
   `serve --new-link` again and re-open the new link on your own devices.

## Keeping it running

The dashboard is a foreground process. To keep it up across reboots, run it as a
Windows Scheduled Task (like the `supervise` analyzer task) pointing at:

```
.venv\Scripts\python.exe -m traffic_logger.main serve --config config\config.run.local.yaml
```

Because it's read-only over the analyzer's data, it's safe to start/stop/restart at
any time independently of capture.
