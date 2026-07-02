# Single-Box Deployment (Windows + NVIDIA GPU)

The permanent deployment is **one Windows PC doing everything**: it records the
camera's 4K RTSP stream to a local ring buffer, analyzes it live on the GPU,
cuts evidence clips, and serves the dashboard. There is no mini-PC, no Docker,
and no WSL in the loop.

> The original two-box plan (Linux mini-PC capture appliance + separate
> analysis box, via Docker Compose) still exists in the repo (see the
> [appendix](#appendix-legacy-mini-pcdocker-path)), but it is **not** what
> runs.

| Piece | What it does |
| --- | --- |
| Camera | 4K PoE RTSP camera (reference: iSee CCIPB812-Z-4K, Dahua-style URLs) |
| `traffic-log run` | Records the 4K main stream (H.265 **stream-copy**, no CPU encode) to the ring **and** analyzes it live (YOLOv8 on CUDA) |
| `traffic-log supervise` | Keeps `run` alive 24/7 (or daylight-only), restarts it on crash |
| `traffic-log serve` | Web dashboard behind a secret-link gate, exposed via `tailscale funnel` |

## 0. Prerequisites

- Windows 10/11 with an NVIDIA GPU (reference box: RTX 4080; analysis at
  4K/30fps uses a fraction of it).
- **Disk:** the ring is the big consumer. The camera's H.265 is variable-bitrate
  (16 Mbps configured ceiling) and averages only a few Mbps on a mostly-static
  street scene, ≈ 30–35 GB/day in practice; `recording.ring_max_gb: 300` keeps
  roughly 9 days of footage. Event clips live outside the ring and grow slowly
  (~30 s H.264 per event).
- Python **3.11+**, git, and **ffmpeg on PATH** (`ffmpeg -version` must work in
  the shell that runs the service).
- The camera reachable on the LAN, ideally wired PoE. Dahua-style stream URLs:

  ```
  main (4K):  rtsp://USER:PASS@CAMERA_IP:554/cam/realmonitor?channel=1&subtype=0
  sub (D1):   rtsp://USER:PASS@CAMERA_IP:554/cam/realmonitor?channel=1&subtype=1
  ```

## 1. Install

```powershell
git clone https://github.com/adamopenweb/traffic-safety-clip-logger.git
cd traffic-safety-clip-logger
python -m venv .venv
.venv\Scripts\activate

# CUDA torch first (pick the right index URL for your CUDA version at pytorch.org),
# then the project with the CV + web + dev extras:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .[analyze,web,dev]

pytest -q          # the suite needs no camera or GPU
```

## 2. Run config (credentials stay out of git)

Copy the documented template and fill in the camera credentials:

```powershell
copy config\config.camera.yaml config\config.run.local.yaml
```

`config/*.local.yaml` is **gitignored**: real RTSP credentials and the web
secrets only ever live there. Key choices in the live config:

- `recording.source` **and** `analysis.source` both point at the 4K main stream
  (`subtype=0`). Recording stream-copies it; analysis downscales to
  `analyze_max_width: 1280` before the de-warp so the pipeline sustains 30 fps.
  (Analyzing the D1 sub-stream instead also works, but it tracks fast
  near-lane vehicles less reliably; that's why the deployment moved to 4K.)
- Windows-friendly relative paths: `ring_path: "data/ring"`,
  `segment_index_path: "data/index/segments.sqlite"`, `output_path: "data/events"`.
- Three-tier evidence thresholds under `events.relative_speeding` (as deployed):
  `absolute_kmh_threshold: 55` → text log row for every violation;
  `screenshot_kmh_threshold: 65` → adds a still of the car;
  `clip_kmh_threshold: 70` → full 30 s clip + annotated render.

Smoke test (30 s of record + analyze, then check the ring):

```powershell
traffic-log run --config config\config.run.local.yaml --max-seconds 30
dir data\ring          # expect dated folders of 10s segment_YYYYMMDD-HHMMSS.mp4
traffic-log health --config config\config.run.local.yaml    # exit 0 = fresh segments
```

## 3. Calibrate (once per camera position)

1. **De-warp:** tune `calibration.undistort.k1` until known-straight curbs are
   straight (`scripts/dewarp_check.py`, `scripts/undistort_tune.py` help). k1 is
   resolution-independent (same value for sub and 4K).
2. **Road quad:** grab a frame, click the 4 road corners on the *de-warped*
   image, write them back:

   ```powershell
   traffic-log calibrate --source frame.png --config config\config.run.local.yaml --write
   ```

3. **Metric speed:** set `calibration.units: "meters"` and the measured
   `target_width_units` / `target_length_units` (metres the quad spans across /
   along the road; Google Maps "measure distance" works).
4. **Validate with GPS drive-bys:** drive past at a known GPS speed, then label
   the pass on the dashboard's **speed test** page (it lists every measured
   pass, including sub-threshold ones). Use the per-lane errors to set
   `calibration.speed_across_correction` (near/far factors). Re-check after any
   camera move.

## 4. Permanent run: `supervise` + Task Scheduler

`supervise` starts `run`, restarts it if it dies, and (unless `all_day`) gates
it to the civil-twilight window computed from `schedule.latitude/longitude`.
The deployed config runs 24/7 (`schedule.all_day: true`; the camera has a
usable night exposure).

Autostart is a **Task Scheduler** job (e.g. `TrafficSupervisor`, trigger: at
logon) that runs `scripts\run_supervisor.vbs`. The VBS launches the supervisor
with **no console window**, self-locates the repo root, and appends all output
to `data\logs\supervise.log`:

```
wscript.exe C:\path\to\repo\scripts\run_supervisor.vbs
```

Note the VBS hardcodes `config\config.run.local.yaml`; keep the live config at
that name. To check on it:

```powershell
Get-Content data\logs\supervise.log -Tail 50
traffic-log health --config config\config.run.local.yaml
```

## 5. Dashboard

```powershell
traffic-log serve --new-link --config config\config.run.local.yaml   # mint the unlock link
traffic-log serve --config config\config.run.local.yaml              # run it (port 8090)
tailscale funnel 8090                                                # expose publicly
```

The dashboard 404s every request until a device visits the secret unlock link
once; that sets a signed 30-day cookie. `--new-link` rotates **both** the token
and the session secret, so old links *and* existing device sessions are revoked.
`serve` only reads the analyzer's stores, so it is safe to start/stop
independently of the capture+analysis process. See `docs/dashboard.md` for the
pages.

## 6. Operations

- **Reports:** `traffic-log speed-report --days 7 --csv out.csv` (violations
  summary + evidence CSV); `traffic-log police-report --hours 24` (if police
  recognition is enabled; currently off in the deployed config).
- **Ring retention:** pruning is automatic (oldest-first to `ring_max_gb`).
  `traffic-log prune-ring` runs one manual pass.
- **Bad speed readings:** speeds are validated at write time by the invariant
  predicates in `analyze/pass_validity.py`; dashboards and reports read only
  validated speeds. If a new artifact class shows up (an impossible "top
  speed"), **add a predicate there and re-run `scripts/revalidate_passes.py`**.
  Do not add filtering to the web/read path.
- **Manual data surgery** (deleting events, purging history) is logged in
  `docs/maintenance-log.md`; keep that up to date, since `data/` is gitignored
  and leaves no git trail.
- **Manual clip export:** `traffic-log export-event --start-ts <a> --end-ts <b>`
  cuts a window from the ring. If a recording gap overlaps a clip window, the
  export clamps to the contiguous run containing the trigger and records
  `gap_truncated` in the metadata.

## Appendix: legacy mini-PC/Docker path

The original Milestone-7 plan, a Linux mini-PC running the slim capture image
(`Dockerfile.capture`) with analysis elsewhere, is still supported by the
repo: `docker-compose.yml` (capture/analyze/analyze-gpu services, compose
healthcheck wired to `traffic-log health`), `config/config.mini_pc.yaml`,
`systemd/traffic-capture.service`, and `scripts/setup_mini_pc.sh` /
`scripts/install_ubuntu_deps.sh`. Camera probing for v4l2 devices
(`traffic-log probe-camera`) belongs to that path. It is unmaintained relative
to the single-box deployment but kept for a future second site.
