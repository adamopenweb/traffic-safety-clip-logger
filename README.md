# Traffic Safety Clip Logger

[![tests](https://github.com/adamopenweb/traffic-safety-clip-logger/actions/workflows/ci.yml/badge.svg)](https://github.com/adamopenweb/traffic-safety-clip-logger/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

A local-first system that watches a residential street 24/7, measures the
speed of **every** passing vehicle against a GPS-validated calibration, and
automatically saves tiered evidence (from a log row up to an annotated 4K
clip) for the ones driving dangerously.

**Deployed and running around the clock** on a single Windows PC (RTX 4080)
with a 4K PoE camera: one process stream-copies the camera's H.265 feed into a
~9-day ring buffer *and* analyzes it live on the GPU; a supervisor keeps it
alive; a web dashboard behind a secret-link gate reports results as
*"X% of N cars"*, because every pass is counted, not just the violators. See
**[DEPLOY.md](DEPLOY.md)** for the bring-up.

> **This repo is a portfolio snapshot of a private working system, built
> end-to-end with AI coding agents.** If that's what brought you here, start
> with:
>
> - **[How this was built (human + AI)](docs/building-with-ai.md)**: the
>   workflow, and the repo artifacts that document it
> - **[Engineering challenges & solutions](docs/challenges.md)**: the war
>   stories. Cross-stream sync drift, defensible speed measurement, an honest
>   ML failure, Windows deployment.
> - **[`traffic.md`](traffic.md)**: the original ~1,300-line spec that drove
>   development, written before any code
> - **[AI architecture review](docs/engineering-review-2026-06.md)** → its
>   [migration plan](docs/unified-store-plan.md) → the shipped result
>
> This is a point-in-time snapshot; active development continues in the
> private original, and improvements are synced here periodically.

*(Dashboard screenshots are coming. Every capture shows the real street, so
they need curation/redaction first.)*

**At a glance:** ~140,000 vehicle passes measured · GPS-drive-by-validated
km/h · three evidence tiers (log row → still → annotated clip) · 300+ tests
that need no camera, GPU, or CV stack (see the note under
[Requirements](#requirements)).

## Unsafe behaviors targeted

1. Vehicles moving unusually fast relative to normal traffic.
2. Vehicles using the shared center turning lane as a passing lane.
3. Loud engine / aggressive acceleration noise (optional, audio-dependent).

## Architecture

Everything runs on **one Windows box** with an NVIDIA GPU; the camera does the
video encoding:

| Piece | Role |
| --- | --- |
| 4K PoE RTSP camera (iSee CCIPB812-Z-4K) | Hardware H.265 encode; main (4K) + sub (D1) streams |
| `run`: recorder thread | Stream-copies the 4K main stream into 10s ring segments (~300 GB, ~9 days) |
| `run`: live analyzer | Same 4K stream, downscaled + de-warped → YOLOv8 (CUDA) + ByteTrack + rules |
| `run`: clip exporter thread | Cuts 30s evidence clips from the ring after each event's post-roll; renders annotated copies |
| `supervise` | Keeps `run` alive (24/7 or civil-twilight window), restarts on crash |
| `serve` | FastAPI dashboard over the analyzer's stores, behind a 404-everything secret-link gate |

The original two-box design (Linux mini-PC capture appliance + separate
analysis box, via Docker) is retained in the repo but is not the active
deployment; see the appendix in [DEPLOY.md](DEPLOY.md).

## Why Windows?

Most projects like this reach for a Linux SBC or a mini-PC appliance, and the
original plan here did too (the Docker/mini-PC scaffolding still lives in this
repo as the legacy path). The deployment ended up on Windows for a simple,
pragmatic reason: **the ideal machine already existed.** An always-on Windows
PC with an RTX 4080 was sitting a few metres from the camera mount, with more
GPU than the pipeline can use.

That decision held up because the workload turned out to be a natural fit:

- **The camera does the hard part.** Recording is a stream-copy of the
  camera's hardware H.265; the PC never encodes video, it just writes
  segments and runs inference. CUDA PyTorch, ffmpeg, and Python are all
  first-class on Windows.
- **One box beats two.** The single-box design eliminated a whole class of
  problems the two-box plan carried: no cross-machine video transport, no
  fleet of configs, no "which clock is right" between recorder and analyzer.
- **The costs were real but bounded.** Windows charged for the decision in
  specific, fixable ways: ffmpeg's segment muxer can't use glibc's `%s`
  filename token, wall-clock filenames meet DST, autostart means Task
  Scheduler plus a hidden-window VBScript, stopping the analyzer means killing
  a process *tree*, and Windows ships no timezone database. Each one is
  documented, with its fix, in [docs/challenges.md](docs/challenges.md#6-the-windows-specific-deployment-problems).

The result: hardware cost of the compute tier, $0; and the machine still does
its day job.

## Requirements

- Python **3.11+**. The core install and tests are pure-Python and run
  anywhere; the CV stack (ultralytics, supervision, opencv, torch) is behind
  the `analyze` extra and needs a CUDA torch build for GPU inference.
- **No camera, GPU, or CV stack is needed to run the test suite.** Tracking
  (ByteTrack), debug video, and the whole offline pipeline are exercised under
  `pytest` with a scripted detector and synthetic video (that's what CI runs).
- ffmpeg on PATH (recording, clip export, thumbnails).
- Docker is only needed for the legacy mini-PC path.

## Quick start (development)

```bash
pip install -e .[dev]      # core + pytest only (no heavy CV install)
traffic-log --help         # lists all subcommands
traffic-log test --source samples/street-test.mp4 --config config/config.dev.yaml
pytest -q                  # real tests pass; later-milestone tests are skipped
```

A missing `--source` file still exits 0 (the M0 `test` runs a dry stub), so you
don't need a sample video committed to try it.

## Docker (legacy mini-PC path only)

The active deployment runs natively on Windows, with no containers. The
compose services remain for the original two-box design:

```bash
docker compose build capture analyze   # CPU image (mini-PC)
docker compose build analyze-gpu       # GPU image (needs NVIDIA toolkit)
```

The CPU image is `Dockerfile`; the GPU image is `Dockerfile.gpu` (PyTorch CUDA
base). The `capture`/`analyze` services mount `/dev/video0` and are mini-PC
only; `analyze-gpu` runs the offline stub against a sample clip.

## CLI commands

| Command | Purpose | Status |
| --- | --- | --- |
| `run` | Record the ring + analyze live together (**the deployed mode**) | ✅ M7 |
| `supervise` | Keep `run` alive 24/7 (or daylight-only by lat/long), auto-restart | ✅ M7 |
| `serve` | Web dashboard (FastAPI) behind a secret-link gate; `--new-link` rotates access | ✅ |
| `analyze` | Analysis only: RTSP stream / device / file | ✅ M7 |
| `capture` | Recording only: camera → ring-buffer segments | ✅ M1 |
| `calibrate --source <img/video>` | 4-point calibration → lane-band preview | ✅ M3 |
| `test --source <file>` | Offline detection + tracking on a saved video | ✅ M2 |
| `export-event --start-ts --end-ts` | Export a clip for a time window from the ring | ✅ M6 |
| `speed-report` | Violation summary + evidence CSV from the speed log | ✅ |
| `police-report` | Police-vehicle sightings over a window (when enabled) | ✅ |
| `prune-ring` | Prune the ring buffer to its size cap | ✅ M1 |
| `health` | Exit 0 if recording is fresh, else 1 | ✅ M7 |
| `probe-camera` | List `/dev/video*` devices (legacy v4l2 path) | ✅ M1 |

### Recording pipeline (M1)

The recorder runs ffmpeg's segment muxer to write fixed-length segments into
`<ring_path>/incoming/segment_<YYYYMMDD-HHMMSS>.mp4` (local wall-clock names,
portable to Windows; decoding back to unix time is timezone- and DST-fold-aware,
and the fall-back repeat hour can never overwrite an indexed segment). An RTSP
source (the deployed case) is **stream-copied**: the camera's hardware encoder
does the work. A v4l2 source is copied when it's already H.264 and re-encoded
when raw. An indexer loop moves each completed segment into its dated folder
`<ring_path>/YYYY-MM-DD/`, probes it with ffprobe, and records
`path, start/end ts, duration, size, codec, resolution, fps` in the SQLite
segment index. The segment currently being written never enters the index, so
ring pruning (oldest-first, down to `ring_max_gb`) can never delete the active
segment or any event clip. The ffmpeg subprocess is supervised and restarted
with backoff on failure, so Wi-Fi/camera hiccups don't stop recording.

### Offline analysis (M2)

`traffic-log test --source <file>` reads a video with OpenCV, samples frames at
`analysis.inference_fps`, and runs:

1. **Detection.** `Detector` is abstract; `YoloDetector` wraps Ultralytics YOLO
   and filters to vehicle classes, returning `supervision.Detections`. A
   `ScriptedDetector` provides deterministic detections for tests/demos without
   torch.
2. **Tracking.** `VehicleTracker` wraps Supervision `ByteTrack` and feeds
   tracked boxes into a pure `TrackStore`. Each `Track` keeps timestamped bbox
   history, bottom-center road-contact points, a camera-relative direction
   estimate, and age: the inputs the M4/M5 rules consume.
3. **Debug output.** When `analysis.save_debug_video` is set, an annotated MP4
   with boxes + track-id/direction labels is written to `data/debug/`.

If the CV stack isn't installed, `test` falls back to a dependency-free stub so
the command still runs on a core-only box.

### Calibration and lane bands (M3)

```bash
# Click 4 road corners interactively (needs a GUI)...
traffic-log calibrate --source samples/street-test.mp4 --config config/config.dev.yaml
# ...or pass them non-interactively and write them into the config:
traffic-log calibrate --source frame.png --points "120,160 520,160 600,440 40,440" --write
```

A NumPy DLT homography (`analyze/project.py`) maps the clicked quadrilateral to
the normalized unit-square road plane: x runs across the road (drives lane
bands), y runs along it (drives speed in M4). The road width is split into five
bands by configurable ratios (`calibration.lane_model`), normalized to sum to 1.
During analysis each track's bottom-center point is projected and assigned a
lane band per frame, building a per-track lane history the M5 center-lane rule
will consume. `calibrate` writes a preview image (`data/calibration/`) with the
bands overlaid and prints a pasteable `source_points` snippet.

### Relative speeding (M4)

When calibration is present, each processed frame estimates a track's smoothed
speed (normalized units/sec) from its projected ground-point path
(`analyze/metrics.py`) and feeds it into a **rolling per-direction baseline**
(median + 85/90/95/97th percentiles, time-windowed). The
`RelativeSpeedingRule` flags a track whose speed percentile stays above the
threshold for a minimum duration. Both the percentile threshold and the minimum
duration are interpolated from the single `events.aggressiveness` knob
(`*_strict` → `*_sensitive`). Each candidate carries evidence (speed,
percentile, rolling median, threshold, duration) and a `warmup` flag while the
baseline is still immature; a per-track cooldown prevents repeat triggers.
Candidate events are reported in the analysis summary; turning them into saved
30-second clips with metadata + thumbnails is Milestone 6.

### Center-lane passing (M5)

`CenterLanePassRule` consumes each track's lane-band history and along-road
progress to detect two patterns:

- **Pattern A (fast center-lane traversal):** a moving vehicle dwells in the
  `center_turn_lane` for ≥ the (aggressiveness-tuned) minimum time while its
  speed percentile is above the center-lane threshold.
- **Pattern B (overtake through the center lane, the stronger signal):** a
  candidate that starts *behind* another same-direction vehicle, moves through
  the center lane, and ends *ahead* of it within the overtake window. Evidence
  records the passed vehicle's track id, the before/after relative positions,
  the lane sequence, dwell time, and speed percentile.

### Road-plane coordinate convention

Calibration maps the road to a normalized plane where **x runs across the road**
(it selects the lane band) and **y runs along the road** (travel/length axis).
Once calibrated, a track's **direction** and overtake **progress** are measured
along y, so a vehicle staying within one lane still has a well-defined direction
and ordering versus other vehicles. Click the four corners so the road's width
maps to the x axis (the `calibrate` preview makes this easy to verify). Without
calibration, direction falls back to image-x motion.

#### Relative vs metric (km/h) speed

By default speed is **relative** (normalized units/sec) and the speeding rule
ranks each vehicle against the rolling per-direction distribution, with no road
measurements needed. To report **km/h**, set in the calibration config:

```yaml
calibration:
  units: "meters"
  target_width_units:  <metres the quad spans ACROSS the road>
  target_length_units: <metres the quad spans ALONG the road>
```

Measure those two distances once (e.g. Google Maps "measure distance" over the
calibration quad's footprint). Then `speed_kmh` appears in event metadata and
the debug video labels, and `peak_kmh` in the analysis summary. A known-speed
drive-by is a good way to validate the metric calibration. Because the planar
homography is an approximation, absolute km/h is most accurate in the lane used
to validate; relative speeding is robust regardless.

### Event clip export (M6)

The `EventManager` turns rule candidates into saved events: it scores them,
deduplicates repeats, and **merges** candidates that share a track within
`events.merge_window_seconds` into one `FinalEvent` whose `event_types` lists
every contributing rule. For each final event the analyzer exports, into
`data/events/YYYY-MM-DD/<event_type>/<YYYYMMDD_HHMMSS>_<event_type>_<id>.*`:

- a **`.mp4`** clip covering `[trigger − pre_roll, trigger + post_roll]`
  (default 30s), trimmed with ffmpeg from the source video (offline) or by
  concatenating the overlapping **ring segments** from the segment index
  (live / `export-event`);
- a **`.json`** metadata sidecar (event id/types, timestamps, score, primary
  track, per-track direction/lane-sequence/speed, all triggers' evidence, and a
  config snapshot);
- a **`.jpg`** thumbnail extracted at `thumbnail_time_offset_seconds`.

`traffic-log test --source <file>` now writes real event clips; `traffic-log
export-event --start-ts <a> --end-ts <b>` exports a manual clip from the ring.
If a recording gap (ffmpeg reconnect) overlaps a clip window, the export clamps
to the contiguous ring run containing the trigger and records `gap_truncated`
in the metadata, so timelines and annotation boxes stay correct.

### Live deployment (M7)

`supervise` keeps a `run` process alive around the clock (Task Scheduler +
`scripts/run_supervisor.vbs` on the deployed box). `run` records the 4K ring
and analyzes live; each finalized event is relabeled with its **validated
full-track speed** (the GPS-checked metric; partial-track trigger speeds read
high) and dispatched into **three evidence tiers** by km/h: a text log row for
every violation over the gate, plus a still image of the car at the mid tier,
plus a full 30 s clip with an annotated render at the top tier. Every completed
vehicle pass, speeding or not, is written to the unified store as the traffic
denominator; speed validity is enforced at write time by the invariant
predicates in `analyze/pass_validity.py` (re-runnable over history via
`scripts/revalidate_passes.py`).

`serve` is the dashboard over those stores: today/stats/browse pages, a
top-speeds leaderboard, and the GPS speed-test page used to validate
calibration. Auth is a
404-everything gate unlocked by a secret link (`serve --new-link` rotates the
link *and* revokes existing sessions); expose it with `tailscale funnel`. See
[docs/dashboard.md](docs/dashboard.md).

All commands accept `--config <path>` (default `config/config.mini_pc.yaml`;
the deployed box passes `config/config.run.local.yaml`) and `--log-level`.

## Configuration

Tracked config files under `config/`:

- `config.example.yaml`: fully documented reference.
- `config.camera.yaml`: template for the deployed 4K RTSP camera (fill in
  credentials, then copy to a `*.local.yaml`).
- `config.dev.yaml`: offline file analysis on the GPU box, no recording.
- `config.mini_pc.yaml`: legacy Ubuntu appliance profile (Docker path).

The **live run config is `config/config.run.local.yaml`**; `config/*.local.yaml`
is gitignored because it holds the camera's RTSP credentials and the dashboard's
`web.access_token` / `web.session_secret`.

A single **aggressiveness** knob (`events.aggressiveness`, 0.0 strict → 1.0
sensitive) interpolates every rule's thresholds. Target: ≤ ~5 useful clips/day.

## Sample video workflow

1. Grab a recorded segment (or several) from the ring under `data/ring/<date>/`.
2. Copy the file into `samples/`.
3. Run offline analysis: `traffic-log test --source samples/<file>.mp4 --config config/config.dev.yaml`.
4. Inspect debug output and event clips under `data/events/`.

## Milestone roadmap

0. **Bootstrap** → 1. Camera recording / ring buffer → 2. Offline detection +
tracking → 3. Calibration + lane bands → 4. Relative speeding → 5. Center-lane
passing → 6. Event clip export → 7. **Live deployment** (landed as a single
Windows box rather than the planned mini-PC POC), all complete. Remaining:
8. Optional audio (loud-engine rule, stubbed).

## Licensing

Project code is MIT (see [LICENSE](LICENSE)). Optional ML dependencies keep
their own licenses; notably, **Ultralytics YOLOv8 is AGPL-3.0**, which carries
source-disclosure obligations for combined works, including network use. This
project imports it as an optional runtime dependency and vendors none of it,
and the `Detector` interface is abstract precisely so YOLO can be swapped out.
Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before any commercial
use.

## Privacy & responsible use

Personal traffic-safety logging and review only. **No** license-plate
recognition, **no** driver/face identification, **no** blurring by default. The
software does not claim legal-grade evidence quality or enforcement-grade speed
accuracy. These boundaries were designed in from the first line of the spec
(see the "Privacy / Safety Defaults" and "Out of Scope" sections of
[`traffic.md`](traffic.md)), not retrofitted.

If you deploy something like this yourself, the working rules this project
operates by:

- **Keep footage local.** Recording, analysis, and storage stay on your own
  hardware. Nothing is uploaded anywhere; the dashboard only reads local
  stores and sits behind an authenticated gate.
- **Point it at public space only.** A public roadway carries no reasonable
  expectation of privacy; aim and calibrate so private property (yards,
  windows, doorways) stays out of the analysis region and, as far as
  practical, out of frame.
- **Retention is bounded by design.** The ring buffer prunes itself (~9 days
  here). Event clips are the only long-lived footage, and they are a curated
  handful per day, not an archive.
- **Never publish identifiable footage.** Redact faces and plates before any
  clip or screenshot leaves your machine. This repository publishes none.
- **No automated enforcement.** Output is for human review and community
  conversation (traffic-calming requests, city engagement), never automated
  reporting of individuals. The spec lists this as explicitly out of scope.
- **Know your local law.** This deployment operates under Canadian/Ontario
  norms for video recording of public space (and records no audio); check
  the rules where you live before deploying.
