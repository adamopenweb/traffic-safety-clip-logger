# claude.md — Traffic Safety Clip Logger

> *This is the original build brief that drove development, preserved as
> written. Some assumptions were superseded during the project: notably, the
> Ubuntu mini-PC + Orbbec Astra Mini capture appliance described below was
> replaced by a single Windows box recording a 4K RTSP camera (see the README
> and DEPLOY.md for what actually shipped), and the ring buffer grew from
> 200 GB to 300 GB. The spec is kept as-is because the delta between plan and
> shipped system is part of the story.*

## Project Summary

Build a local-first traffic safety video logging system for a residential street. The system continuously records video from an outdoor camera, analyzes traffic for unsafe driving behavior, and automatically saves short evidence clips into event folders for later review.

This is a proof-of-concept system intended to reduce manual review of camera footage. The first version should prioritize reliability, simple deployment, and practical detection of obvious events over perfect accuracy.

Primary unsafe behaviors to detect:

1. Vehicles moving unusually fast relative to normal traffic.
2. Vehicles using the shared center turning lane as a passing lane.
3. Loud engine / aggressive acceleration noise as a supporting signal or standalone event.

The system should be Linux-first, Dockerized, and runnable on an Ubuntu mini-PC near the camera. Development will happen on a Windows 11 desktop with an RTX 4080 using WSL2 Ubuntu.

---

## Known User Environment

### Camera

Initial proof-of-concept camera:

* Orbbec Astra Mini
* Use RGB/color only for MVP
* Color stream supports:

  * 1280x960 @ 30fps
  * RGB888 / YUV422
* IR stream also supports 1280x960 @ 30fps, but is out of scope for daytime MVP
* Depth is out of scope for MVP:

  * 1280x960 depth only 5fps
  * 640x480 depth can do 30fps
  * Outdoor daylight makes depth unreliable

Camera installation assumptions:

* Mounted outside or near front porch
* Approximately 20ft high
* Approximately 20–30 degree downward angle
* Camera roughly perpendicular to road
* Typical vehicle distance: 40–60ft
* Daytime-only detection for MVP

### Road Layout

The camera watches a busy 3-lane road with bike lanes:

* One travel lane each direction
* One shared center turning lane
* Bike lanes on both sides
* Road is mostly flat and visible in-frame
* Entire road is treated as a community safety zone
* Main issues:

  * Speeding
  * Passing in the center turning lane
  * Loud engine noise / aggressive driving

### Hardware / Deployment

Deployment target:

* Ubuntu Linux mini-PC near the camera
* Camera physically attached to mini-PC via USB
* Mini-PC connected by 5 GHz Wi-Fi, approximately 20ft from router
* Local recording on mini-PC is required so Wi-Fi interruptions do not lose evidence
* Ring buffer cap: 200GB
* Event clip storage: no maximum for now

Development target:

* Windows 11 desktop with NVIDIA RTX 4080 16GB
* Use WSL2 Ubuntu as canonical development environment
* Dockerized Linux-first application
* Do not build as a native Windows app
* MacBook Air can be used only as optional remote-control/editor machine, not as canonical dev target

Optional later:

* RTX 4080 machine may consume RTSP stream from mini-PC for live heavy inference
* RTX 4080 machine may run offline analysis on copied clips
* Mini-PC may eventually run enough analysis locally if performance is acceptable

---

## Core Design Decision

The system must separate:

1. **Capture / Recording Appliance**

   * Runs on Ubuntu mini-PC
   * Records from Astra Mini
   * Maintains 200GB ring buffer
   * Exposes optional RTSP H.264 stream
   * Exports event clips

2. **Analysis / Development Environment**

   * Developed in WSL2 Ubuntu on RTX 4080 machine
   * Must support offline analysis on recorded video files
   * Must support optional live analysis from RTSP stream
   * Uses Supervision + YOLO + ByteTrack for detection/tracking

The camera should not be plugged directly into the Windows box for MVP. Avoid WSL USB camera passthrough complexity.

---

## MVP Scope

### Must Have

* Continuous video recording from Astra Mini RGB stream
* H.264 segment recording to local disk
* 200GB ring buffer pruning
* Simple folder-based event output
* Offline video analysis mode
* Vehicle detection and tracking
* Perspective calibration using a 4-point road surface selection
* Relative speed estimation
* Lane-band inference:

  * left bike lane
  * left travel lane
  * center turning lane
  * right travel lane
  * right bike lane
* Event detection:

  * relative speeding
  * center lane passing / high-speed center lane traversal
  * loud engine audio event, if mic/audio available
* 30-second event clip export
* JSON metadata sidecar per event
* Thumbnail per event
* Docker Compose deployment
* Basic logs and health checks

### Nice to Have

* RTSP H.264 stream from mini-PC
* Static HTML event index
* Audio-based event score boosting
* Optional live analysis from RTSP stream on RTX 4080 machine
* Scene health check for obstructed/blurry camera
* Basic systemd units or Docker restart policies

### Out of Scope for MVP

* License plate recognition
* Driver identification
* Face/plate blurring by default
* Multi-camera support
* Nighttime optimization
* Depth-based analysis
* Legal-grade speed measurement
* Automated reporting to authorities
* Native Windows implementation

---

## Privacy / Safety Defaults

Do not implement plate recognition or identity detection.

Do not default to face or plate blurring for this POC.

The software should be framed as a personal traffic safety logging and review system. It should not claim legal-grade evidence quality or enforcement-grade speed accuracy.

---

## Network Assumptions

The mini-PC will use 5 GHz Wi-Fi approximately 20ft from router.

Do not send raw video over the network.

Raw 1280x960 RGB @ 30fps is roughly 845 Mbps before overhead and is not appropriate for Wi-Fi.

Use compressed H.264 for any network stream.

Default stream settings:

```yaml
network:
  mode: "wifi_poc"
  expose_rtsp: true
  rtsp_port: 8554
  stream_codec: "h264"
  stream_bitrate_mbps: 8
  stream_max_bitrate_mbps: 12
  keyframe_interval_seconds: 1
```

Wi-Fi interruptions must not stop local recording. If live RTSP analysis disconnects, reconnect automatically.

---

## Technology Stack

Use Python 3.11+.

Required libraries/tools:

* Docker / Docker Compose
* ffmpeg
* OpenCV
* NumPy
* PyYAML
* Supervision by Roboflow
* Ultralytics YOLO for initial detector
* ByteTrack through Supervision where practical
* SQLite or JSONL for segment index
* pytest for rule logic tests

Preferred CV pattern:

* Detector outputs are converted to `supervision.Detections`
* Tracking uses `supervision.ByteTrack`
* Perspective transform uses OpenCV / Supervision-friendly primitives
* Lane bands and unsafe behavior rules are custom project logic

Important licensing note:

* YOLOv8 / Ultralytics licensing may matter if this becomes commercial.
* For this personal POC, YOLOv8 is acceptable.
* Keep detector abstraction clean so another model can replace YOLO later.

---

## Repository Layout

```text
traffic-safety-logger/
  claude.md
  README.md
  pyproject.toml
  docker-compose.yml
  Dockerfile
  Dockerfile.gpu
  Makefile

  config/
    config.example.yaml
    config.dev.yaml
    config.mini_pc.yaml

  scripts/
    install_ubuntu_deps.sh
    setup_wsl_dev.sh
    setup_mini_pc.sh
    test_camera_formats.sh

  systemd/
    traffic-capture.service
    traffic-analyze.service

  src/
    traffic_logger/
      __init__.py

      main.py
      config.py

      capture/
        __init__.py
        recorder.py
        segment_index.py
        ring_pruner.py
        camera_probe.py
        rtsp_server.py

      analyze/
        __init__.py
        detector.py
        tracker.py
        project.py
        lane_model.py
        metrics.py
        offline.py
        live.py

        rules/
          __init__.py
          base.py
          relative_speeding.py
          center_lane_pass.py
          loud_engine.py

      audio/
        __init__.py
        capture.py
        loudness.py

      events/
        __init__.py
        manager.py
        exporter.py
        metadata.py
        thumbnail.py

      util/
        __init__.py
        logging.py
        time.py
        ffmpeg.py
        paths.py

  tests/
    test_lane_model.py
    test_relative_speeding.py
    test_center_lane_pass.py
    test_ring_pruner.py
    test_event_manager.py

  samples/
    README.md

  data/
    .gitkeep
```

`data/` must be gitignored except `.gitkeep`.

---

## Docker Requirements

The app must be Dockerized.

Support at least two modes:

1. CPU / portable image
2. GPU-capable image for RTX 4080 development/testing

### Docker Compose Services

Minimum services:

```yaml
services:
  capture:
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    volumes:
      - ./config:/app/config
      - ./data:/data
    devices:
      - "/dev/video0:/dev/video0"
    command: ["traffic-log", "capture", "--config", "/app/config/config.mini_pc.yaml"]

  analyze:
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    volumes:
      - ./config:/app/config
      - ./data:/data
    command: ["traffic-log", "analyze", "--config", "/app/config/config.mini_pc.yaml"]
```

For GPU development on WSL2 RTX 4080:

```yaml
services:
  analyze-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
    volumes:
      - ./config:/app/config
      - ./data:/data
      - ./samples:/samples
    command: ["traffic-log", "test", "--source", "/samples/street-test.mp4", "--config", "/app/config/config.dev.yaml"]
```

Do not assume CUDA is available in the mini-PC runtime.

---

## CLI Requirements

Implement a CLI command called `traffic-log`.

Required commands:

```bash
traffic-log probe-camera --config config/config.mini_pc.yaml
traffic-log capture --config config/config.mini_pc.yaml
traffic-log analyze --config config/config.mini_pc.yaml
traffic-log run --config config/config.mini_pc.yaml
traffic-log calibrate --config config/config.mini_pc.yaml
traffic-log test --source samples/street-test.mp4 --config config/config.dev.yaml
traffic-log export-event --start-ts 123 --end-ts 456 --config config/config.mini_pc.yaml
traffic-log prune-ring --config config/config.mini_pc.yaml
```

### `probe-camera`

Should run on mini-PC and print:

* available `/dev/video*` devices
* supported resolutions
* supported pixel formats
* supported frame rates

Use `v4l2-ctl` if available, or OpenCV fallback.

### `capture`

Captures camera stream and writes continuous H.264 segments to ring buffer.

### `analyze`

Runs live analysis on either:

* local camera stream
* RTSP stream
* existing ring-buffer segments

The MVP can initially support analyzing from a file before live stream support.

### `run`

Runs capture and analyze in one process group for simple local testing. Docker Compose may run them separately.

### `calibrate`

Interactive helper:

* Load a frame from camera or sample image
* User clicks four road-surface corners
* Save `source_points` to config
* Compute and save transform config
* Generate a preview image with projected lane bands

### `test`

Run offline analyzer against a saved video file.

This is required for development on the RTX 4080 box without a live camera.

---

## Config File

Create `config/config.example.yaml`.

```yaml
app:
  name: "traffic-safety-logger"
  timezone: "America/Toronto"
  log_level: "INFO"

camera:
  profile: "astra_mini"
  source: "/dev/video0"
  capture_resolution: [1280, 960]
  capture_fps: 30
  pixel_format_preference: ["YUYV", "MJPG", "RGB3"]
  rotate_degrees: 0
  flip_horizontal: false
  flip_vertical: false

recording:
  enabled: true
  codec: "h264"
  target_bitrate_mbps: 10
  max_bitrate_mbps: 15
  segment_seconds: 10
  ring_max_gb: 200
  ring_path: "/data/ring"
  segment_index_path: "/data/index/segments.sqlite"

network:
  expose_rtsp: true
  rtsp_port: 8554
  rtsp_path: "/street"
  stream_codec: "h264"
  stream_bitrate_mbps: 8
  stream_max_bitrate_mbps: 12
  keyframe_interval_seconds: 1
  reconnect: true

analysis:
  enabled: true
  source: "file_or_stream"
  inference_fps: 12
  inference_input_size: 640
  device: "auto"
  save_debug_video: false
  save_debug_frames: false

models:
  detector_type: "yolov8"
  yolo_model: "yolov8s.pt"
  confidence_threshold: 0.35
  iou_threshold: 0.5
  vehicle_classes:
    - "car"
    - "truck"
    - "bus"
    - "motorcycle"

tracking:
  tracker_type: "bytetrack"
  track_activation_threshold: 0.25
  lost_track_buffer: 30
  minimum_matching_threshold: 0.8
  minimum_consecutive_frames: 3

calibration:
  mode: "relative"
  source_points: []
  target_width_units: 1.0
  target_length_units: 1.0
  lane_model:
    enabled: true
    bike_lane_width_ratio: 0.12
    travel_lane_width_ratio: 0.28
    center_lane_width_ratio: 0.20
    auto_normalize: true

events:
  output_path: "/data/events"
  clip_total_seconds: 30
  pre_roll_seconds: 10
  post_roll_seconds: 20
  thumbnail_time_offset_seconds: 15
  cooldown_seconds: 8
  merge_window_seconds: 12

  aggressiveness: 0.3

  relative_speeding:
    enabled: true
    percentile_threshold_strict: 0.97
    percentile_threshold_sensitive: 0.90
    min_duration_seconds_strict: 0.8
    min_duration_seconds_sensitive: 0.4
    min_tracks_for_baseline: 20
    rolling_window_minutes: 60

  center_lane_pass:
    enabled: true
    center_lane_min_time_seconds_strict: 0.8
    center_lane_min_time_seconds_sensitive: 0.4
    speed_percentile_threshold_strict: 0.90
    speed_percentile_threshold_sensitive: 0.80
    detect_overtake: true
    overtake_window_seconds: 6

audio:
  enabled: false
  source: "default"
  sample_rate: 16000
  window_seconds: 0.25
  loud_engine:
    enabled: true
    db_over_baseline_strict: 12
    db_over_baseline_sensitive: 6
    min_duration_seconds: 0.5
    baseline_window_minutes: 10
    boost_visual_event_score: true

privacy:
  blur_faces: false
  blur_plates: false
  plate_recognition: false

debug:
  draw_tracks: true
  draw_lane_bands: true
  draw_event_labels: true
```

---

## Camera Capture Requirements

### Astra Mini RGB MVP

Use only the color stream.

Preferred capture:

* 1280x960
* 30fps
* YUYV/YUV422 if stable
* MJPG if available and more stable
* RGB fallback if necessary

The app must include a camera probe command and logs showing the actual selected format.

### Recording Pipeline

Use ffmpeg or GStreamer to encode and segment.

Segment filename format:

```text
/data/ring/YYYY-MM-DD/segment_<start_unix_ms>.mp4
```

Each segment must be recorded in the segment index with:

* path
* start timestamp
* end timestamp
* duration
* file size
* codec
* resolution
* fps

### Ring Buffer Pruning

Maintain max ring buffer size of 200GB.

Pruning rules:

* Delete oldest segments first
* Update segment index
* Never delete event clips
* Avoid deleting active segment currently being written

---

## Event Clip Export Requirements

Default clip:

* 30 seconds total
* 10 seconds pre-roll
* 20 seconds post-roll

When event triggers at `trigger_ts`, export:

```text
[trigger_ts - 10s, trigger_ts + 20s]
```

Use ffmpeg to concatenate/trim from ring buffer segments.

Output folder:

```text
/data/events/YYYY-MM-DD/<event_type>/
```

Output files:

```text
<YYYYMMDD_HHMMSS>_<event_type>_<event_id>.mp4
<YYYYMMDD_HHMMSS>_<event_type>_<event_id>.json
<YYYYMMDD_HHMMSS>_<event_type>_<event_id>.jpg
```

Events must not be lost if multiple triggers occur close together.

Implement event deduplication:

* Same event type + same primary track ID within cooldown should merge
* Multiple event types within `merge_window_seconds` should optionally create one combined event
* Metadata should include all labels/triggers

---

## Metadata Schema

Each event must have JSON sidecar metadata:

```json
{
  "event_id": "uuid",
  "event_type": "center_lane_pass",
  "event_types": ["center_lane_pass", "relative_speeding"],
  "created_at": "2026-06-08T13:00:00-04:00",
  "start_ts": 1780923590.0,
  "trigger_ts": 1780923600.0,
  "end_ts": 1780923620.0,
  "clip_path": "/data/events/2026-06-08/center_lane_pass/...",
  "thumbnail_path": "/data/events/2026-06-08/center_lane_pass/...",
  "score": 0.87,
  "primary_track_id": 42,
  "tracks": [
    {
      "track_id": 42,
      "direction": "left_to_right",
      "lane_band_sequence": ["travel_lane_a", "center_turn_lane", "travel_lane_a"],
      "speed": {
        "mode": "relative",
        "value": 0.71,
        "percentile": 0.96,
        "units": "normalized_units_per_second"
      }
    }
  ],
  "evidence": {
    "rule": "center_lane_overtake",
    "center_lane_time_seconds": 1.2,
    "speed_percentile": 0.96,
    "overtake_detected": true,
    "audio_overlap": false
  },
  "config_snapshot": {
    "aggressiveness": 0.3,
    "clip_total_seconds": 30
  }
}
```

---

## Detection and Tracking

### Detection

Use Ultralytics YOLO initially.

Start with:

* `yolov8s.pt` for GPU development
* `yolov8n.pt` as fallback for mini-PC CPU testing

Detector must be abstracted behind a common interface:

```python
class Detector:
    def detect(self, frame: np.ndarray) -> sv.Detections:
        ...
```

### Tracking

Use Supervision ByteTrack where practical.

Each tracked object should maintain:

* track ID
* timestamped bbox history
* bottom-center point history
* projected ground-plane coordinate history
* lane-band history
* direction estimate
* speed estimate
* confidence / age

Use bottom-center of vehicle bbox as the default road-contact point.

---

## Perspective Transform / Calibration

Use a four-point road-plane calibration.

The user should click four points on a frame representing the visible road surface quadrilateral.

Save as:

```yaml
calibration:
  source_points:
    - [x1, y1]
    - [x2, y2]
    - [x3, y3]
    - [x4, y4]
```

Create a normalized target plane:

```yaml
calibration:
  target_width_units: 1.0
  target_length_units: 1.0
```

For the MVP, speed does not need to be km/h. Relative speed is acceptable.

The normalized transform should still make lane-band classification and same-direction comparisons more stable.

Later, if real road dimensions are provided, the target plane can be scaled to meters.

---

## Lane Band Model

Do not require manually drawing lane polygons.

After perspective transform, divide road width into five bands:

1. `bike_lane_a`
2. `travel_lane_a`
3. `center_turn_lane`
4. `travel_lane_b`
5. `bike_lane_b`

Default ratio model:

```yaml
bike_lane_width_ratio: 0.12
travel_lane_width_ratio: 0.28
center_lane_width_ratio: 0.20
travel_lane_width_ratio: 0.28
bike_lane_width_ratio: 0.12
```

Normalize ratios so total width = 1.0.

Each track should receive a lane band per frame based on its projected bottom-center point.

Direction naming can be camera-relative:

* `left_to_right`
* `right_to_left`

No need to name actual east/west directions for MVP.

---

## Speed Estimation

Use relative speed estimation first.

For each track:

* Project bottom-center points into normalized road plane
* Compute displacement over time
* Smooth using a short rolling window
* Calculate normalized units per second
* Maintain speed baselines separately for each direction

For each direction, maintain rolling statistics:

* median speed
* 85th percentile
* 90th percentile
* 95th percentile
* 97th percentile

Do not require km/h in MVP.

---

## Aggressiveness Knob

Config value:

```yaml
events:
  aggressiveness: 0.3
```

Range:

* `0.0` = strict / fewer clips
* `1.0` = sensitive / more clips

Map aggressiveness into thresholds.

Example mapping:

```python
def lerp(strict, sensitive, aggressiveness):
    return strict + (sensitive - strict) * aggressiveness
```

Use it for:

* speeding percentile threshold
* center lane speed threshold
* center lane minimum dwell time
* event cooldown
* audio loudness threshold

Initial target: approximately 5 or fewer useful clips per day.

---

## Event Rule: Relative Speeding

Detect vehicles moving unusually fast compared to normal traffic in the same direction.

Trigger if:

* Track has existed for at least minimum consecutive frames
* Direction is known
* Enough baseline data exists, or fallback threshold is used
* Track speed percentile exceeds threshold
* Condition persists for minimum duration

Metadata evidence:

* track ID
* direction
* speed value
* percentile
* rolling median
* threshold used
* duration

If baseline is not mature, allow “warmup mode” but mark confidence lower.

---

## Event Rule: Center Lane Pass

This is the most important custom rule.

Detect two patterns:

### Pattern A — Fast Center Lane Traversal

Trigger if:

* Vehicle enters `center_turn_lane`
* Vehicle remains in center lane for minimum dwell time
* Vehicle speed percentile is above center-lane threshold
* Track is moving through the scene, not merely stopping/turning

This captures common passing/aggressive use even if overtake confirmation is hard.

### Pattern B — Overtake Through Center Lane

Trigger stronger event if:

* Candidate vehicle starts behind another vehicle moving same direction
* Candidate moves into center turn lane
* Candidate becomes ahead of the other vehicle within an overtake window
* Candidate spends sufficient fraction of that window in center lane

Metadata evidence:

* candidate track ID
* passed vehicle track ID if known
* lane sequence
* relative position before/after
* center lane dwell time
* speed percentile
* overtake confidence

If uncertain, still allow a lower-confidence event based on Pattern A.

---

## Event Rule: Loud Engine

Audio is optional for MVP because the camera may not provide useful audio.

If a USB mic or other audio input is available:

* Capture mono audio
* Compute RMS / peak loudness in short windows
* Maintain rolling baseline
* Trigger if loudness exceeds baseline by threshold for minimum duration

Use loud engine detection in two ways:

1. Standalone event: `loud_engine`
2. Score booster when overlapping with:

   * relative speeding
   * center lane pass

Metadata evidence:

* dB over baseline
* duration
* overlap with visual event
* timestamp window

---

## Event Manager

The event manager receives rule outputs and decides when to export clips.

Responsibilities:

* Score events
* Deduplicate repeated triggers
* Merge nearby triggers
* Schedule clip export
* Write metadata
* Generate thumbnail

Rules should emit candidate events; event manager decides final saved clips.

---

## Simple Folder Review

No dashboard required for MVP.

Event folder structure should be easy to browse manually:

```text
/data/events/
  2026-06-08/
    center_lane_pass/
      20260608_140355_center_lane_pass_a1b2c3.mp4
      20260608_140355_center_lane_pass_a1b2c3.json
      20260608_140355_center_lane_pass_a1b2c3.jpg
    relative_speeding/
    loud_engine/
```

Optional later:

* `traffic-log build-index` to generate static HTML
* Local FastAPI dashboard

---

## Development Workflow

### Canonical Dev Environment

Use WSL2 Ubuntu on the Windows 11 RTX 4080 box.

Development should work without the camera attached.

Start with offline sample files.

Expected commands:

```bash
make setup-dev
make test
traffic-log test --source samples/street-test.mp4 --config config/config.dev.yaml
```

### Mini-PC Deployment

Expected commands:

```bash
make setup-mini-pc
docker compose up -d capture
docker compose up -d analyze
```

The mini-PC must be able to run capture without the 4080 machine being online.

### Sample Video Workflow

The project should make it easy to:

1. Record 5–10 minutes from the camera on mini-PC
2. Copy the file to `samples/`
3. Run offline analysis on RTX 4080 box
4. Inspect debug output and event clips

---

## Milestones

### Milestone 0 — Project Bootstrap

Deliver:

* Repo structure
* Python package
* CLI skeleton
* Config loading
* Logging
* Dockerfile
* Docker Compose skeleton
* README with setup instructions

Acceptance:

* `traffic-log --help` works
* `traffic-log test --source samples/example.mp4` runs stub pipeline
* pytest runs

---

### Milestone 1 — Camera Bring-Up and Recording

Deliver:

* `probe-camera`
* camera format detection
* continuous H.264 segment recording
* segment index
* 200GB ring pruning

Acceptance:

* On mini-PC, records Astra Mini RGB 1280x960 @ 30fps
* Writes 10-second segments
* Segment index contains correct timestamps
* Ring pruning deletes oldest segments above cap

---

### Milestone 2 — Offline Detection and Tracking

Deliver:

* YOLO detector wrapper
* Supervision Detections integration
* ByteTrack tracking
* Offline video analyzer
* Debug video/frame output with boxes and track IDs

Acceptance:

* Given a sample road video, vehicles are detected
* Track IDs persist across frames reasonably
* Debug output can be generated

---

### Milestone 3 — Calibration and Lane Bands

Deliver:

* interactive 4-point calibration
* perspective transform
* lane band model
* lane-band overlay preview

Acceptance:

* User can click road quadrilateral
* System generates preview image with five lane bands
* Tracks get assigned lane bands over time

---

### Milestone 4 — Relative Speeding Detection

Deliver:

* projected speed estimation
* per-direction rolling baseline
* relative speeding rule
* candidate event output

Acceptance:

* Offline sample video produces speed estimates
* Obviously fast vehicles receive high percentile scores
* Rule emits candidate events with evidence

---

### Milestone 5 — Center Lane Passing Detection

Deliver:

* center lane dwell detector
* optional overtake detector
* center lane pass event output
* evidence metadata

Acceptance:

* Vehicles using center lane at speed are flagged
* If overtake pattern is visible, metadata marks stronger event
* False positives can be reduced using aggressiveness knob

---

### Milestone 6 — Event Clip Export

Deliver:

* event manager
* ffmpeg clip exporter
* 30-second clips with pre/post roll
* JSON metadata sidecar
* thumbnail generation

Acceptance:

* Trigger creates playable MP4 event clip
* Metadata sidecar matches clip
* Duplicate triggers are merged reasonably

---

### Milestone 7 — Mini-PC Live POC

Deliver:

* Docker Compose capture service
* Docker Compose analyze service
* auto-restart
* logs
* optional RTSP stream

Acceptance:

* Mini-PC runs unattended
* Camera records locally
* Events save into folder
* Wi-Fi interruption does not stop local recording

---

### Milestone 8 — Optional Audio

Deliver if hardware available:

* audio capture
* loudness baseline
* loud engine rule
* visual event score boost

Acceptance:

* Loud engine events can be detected
* Overlapping audio boosts center-lane/speeding score

---

## Testing Requirements

Unit tests:

* lane band assignment
* speed percentile logic
* aggressiveness threshold mapping
* center lane dwell detection
* overtake detection with synthetic tracks
* event deduplication
* ring pruning

Integration tests:

* offline analysis on a short sample clip
* event export from test segment index
* metadata sidecar validation

Do not require live camera for automated tests.

---

## Error Handling

The system should handle:

* camera disconnect
* RTSP disconnect
* ffmpeg process crash
* corrupted segment
* low disk space
* missing calibration
* no vehicles detected
* no baseline speed data yet

Behavior:

* log clearly
* keep capture alive when possible
* restart crashed subprocesses
* never delete event clips as part of ring pruning

---

## Logging

Use structured logs.

Include:

* selected camera device and format
* segment start/end
* prune decisions
* model load status
* inference fps
* tracking count
* event trigger reason
* clip export success/failure

---

## Performance Targets

Mini-PC:

* Must record 1280x960 @ 30fps H.264 reliably
* Analysis may run at reduced FPS or be disabled if too slow

RTX 4080 WSL2 dev:

* Should handle YOLO inference comfortably
* Offline analysis should support testing different model sizes

Inference defaults:

* capture: 30fps
* analysis: 12fps
* model input size: 640
* detector: YOLOv8s on GPU, YOLOv8n fallback

---

## Immediate Build Priority

Start with this sequence:

1. Bootstrap repo and CLI
2. Build offline test pipeline
3. Build camera recording / ring buffer
4. Add YOLO + Supervision tracking offline
5. Add calibration + lane bands
6. Add relative speed and center-lane rules
7. Add event export
8. Deploy to mini-PC

Do not start with dashboard, LLMs, or depth.

---

## Key Implementation Notes for Claude Code

* Keep modules small and testable.
* Build offline mode first so development does not depend on the camera.
* Do not hardcode paths outside config.
* Do not assume GPU is available except in GPU-specific Docker/dev mode.
* Keep detector interface abstract.
* Keep rule engine independent of YOLO details; rules should consume track histories.
* Event exporter should operate from segment index, not from live frames.
* Prefer simple, reliable pieces over clever realtime architecture.
* Make the first useful POC save too many clips rather than missing obvious events, then tune aggressiveness down.

---

## Final MVP Acceptance Criteria

The MVP is successful when:

1. The Astra Mini records continuous daytime street footage on the Ubuntu mini-PC.
2. The ring buffer is capped at 200GB.
3. Offline analysis on the RTX 4080 box can detect and track vehicles.
4. Calibration produces stable lane bands.
5. The system can identify likely center-lane passing and relative speeding events.
6. Events export as 30-second clips with JSON metadata and thumbnail.
7. The mini-PC can run unattended using Docker Compose.
8. The event clips are easy to browse in folders.
