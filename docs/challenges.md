# Engineering Challenges & How They Were Solved

This project looks simple from a distance ("point a camera at the street, run
YOLO, save clips"), but almost every stage hid a real engineering problem. This
document walks through the ones that shaped the system, in roughly the order
they hurt. Each section covers what went wrong, what was tried, and what
actually shipped. File references point into this repo; dates refer to the
private development log this repo is a snapshot of.

A theme to watch for: the first fix is almost never the shipped fix. Several of
these arcs went through two or three plausible-but-wrong solutions that were
measured, found wanting, and **reverted**. The reverts are documented as
deliberately as the fixes.

---

## 1. Annotation boxes that wouldn't stay on the cars (cross-stream sync drift)

**Problem.** Detection runs on one RTSP connection; the evidence clip is cut
from a second connection recording the 4K main stream. The two pipelines have
different, *drifting* latency. Boxes burned onto the clip using live tracking
data trailed the vehicles, sometimes badly: the measured offset wandered from
−0.05 s to +2.0 s within a single evening, and was once observed past +3.5 s.

**The ladder of attempts:**

1. *Static offset* (0.75 s). Worked for a day, then drifted.
2. *Per-clip auto-align:* detect the flagged car in a few sampled 4K frames and
   solve for this clip's offset (`events/overlay_render.py`,
   `estimate_sync_offset`). Much better, but a failed solve still fell back to
   a stale constant.
3. *Rolling-median smoothing* of recent solves. **Reverted:** the true latency
   genuinely jitters per clip, so overriding an accurate low solve with the
   median pushed boxes *off* the cars. The lesson (trust the per-clip
   measurement, keep the median only as a fallback for failed or railed
   solves) is written into the `smooth_offset` docstring in
   `events/ring_clip_exporter.py`.
4. *Search-window whack-a-mole:* widening/tightening the solver's search radius
   as new drift extremes appeared. Each tune fixed the last incident, not the
   problem.

**What shipped.** Stop synchronizing across streams entirely. The annotation
renderer now **re-detects vehicles on the clip's own frames**
(`render_annotated_clip_detected` in `events/overlay_render.py`), so a box
drawn on frame *i* is on the car in frame *i* by construction; there is no
offset to drift. The live track is used only to *identify* which detection is
the flagged vehicle (a trajectory match that tolerates seconds of offset). The
offset-projection path survives as a fallback when the detector is unavailable.

**Takeaway:** when a calibration constant needs constant re-tuning, the
architecture is wrong. The durable fix removed the need for the constant.

---

## 2. Speed measurements you could defend (noise, jumps, and impossible cars)

**Problem.** Speeds come from projecting bounding-box road-contact points
through a homography and differencing positions over time. Every noise source
lands directly in km/h: a steady ~50 km/h car peaked at 56+ on the rolling
window; an endpoint detection jump made a 0.6 s track read **111 km/h**;
a multi-frame jump made a 0.9 s track read **152 km/h, about twice the length
of the visible road**. For a project whose output is "here is evidence of
speeding," bogus numbers are fatal.

**Layered fixes** (each layer earned by a specific artifact class), all in
`analyze/metrics.py`:

1. **Steady end-to-end speed** over the whole trimmed track replaced the peaky
   rolling window for gating decisions.
2. **Jump guard:** cross-check the end-to-end speed against the *median*
   per-frame step speed. A single endpoint jump inflates the first but can't
   move the second; if they disagree by 1.5×, use the median.
3. **Span guard:** a track that appears to traverse more than ~1.6× the
   calibrated road plane left the field of view. Physically impossible, so
   reject it.
4. **Trigger-bias fix:** live events fire mid-crossing, when the partial-track
   estimate reads systematically ~7 km/h high. Events are now relabeled at
   dispatch with the completed track's validated speed (`_relabel_event_speed`
   in `analyze/live.py`): one correction that fixes the filename, the log row,
   and the clip decision together.

**The architectural fix.** Guards kept accreting in the dashboard's *read*
path. Each new artifact class added another filter clause with a docstring
narrating its history (~117 lines at peak). That was data-repair masquerading
as read logic. The endgame, shipped as a two-phase migration:

- Raw measurements are always stored; a **validated speed** is derived at write
  time from a single module of physical-plausibility invariants
  (`analyze/pass_validity.py`).
- History is **re-validated** by re-running the same predicates over stored
  rows (`scripts/revalidate_passes.py`). It's idempotent and repeatable, so a
  newly discovered artifact class costs one predicate plus one script run, not
  another read-time filter.
- Consumers read only validated speeds; the read-path guards were deleted.
- The dividing line is principled: **scene physics** (a crossing takes ≥0.4 s,
  spans ≤ ~2× the road) is validity and lives at write time; **policy** (what
  counts as "fast") stays read-time configuration.

The migration was verified by re-validating ~140,000 stored passes and
confirming zero drift against the old read-time filters before deleting them.

---

## 3. Fast near-lane cars kept fragmenting (throughput vs. resolution)

**Problem.** Analysis originally ran on the camera's D1 (704×480) sub-stream.
Fast vehicles in the near lane crossed the frame in ~0.5 s and fragmented into
multiple short tracks: ~12% of near-lane passes were unmeasurable vs ~1% in
the far lane.

**What shipped.** Analyze the 4K main stream at 30 fps. That was initially
impossible: measured inside the loaded live pipeline, the lens de-warp remap
on a full 4K frame cost ~44 ms, capping the pipeline at ~14 fps. The fix is
ordering. **Downscale to 1280 px wide first, then de-warp** (~1.5 ms for
both). Detection is unaffected (YOLO resizes internally anyway) and the speed
scale is preserved because the homography works in normalized coordinates;
only the calibration points scale (`analyze/live.py`,
`analysis.analyze_max_width`).

**The frame budget, measured.** At 30 fps the analysis loop owns **33.3 ms
per frame**. Stage medians, benchmarked on the deployment box (RTX 4080)
*while the production analyzer was running*:

| Stage | Where it runs | Median |
| --- | --- | --- |
| H.265→pixels video encode | the camera's hardware encoder | 0 ms (not our silicon) |
| 4K frame decode | dedicated grabber thread | ~10 ms, off the loop |
| Downscale 3840→1280 px | analysis loop | 0.6 ms |
| Lens de-warp remap @1280 | analysis loop | 0.9 ms |
| YOLOv8s inference @1280 (CUDA) | analysis loop | 6.5 ms |
| ByteTrack + track store + road projection | analysis loop | <0.1 ms |
| Rules + overlay bookkeeping | analysis loop | <0.1 ms |
| **Analysis-loop total** | | **~8 ms of 33.3** |

(The rejected full-4K remap measures ~7 ms standalone on an unloaded CPU; the
~44 ms figure above is what it cost *inside* the contended pipeline, which is
the number that matters.)

The budget is managed less by making stages fast than by **deciding what
never enters the loop at all**: encoding stays on the camera, decoding stays
on the grabber thread (which also implements the drop policy — the loop
always takes the freshest frame and never queues), clip export and the
annotated 4K render run on their own worker, and the police classifier ran as
an async pipeline with bounded queues that drop under backpressure. The ~4×
headroom that leaves is not waste; it's what absorbs bursts (multi-car frames,
exporter contention during an event) without dropping to the fragmentation
regime this whole section exists to avoid. The ring-mode revert below is the
measured counterexample of what happens when the loop carries too much.

**Related experiment, reverted.** Reading frames from the recorded ring instead
of a live RTSP connection would fix annotation sync *structurally* (analyze
the exact frames that get exported). Tried, measured at ~0.73× real time with
unbounded lag growth (the live process also carries the render worker and
classifiers that the offline benchmark didn't), and **reverted** with the
numbers recorded in the run config as a postmortem. The stream path stayed;
the structural fix for annotations came from challenge #1 instead.

---

## 4. Evidence clips nobody could play (HEVC, and disk economics)

**Problem.** The ring records the camera's native H.265, and copying it is
free. But short clips stream-copied from the HEVC ring produced MP4s that no
browser (not even the iOS Files app) would play: `hev1` tagging plus
non-monotonic timestamps from segment concatenation. Also, at ~5 clips/day the
disk cost of 30 s 4K clips for *every* violation was disproportionate: most
violations only need to be counted, not watched.

**What shipped: three evidence tiers by measured speed** (thresholds from
config; see `handle_run` in `cli_handlers.py`):

| Tier | Evidence | Cost |
| --- | --- | --- |
| Over the gate (55+) | Text row in the speed log; every violation is counted | ~nothing |
| Mid (65+) | A still image of the vehicle, cropped from its best live frame | ~100 KB |
| Egregious (70+) | Full 30 s clip re-encoded to H.264 (+faststart, CRF-tuned) **plus** an annotated render | ~10–20 MB |

The 583 already-written unplayable HEVC clips were pruned in a logged,
dry-run-first cleanup (`scripts/prune_hevc_videos.py`,
`docs/maintenance-log.md`); each kept its still and metadata, becoming an
image-only record.

---

## 5. Police-vehicle recognition: an honest failure

**Goal.** Flag marked police vehicles so the reports could answer a fair
community question: how often does the street actually get patrolled?

**What was tried** (`analyze/police_classifier.py`):

1. **Zero-shot CLIP** (OpenCLIP ViT-B-32) scoring vehicle crops against
   police-vs-civilian prompt sets.
2. **Prompt engineering that keyed on hardware, not colour** (light bars,
   livery, push bumpers) after plain dark SUVs kept reading as "black police
   SUV". Explicit civilian negatives were added for dark sedans/SUVs, and a
   garbage-truck negative after a waste-collection truck scored 0.66–0.77
   (it dropped to 0.03). Prompt tuning cut the civilian median score 0.29→0.12,
   with a calibration script suite (`scripts/police_prompt_tune.py`,
   `scripts/police_sub_calib.py`).
3. **A two-stage cascade:** the cheap sub-stream score only *escalates*
   candidates; escalated tracks are re-cropped from full-resolution 4K ring
   frames and re-scored before anything is logged (a black SUV spiking 0.68 on
   the low-res sub read ~0.08 at 4K). Async workers with bounded queues and
   backpressure-drop keep all of it off the frame loop.

**Outcome: disabled.** Over a 24 h validation window, 4 of 4 "confirmed"
sightings were civilian dark/silver SUVs and sedans. Conclusion, recorded in
the config where the feature was switched off: zero-shot CLIP cannot separate
"marked cruiser" from "dark SUV" on these crops at this distance, and the 4K
confirm stage doesn't fix it. The code remains (it's a clean async-cascade
design) but the feature is off pending a livery/light-bar detector or a
fine-tuned classifier.

**Postscript.** A later experiment with a local VLM (Qwen3-VL 8B via Ollama;
see `docs/vlm-vehicle-id-experiment.md`) got emergency-vehicle identification
essentially *for free* while being evaluated for make/model tagging: the
exact signal the CLIP pipeline failed to build. It's parked on cost grounds
(6–18 s per image, and GPU contention with recording), but it reframes the
problem from "tune CLIP harder" to "run a better model post-hoc on the few
clips that matter."

---

## 6. The Windows-specific deployment problems

The system runs 24/7 on a Windows box (see the README's "Why Windows?"). That
bought a GPU for free and cost a series of platform problems, each small, none
optional:

- **No `%s` in Windows `strftime`.** ffmpeg's segment muxer named ring files
  with glibc's unix-seconds token, which doesn't exist on Windows. Segments
  are now named with portable local wall-clock timestamps
  (`segment_YYYYMMDD-HHMMSS.mp4`); the parser accepts both forms
  (`capture/recorder.py`).
- **Wall-clock names meet DST.** Local timestamps are ambiguous during the
  November fall-back hour. Decoding is timezone-explicit and fold-aware, and
  the finalizer refuses to overwrite an already-indexed segment, so the repeat
  hour can never silently destroy footage.
- **Windows ships no timezone database.** `tzdata` is a hard dependency so
  event timestamps localize instead of silently falling back to UTC
  (`pyproject.toml` has the note).
- **Killing a process tree.** Stopping the analyzer must also stop its ffmpeg
  children; on Windows that's `taskkill /PID … /T /F`, wrapped with fallbacks
  (`_kill_tree` in `cli_handlers.py`).
- **Running headless at logon.** The supervisor autostarts via Task Scheduler
  running a VBScript (`scripts/run_supervisor.vbs`) that launches Python with
  no console window, self-locates the repo, and appends everything to a log.
  It also forces **unbuffered** output so a crash's final traceback isn't
  lost in a block buffer; that exact failure mode happened, and is documented
  in `_run_child_spec` in `cli_handlers.py`.

---

## 7. Making km/h numbers true (GPS drive-by calibration)

**Problem.** A 4-point planar homography maps the road to a normalized plane,
but it isn't perfectly metric: a real metre in the far lane maps to fewer
ground units than a metre in the near lane. Far-lane speeds read as low as
0.65× true.

**What shipped.** Physical calibration against reality: repeated drive-bys at
known GPS speeds, labeled through a dedicated **speed-test page** on the
dashboard (it lists every measured pass, including slow ones, so a calibration
run needs no special mode; see `web/speedtest.py`). The measured per-lane error
ratios feed an interpolated **across-road correction factor**
(`across_speed_factor` in `analyze/metrics.py`) that divides out the
perspective non-uniformity by each track's mean road position. Residual error
after correction: a few percent, re-checked after any camera move.

---

## 8. Shorter ones worth a paragraph

- **Recording gaps vs. clip export.** Concatenating ring segments assumed
  contiguity; a gap (ffmpeg reconnect) inside a clip window silently shifted
  the footage and the annotation timeline. Rather than splice across gaps, the
  export **clamps to the contiguous run containing the trigger** and records
  what was truncated in metadata. Correct by construction, honest about
  what's missing (`events/ring_clip_exporter.py`).
- **A dashboard on the public internet.** Instead of a login page (which
  advertises that an app exists), the dashboard returns an **identical 404 to
  every unauthenticated request**: `/`, `/api/stats`, `/.env`, everything. One
  unguessable 256-bit link sets a signed cookie; rotating the link revokes both
  old links and existing sessions (`web/auth.py`, `web/app.py`).
- **Ring safety invariant.** The segment being written never enters the index,
  and pruning is index-driven, so the pruner can never delete the active
  segment. That holds structurally rather than by luck (`capture/recorder.py`,
  `capture/ring_pruner.py`).
- **Center-lane false positives.** Slow cars legitimately entering the shared
  turn lane were flagged as overtakes on noise-level position jitter. Fixed by
  requiring a real along-road gain for an overtake and restricting the rule to
  the travel direction with no legitimate reason to be in that lane
  (`analyze/rules/center_lane_pass.py`).
