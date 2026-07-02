# Engineering Review: Traffic Safety Clip Logger

_Date: 2026-06-20 · Branch reviewed: `permanent-camera-single-box` · ~7,200 LOC across 48 modules_

> Objective architecture review from an engineering-excellence POV: would we design
> it differently if we rebuilt from scratch today, given the future plans to track
> license plates and per-vehicle-type statistics?

## Verdict up front

This is a **well-built single-purpose instrument that has grown organically,
feature-by-feature, into something broader than its original frame.** The bones are
genuinely good — clean pure-functional cores, immutable event types, sensible
threading. But it was designed as a *clip logger* (detect an unsafe pass → cut a
video → write a sidecar), and the future plans (license plates, per-vehicle
identity, "which vehicle types speed when") are a *fleet-analytics* problem. Those
are different shapes. You wouldn't throw this away, but if you rebuilt today knowing
where it's going, **three decisions would change**, and they all trace to the same
root: **there is no vehicle, and there is no database.**

## What's genuinely good (keep these)

- **Pure cores are well-isolated and testable.** `project.py` (homography),
  `metrics.py` (speed/baseline), `lane_model.py`, `ring_pruner.py` are pure math
  with no CV/IO dependencies. The hard part to get right, and it's right.
- **Immutable event payloads.** `Observation` → `Track` → `CandidateEvent` →
  `FinalEvent` is a clean one-way dataflow. No back-references, no callbacks into
  tracks, no globals in the analyze path.
- **Time-addressable ring buffer.** Events carry absolute `trigger_ts`; segments are
  indexed by `[start_ts, end_ts]`; clip export is a clean `get_overlapping()` time
  query. The event→footage mapping is the cleanest subsystem in the project.
- **The evidence dict is free-form.** Every rule dumps a rule-specific `evidence`
  blob into the metadata JSON — the one place the schema is already future-proof.
- **Deferred police classification.** CLIP scoring is off the frame loop on a worker
  thread with backpressure-drop. Correct instinct — never block decode on the GPU
  classifier.

## The three things I'd design differently from scratch

### 1. There is no `Vehicle` entity, and the data lives in three disconnected stores

The big one for the future plans. Today the same drive-by is written to **three
independent places by three independent writers**, each re-extracting
`ts` / `direction` / `track_id` from the event in its own ad-hoc way:

| Store | Written by | Format |
|---|---|---|
| Event metadata | `metadata.py` | JSON sidecar per clip |
| `speed_log.sqlite` | `cli_handlers.py` | flat table, 1 row/event |
| `police_sightings.sqlite` | `police_classifier.py` | flat table, 1 row/track |

There is no row that says "*this vehicle*." `track_id` is **ephemeral — it resets to
1 every session.** So today you fundamentally cannot answer "has this car sped past
before?" — not because the feature isn't built, but because nothing persists an
identity to hang it on.

License plates and per-vehicle tracking are *exactly* the questions that need that
missing entity. Plate recognition only pays off if a plate is a stable key you can
aggregate against across days and cameras. Right now there's nowhere for it to live
except bolted onto three separate schemas via three separate `ALTER TABLE`s (which is
literally how `vehicle_type` was added — the seam is already showing).

**From scratch:** one **relational store** (SQLite is still fine — local-first,
single-box) with a real schema:
`sightings(id, vehicle_id?, plate?, ts, camera_id, direction, speed_kmh,
vehicle_type, clip_path, evidence_json)`, and an `events` table referencing it. The
flat logs and the JSON sidecar become *projections/exports* of that, not the source
of truth. A plate or an ML re-id embedding becomes the optional `vehicle_id` foreign
key. Highest-leverage change for where this is headed, and contained — the writers
already converge on `FinalEvent`; give them one `EventRecord.from_final_event()`
instead of three bespoke extractions.

### 2. `live.py` and `offline.py` are ~70% duplicated — no pipeline abstraction

`run_live()` (201 lines) and `run_offline()` (283 lines) independently re-implement
the same orchestration: detect → track → project → metrics → rules → manager → emit.
Rule init is copy-pasted (`live.py:113` ≈ `offline.py:140`), the eval loop is
copy-pasted, box-annotation label-building is copy-pasted, and `_kmh()` is
reimplemented in *both* rule files.

Practical cost: **every new capability — plate OCR, a new vehicle classifier, a new
rule — has to be wired into two long functions and kept in sync by hand.** A steady
tax that compounds with exactly the feature growth planned.

**From scratch:** a `Pipeline.process_frame(frame, ts) -> list[FinalEvent]` class.
Live and offline become a *frame source* and a *result sink* around the same
pipeline. Offline replay (re-running new rules against recorded footage — wanted for
calibrating plate/type detection without waiting for live traffic) drops out for
free. Lowest-risk refactor on the list; do it before adding plates, not after.

### 3. The rule interface isn't actually an interface

`base.py` defines a `Rule` ABC, but the two rules have **incompatible signatures** —
`RelativeSpeedingRule.evaluate(track, speed, direction, estimator, ts)` vs
`CenterLanePassRule.evaluate(tracks, estimator, ts)` (one wants a single track, the
other the whole list). The caller has to know each rule's shape, which is why the
eval loop can't be generic and gets copy-pasted. Adding a plate-match or per-type
rule means another bespoke call site in two files.

**From scratch:** one `RuleInput` dataclass (tracks, estimator, ts, frame-context)
passed to every `rule.evaluate(input) -> list[CandidateEvent]`. Rules become a
genuine list you iterate, and #2's pipeline gets simple.

## Secondary observations (note, not redesign-drivers)

- **Police module is the complexity hot-spot.** `police_classifier.py` (579 lines)
  runs **three threads + two queues + six shared dicts** (`PoliceSession` +
  `PoliceTagger` + `Police4KConfirmer`). Works, but the one place to expect a
  heisenbug, and the "forget track before its async 4K job lands" race is currently
  defended only by a timing delay (`ready_delay += 6.0`), not a real handshake. When
  adding plate OCR, resist adding a *fourth* async stage here — the pattern (sample
  crop → defer to worker → confirm on 4K → write sighting) is **general**; extract a
  reusable "async per-track enricher" rather than copy it for plates.
- **Speed calibration is correct but scattered.** km/h derivation is a chain
  (`track_speed` → `meters_per_unit` → `KMH_PER_MS` → `across_speed_factor` with
  hardcoded `near_gx/far_gx` interpolation) reached into directly from `metrics.py`
  *and* duplicated in `steady_speed_kmh` *and* in offline's peak loop. The
  GPS-validated correction is real engineering but lives as constants spread across
  modules. One `SpeedModel` object owning the calibration would make it swappable and
  stop the triplication.
- **Config is loose by design — fine now, a liability at scale.** Only `log_level` is
  validated; everything else is unvalidated dicts reached into by string keys
  (`calibration.speed_across_correction.near_gx`). Deliberate M0 choice that's served
  well, but typo'd keys fail silently as "feature off," and there are now ~10
  sections. Typesample the high-traffic sections (analysis, calibration, events) into
  dataclasses; leave the rest loose.
- **Track store never evicts departed tracks**, and each rule keeps its own per-track
  `_state` dict that's never cleared. Fine for a daylight-bounded run; a memory creep
  to watch if this ever goes continuous multi-day.

## If I were scoping the rebuild

I wouldn't. Three contained refactors **in this order, before** plate work — each
independently valuable, each de-risking the next:

1. **Unify the data model** → one SQLite store + `EventRecord`; logs/JSON become
   projections. _(Unlocks plates + per-vehicle stats; nothing else does.)_
2. **Extract `Pipeline` + `RuleInput`** → kills the live/offline duplication; gives
   offline replay for tuning new detectors.
3. **Generalize the async per-track enricher** out of the police module → plate OCR
   becomes a config entry, not a fourth thread.

The thing to internalize: **this codebase is excellent at "detect an event, cut a
clip." It has no concept of "a vehicle that exists over time."** Every future plan is
about vehicles over time. That gap — not code quality, which is good — is what a
from-scratch design would close first.
