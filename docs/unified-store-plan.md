# Plan: Unified Data Store

_Date: 2026-06-20 · Review item #1 from `engineering-review-2026-06.md`_

> Consolidate the three disconnected persistence stores into one relational store
> built around a stable `passes` (vehicle drive-by) entity, so that license-plate
> recognition and per-vehicle-type statistics have a place to live.

## What exists today (the three writers, precisely)

| Store | Written from | Grain | Keyed on |
|---|---|---|---|
| `metadata` JSON sidecar | `live.py:376` (analyze-only), `offline.py:363`, `ring_clip_exporter.py:234` (recording) — **3 paths** | one unsafe-driving *event* | `event_id` (uuid) |
| `speed_log.sqlite` | `cli_handlers.py:184`, inside the `on_event` closure | one *speeding pass* (clipped or not) | nothing stable — bare `ts` |
| `police_sightings.sqlite` | `police_classifier.py:530` + `:370` (4K worker) | one *police pass* | session `track_id` |

Three consequences that shape the plan:

1. **The two SQLite writers never meet.** Speeding writes inline through `on_event`
   mid-track; police writes from a worker thread at track-finalize (on a delay). A
   police car that speeds is two unrelated rows in two databases — the only link is
   `track_id`, which **resets to 1 every run.**
2. **There is no traffic denominator.** Only speeders and police get a row. Nothing
   logs "a truck passed and wasn't speeding." The stated goals — *"which vehicle
   types are more likely to speed," "rate of speeding by type"* — are **ratios**, and
   the denominator (total passes by type) is **not persisted anywhere today.** This
   is the real gap, bigger than the duplication.
3. **`metadata` JSON is written from 3 code paths** — any new field is a 3-place edit.

## The unifying idea: one `passes` row per vehicle, upserted

A **pass** = one completed vehicle track (one drive-by). Both SQLite writers, plus a
future plate/type writer, collaborate on the *same* pass row via an **upsert keyed on
`(session_id, track_id)`** — whoever fires first creates it, the others update it.
That single decision lets the inline speeding-writer and the async police-writer stop
being separate databases without imposing any ordering between them.

```sql
sessions(session_id TEXT PK, started_at REAL, camera_id TEXT, config_hash TEXT)

passes(                              -- one drive-by; THE traffic denominator
  id INTEGER PK, session_id TEXT, track_id INTEGER,
  first_ts REAL, last_ts REAL, direction TEXT,
  vehicle_type TEXT,                 -- car/truck/bus/...  <- enables per-type stats
  max_speed_kmh REAL, steady_speed_kmh REAL, was_speeding INTEGER,
  is_police INTEGER, police_confidence REAL,
  plate TEXT, vehicle_id INTEGER,    -- future, nullable
  UNIQUE(session_id, track_id))      -- <- the upsert key

events(                              -- one clip; references a pass
  id INTEGER PK, event_id TEXT UNIQUE, session_id TEXT,
  primary_pass_id INTEGER REFERENCES passes(id),
  event_type TEXT, event_types TEXT, trigger_ts REAL, score REAL,
  clipped INTEGER, clip_path TEXT, thumbnail_path TEXT, evidence_json TEXT)

vehicles(id INTEGER PK, plate TEXT, first_seen REAL, ...)   -- future identity
```

Legacy reports keep working through **compatibility views** during migration:
`speed_events` = events joined to passes where type is speeding; `police_sightings` =
`SELECT * FROM passes WHERE is_police=1`. So `speed-report` / `police-report` don't
change until we choose to repoint them.

## Staging (each phase ships independently; live operation never disturbed)

**Phase 0 — Store module, additive, zero behavior change.**
New `events/store.py`: `TrafficStore` over one `data/index/traffic.sqlite`, stdlib
`sqlite3` + a lock (same pattern as `speed_log.py`/`police_log.py`), upsert on
`(session_id, track_id)`. `PassRecord` / `EventRecord` dataclasses +
`from_final_event()`. Unit tests against `:memory:`. **Nothing wired in.** Risk: none.

**Phase 1 — Dual-write (shadow).**
Wire `TrafficStore` *alongside* the existing three writers — write both. Legacy logs
stay the source of truth; the new DB is validated against them over a few days of
real traffic. Two write points: the `on_event` closure (`cli_handlers.py:179`) and
`police_classifier._finalize` both call `store.upsert_pass(...)`; `on_event` also
calls `store.add_event(...)`. Risk: low — purely additive, legacy untouched, easy
rollback.

**Phase 2 — Repoint readers + backfill.**
One-time importer backfills existing `speed_log.sqlite` + `police_sightings.sqlite`
(validated GPS/police history) into `passes` as synthetic rows. Point
`speed-report`/`police-report` at the views. Stop writing legacy logs (or keep as
deprecated shadow one more cycle). Risk: medium — reader cutover; mitigated by views
emulating old schemas exactly.

**Phase 3 — The denominator (the actual unlock).**
Emit a `passes` row for **every** finalized track, not just speeders/police. Needs a
per-track-finalize hook in the analyze loop — the *same* machinery as review item #3
(the async per-track enricher). So **Phase 3 rides on the enricher refactor**, not
before it. Once it lands, "% of trucks that speed by hour" is a single `GROUP BY
vehicle_type` query. Risk: medium — touches the live loop; do it after the enricher
extraction.

**Phase 4 — Identity-ready (no-op until plates exist).**
`plate` / `vehicle_id` columns + `vehicles` table already in the schema from Phase 0;
Phase 4 just adds the upsert-by-plate path when ALPR is built. Zero work now beyond
having reserved the columns.

## The one design risk to decide now

**Concurrency on the pass row.** The speeding writer (main thread, inline) and police
writer (worker thread, delayed) both touch the same `(session_id, track_id)`. Handled
with SQLite `INSERT … ON CONFLICT(session_id,track_id) DO UPDATE` under a single
connection + lock — order-independent, no handshake. This means the police worker and
the main loop share one DB connection; `TrafficStore` uses the same
`check_same_thread=False` + lock that `PoliceLog` already uses (`police_log.py:94`).

## Scope boundary

Phases 0–2 are the contained, low-risk core that delivers the unified store and kills
the cross-database split **without touching the live loop at all** (only
`cli_handlers` wiring + the police finalize call). Phase 3 — the denominator — is
where per-type stats become answerable, and it's deferred to land with the enricher
refactor.
