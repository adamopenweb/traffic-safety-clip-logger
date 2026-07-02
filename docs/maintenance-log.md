# Maintenance Log

An audit trail of manual data operations on the evidence store (`data/` is
gitignored, so deletions there leave no git diff — they are recorded here instead).

## 2026-06-20 — Purge pre-calibration events

**Action:** Deleted all event records in `data/events/` from **before speed
calibration went live**.

- **Cutoff:** `trigger_ts < 1781909909.343` — i.e. before the first GPS-calibrated
  (`absolute_speeding`) event at **2026-06-19 18:58:29 EDT**.
- **Removed:** 521 events / 2,039 files / **51.22 GB**.
  - All of 2026-06-14 → 2026-06-18, plus 2026-06-19 daytime (before 18:58).
- **Kept:** 334 calibrated events (2026-06-19 evening + 2026-06-20).
- **Untouched:** `data/keepers/` (incl. the 89 km/h police-cruiser clip), the
  `speed_log` / `police_sightings` SQLite logs.

**Rationale:** Those events predate the GPS-validated km/h gate, so their speeds were
the old relative-percentile estimates (not defensible absolute km/h). They were noise
for the community-safety case. The remaining `data/events` holds only
GPS-validated-km/h evidence.

**Reversibility:** None — the event clips/metadata are permanently deleted. The
underlying 4K ring footage for that period had already aged out of the 200 GB ring
(~4 days retention; the ring was later grown to 300 GB ≈ 9 days), so it was not
recoverable regardless.

## 2026-06-22 — Delete unplayable HEVC event clips

**Action:** Deleted **583 HEVC clean-clip `.mp4` files** (1.29 GB) via
`scripts/prune_hevc_videos.py --apply`. Each event's `.jpg` still and `.json`
metadata were **kept**, so those events become image-only records.

**Why:** The mid-tier short clips (65–69 km/h) were stream-copied from the HEVC ring,
producing files that won't play in any browser — or even the iOS Files app (`hev1`
tagging + non-monotonic DTS from the segment concat). A few early full clips were
HEVC too. Going forward the mid-tier saves a still image instead of a clip (commit
`0ef453d`), so this cleans the back catalog to match. H.264 `_annotated.mp4` files
were untouched (they play), so full-clip events that had an annotated render keep a
playable video.

**Reversibility:** None — the clips are permanently deleted (and the source ring
footage has aged out). The stills + metadata remain as the record.
