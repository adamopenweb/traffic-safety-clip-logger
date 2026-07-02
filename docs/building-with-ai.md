# How This Was Built (Human + AI)

This project was built end-to-end in collaboration with AI coding agents
(primarily Claude Code). That's not a footnote; it's the reason a
one-person side project ships with a supervised recorder, a calibrated
CV pipeline, a validated data store, a hardened web dashboard, and a
300-test suite. This document explains the working method, and points at the
artifacts in this repo that the process left behind, so the claim is
inspectable rather than taken on faith.

The short version: **the AI wrote most of the code; the human owned the
product, the physics, the verification, and the judgment calls.** Neither half
works alone.

---

## 1. Spec first: the brief *is* in this repo

Development started not with code but with [`traffic.md`](../traffic.md), a
~1,300-line design brief written *for* the AI before implementation began. Its
first heading is literally `# claude.md — Traffic Safety Clip Logger`. It
specifies the goals and non-goals (explicitly: no plate recognition, no face
ID), the module layout, the config schema, the event metadata schema, a
milestone plan (M0–M8), acceptance criteria per milestone, and closes with
"Key Implementation Notes for Claude Code": standing instructions like *keep
modules small and testable*, *build offline mode first*, *keep the detector
interface abstract*.

That up-front investment is what made agent-driven development compound
instead of wander: every session started from a shared, versioned definition
of "correct."

## 2. Milestones as the unit of work

The spec's milestone plan was executed in order (bootstrap → recorder/ring →
detection+tracking → calibration → speeding → center-lane rule → clip export →
live deployment), each landing with its tests before the next began. The
README's pipeline sections still mirror that structure. Two design rules from
the spec paid off repeatedly:

- **Offline-first:** the whole pipeline runs against a video file with a
  scripted (non-ML) detector, so every stage was testable without a camera,
  a GPU, or the CV stack. That's also why CI on this repo is green with no
  special hardware.
- **Pure logic separated from I/O:** rules, metrics, lane math, pruning
  selection, and validity predicates are dependency-free modules with direct
  unit tests; ffmpeg/OpenCV/SQLite live at the edges.

## 3. The review → plan → refactor loop

The strongest pattern in the process is also the most documented one:

1. [`engineering-review-2026-06.md`](engineering-review-2026-06.md) is a
   from-scratch AI architecture review of the working system, commissioned
   deliberately at the "it works, now what's wrong with it" stage. Its central
   finding: the system was excellent at "detect an event, cut a clip" but had
   **no concept of a vehicle that exists over time**, which everything planned
   next would need.
2. [`unified-store-plan.md`](unified-store-plan.md) is the follow-on migration
   plan (its header cites the review item it answers): one relational store
   built around a *pass* (a vehicle drive-by) that multiple writers merge into.
3. The shipped result: every vehicle pass is logged as the traffic
   denominator (`analyze/pass_recorder.py`, `events/store.py`), so the
   dashboard reports "X% of N cars were speeding" instead of bare counts; and
   the write-time validity architecture (`analyze/pass_validity.py` +
   `scripts/revalidate_passes.py`) replaced ~117 lines of accreted read-time
   filtering. The migration re-validated ~140,000 stored rows and was checked
   for zero drift against the old behavior before the old code was deleted.

Separately, periodic **full code reviews** ran as their own sessions: one
pre-release review produced a ranked findings list (memory growth in a 24/7
process, a test-collection failure that silently disabled the release gate, a
session-revocation gap in the dashboard auth, DST data-loss, gap-handling in
clip export), all of which were fixed within days. `docs/challenges.md` §8
and the auth/export code carry the results.

## 4. The human's half of the loop

AI output is a proposal, not a decision. Concrete examples of where the human
side was load-bearing:

- **Overriding a recommendation with local knowledge.** A review recommended
  switching ring-segment filenames to epoch naming, reasoning from
  Docker/UTC timezone-mismatch risks. The deployment is a single Windows box
  where that failure mode doesn't exist and where the epoch format isn't even
  available (`strftime %s` is glibc-only), so the fix shipped as fold-aware
  local-time decoding instead. The reviewer's framing was wrong because its
  deployment assumption was wrong; catching that is the operator's job.
- **Physical ground truth.** No agent can drive a car past the camera at a
  known GPS speed. The metric calibration (`docs/challenges.md` §7), with its
  repeated drive-bys, per-lane error measurement, and across-road correction,
  is human fieldwork that turned plausible numbers into validated ones.
- **Judging results, not just code.** The police classifier (§5 of the
  challenges doc) passed every code review; it was *watching its actual
  output for 24 hours* (4 of 4 confirmed sightings were civilian SUVs) that
  killed the feature. The disable decision, with its evidence, is recorded
  where the flag was turned off.
- **Deciding what the product is.** Three evidence tiers instead of
  clip-everything, "count every car" as a first-class requirement, no plate
  recognition ever: product judgments the spec encodes and the agent then
  serves.

## 5. Discipline that makes agent work auditable

Working with agents at this pace only stays safe if decisions leave a trail:

- **Experiments end in a written verdict.**
  [`vlm-vehicle-id-experiment.md`](vlm-vehicle-id-experiment.md) is a complete
  hypothesis → method → findings → cost → **"parked"** report for using a
  local VLM for vehicle identification.
- **Reverts get postmortems.** The ring-frame-source experiment and the
  offset-smoothing rollback both survive as documented decisions with the
  measurements that killed them (see `docs/challenges.md` §1 and §3), not as
  silently deleted code.
- **Un-versioned data operations get logged.**
  [`maintenance-log.md`](maintenance-log.md) records every manual operation on
  the (git-ignored) data store: what was deleted, the cutoff, the counts, the
  rationale, and whether it's reversible. `data/` leaves no git history, so
  this log is the audit trail.
- **Non-obvious code carries its "why."** The codebase reads as a decision
  journal: `pass_validity.py`'s module docstring explains the
  write-time-validity argument; `smooth_offset` in
  `events/ring_clip_exporter.py` documents why the median override was
  removed; `capture/recorder.py` explains the filename-format tradeoff. When
  an agent (or a human) revisits code months later, the reasoning is *in* the
  file, which is precisely what makes iterative AI development converge
  instead of thrash.

## 6. What I'd tell someone adopting this workflow

1. **Write the spec like the agent is a contractor you'll never meet.** The
   1,300 lines of `traffic.md` were the highest-leverage hours in the project.
2. **Make everything testable without the hardware.** The scripted detector
   and synthetic-video fixtures meant the agent could verify its own work;
   "run the suite" beats "trust me" every time.
3. **Schedule adversarial reviews of your own system.** A fresh-context
   architecture review found the structural gap (no vehicle entity) that
   feature-by-feature development never would have.
4. **Keep the human on the physics and the outcomes.** Calibration, watching
   real output, deployment context: the agent can't know what it can't
   observe, and most of my overrides came from exactly there.
5. **Demand written verdicts.** Experiments, reverts, data surgery: if it
   isn't written down, the next session (human or AI) re-litigates it.
