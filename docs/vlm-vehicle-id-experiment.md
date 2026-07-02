# Experiment — VLM vehicle ID (Qwen3-VL via Ollama)

**Date:** 2026-06-22 · **Status:** parked (needs compute off the recording box)

Quick experiment to see whether a local vision-language model can (a) classify body
type better than YOLO — which only knows COCO `car`/`truck`/`bus`/`motorcycle` and so
lumps SUVs, vans, and pickups into "car"/"truck" — and (b) guess make/model with a
usable confidence. Model: `qwen3-vl:8b` (~6 GB) served by Ollama on the 4080.

Throwaway scripts live in gitignored `data/vlm_experiment/` (`run.py` = sub-stream
thumbnails, `run_4k.py` = 4K crops). Not part of the package.

## Findings

**Body type — clear win over YOLO.** Qwen reliably distinguishes
sedan / hatchback / SUV / crossover / minivan / cargo-van / pickup / box-truck — the
exact granularity YOLO lacks. Examples where YOLO was generic and Qwen was specific:
- 85 km/h "car" → **SUV** (white EMS unit, with markings called out)
- 68 km/h "car" → **sedan, marked police Charger**
- "truck" → **pickup (Ford F-150)** vs **fire truck** (a marked fire engine)

**Make/model — depends heavily on image resolution.**
- *Sub-stream thumbnails (~400 px crops):* hit-or-miss. A black VW Golf came back
  "sedan / Hyundai Genesis" — wrong body type **and** wrong make.
- *4K crops (native-res, padded):* much better. The same Golf → "hatchback /
  **Volkswagen Golf** (0.85)" — both correct, higher confidence. Other 4K reads:
  Honda Civic, Ford Transit (1.0), Ford Escape, Chrysler Grand Caravan — all plausible.
- Confidence is **well-calibrated**: 0.6–0.85 when it commits, drops to ~0.1 on blurry
  crops, so it can be thresholded rather than blindly trusted.

**Bonus — free emergency-vehicle flag.** Without being asked, Qwen flagged the fire
truck, the EMS SUV, and two police cars in its `notes` field — including a 68 km/h
"car" the system had logged as an ordinary vehicle. This is the police/EMS signal we
deliberately chose **not** to build a complex detector for; the VLM gives it for free.

**Cost:** ~6–18 s/image on the 4080 with the model resident. Too slow for every car in
real time, but fine as a **post-hoc pass on excessive speeders only**.

## Caveats / open items

- **Crop the right vehicle.** `run_4k.py` cropped the *largest* vehicle in the clip,
  which is not always the event's primary track (an "EMS SUV" clip returned a gray
  Civic that passed in the same window — the read was correct, just the wrong car). A
  real feature must crop from the **primary track's box** (overlay sidecar) — which
  means reusing the sub→4K dewarp mapping, skipped here for speed.
- **GPU contention breaks recording-time annotation.** Running the VLM (or YOLO batch
  inference) on the same 4080 as the live `traffic-log run` jitters frame timing and
  drifts the sub↔4K sync offset mid-clip → misaligned annotation boxes. Do not run VLM
  experiments on the recording box while it is capturing; use a capture gap or separate
  hardware. (See the annotation-sync notes; misaligned clips are re-fixable with
  `reannotate.py` once the GPU is free.)

## Verdict

Worth pursuing — primarily to fix YOLO's SUV/van/pickup blindness, secondarily for
make/model and a free police/EMS flag — **once the model has compute that doesn't
compete with recording** (second GPU, CPU batch, or a separate machine pulling clips).
