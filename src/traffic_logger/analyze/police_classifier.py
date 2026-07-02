"""Zero-shot police-vehicle recognition (OpenCLIP).

COCO YOLO only knows car/truck/bus, so "is this a *marked police* vehicle?" is a
separate fine-grained step. We answer it zero-shot: embed a vehicle crop with an
OpenCLIP image encoder and compare it against two text prompt sets -- a police
set ("a marked police car with a light bar", ...) and a civilian set ("an
ordinary parked/moving car", ...) -- returning the softmax probability of the
police class. No training data; best-effort, and only on MARKED units.

Two pieces:

* :class:`PoliceClassifier` -- the model wrapper: ``score(bgr_crop) -> prob``.
  Lazy-builds the model so importing this module stays cheap and torch only
  loads when policing is enabled. Prompt embeddings are ensembled (mean of each
  set) once at build time.
* :class:`PoliceTagger` -- an async front end: the 30 fps analysis loop calls
  ``submit(track_id, crop)`` (non-blocking, drops under backpressure) and a
  background worker runs CLIP and accumulates per-track scores. On track
  departure the loop calls ``pop(track_id)`` to read the aggregated result.

The pure score-aggregation / threshold logic lives in
``events/police_log.py`` (``aggregate_score`` / ``decide_police``) so it is
testable without torch.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..util.logging import get_logger

log = get_logger(__name__)

# Default zero-shot prompt sets. Ensembled (averaged) per class. Tunable via the
# events.police.{police_prompts,civilian_prompts} config keys.
# Key the police set on unambiguous police HARDWARE (light bar, livery, push
# bumper) rather than colour/body-type -- on low-res sub-stream crops a plain
# black SUV otherwise reads as a "black police SUV". The civilian set explicitly
# covers dark/plain SUVs+sedans so body colour alone never tips it. Tuned on real
# sub-stream crops (scripts/police_sub_calib.py): median 0.29->0.12 vs the naive
# colour-based prompts. Override via events.police.{police,civilian}_prompts.
DEFAULT_POLICE_PROMPTS = [
    "a police car with a roof-mounted emergency light bar",
    "a police SUV with POLICE text and reflective livery decals on the doors",
    "a law enforcement patrol vehicle with a push bumper and roof lights",
    "a police cruiser with blue and red emergency lights",
]
DEFAULT_CIVILIAN_PROMPTS = [
    "an ordinary civilian car with no markings",
    "a plain black SUV with no light bar",
    "a dark colored sedan or SUV",
    "a compact crossover SUV like a Toyota RAV4 or Honda CR-V",
    "a pickup truck, minivan, or delivery van",
    # Big industrial vehicles read as "patrol vehicle with roof lights/push
    # bumper" without these -- a GFL garbage truck scored 0.66-0.77 at 4K until
    # added (then 0.03). Validated against real sightings 2026-06-19.
    "a garbage truck or waste collection truck",
    "a large work truck, dump truck, or construction vehicle",
    "a city bus or a commercial box truck",
]


class PoliceClassifier:
    """OpenCLIP zero-shot scorer: a vehicle crop in, a police probability out."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
        police_prompts: Optional[Sequence[str]] = None,
        civilian_prompts: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.police_prompts = list(police_prompts or DEFAULT_POLICE_PROMPTS)
        self.civilian_prompts = list(civilian_prompts or DEFAULT_CIVILIAN_PROMPTS)
        self._model = None
        self._preprocess = None
        self._class_text = None  # (2, D) normalized class embeddings: [police, civ]
        self._logit_scale = 100.0
        self._lock = threading.Lock()  # serialize score() across worker threads

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA unavailable for police classifier; using CPU.")
            self.device = "cpu"
        # The OpenAI CLIP weights were trained with the QuickGELU activation;
        # loading them into a plain-GELU model degrades the embeddings (open_clip
        # warns). Use the matching ``-quickgelu`` architecture transparently so
        # the config can stay readable as "ViT-B-32".
        arch = self.model_name
        if self.pretrained == "openai" and not arch.endswith("-quickgelu"):
            arch = f"{arch}-quickgelu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=self.pretrained, device=self.device,
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(arch)
        with torch.no_grad():
            class_embs = []
            for prompts in (self.police_prompts, self.civilian_prompts):
                tok = tokenizer(prompts).to(self.device)
                feats = model.encode_text(tok)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                class_embs.append(feats.mean(dim=0))  # ensemble the set
            class_text = torch.stack(class_embs)
            class_text = class_text / class_text.norm(dim=-1, keepdim=True)
        self._model = model
        self._preprocess = preprocess
        self._class_text = class_text
        try:
            self._logit_scale = float(model.logit_scale.exp().item())
        except Exception:  # noqa: BLE001 - fall back to the CLIP default
            self._logit_scale = 100.0
        log.info("Police classifier ready (%s/%s on %s).",
                 self.model_name, self.pretrained, self.device)

    def score(self, bgr_crop) -> float:
        """Police-class probability (0..1) for a BGR vehicle crop."""
        import numpy as np
        import torch
        from PIL import Image

        if bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
            return 0.0
        with self._lock:  # one model, possibly two worker threads (sub + 4K)
            self._ensure_model()
            rgb = bgr_crop[:, :, ::-1]  # BGR -> RGB
            img = Image.fromarray(np.ascontiguousarray(rgb))
            tensor = self._preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self._model.encode_image(tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = self._logit_scale * feats @ self._class_text.T  # (1, 2)
                probs = logits.softmax(dim=-1)[0]
            return float(probs[0].item())  # index 0 == police class


class PoliceTagger:
    """Async per-track police scoring on a background worker thread.

    The analysis loop submits ``(track_id, crop)`` samples without blocking;
    the worker runs the (GPU) classifier and accumulates scores per track. When
    a track departs, the loop pops its aggregated scores for the sighting record.
    A bounded queue means that if the worker ever falls behind, new samples are
    dropped rather than stalling detection.
    """

    def __init__(self, classifier: PoliceClassifier, *, max_queue: int = 64) -> None:
        self.classifier = classifier
        self._q: "queue.Queue[tuple[int, object]]" = queue.Queue(maxsize=max_queue)
        self._scores: Dict[int, List[float]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.dropped = 0

    def start(self) -> "PoliceTagger":
        self._thread = threading.Thread(target=self._run, name="police-tagger", daemon=True)
        self._thread.start()
        return self

    def submit(self, track_id: int, crop) -> None:
        """Queue a crop for scoring; drop it if the worker is backed up."""
        try:
            self._q.put_nowait((track_id, crop))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                track_id, crop = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                s = self.classifier.score(crop)
            except Exception:  # noqa: BLE001 - one bad crop must not kill scoring
                log.exception("Police scoring failed for track %s", track_id)
                continue
            with self._lock:
                self._scores.setdefault(track_id, []).append(s)

    def scores_for(self, track_id: int) -> List[float]:
        with self._lock:
            return list(self._scores.get(track_id, []))

    def pop(self, track_id: int) -> List[float]:
        """Return and clear a track's accumulated scores (call on finalize)."""
        with self._lock:
            return self._scores.pop(track_id, [])

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


@dataclass
class Confirm4KJob:
    """A track escalated to 4K confirmation: metadata fixed at finalize + the
    best (wall_ts, de-warped-sub bbox) frames to re-crop from the ring at 4K."""

    track_id: int
    direction: Optional[str]
    ts: float
    first_ts: float
    max_speed_rel: Optional[float]
    max_speed_kmh: Optional[float]
    was_speeding: bool
    sub_conf: float
    candidates: List[Tuple[float, Tuple[float, float, float, float]]] = field(default_factory=list)


class Police4KConfirmer:
    """Second stage: confirm an escalated track on full-res 4K ring frames.

    The live sub-stream score is a cheap pre-filter; here we pull the vehicle's
    best frame(s) from the 4K ring, re-distort+scale the sub bbox onto the raw 4K
    (``StreamProjector``), crop, and re-score. A plain dark SUV that spiked on the
    low-res sub (~0.68) reads correctly (~0.08) at 4K, so this is what gives the
    count its precision. Runs on its own worker thread so the ffmpeg frame pulls
    never touch the analysis loop. Only confirmed sightings are logged.
    """

    def __init__(self, config, classifier: "PoliceClassifier", log_db,
                 sub_size: Tuple[int, int], *, max_queue: int = 128) -> None:
        pol = config.events.get("police") or {}
        cal = config.calibration
        ucfg = cal.get("undistort") or {}
        self.classifier = classifier
        self.log = log_db
        self.sub_size = sub_size
        self.threshold = float(pol.get("confidence_threshold", 0.55))
        self.min_crop_px = int(pol.get("min_crop_px", 40))
        self.index_path = config.recording.get("segment_index_path",
                                               "/data/index/segments.sqlite")
        # A confirm job can't run until the ring segment holding the vehicle's
        # best frame is finalized + indexed -- that lands ~segment_seconds after
        # the frame, plus indexing lag. Defer each job until then, else every
        # escalated track pulls 0 frames and is wrongly rejected.
        self.ready_delay = float(config.recording.get("segment_seconds", 10)) + 6.0
        self._k1 = float(ucfg.get("k1", 0.0))
        self._k2 = float(ucfg.get("k2", 0.0))
        self._roll = float(ucfg.get("roll_degrees", 0.0))
        self._homography = cal.get("overlay_homography")
        self._projector = None  # built lazily once 4K size is known
        self._tmp = None
        self._q: "queue.Queue[Confirm4KJob]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.dropped = 0
        self.confirmed = 0
        self.rejected = 0

    def start(self) -> "Police4KConfirmer":
        self._thread = threading.Thread(target=self._run, name="police-4k", daemon=True)
        self._thread.start()
        return self

    def submit(self, job: Confirm4KJob) -> None:
        try:
            self._q.put_nowait(job)
        except queue.Full:
            self.dropped += 1
            log.warning("4K confirm queue full; dropped track %d.", job.track_id)

    def _ready_at(self, job: Confirm4KJob) -> float:
        latest = max((wt for wt, _ in job.candidates), default=0.0)
        return latest + self.ready_delay

    def _run(self) -> None:
        pending: List[Confirm4KJob] = []
        while not self._stop.is_set():
            try:
                pending.append(self._q.get(timeout=0.25))
            except queue.Empty:
                pass
            now = time.time()
            ready = [j for j in pending if self._ready_at(j) <= now]
            pending = [j for j in pending if self._ready_at(j) > now]
            for job in ready:
                self._process(job)
        # Drain at shutdown: pull anything still queued, then process all pending
        # (best-effort -- a just-departed car's segment may not be indexed yet).
        while True:
            try:
                pending.append(self._q.get_nowait())
            except queue.Empty:
                break
        for job in pending:
            self._process(job)

    def _projector_for(self, main_w: int, main_h: int):
        if self._projector is None:
            from ..events.overlay_render import StreamProjector

            self._projector = StreamProjector(
                self.sub_size, (main_w, main_h), k1=self._k1, k2=self._k2,
                roll_degrees=self._roll, homography=self._homography)
        return self._projector

    def _crop_4k(self, wall_ts: float, bbox) -> Optional["object"]:
        """Pull the 4K frame covering wall_ts from the ring and crop the vehicle."""
        import subprocess

        import cv2

        from ..capture.segment_index import SegmentIndex
        from ..util.ffmpeg import ffmpeg_path
        from ..util.paths import ensure_dir

        with SegmentIndex(self.index_path) as index:
            segs = index.get_overlapping(wall_ts, wall_ts)
        seg = next((s for s in segs if s.start_ts <= wall_ts <= s.end_ts), None)
        if seg is None:
            return None
        if self._tmp is None:
            from pathlib import Path

            self._tmp = str(ensure_dir(Path("data") / "tmp") / "police_4k_probe.jpg")
        offset = max(0.0, wall_ts - seg.start_ts)
        try:
            subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error", "-ss",
                            f"{offset:.3f}", "-i", seg.path, "-frames:v", "1", self._tmp],
                           check=True)
        except Exception:  # noqa: BLE001 - missing/locked segment -> skip this frame
            return None
        frame = cv2.imread(self._tmp)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        proj = self._projector_for(w, h)
        corners = proj.project_bbox(tuple(bbox))
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
        x2, y2 = min(w, int(max(xs))), min(h, int(max(ys)))
        if x2 - x1 < self.min_crop_px or y2 - y1 < self.min_crop_px:
            return None
        return frame[y1:y2, x1:x2]

    def _process(self, job: Confirm4KJob) -> None:
        from ..events.police_log import Sighting, decide_police

        scores = []
        for wall_ts, bbox in job.candidates:
            crop = self._crop_4k(wall_ts, bbox)
            if crop is not None:
                scores.append(self.classifier.score(crop))
        # One clean 4K view is reliable (the sub stage already pre-filtered).
        is_police, conf = decide_police(scores, self.threshold, min_samples=1)
        if not is_police:
            self.rejected += 1
            log.info("4K rejected track=%d (sub=%.2f, 4K=%.2f over %d frame(s)).",
                     job.track_id, job.sub_conf, conf, len(scores))
            return
        self.confirmed += 1
        self.log.add(Sighting(
            ts=job.ts, first_ts=job.first_ts, track_id=job.track_id,
            direction=job.direction, confidence=conf, is_police=True,
            max_speed_rel=job.max_speed_rel, max_speed_kmh=job.max_speed_kmh,
            was_speeding=job.was_speeding))
        log.info("POLICE confirmed track=%d 4K-conf=%.2f (sub=%.2f) dir=%s speeding=%s%s",
                 job.track_id, conf, job.sub_conf, job.direction, job.was_speeding,
                 f" {job.max_speed_kmh:.0f}km/h" if job.max_speed_kmh is not None else "")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15.0)  # let in-flight + drained jobs finish


def _max_speed(history, window_seconds, scale) -> Optional[float]:
    """Peak smoothed speed over a track's ground-point history (None if N/A)."""
    from .metrics import track_speed

    if len(history) < 2:
        return None
    best = None
    for i in range(2, len(history) + 1):
        s = track_speed(history[:i], window_seconds, scale=scale)
        if s is not None and (best is None or s > best):
            best = s
    return best


class PoliceSession:
    """Drives police sampling + sighting finalize on top of the live loop.

    Keeps the hot loop terse: the loop calls :meth:`observe_frame` every frame,
    :meth:`note_event` when an event fires, and :meth:`close` at shutdown. A
    vehicle whose track stops appearing for a while is *finalized* -- its
    accumulated CLIP scores are collapsed to one confidence and, if it clears the
    threshold, written to the :class:`~...events.police_log.PoliceLog`.
    """

    def __init__(self, config, tagger: PoliceTagger, log_db, *, inference_fps: float) -> None:
        pol = config.events.get("police") or {}
        self.config = config
        self.tagger = tagger
        self.log = log_db
        self.threshold = float(pol.get("confidence_threshold", 0.55))
        self.min_samples = int(pol.get("min_samples", 2))
        self.min_track_age = int(pol.get("min_track_age", 5))
        self.sample_count = int(pol.get("sample_count", 4))
        self.sample_every = max(1, int(pol.get("sample_every_frames", 6)))
        self.min_crop_px = int(pol.get("min_crop_px", 40))
        self.speed_window = float(config.analysis.get("speed_window_seconds", 0.5))
        # Two-stage cascade: when confirm_4k is on, the live sub score only acts
        # as a pre-filter -- any track whose sub-mean clears escalate_threshold is
        # re-checked on full-res 4K ring frames (Police4KConfirmer) before being
        # logged. confirm_samples is how many of the vehicle's best frames to pull.
        self.confirm_4k = bool(pol.get("confirm_4k", False))
        self.escalate_threshold = float(pol.get("escalate_threshold", 0.35))
        self.confirm_samples = max(1, int(pol.get("confirm_samples", 2)))
        self.confirmer = None
        self.sub_size = None
        # Finalize a track once it's been gone comfortably longer than ByteTrack's
        # lost_track_buffer, so a briefly-occluded car isn't split into two.
        self.finalize_after = max(int(inference_fps * 3), 45)
        self._taken: Dict[int, int] = {}
        self._last_sample: Dict[int, int] = {}
        self._last_seen: Dict[int, int] = {}
        self._last_seen_wall: Dict[int, float] = {}
        self._first_wall: Dict[int, float] = {}
        # track_id -> list of (bbox_area, wall_ts, de-warped-sub bbox) for the 4K
        # re-crop: the larger the sub bbox, the cleaner the 4K view.
        self._best: Dict[int, List[Tuple[float, float, Tuple[float, float, float, float]]]] = {}
        self._speeding_ids: set[int] = set()

    def observe_frame(self, frame, tracks, processed: int, wall_now: float) -> None:
        h, w = frame.shape[:2]
        if self.sub_size is None:
            self.sub_size = (w, h)
            if self.confirm_4k:
                try:
                    self.confirmer = Police4KConfirmer(
                        self.config, self.tagger.classifier, self.log, self.sub_size).start()
                    log.info("Police 4K confirmation enabled (escalate>=%.2f).",
                             self.escalate_threshold)
                except Exception:  # noqa: BLE001 - fall back to sub-only decision
                    log.exception("4K confirmer unavailable; using sub-stream decision.")
                    self.confirmer = None
        for t in tracks:
            tid = t.track_id
            self._last_seen[tid] = processed
            self._last_seen_wall[tid] = wall_now
            self._first_wall.setdefault(tid, wall_now)
            if t.age < self.min_track_age:
                continue
            if self._taken.get(tid, 0) >= self.sample_count:
                continue
            if processed - self._last_sample.get(tid, -10 ** 9) < self.sample_every:
                continue
            bbox = t.latest_bbox
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 - x1 < self.min_crop_px or y2 - y1 < self.min_crop_px:
                continue
            self.tagger.submit(tid, frame[y1:y2, x1:x2].copy())
            self._taken[tid] = self._taken.get(tid, 0) + 1
            self._last_sample[tid] = processed
            # Remember this frame as a 4K re-crop candidate (full float sub bbox).
            area = float((x2 - x1) * (y2 - y1))
            self._best.setdefault(tid, []).append((area, wall_now, bbox))

    def note_event(self, fe) -> None:
        """Record the track ids of a speeding event for later cross-reference."""
        if "relative_speeding" in getattr(fe, "event_types", ()):
            if fe.primary_track_id is not None:
                self._speeding_ids.add(int(fe.primary_track_id))
            for tid in getattr(fe, "track_ids", ()):
                self._speeding_ids.add(int(tid))

    def finalize_departed(self, processed: int, store, meters_per_unit) -> None:
        gone = [tid for tid, seen in self._last_seen.items()
                if processed - seen > self.finalize_after]
        for tid in gone:
            self._finalize(tid, store, meters_per_unit)

    def _finalize(self, tid: int, store, meters_per_unit) -> None:
        from ..events.police_log import Sighting, aggregate_score, decide_police

        scores = self.tagger.pop(tid)
        best = sorted(self._best.pop(tid, []), key=lambda c: c[0], reverse=True)
        sub_conf = aggregate_score(scores)
        # Track-level facts, fixed now (the worker must not touch the live store).
        track = store.get(tid) if store is not None else None
        direction = track.direction() if track is not None else None
        gh = track.ground_point_history() if track is not None else []
        max_rel = _max_speed(gh, self.speed_window, (1.0, 1.0))
        max_kmh = None
        if meters_per_unit is not None:
            ms = _max_speed(gh, self.speed_window, meters_per_unit)
            max_kmh = ms * 3.6 if ms is not None else None
        ts = self._last_seen_wall.get(tid, 0.0)
        first_ts = self._first_wall.get(tid, ts)
        was_speeding = tid in self._speeding_ids
        self._forget(tid)

        # Stage 2: escalate promising tracks to 4K confirmation (logged there).
        if self.confirmer is not None:
            if scores and best and sub_conf >= self.escalate_threshold:
                cands = [(wt, bbox) for _area, wt, bbox in best[:self.confirm_samples]]
                self.confirmer.submit(Confirm4KJob(
                    track_id=tid, direction=direction, ts=ts, first_ts=first_ts,
                    max_speed_rel=max_rel, max_speed_kmh=max_kmh,
                    was_speeding=was_speeding, sub_conf=sub_conf, candidates=cands))
            return

        # Sub-only decision (confirm_4k off): mean over >= min_samples frames.
        is_police, _ = decide_police(scores, self.threshold, self.min_samples)
        if not scores or not is_police:
            return
        self.log.add(Sighting(
            ts=ts, first_ts=first_ts, track_id=tid, direction=direction,
            confidence=sub_conf, is_police=True, max_speed_rel=max_rel,
            max_speed_kmh=max_kmh, was_speeding=was_speeding))
        log.info("POLICE sighting track=%d conf=%.2f dir=%s speeding=%s%s",
                 tid, sub_conf, direction, was_speeding,
                 f" {max_kmh:.0f}km/h" if max_kmh is not None else "")

    def _forget(self, tid: int) -> None:
        for d in (self._taken, self._last_sample, self._last_seen,
                  self._last_seen_wall, self._first_wall, self._best):
            d.pop(tid, None)

    def close(self, store, meters_per_unit) -> None:
        """Finalize every still-active track, then stop the workers + close the log."""
        for tid in list(self._last_seen.keys()):
            self._finalize(tid, store, meters_per_unit)
        # Stop the 4K confirmer first -- it drains its queue (writing those
        # sightings) and still needs the classifier + log, which we close after.
        if self.confirmer is not None:
            self.confirmer.stop()
        self.tagger.stop()
        try:
            self.log.close()
        except Exception:  # noqa: BLE001
            pass


def build_police_tagger(config) -> Optional[PoliceTagger]:
    """Construct a started :class:`PoliceTagger` if ``events.police.enabled``.

    Returns None when policing is off or the classifier can't be built, so the
    caller (the live loop) simply skips all police work.
    """
    pol = config.events.get("police") or {}
    if not pol.get("enabled", False):
        return None
    try:
        device = str(pol.get("device") or config.analysis.get("device", "cuda"))
        classifier = PoliceClassifier(
            model_name=str(pol.get("model", "ViT-B-32")),
            pretrained=str(pol.get("pretrained", "openai")),
            device=device,
            police_prompts=pol.get("police_prompts"),
            civilian_prompts=pol.get("civilian_prompts"),
        )
        return PoliceTagger(classifier).start()
    except Exception:  # noqa: BLE001 - never let police setup break the run
        log.exception("Could not start police tagger; continuing without it.")
        return None
