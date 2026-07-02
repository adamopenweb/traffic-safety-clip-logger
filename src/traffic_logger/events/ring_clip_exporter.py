"""Live event -> ring-buffer clip exporter (Milestone 7).

When the live analyzer flags an event, the clip we want is
``[trigger - pre_roll, trigger + post_roll]`` — but the event fires right around
the trigger, so the post-roll footage hasn't been recorded yet. Each event is
therefore queued and exported a little after its post-roll has elapsed, by which
point the recorder has written + indexed the covering ring segments.

Runs on its own daemon thread so the analysis loop never blocks on ffmpeg, and
so a slow export can't stall detection. Exports are best-effort: a missing ring
window is logged and skipped, and one failing export never kills the thread.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from ..config import Config
from ..util.logging import get_logger

log = get_logger(__name__)

# Extra wall-clock margin after post_roll before exporting, so the ring segment
# holding the post-roll tail has been finalized + indexed by the recorder.
_EXPORT_MARGIN_SECONDS = 3.0


def _median(values: Sequence[float]) -> Optional[float]:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def smooth_offset(solved, history, *, center, radius, step, fallback, min_history=4):
    """Pick this clip's sync offset, trusting the per-clip solve.

    The auto-align solver is accurate *per clip* -- the true sub->4K latency genuinely
    jitters between events (0.5s to 1.5s+ within a day), so a clip whose offset solves
    to 0.5s really does need 0.5s, not the run's average. Repeated visual checks confirm
    it: re-rendering at the raw solve lands the boxes on the cars; substituting any
    consensus median pushes them off. So a **non-railed solve is used as-is** -- no
    outlier override (an earlier "wild outlier" guard kept replacing good low solves with
    a higher stale median and trailing the boxes; that did more harm than the rare bad
    solve it guarded against).

    The recent-solve history exists only as a *fallback*: when a solve rails at the search
    edge (clamped -> unreliable) or fails entirely, we use the recent median (or
    ``fallback`` before any history). Every non-railed solve is recorded so that fallback
    tracks a real latency drift instead of freezing. Returns ``(chosen, record)``.
    """
    median = _median(history) if len(history) >= min_history else None
    if solved is None:
        return (median if median is not None else fallback), False
    railed = abs(solved - center) >= radius - step
    if railed:
        return (median if median is not None else solved), False
    return solved, True              # trust the accurate per-clip solve


@dataclass
class _PendingClip:
    """A queued event: exported once its post-roll is on disk.

    ``snapshots`` is the overlay slice captured *eagerly* the moment the event
    becomes ready -- before the (slow, serial) 4K render -- so a burst backlog
    can't let those snapshots scroll out of the rolling buffer first. ``mode`` is
    ``"clip"`` (full video) or ``"screenshot"`` (a single 4K still, the mid-tier).
    """

    ready_at: float
    final_event: object
    wall_trigger: float
    snapshots: Optional[list] = None
    captured: bool = False
    mode: str = "clip"


class RingClipExporter:
    """Turns live ``FinalEvent``s into 30s evidence clips cut from the ring."""

    def __init__(
        self,
        config: Config,
        *,
        margin: float = _EXPORT_MARGIN_SECONDS,
        overlay_buffer=None,
    ) -> None:
        self.config = config
        ev = config.events
        self.pre = float(ev.get("pre_roll_seconds", 10))
        self.post = float(ev.get("post_roll_seconds", 20))
        # Mid-tier (screenshot_kmh_threshold..clip_kmh_threshold): save just a still of
        # the violator, no clip -- identifies the car at a tiny fraction of the disk and
        # avoids the ring's HEVC (unplayable in browsers/iOS once stream-copied).
        # x264 CRF for the re-encoded full clips (clean + annotated). Higher = smaller.
        # 30 ~= 2.5x smaller than the x264 default (23) with vehicle detail intact at
        # this camera distance; null keeps the default.
        self.clip_crf = ev.get("clip_encode_crf", 30)
        self.margin = float(margin)
        self.index_path = config.recording.get("segment_index_path", "/data/index/segments.sqlite")
        self.output_path = Path(ev.get("output_path", "/data/events"))
        self.thumb_offset = float(ev.get("thumbnail_time_offset_seconds", 15))
        self.timezone = config.app.timezone
        # Approach B: when an overlay buffer is supplied (and annotate_clips is on)
        # write a companion clip with track boxes + speed burned onto the 4K.
        self.overlay_buffer = overlay_buffer
        self.annotate = bool(ev.get("annotate_clips", True)) and overlay_buffer is not None
        # Boxes come from the sub-stream, the clip from the 4K main; the two
        # pipelines differ in latency so a positive offset advances the overlay
        # lookup to where the car actually is in the 4K frame (else boxes trail).
        self.sync_offset = float(ev.get("annotate_sync_offset_seconds", 0.0))
        # Per-clip auto-alignment: the sub<->4K latency jitters between events, so
        # instead of one static offset we detect the flagged car in a few sampled
        # 4K frames and solve for this clip's offset. sync_offset is then just the
        # search centre + fallback. A lazily-built detector (its own instance, so
        # it never contends with the analysis thread's model) does the detection.
        self.auto_align = bool(ev.get("annotate_auto_align", True))
        # Detection-based annotation: instead of projecting LIVE sub-stream boxes onto
        # the 4K clip (which needs the drift-prone sub<->4K sync offset), re-detect on
        # the clip's OWN frames so boxes are aligned by construction. The live track is
        # used only to identify the flagged car. Off by default (opt in per config).
        self.detect_annotate = bool(ev.get("annotate_from_detection", False))
        self._prep = None  # lazily-built downscale+de-warp transform for re-detection
        # Half-width of the offset search around sync_offset. Wide enough that the
        # solver doesn't clamp at the edge (fast cars amplify any residual lag, and
        # the true offset has been seen past 1.3s).
        self.align_radius = float(ev.get("annotate_align_radius_seconds", 0.8))
        # Detection-based annotation resolves the flagged car by IDENTITY (matching the
        # live track to clip detections over an offset search), so it tolerates a large
        # sub<->4K offset -- unlike the offset-projection path. The two 4K RTSP
        # connections drift over a day (seen past +3.5s), so the identity search must be
        # wide enough not to rail at its edge (a railed solve lands the box on the wrong
        # frame -> boxes lag). Centred at 0 (the drift is not around the static offset)
        # with generous half-width; the min-cost match still pins the true offset.
        self.detect_radius = float(ev.get("annotate_detect_radius_seconds", 6.0))
        # Each clip trusts its own non-railed solve (the latency really does jitter per
        # clip); the recent-solve median is only a fallback for railed/failed solves.
        # Every non-railed solve is recorded so that fallback tracks a real latency drift.
        self._offset_history: "deque" = deque(maxlen=9)
        self._detector = None  # None=unbuilt, False=unavailable, else the detector
        # Persist the overlay slice + render params per event so the offset can be
        # retuned offline (scripts/reannotate.py) without waiting for live events.
        self.save_overlay = bool(ev.get("annotate_save_overlay", True))
        cal = config.calibration
        ucfg = (cal.get("undistort") or {})
        self._k1 = float(ucfg.get("k1", 0.0))
        self._k2 = float(ucfg.get("k2", 0.0))
        self._roll = float(ucfg.get("roll_degrees", 0.0))
        self._homography = cal.get("overlay_homography")
        self.exported = 0
        self._pending: List[_PendingClip] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None

    def enqueue(self, final_event, wall_trigger_ts: float) -> None:
        """Queue an event; its clip is exported once the post-roll is recorded."""
        ready_at = wall_trigger_ts + self.post + self.margin
        with self._lock:
            self._pending.append(_PendingClip(ready_at, final_event, wall_trigger_ts))
        log.info(
            "Event %s queued for ring clip (ready in ~%.0fs).",
            final_event.event_type, max(0.0, ready_at - time.time()),
        )

    def enqueue_screenshot(self, final_event, wall_trigger_ts: float) -> None:
        """Queue a mid-tier event for an image-only save (the violator still, no clip).

        Below the clip threshold we keep just a still of the car -- no video. The image
        is the live sub-frame crop captured at dispatch, so nothing needs the post-roll;
        it's ready immediately.
        """
        with self._lock:
            self._pending.append(
                _PendingClip(wall_trigger_ts, final_event, wall_trigger_ts, mode="screenshot"))
        log.info("Event %s queued for violator still (image, no clip).",
                 final_event.event_type)

    def start(self) -> "RingClipExporter":
        self._thread = threading.Thread(target=self._run, name="clip-exporter", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain(time.time())
            self._stop.wait(1.0)
        # Final pass on shutdown: export anything already ready; drop the rest
        # (their post-roll isn't on disk yet, so a clip would be truncated).
        self._drain(time.time())
        with self._lock:
            dropped = len(self._pending)
            self._pending = []
        if dropped:
            log.info("Clip exporter stopped with %d event(s) whose post-roll wasn't recorded yet.", dropped)

    def _drain(self, now: float) -> None:
        ready: List[_PendingClip] = []
        with self._lock:
            keep: List[_PendingClip] = []
            for item in self._pending:
                (ready if item.ready_at <= now else keep).append(item)
            self._pending = keep
        # Phase 1 (fast): capture every ready item's overlay slice NOW, before any
        # slow render runs. Overlay snapshots live in a bounded rolling buffer, so
        # if we waited to slice inside the serial render a burst backlog could let
        # them scroll out first (the clean clip is safe -- it comes from the ring).
        # (Screenshot items need no overlay -- there's no clip to annotate.)
        if self.overlay_buffer is not None:
            for item in ready:
                if not item.captured and self.annotate and item.mode != "screenshot":
                    item.snapshots = self._grab_overlay(item.wall_trigger)
                    item.captured = True
        # Phase 2 (slow): export the clean clip + render the annotation serially.
        for item in ready:
            try:
                if item.mode == "screenshot":
                    self._export_screenshot(item.final_event, item.wall_trigger)
                else:
                    self._export(item.final_event, item.wall_trigger, item.snapshots)
                self.exported += 1
            except Exception:  # noqa: BLE001 - one bad export must not kill the thread
                log.exception("Failed to export ring %s for event %s", item.mode,
                              getattr(item.final_event, "event_id", "?"))

    def _grab_overlay(self, wall_trigger: float) -> Optional[list]:
        """Slice the overlay buffer for an event's clip window (incl. sync pad)."""
        if self.overlay_buffer is None or self.overlay_buffer.frame_size is None:
            return None
        start_ts = wall_trigger - self.pre
        end_ts = wall_trigger + self.post
        pad = abs(self.sync_offset) + 0.5
        return self.overlay_buffer.slice(start_ts - pad, end_ts + pad)

    def _write_thumbnail(self, final_event, clip_path, thumb_path, fallback_offset) -> None:
        """Crop the thumbnail to the violator's best sub frame (captured live, so it's
        pixel-accurate with no sub<->4K offset); fall back to a clip-frame grab when no
        frame was captured or the crop fails."""
        from .thumbnail import generate_thumbnail, save_cropped_thumbnail

        src = getattr(final_event, "thumb_source", None)
        if src is not None:
            try:
                frame, bbox = src
                if save_cropped_thumbnail(frame, bbox, thumb_path) is not None:
                    return
            except Exception:  # noqa: BLE001 - any issue -> fall back to the clip grab
                pass
        generate_thumbnail(clip_path, thumb_path, fallback_offset)

    def _export(self, final_event, wall_trigger: float, snapshots=None) -> None:
        from ..capture.segment_index import SegmentIndex
        from ..util.paths import ensure_dir
        from ..util.time import format_date_dir, format_segment_stamp, iso_now
        from .exporter import clamp_to_trigger_run, export_from_segments
        from .metadata import build_metadata, event_stem, write_metadata

        req_start = wall_trigger - self.pre
        req_end = wall_trigger + self.post
        with SegmentIndex(self.index_path) as index:
            segments = index.get_overlapping(req_start, req_end)
        if not segments:
            log.warning(
                "No ring segments cover event %s [%.0f, %.0f]; skipping clip.",
                final_event.event_type, req_start, req_end,
            )
            return
        # A recording gap inside the window would shift the concatenated footage (and the
        # annotation frame->time mapping). Keep only the contiguous run holding the trigger
        # and clamp the window to it; skip entirely if the trigger itself fell in a gap.
        run = clamp_to_trigger_run(segments, req_start, req_end, wall_trigger)
        if run is None:
            log.warning("Event %s trigger fell in a recording gap; skipping clip.",
                        final_event.event_type)
            return
        start_ts, end_ts = run.start_ts, run.end_ts
        if run.truncated_pre or run.truncated_post:
            log.info("Event %s clip shortened by a recording gap (-%.1fs pre, -%.1fs post).",
                     final_event.event_type, run.truncated_pre, run.truncated_post)

        ms = int(wall_trigger * 1000)
        stamp = format_segment_stamp(ms, self.timezone)
        short = final_event.event_id[:8]
        out_dir = ensure_dir(
            self.output_path / format_date_dir(ms, self.timezone) / final_event.event_type
        )
        stem = event_stem(stamp, final_event.event_type, short,
                          (c.evidence for c in final_event.candidates))
        clip_path = out_dir / f"{stem}.mp4"
        export_from_segments(run.segments, start_ts, end_ts, clip_path, crf=self.clip_crf)

        thumb_path = out_dir / f"{stem}.jpg"
        self._write_thumbnail(final_event, clip_path, thumb_path,
                              min(self.thumb_offset, (end_ts - start_ts) / 2))

        meta = build_metadata(
            final_event, clip_path=str(clip_path), thumbnail_path=thumb_path, config=self.config,
            created_at=iso_now(self.timezone), start_ts=start_ts,
            trigger_ts=wall_trigger, end_ts=end_ts,
            truncated_pre=run.truncated_pre, truncated_post=run.truncated_post,
        )
        write_metadata(meta, out_dir / f"{stem}.json")
        log.info("Exported event clip %s -> %s", final_event.event_type, clip_path)

        if self.annotate:
            self._annotate_clip(final_event, clip_path, out_dir / f"{stem}_annotated.mp4",
                                start_ts, snapshots)

    def _export_screenshot(self, final_event, wall_trigger: float) -> None:
        """Mid-tier: save just a still of the violator -- no video clip.

        Below the clip threshold we keep only an image: it identifies the car at a tiny
        fraction of the disk and avoids the ring's HEVC entirely (which browsers and even
        iOS can't play once segment-concatenated/stream-copied). The image is the live
        sub-frame crop of the violator (pixel-accurate, no sub<->4K offset); if no live
        frame was captured it falls back to a single 4K ring still at the trigger.
        """
        from ..util.paths import ensure_dir
        from ..util.time import format_date_dir, format_segment_stamp, iso_now
        from .metadata import build_metadata, event_stem, write_metadata

        ms = int(wall_trigger * 1000)
        stamp = format_segment_stamp(ms, self.timezone)
        short = final_event.event_id[:8]
        out_dir = ensure_dir(
            self.output_path / format_date_dir(ms, self.timezone) / final_event.event_type
        )
        stem = event_stem(stamp, final_event.event_type, short,
                          (c.evidence for c in final_event.candidates))
        thumb_path = out_dir / f"{stem}.jpg"
        if not self._save_still(final_event, wall_trigger, thumb_path):
            log.warning("No still could be saved for %s; skipping screenshot event.",
                        final_event.event_type)
            return
        meta = build_metadata(
            final_event, clip_path="", thumbnail_path=str(thumb_path), config=self.config,
            created_at=iso_now(self.timezone), start_ts=wall_trigger,
            trigger_ts=wall_trigger, end_ts=wall_trigger, media_kind="screenshot",
        )
        write_metadata(meta, out_dir / f"{stem}.json")
        log.info("Saved violator still (no clip) for %s -> %s",
                 final_event.event_type, thumb_path)

    def _save_still(self, final_event, wall_trigger: float, out_path) -> bool:
        """Save the violator image: the live sub-frame crop when captured (no offset),
        else one 4K frame grabbed from the ring at the trigger. Returns success."""
        from .thumbnail import save_cropped_thumbnail

        src = getattr(final_event, "thumb_source", None)
        if src is not None and save_cropped_thumbnail(src[0], src[1], out_path) is not None:
            return True
        # fallback: one frame from the covering ring segment near the trigger
        import subprocess

        from ..capture.segment_index import SegmentIndex
        from ..util.ffmpeg import ffmpeg_path
        with SegmentIndex(self.index_path) as index:
            segs = index.get_overlapping(wall_trigger - 0.5, wall_trigger + 0.5)
        if not segs:
            return False
        seg = segs[0]
        offset = max(0.0, wall_trigger - seg.start_ts)
        ff = ffmpeg_path() or "ffmpeg"
        subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", f"{offset:.3f}", "-i", str(seg.path),
                        "-frames:v", "1", "-q:v", "2", str(out_path)], capture_output=True)
        return Path(out_path).exists() and Path(out_path).stat().st_size > 0

    def _annotate_clip(self, final_event, clean_clip, out_path, start_ts, snapshots) -> None:
        """Best-effort: burn track boxes + speed onto a companion 4K clip.

        ``snapshots`` was captured eagerly when the event became ready (see
        :meth:`_drain`). Never raised to the caller -- a failed annotation must
        leave the clean evidence clip and metadata untouched.
        """
        from .overlay_render import render_annotated_clip, write_overlay_sidecar
        from .speed_log import event_speed_and_direction

        sub_size = self.overlay_buffer.frame_size
        if sub_size is None:
            log.warning("Overlay frame size unknown; skipping annotation for %s.",
                        final_event.event_type)
            return
        if not snapshots:
            log.warning("No overlay snapshots captured for %s; skipping annotation.",
                        final_event.event_type)
            return
        primary_id = final_event.primary_track_id
        passed_ids = [tid for tid in final_event.track_ids if tid != primary_id]
        # The flagged car's label shows the exact gate (steady) speed -- the same
        # number in the filename + speed report -- not the jumpy per-frame readout.
        sd = event_speed_and_direction(final_event)
        primary_speed_kmh = sd[0] if sd else None

        # Preferred path: draw boxes from detection on the clip's own frames (no sync
        # offset to drift). Falls through to the offset-projection method below if the
        # detector is unavailable or the re-detect can't run for this clip.
        if self.detect_annotate and self._try_detect_annotate(
            final_event, clean_clip, out_path, start_ts, snapshots, sub_size,
            primary_id, passed_ids, primary_speed_kmh,
        ):
            return

        # Resolve this clip's offset: auto-align against the 4K when possible,
        # else fall back to the static (search-centre) offset.
        offset = self._resolve_offset(clean_clip, snapshots, sub_size, start_ts, primary_id,
                                      final_event.event_type)

        if self.save_overlay:
            try:
                write_overlay_sidecar(
                    str(out_path).replace("_annotated.mp4", "_overlay.json"), snapshots,
                    clean_clip=str(clean_clip), sub_size=sub_size, start_ts=start_ts,
                    k1=self._k1, k2=self._k2, roll_degrees=self._roll, homography=self._homography,
                    primary_id=primary_id, passed_ids=passed_ids, sync_offset=offset,
                    primary_speed_kmh=primary_speed_kmh,
                )
            except Exception:  # noqa: BLE001 - sidecar is debug-only; never block the clip
                log.exception("Failed to write overlay sidecar for %s", final_event.event_type)
        try:
            render_annotated_clip(
                clean_clip, out_path, snapshots, sub_size,
                start_ts=start_ts, k1=self._k1, k2=self._k2, roll_degrees=self._roll,
                homography=self._homography, primary_id=primary_id, passed_ids=passed_ids,
                primary_speed_kmh=primary_speed_kmh, sync_offset=offset, crf=self.clip_crf,
            )
        except Exception:  # noqa: BLE001 - annotation is additive; never kill the export
            log.exception("Failed to render annotated clip for event %s",
                          getattr(final_event, "event_id", "?"))

    def _build_prep(self):
        """Lazily build the analysis frame transform (downscale + de-warp) so the
        re-detection sees the same de-warped space the boxes/projector use."""
        if self._prep is not None:
            return self._prep
        import cv2

        from ..analyze.undistort import build_undistorter

        und = build_undistorter(self.config.calibration)
        analyze_w = int(self.config.analysis.get("analyze_max_width", 0) or 0)

        def prep(frame):
            h, w = frame.shape[:2]
            if analyze_w and analyze_w < w:
                frame = cv2.resize(frame, (analyze_w, max(1, round(h * analyze_w / w))),
                                   interpolation=cv2.INTER_LINEAR)
            if und is not None:
                frame = und(frame)
            return frame

        self._prep = prep
        return prep

    def _try_detect_annotate(self, final_event, clean_clip, out_path, start_ts, snapshots,
                             sub_size, primary_id, passed_ids, primary_speed_kmh) -> bool:
        """Render the annotation by detecting on the clip's own frames. Returns True
        on success; False (detector unavailable / too little track / render failed) so
        the caller falls back to the offset-projection method."""
        detector = self._align_detector()
        if detector is None:
            return False
        from .overlay_render import (render_annotated_clip_detected, track_center_path,
                                     write_overlay_sidecar)

        primary_path = track_center_path(snapshots, primary_id)
        if len(primary_path) < 2:
            return False
        passed_paths = {tid: track_center_path(snapshots, tid) for tid in passed_ids}
        if self.save_overlay:
            try:
                write_overlay_sidecar(
                    str(out_path).replace("_annotated.mp4", "_overlay.json"), snapshots,
                    clean_clip=str(clean_clip), sub_size=sub_size, start_ts=start_ts,
                    k1=self._k1, k2=self._k2, roll_degrees=self._roll, homography=self._homography,
                    primary_id=primary_id, passed_ids=passed_ids, sync_offset=self.sync_offset,
                    primary_speed_kmh=primary_speed_kmh,
                )
            except Exception:  # noqa: BLE001 - sidecar is debug-only; never block the clip
                log.exception("Failed to write overlay sidecar for %s", final_event.event_type)
        try:
            result = render_annotated_clip_detected(
                clean_clip, out_path, detector=detector, prep=self._build_prep(),
                sub_size=sub_size, start_ts=start_ts,
                k1=self._k1, k2=self._k2, roll_degrees=self._roll, homography=self._homography,
                primary_path=primary_path, passed_paths=passed_paths,
                primary_speed_kmh=primary_speed_kmh, primary_track_id=primary_id,
                search_center=0.0, radius=self.detect_radius,
                crf=self.clip_crf,
            )
            return result is not None
        except Exception:  # noqa: BLE001 - any failure -> fall back to the offset method
            log.exception("Detection-annotate failed for %s; falling back to offset method.",
                          final_event.event_type)
            return False

    def _resolve_offset(self, clean_clip, snapshots, sub_size, start_ts, primary_id, etype) -> float:
        """Auto-align this clip's offset, smoothed against recent good solves."""
        if not self.auto_align:
            return self.sync_offset
        detector = self._align_detector()
        if detector is None:
            return self.sync_offset
        solved = None
        try:
            from .overlay_render import estimate_sync_offset

            solved = estimate_sync_offset(
                clean_clip, snapshots, sub_size, start_ts=start_ts, primary_id=primary_id,
                detector=detector, k1=self._k1, k2=self._k2, roll_degrees=self._roll,
                homography=self._homography, search_center=self.sync_offset,
                radius=self.align_radius,
            )
        except Exception:  # noqa: BLE001 - treat as a failed solve below
            log.exception("Auto-align raised for %s.", etype)

        chosen, record = smooth_offset(
            solved, list(self._offset_history), center=self.sync_offset,
            radius=self.align_radius, step=0.05, fallback=self.sync_offset)
        if record:
            self._offset_history.append(solved)
        n = len(self._offset_history)
        if record and chosen == solved:
            log.info("Auto-aligned %s offset -> %.2fs (rolling n=%d).", etype, chosen, n)
        else:
            shown = f"{solved:.2f}s" if solved is not None else "fail"
            note = "recorded; using median" if record else "rejected"
            log.info("%s offset solve=%s %s -> %.2fs (rolling n=%d).",
                     etype, shown, note, chosen, n)
        return chosen

    def _align_detector(self):
        """Lazily build a detector for auto-alignment (its own instance)."""
        if self._detector is False:
            return None
        if self._detector is None:
            try:
                from ..analyze.detector import build_detector

                self._detector = build_detector(self.config)
                log.info("Auto-align detector ready.")
            except Exception:  # noqa: BLE001 - no detector -> static offset
                log.exception("Auto-align detector unavailable; using static offset.")
                self._detector = False
                return None
        return self._detector

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30.0)  # let an in-flight export finish
