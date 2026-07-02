"""Event clip exporter.

Trims a time window into a playable MP4, either from a single source video
(offline analysis, where timestamps are offsets into the file) or by
concatenating overlapping ring-buffer segments (live deployment, where the
segment index records absolute timestamps). Operates from the segment index,
not live frames (spec "Key Implementation Notes").

Default clip geometry (spec "Event Clip Export"): 30s total = 10s pre-roll +
20s post-roll around the trigger.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from ..util.ffmpeg import ffmpeg_path
from ..util.logging import get_logger
from ..util.paths import ensure_dir

log = get_logger(__name__)


@dataclass(frozen=True)
class ClipWindow:
    start: float
    trigger: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def clip_window(trigger_ts: float, pre_roll: float, post_roll: float) -> ClipWindow:
    """[trigger - pre_roll, trigger + post_roll]."""
    return ClipWindow(start=trigger_ts - pre_roll, trigger=trigger_ts, end=trigger_ts + post_roll)


@dataclass(frozen=True)
class TriggerRun:
    """The contiguous recording run that actually contains the trigger, plus how much
    pre/post-roll a recording gap cost."""

    segments: list
    start_ts: float          # clamped clip start (>= the requested start)
    end_ts: float            # clamped clip end (<= the requested end)
    truncated_pre: float     # requested pre-roll lost to a gap before the trigger
    truncated_post: float    # requested post-roll lost to a gap after the trigger


def clamp_to_trigger_run(
    segments: Sequence,
    abs_start: float,
    abs_end: float,
    trigger_ts: float,
    *,
    gap_tolerance: float = 0.5,
) -> Optional[TriggerRun]:
    """Restrict an export window to the contiguous recording run holding the trigger.

    Ring recording can gap (the recorder restarts ffmpeg with backoff on a camera/Wi-Fi
    drop). :func:`export_from_segments` concatenates + trims on the assumption of a
    continuous timeline, so a gap *inside* the window shifts the footage earlier -- the
    clip covers the wrong wall-clock range and, worse, the annotation renderer (which maps
    frame ``i`` -> ``start_ts + i/fps``) draws boxes at the wrong times after the gap.

    Rather than build spliced-timeline bookkeeping for an event that happens a few times a
    year, keep only the run that contains the trigger (the evidence) and clamp the window
    to it: within a run the timeline is continuous, so the offset math and the annotation
    mapping stay correct *by construction*. Lost pre/post-roll is reported so the caller can
    record it honestly. Returns ``None`` when no run covers the trigger (recording was down
    at the moment of the event) -- the caller then skips the clip like the no-coverage path.
    Pure -- unit-tested without ffmpeg.
    """
    segs = sorted(segments, key=lambda s: s.start_ts)
    if not segs:
        return None
    runs: List[list] = [[segs[0]]]
    for s in segs[1:]:
        if s.start_ts - runs[-1][-1].end_ts > gap_tolerance:
            runs.append([s])          # a gap wider than tolerance -> new run
        else:
            runs[-1].append(s)
    for run in runs:
        if run[0].start_ts <= trigger_ts <= run[-1].end_ts:
            clip_start = max(abs_start, run[0].start_ts)
            clip_end = min(abs_end, run[-1].end_ts)
            return TriggerRun(
                segments=run, start_ts=clip_start, end_ts=clip_end,
                truncated_pre=round(max(0.0, clip_start - abs_start), 3),
                truncated_post=round(max(0.0, abs_end - clip_end), 3))
    return None


def _run_ffmpeg(cmd: List[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip() or result.returncode}")


def export_from_source(
    source: str | Path,
    start_offset: float,
    duration: float,
    out_path: str | Path,
) -> Path:
    """Extract [start_offset, start_offset+duration] from a single video file.

    Used for offline analysis where the trigger time is an offset into the
    source. Re-encodes for frame-accurate, self-contained clips.
    """
    ff = ffmpeg_path() or "ffmpeg"
    start = max(0.0, start_offset)
    out = Path(out_path)
    ensure_dir(out.parent)
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-an", str(out),
    ]
    _run_ffmpeg(cmd)
    return out


def export_from_segments(
    segments: Sequence,
    abs_start: float,
    abs_end: float,
    out_path: str | Path,
    *,
    copy: bool = False,
    crf: int | None = None,
) -> Path:
    """Concatenate overlapping ring segments and trim to [abs_start, abs_end].

    ``segments`` are SegmentRecord-like objects (``.path``, ``.start_ts``),
    sorted oldest-first, that overlap the window. Timestamps are absolute unix
    seconds (as stored in the segment index).

    ``copy=True`` stream-copies the source codec (no re-encode): ~3x smaller,
    lossless, near-zero CPU -- but the trim snaps to the nearest keyframe (the cut
    can't land mid-GOP), so the clip's start drifts up to ~1 GOP. That's fine for a
    self-contained context clip but NOT for clips fed to the annotation auto-align,
    which needs a frame-accurate start; those must re-encode (the default).
    """
    segs = sorted(segments, key=lambda s: s.start_ts)
    if not segs:
        raise ValueError("No segments cover the requested window")

    offset = max(0.0, abs_start - segs[0].start_ts)
    duration = max(0.0, abs_end - abs_start)
    out = Path(out_path)
    ensure_dir(out.parent)

    # ffmpeg concat demuxer list. Paths must be ABSOLUTE: the concat demuxer
    # resolves relative entries against the list file's directory (a temp dir),
    # not the working dir, so a relative ring path (e.g. data/ring/...) wouldn't
    # be found. resolve() also normalizes forward slashes, portable on Windows.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        for seg in segs:
            fh.write(f"file '{Path(seg.path).resolve().as_posix()}'\n")
        list_path = fh.name
    try:
        ff = ffmpeg_path() or "ffmpeg"
        if copy:
            codec = ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        else:
            codec = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
            if crf is not None:
                codec += ["-crf", str(int(crf))]
        cmd = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-ss", f"{offset:.3f}", "-t", f"{duration:.3f}",
            # +faststart: moov atom up front so a web <video> player starts streaming
            # immediately rather than seeking the index at the end of the file first.
            *codec, "-movflags", "+faststart", "-an", str(out),
        ]
        _run_ffmpeg(cmd)
    finally:
        Path(list_path).unlink(missing_ok=True)
    return out
