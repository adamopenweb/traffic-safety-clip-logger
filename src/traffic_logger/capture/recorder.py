"""Camera recorder.

Captures the camera stream and writes continuous H.264 segments to the ring
buffer, recording each completed segment in the segment index.

Design (chosen for robustness + testability):

* ffmpeg's ``segment`` muxer writes fixed-length files into an ``incoming/``
  staging directory, named ``segment_<unix_ms>.mp4`` via strftime ``%s000``.
* A lightweight indexer loop notices *completed* segments (every file except
  the newest, which is still being written), moves each into the dated ring
  folder ``<ring>/YYYY-MM-DD/``, probes it with ffprobe, and inserts a row into
  the index. The active segment therefore never appears in the index, so the
  pruner can never delete it.
* The ffmpeg subprocess is supervised: if it exits unexpectedly (camera
  disconnect, crash) it is restarted with backoff. Local recording must survive
  transient failures.

The pure pieces (`build_capture_command`, `parse_start_ms`, `completed_segments`,
`finalize_segment`) are unit/integration tested; ``Recorder.run`` is not (it
needs a live camera) but is exercised manually via a synthetic ffmpeg source.
"""

from __future__ import annotations

import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from ..util.ffmpeg import ffmpeg_path, ffprobe_segment
from ..util.logging import get_logger
from ..util.paths import ensure_dir
from ..util.time import format_date_dir
from .ring_pruner import prune_ring
from .segment_index import SegmentIndex, SegmentRecord

log = get_logger(__name__)

# Two segment filename forms, both decode to a unix-ms start time:
#   segment_<digits>.mp4          legacy: digits are unix milliseconds (ffmpeg %s000)
#   segment_YYYYMMDD-HHMMSS.mp4    portable: local wall-clock — Windows strftime
#                                  lacks the unix-seconds %s token, so this form
#                                  lets the recorder run natively on Windows too.
_SEGMENT_RE = re.compile(r"^segment_(\d+)\.mp4$")
_SEGMENT_DT_RE = re.compile(r"^segment_(\d{8})-(\d{6})\.mp4$")

# v4l2 pixel-format config tokens -> ffmpeg -input_format names.
_V4L2_FORMAT_MAP = {
    "YUYV": "yuyv422",
    "YUV422": "yuyv422",
    "MJPG": "mjpeg",
    "MJPEG": "mjpeg",
    "RGB3": "rgb24",
    "RGB24": "rgb24",
    "H264": "h264",
    "H.264": "h264",
}

# Pixel formats that are already H.264 from the camera's hardware encoder, so
# recording can stream-copy instead of re-encoding (no CPU encode load).
_H264_FORMATS = {"H264", "H.264"}

INCOMING_DIRNAME = "incoming"


# --------------------------------------------------------------------------
# Pure helpers (tested)
# --------------------------------------------------------------------------
def _v4l2_input_format(token: str) -> Optional[str]:
    return _V4L2_FORMAT_MAP.get(str(token).upper())


def build_input_args(camera_cfg: Dict[str, Any], pixel_format: Optional[str] = None) -> List[str]:
    """Build ffmpeg input args for a v4l2 camera from the camera config."""
    source = camera_cfg.get("source", "/dev/video0")
    res = camera_cfg.get("capture_resolution", [1280, 960])
    width, height = int(res[0]), int(res[1])
    fps = int(camera_cfg.get("capture_fps", 30))

    chosen = pixel_format
    if chosen is None:
        prefs = camera_cfg.get("pixel_format_preference") or []
        chosen = prefs[0] if prefs else None

    args = ["-f", "v4l2"]
    if chosen is not None:
        mapped = _v4l2_input_format(chosen)
        if mapped:
            args += ["-input_format", mapped]
    args += [
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", source,
    ]
    return args


# URL schemes recorded via ffmpeg's network demuxers (stream-copied), not v4l2.
_STREAM_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def is_rtsp_url(source: str) -> bool:
    """True if the recording source is a network stream rather than a v4l2 device."""
    return str(source).startswith(_STREAM_SCHEMES)


def build_rtsp_input_args(source: str) -> List[str]:
    """ffmpeg input args for a network stream. TCP transport for reliable
    recording (UDP drops frames); the camera already H.26x-encodes, so the
    stream is copied to disk without any CPU encode."""
    return ["-rtsp_transport", "tcp", "-i", str(source)]


def build_capture_command(
    config: Config,
    *,
    incoming_dir: str | Path,
    input_args: Optional[List[str]] = None,
    pixel_format: Optional[str] = None,
) -> List[str]:
    """Build the full ffmpeg command line for segmented H.264 recording.

    ``input_args`` may be supplied to override the v4l2 input (e.g. a synthetic
    ``lavfi`` source for tests). ``incoming_dir`` is the staging directory the
    segment muxer writes into.
    """
    rec = config.recording
    cam = config.camera
    seg_seconds = int(rec.get("segment_seconds", 10))
    fps = int(cam.get("capture_fps", 30))
    target_mbps = rec.get("target_bitrate_mbps", 10)
    max_mbps = rec.get("max_bitrate_mbps", 15)
    encode_mode = str(rec.get("encode_mode", "auto")).lower()

    # Recording source: an explicit recording.source (e.g. the camera's 4K main
    # RTSP stream) overrides camera.source — the analyzer may point camera.source
    # at a lighter sub-stream, but recording wants the full-resolution main.
    rec_source = rec.get("source") or cam.get("source", "/dev/video0")
    rtsp = is_rtsp_url(rec_source)

    if input_args is None:
        input_args = build_rtsp_input_args(rec_source) if rtsp else build_input_args(cam, pixel_format)

    # Decide whether to stream-copy or re-encode. The camera's hardware encoder
    # (H.264 over v4l2, or H.26x over RTSP) is copied — re-encoding wastes CPU and
    # on a weak box can't keep 30fps (produces frozen/duplicate frames). A network
    # stream is always already encoded, so copy unless explicitly forced to
    # re-encode; for v4l2, encode_mode "auto" copies an H.264 input, re-encodes raw.
    if rtsp:
        copy_mode = encode_mode != "h264"
    else:
        chosen = pixel_format
        if chosen is None:
            prefs = cam.get("pixel_format_preference") or []
            chosen = prefs[0] if prefs else None
        is_h264_input = str(chosen).upper() in _H264_FORMATS
        copy_mode = encode_mode == "copy" or (encode_mode == "auto" and is_h264_input)

    if copy_mode:
        codec_args = ["-c", "copy"]  # passthrough: no CPU encode
    else:
        codec_args = [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-b:v", f"{target_mbps}M",
            "-maxrate", f"{max_mbps}M",
            "-bufsize", f"{int(max_mbps) * 2}M",
            "-g", str(fps),
            "-force_key_frames", f"expr:gte(t,n_forced*{seg_seconds})",
        ]

    # strftime local wall-clock (portable: Windows strftime lacks the unix-seconds
    # %s token). -segment_atclocktime cuts on whole seconds, so second resolution
    # suffices; parse_start_ms decodes the name back to unix ms.
    output_pattern = str(Path(incoming_dir) / "segment_%Y%m%d-%H%M%S.mp4")

    return [
        ffmpeg_path() or "ffmpeg",
        "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
        *input_args,
        "-an",  # no audio on the v4l2 color stream
        *codec_args,
        "-f", "segment",
        "-segment_time", str(seg_seconds),
        "-segment_format", "mp4",
        "-segment_atclocktime", "1",
        "-reset_timestamps", "1",
        "-strftime", "1",
        output_pattern,
    ]


def parse_start_ms(path: str | Path, tz: Optional[str] = None) -> Optional[int]:
    """Extract the start timestamp (unix ms) from a segment filename.

    Accepts both the legacy unix-ms form and the portable local-wall-clock form. For the
    wall-clock form, pass ``tz`` (the app timezone) to decode it correctly: a naive
    ``datetime.timestamp()`` assumes the *host* tz, which is wrong whenever the host tz
    differs from the app tz, and is ambiguous in the DST fall-back hour. Binding the
    configured tz with ``fold=0`` resolves both -- deterministically taking the earlier of
    the two instants in the repeat hour (its second pass is stored under a suffixed name by
    :func:`finalize_segment`, so no footage is lost to the ambiguity). ``tz=None`` keeps the
    naive host-local decode (fine for ordering-only callers).
    """
    name = Path(path).name
    m = _SEGMENT_RE.match(name)
    if m:
        return int(m.group(1))
    m = _SEGMENT_DT_RE.match(name)
    if m:
        from datetime import datetime

        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        if tz:
            try:
                from zoneinfo import ZoneInfo

                dt = dt.replace(tzinfo=ZoneInfo(tz), fold=0)
            except Exception:  # noqa: BLE001 - unknown tz -> fall back to naive host-local
                pass
        return int(dt.timestamp() * 1000)
    return None


def completed_segments(incoming_files: List[Path]) -> List[Path]:
    """Given the incoming segment files, return those safe to finalize.

    The newest file (highest start timestamp) is still being written by ffmpeg,
    so it is excluded. Files without a parseable timestamp are ignored.
    """
    dated = [(parse_start_ms(p), p) for p in incoming_files]
    dated = [(ms, p) for ms, p in dated if ms is not None]
    dated.sort(key=lambda t: t[0])
    return [p for _, p in dated[:-1]]  # all but the newest


def finalize_segment(
    index: SegmentIndex,
    incoming_path: Path,
    ring_root: Path,
    timezone: Optional[str] = None,
) -> Optional[SegmentRecord]:
    """Move a completed segment into its dated folder and index it.

    Returns the inserted :class:`SegmentRecord`, or None if the file could not
    be probed (corrupted/zero-length segment is logged and skipped).
    """
    start_ms = parse_start_ms(incoming_path, timezone)
    if start_ms is None:
        return None
    start_ts = start_ms / 1000.0

    # Probe BEFORE moving into the ring. A corrupt/truncated segment (e.g. the
    # recorder was hard-killed mid-write) can't be indexed, and because pruning is
    # index-driven, a moved-but-unindexed file would leak disk forever outside
    # retention accounting. So probe in place and, on failure, discard the unusable
    # file rather than orphaning it in the ring (or re-probing it every cycle).
    try:
        meta = ffprobe_segment(incoming_path)
    except Exception as exc:  # noqa: BLE001 - corrupt/short segment shouldn't kill capture
        log.warning("Discarding unprobeable segment %s: %s", incoming_path.name, exc)
        try:
            incoming_path.unlink()
        except OSError:
            pass
        return None

    date_dir = ensure_dir(ring_root / format_date_dir(start_ms, timezone))
    dest = date_dir / incoming_path.name
    if dest.exists():
        # The local-time stamp collided with an already-stored segment -- most likely the
        # DST fall-back repeat hour (01:00-02:00 recurs), or a backward clock step. Moving
        # over it would silently destroy an already-indexed hour of footage AND leave the
        # index row pointing at the second pass. Suffix the new file instead: its wall time
        # stays ambiguous for that hour, but no footage is lost (the property that matters).
        stem, suf = dest.stem, dest.suffix
        i = 1
        while dest.exists():
            dest = date_dir / f"{stem}_dup{i}{suf}"
            i += 1
        log.warning("Segment name collision for %s; storing as %s (DST fall-back / clock step?).",
                    incoming_path.name, dest.name)
    shutil.move(str(incoming_path), str(dest))

    record = SegmentRecord(
        path=str(dest),
        start_ts=start_ts,
        end_ts=start_ts + meta["duration"],
        duration=meta["duration"],
        size_bytes=meta["size_bytes"],
        codec=meta["codec"],
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
    )
    index.add_segment(record)
    log.info(
        "Indexed segment %s (%.1fs, %dx%d, %s, %d bytes)",
        dest.name, record.duration, record.width, record.height,
        record.codec, record.size_bytes,
    )
    return record


# --------------------------------------------------------------------------
# Recorder lifecycle
# --------------------------------------------------------------------------
class Recorder:
    """Supervises the ffmpeg recording subprocess and the indexer/pruner loop."""

    def __init__(
        self,
        config: Config,
        *,
        input_args: Optional[List[str]] = None,
        pixel_format: Optional[str] = None,
    ) -> None:
        self.config = config
        self._input_args = input_args
        self._pixel_format = pixel_format

        rec = config.recording
        self.ring_root = Path(rec.get("ring_path", "/data/ring"))
        self.incoming_dir = self.ring_root / INCOMING_DIRNAME
        self.index_path = rec.get("segment_index_path", "/data/index/segments.sqlite")
        self.segment_seconds = int(rec.get("segment_seconds", 10))
        self.ring_max_bytes = int(float(rec.get("ring_max_gb", 200)) * (1024 ** 3))
        self.timezone = config.app.timezone

        self._stop = False
        self._proc: Optional[subprocess.Popen] = None
        # Prune at most once per this interval (seconds).
        self._prune_interval = 30.0
        self._last_prune = 0.0

    def request_stop(self, *_args) -> None:
        self._stop = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def run(self) -> int:
        """Record until interrupted. Returns a process exit code."""
        ensure_dir(self.incoming_dir)
        ensure_dir(self.ring_root)
        index = SegmentIndex(self.index_path)

        # Graceful shutdown on SIGINT/SIGTERM.
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):
                pass  # not in main thread / unsupported platform

        backoff = 1.0
        try:
            while not self._stop:
                cmd = build_capture_command(
                    self.config,
                    incoming_dir=self.incoming_dir,
                    input_args=self._input_args,
                    pixel_format=self._pixel_format,
                )
                log.info("Starting recorder: %s", " ".join(cmd))
                self._proc = subprocess.Popen(cmd)
                started = time.time()

                while self._proc.poll() is None and not self._stop:
                    self._index_completed(index)
                    self._maybe_prune(index)
                    time.sleep(1.0)

                rc = self._proc.poll()
                # Finalize whatever completed before exit.
                self._index_completed(index)

                if self._stop:
                    log.info("Recorder stopping (signal received).")
                    break

                # ffmpeg exited on its own — treat as a fault and restart.
                ran = time.time() - started
                backoff = 1.0 if ran > 30 else min(backoff * 2, 30.0)
                log.error(
                    "ffmpeg exited (code=%s) after %.0fs; restarting in %.0fs",
                    rc, ran, backoff,
                )
                time.sleep(backoff)
        finally:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            index.close()
        return 0

    def _index_completed(self, index: SegmentIndex) -> None:
        files = list(self.incoming_dir.glob("segment_*.mp4"))
        for path in completed_segments(files):
            if index.has(str(path)):
                continue
            try:
                finalize_segment(index, path, self.ring_root, self.timezone)
            except Exception:  # noqa: BLE001 - never let indexing crash capture
                log.exception("Failed to finalize segment %s", path)

    def _maybe_prune(self, index: SegmentIndex) -> None:
        now = time.time()
        if now - self._last_prune < self._prune_interval:
            return
        self._last_prune = now
        result = prune_ring(index, self.ring_max_bytes)
        if result.deleted_paths:
            log.info(
                "Pruned %d segment(s), freed %.2f GB (ring now %.2f GB)",
                len(result.deleted_paths),
                result.freed_bytes / (1024 ** 3),
                result.remaining_bytes / (1024 ** 3),
            )
