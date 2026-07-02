"""ffmpeg / ffprobe discovery and metadata helpers.

Detects the binaries and probes recorded segments for the metadata the segment
index needs (duration, codec, resolution, fps, size). Actual encode/segment
invocation is built in ``capture/recorder.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def ffmpeg_path() -> Optional[str]:
    """Absolute path to the ffmpeg binary, or None if not on PATH."""
    return shutil.which("ffmpeg")


def ffprobe_path() -> Optional[str]:
    """Absolute path to the ffprobe binary, or None if not on PATH."""
    return shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    """True if ffmpeg is discoverable on PATH."""
    return ffmpeg_path() is not None


def ffprobe_available() -> bool:
    """True if ffprobe is discoverable on PATH."""
    return ffprobe_path() is not None


def parse_fps(rate: str) -> float:
    """Parse an ffprobe frame-rate string like ``"30/1"`` into a float.

    Returns 0.0 for empty / malformed / zero-denominator values.
    """
    if not rate or rate in ("0/0", "N/A"):
        return 0.0
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0


def parse_ffprobe_json(payload: Dict[str, Any], size_bytes: int) -> Dict[str, Any]:
    """Normalize an ffprobe JSON document into segment metadata.

    Split out from :func:`ffprobe_segment` so the parsing is unit-testable
    without invoking ffprobe.
    """
    streams = payload.get("streams") or [{}]
    stream = streams[0] if streams else {}
    fmt = payload.get("format") or {}

    duration_raw = stream.get("duration") or fmt.get("duration") or 0.0
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError):
        duration = 0.0

    return {
        "codec": stream.get("codec_name", "") or "",
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": parse_fps(stream.get("r_frame_rate", "") or ""),
        "duration": duration,
        "size_bytes": int(size_bytes),
    }


def ffprobe_segment(path: str | Path, timeout: float = 30.0) -> Dict[str, Any]:
    """Probe a media file for codec/resolution/fps/duration + size on disk.

    Raises ``RuntimeError`` if ffprobe is unavailable or fails.
    """
    probe = ffprobe_path()
    if probe is None:
        raise RuntimeError("ffprobe not found on PATH")
    p = Path(path)
    cmd = [
        probe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate:format=duration",
        "-of", "json",
        str(p),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {p}: {result.stderr.strip() or result.returncode}"
        )
    payload = json.loads(result.stdout or "{}")
    size = p.stat().st_size if p.exists() else 0
    return parse_ffprobe_json(payload, size)
