"""Event thumbnail generation.

Extracts a single representative JPG frame from an event clip at a configured
time offset (spec ``events.thumbnail_time_offset_seconds``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..util.ffmpeg import ffmpeg_path
from ..util.paths import ensure_dir


def generate_thumbnail(
    video_path: str | Path,
    out_path: str | Path,
    offset_seconds: float = 15.0,
) -> Path:
    """Write a single JPG frame from ``video_path`` at ``offset_seconds``.

    Falls back to the first frame if seeking past the clip yields nothing.
    """
    ff = ffmpeg_path() or "ffmpeg"
    out = Path(out_path)
    ensure_dir(out.parent)

    def _grab(offset: float) -> bool:
        cmd = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, offset):.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(out),
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        return out.exists() and out.stat().st_size > 0

    if _grab(offset_seconds):
        return out
    # Seek may have landed past the end of a short clip; retry at the start.
    _grab(0.0)
    return out


def save_cropped_thumbnail(
    frame, bbox, out_path: str | Path, *, pad: float = 0.6, min_w: int = 360,
) -> Path | None:
    """Write a 16:9 JPG cropped to a vehicle's ``bbox`` (+ padding) from a sub frame.

    ``bbox`` is the live detection on *this* sub frame, so the crop is pixel-accurate
    on the violator -- no sub<->4K offset. The crop is widened to a card-friendly 16:9
    around the box centre (with a minimum width so a distant car still reads), clamped
    to the frame. Returns the path, or None on any failure so the caller can fall back
    to the clip-frame grab.
    """
    try:
        import cv2

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if x2 <= x1 or y2 <= y1:
            return None
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pw, ph = (x2 - x1) * (1 + 2 * pad), (y2 - y1) * (1 + 2 * pad)
        tw = max(pw, ph * 16.0 / 9.0, float(min_w))
        tw = min(tw, float(w))
        th = min(tw * 9.0 / 16.0, float(h))
        tw = th * 16.0 / 9.0  # re-derive if height was the binding constraint
        left = int(round(min(max(cx - tw / 2, 0.0), w - tw)))
        top = int(round(min(max(cy - th / 2, 0.0), h - th)))
        crop = frame[top:top + int(round(th)), left:left + int(round(tw))]
        if crop.size == 0:
            return None
        out = Path(out_path)
        ensure_dir(out.parent)
        if cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            return out
        return None
    except Exception:  # noqa: BLE001 - best-effort; caller falls back to the clip grab
        return None
