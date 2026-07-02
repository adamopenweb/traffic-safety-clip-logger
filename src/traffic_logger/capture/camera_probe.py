"""Camera probe.

Enumerates the available camera devices and their supported resolutions, pixel
formats, and frame rates. Prefers ``v4l2-ctl`` (Linux mini-PC); falls back to
OpenCV when v4l2-utils is unavailable.

The v4l2-ctl output parser is pure and unit-tested; the actual probe needs
hardware so it is exercised on the mini-PC.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..util.logging import get_logger

log = get_logger(__name__)


@dataclass
class FormatInfo:
    pixel_format: str
    description: str = ""
    # list of {"width": int, "height": int, "fps": [float, ...]}
    sizes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CameraInfo:
    device: str
    formats: List[FormatInfo] = field(default_factory=list)
    backend: str = "v4l2"


# --------------------------------------------------------------------------
# Pure parsing (tested)
# --------------------------------------------------------------------------
_FMT_RE = re.compile(r"\[\d+\]:\s*'(?P<fmt>[^']+)'\s*\((?P<desc>[^)]*)\)")
_SIZE_RE = re.compile(r"Size:\s*Discrete\s*(?P<w>\d+)x(?P<h>\d+)")
_INTERVAL_RE = re.compile(r"Interval:\s*Discrete\s*[\d.]+s\s*\((?P<fps>[\d.]+)\s*fps\)")


def parse_v4l2_formats(text: str) -> List[FormatInfo]:
    """Parse ``v4l2-ctl --list-formats-ext`` output into structured formats."""
    formats: List[FormatInfo] = []
    current: Optional[FormatInfo] = None
    current_size: Optional[Dict[str, Any]] = None

    for line in text.splitlines():
        fmt_m = _FMT_RE.search(line)
        if fmt_m:
            current = FormatInfo(
                pixel_format=fmt_m.group("fmt"),
                description=fmt_m.group("desc").strip(),
            )
            formats.append(current)
            current_size = None
            continue

        size_m = _SIZE_RE.search(line)
        if size_m and current is not None:
            current_size = {
                "width": int(size_m.group("w")),
                "height": int(size_m.group("h")),
                "fps": [],
            }
            current.sizes.append(current_size)
            continue

        int_m = _INTERVAL_RE.search(line)
        if int_m and current_size is not None:
            current_size["fps"].append(float(int_m.group("fps")))

    return formats


def select_format(
    formats: List[FormatInfo],
    preference: List[str],
    resolution: Optional[List[int]] = None,
    fps: Optional[int] = None,
) -> Optional[str]:
    """Pick the first preferred pixel format the camera actually supports.

    Optionally require a specific resolution / fps to be available for that
    format. Returns the chosen pixel_format token, or None if none match.
    """
    available = {f.pixel_format.upper(): f for f in formats}
    for pref in preference:
        info = available.get(str(pref).upper())
        if info is None:
            continue
        if resolution is None:
            return info.pixel_format
        for size in info.sizes:
            if [size["width"], size["height"]] == list(resolution):
                if fps is None or any(abs(f - fps) < 0.5 for f in size["fps"]):
                    return info.pixel_format
    return None


# --------------------------------------------------------------------------
# Probing (needs hardware)
# --------------------------------------------------------------------------
def _list_video_devices() -> List[str]:
    """Return /dev/video* device paths (Linux)."""
    import glob

    return sorted(glob.glob("/dev/video*"))


def _probe_with_v4l2(device: str) -> CameraInfo:
    out = subprocess.run(
        ["v4l2-ctl", "-d", device, "--list-formats-ext"],
        capture_output=True, text=True, timeout=15,
    )
    formats = parse_v4l2_formats(out.stdout) if out.returncode == 0 else []
    return CameraInfo(device=device, formats=formats, backend="v4l2")


def _probe_with_opencv(index: int = 0) -> List[CameraInfo]:
    """Best-effort OpenCV fallback: open camera indices and read their props."""
    try:
        import cv2  # type: ignore
    except ImportError:
        log.warning("OpenCV not installed; cannot probe cameras without v4l2-ctl")
        return []

    cameras: List[CameraInfo] = []
    for idx in range(index, index + 4):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        cameras.append(
            CameraInfo(
                device=str(idx),
                backend="opencv",
                formats=[FormatInfo(
                    pixel_format="unknown",
                    sizes=[{"width": width, "height": height, "fps": [fps] if fps else []}],
                )],
            )
        )
    return cameras


def probe_cameras(config: Any = None) -> List[CameraInfo]:
    """Enumerate cameras, preferring v4l2-ctl, falling back to OpenCV."""
    if shutil.which("v4l2-ctl"):
        devices = _list_video_devices()
        if devices:
            return [_probe_with_v4l2(d) for d in devices]
        log.warning("v4l2-ctl present but no /dev/video* devices found")
    else:
        log.info("v4l2-ctl not found; using OpenCV fallback")
    return _probe_with_opencv()
