"""Tail the recorded 4K ring as the analysis frame source.

Instead of opening a *second* live RTSP connection to the camera (which drops
frames — a concurrent 4K stream stalls, fragmenting fast near-lane tracks), the
analyzer reads the frames the recorder already wrote to the ring. Benefits:

* every recorded frame is processed (no delivery gaps) -> near-lane cars track
  cleanly end-to-end;
* the overlay snapshots are captured from the *exact* frames the clip exporter
  later annotates, so the sub<->4K (and 4K<->4K) sync offset is structurally zero;
* one fewer camera connection.

Cost: analysis runs ~one segment (~10 s) behind real time. Fine for a clip logger.

Each yielded frame carries its true capture wall-clock time (segment start parsed
from the filename + frame_index / fps), so events/passes/overlay stay correctly
timestamped despite the processing lag.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator, Optional, Tuple

from ..capture.recorder import parse_start_ms
from ..util.logging import get_logger

log = get_logger(__name__)


class RingFrameSource:
    """Yield ``(frame, capture_ts)`` for every frame of each finalized ring segment,
    in order, tailing the ring ~1 segment behind real time."""

    def __init__(self, ring_dir: str | Path, *, default_fps: float = 30.0,
                 poll_s: float = 0.5, stop_event: Optional[threading.Event] = None,
                 start_from_latest: bool = True) -> None:
        self.ring_dir = Path(ring_dir)
        self.default_fps = float(default_fps)
        self.poll_s = float(poll_s)
        self.stop = stop_event or threading.Event()
        self.start_from_latest = bool(start_from_latest)
        self._last_start_ms: Optional[int] = None

    def _segments(self):
        segs = []
        for p in self.ring_dir.glob("*/segment_*.mp4"):
            ms = parse_start_ms(p)
            if ms is not None:
                segs.append((ms, p))
        segs.sort(key=lambda x: x[0])
        return segs

    def _next_segment(self):
        """Earliest unprocessed *complete* segment (complete == not the newest,
        which may still be settling). Returns ``(start_ms, path)`` or None."""
        segs = self._segments()
        if not segs:
            return None
        if self._last_start_ms is None:
            # Begin near real time: skip all history, start with the segment that
            # finalizes *after* the current newest.
            self._last_start_ms = segs[-1][0] if self.start_from_latest else (segs[0][0] - 1)
            return None
        newest = segs[-1][0]
        for ms, p in segs:
            if ms > self._last_start_ms and ms < newest:
                return ms, p
        return None

    def frames(self) -> Iterator[Tuple[object, float]]:
        import cv2

        while not self.stop.is_set():
            nxt = self._next_segment()
            if nxt is None:
                self.stop.wait(self.poll_s)
                continue
            start_ms, path = nxt
            start_ts = start_ms / 1000.0
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                log.warning("Ring source: cannot open %s; skipping.", path)
                self._last_start_ms = start_ms
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or self.default_fps
            if fps <= 0:
                fps = self.default_fps
            i = -1
            try:
                while not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    i += 1
                    yield frame, start_ts + i / fps
            finally:
                cap.release()
            self._last_start_ms = start_ms
