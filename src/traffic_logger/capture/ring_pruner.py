"""Ring buffer pruning.

Keeps the local recording buffer under a size cap by deleting the oldest
segments first (spec section "Ring Buffer Pruning"):

    * Delete oldest segments first
    * Never delete the active segment currently being written
    * (Event clips live outside the ring and are never touched here)

The pure selection logic ``select_segments_to_delete`` is unit tested.
``prune_ring`` wires it to the filesystem and segment index: it deletes the
chosen files and removes their rows. Event clips live outside the ring (and
outside the index), so they are never affected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional

from ..util.logging import get_logger

if TYPE_CHECKING:  # avoid a runtime import cycle with segment_index
    from .segment_index import SegmentIndex

log = get_logger(__name__)


@dataclass(frozen=True)
class SegmentInfo:
    """Minimal segment descriptor used by the pruner.

    ``start_ts`` orders segments (oldest first). ``size_bytes`` is summed
    against the cap. ``path`` identifies the file (and the active segment).
    """

    path: str
    start_ts: float
    size_bytes: int


def total_bytes(segments: Iterable[SegmentInfo]) -> int:
    return sum(s.size_bytes for s in segments)


def select_segments_to_delete(
    segments: Iterable[SegmentInfo],
    max_bytes: int,
    active_path: Optional[str] = None,
) -> List[SegmentInfo]:
    """Choose which segments to delete to bring the ring under ``max_bytes``.

    Returns segments in deletion order (oldest first). The ``active_path``
    segment is never selected, even if that means staying over the cap (we
    must not delete the file currently being written).

    Pure function: performs no filesystem I/O.
    """
    ordered = sorted(segments, key=lambda s: s.start_ts)
    current = total_bytes(ordered)
    if current <= max_bytes:
        return []

    to_delete: List[SegmentInfo] = []
    for seg in ordered:
        if current <= max_bytes:
            break
        if active_path is not None and seg.path == active_path:
            # Skip the active segment but keep scanning newer ones.
            continue
        to_delete.append(seg)
        current -= seg.size_bytes
    return to_delete


@dataclass(frozen=True)
class PruneResult:
    deleted_paths: List[str] = field(default_factory=list)
    freed_bytes: int = 0
    remaining_bytes: int = 0


def prune_ring(
    index: "SegmentIndex",
    max_bytes: int,
    active_path: Optional[str] = None,
    dry_run: bool = False,
) -> PruneResult:
    """Bring the ring buffer under ``max_bytes`` by deleting oldest segments.

    Deletes the chosen files from disk and removes their index rows (oldest
    first), never touching the active segment. Returns what was removed. A
    missing-on-disk file is treated as already gone — its row is still removed.
    """
    segments = index.get_all()
    to_delete = select_segments_to_delete(segments, max_bytes, active_path)

    freed = 0
    deleted: List[str] = []
    for seg in to_delete:
        if not dry_run:
            try:
                os.remove(seg.path)
            except FileNotFoundError:
                log.warning("Segment file already missing, removing index row: %s", seg.path)
            except OSError as exc:
                log.error("Failed to delete segment %s: %s", seg.path, exc)
                continue
            index.delete_segment(seg.path)
        freed += seg.size_bytes
        deleted.append(seg.path)

    remaining = index.total_bytes() if not dry_run else (total_bytes(segments) - freed)
    return PruneResult(deleted_paths=deleted, freed_bytes=freed, remaining_bytes=remaining)
