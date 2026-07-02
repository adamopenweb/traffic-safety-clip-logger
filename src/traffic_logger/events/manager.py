"""Event manager.

Receives candidate events from the rules and decides which become saved clips
(spec "Event Manager"): it scores, deduplicates repeated triggers, and merges
nearby triggers into a single combined event so that one vehicle doing several
unsafe things at once produces one clip with all labels — not three.

Merge policy: candidates that share a track and trigger within
``merge_window_seconds`` are grouped. A group becomes a :class:`FinalEvent`
once its merge window has elapsed (``flush``) or at end-of-stream
(``flush_all``). The group's primary type/score/track come from its
highest-scoring candidate; ``event_types`` lists every contributing type.

This module is pure (no ffmpeg/IO); the offline analyzer and the CLI handle
turning a :class:`FinalEvent` into clip + thumbnail + metadata files.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set

from ..analyze.rules.base import CandidateEvent


@dataclass
class FinalEvent:
    event_id: str
    event_type: str
    event_types: List[str]
    trigger_ts: float
    score: float
    primary_track_id: Optional[int]
    track_ids: List[int]
    candidates: List[CandidateEvent]
    # Best sub frame + bbox of the violator (frame, bbox), attached live for the
    # thumbnail crop; None when unavailable (falls back to a clip-frame grab).
    thumb_source: Optional[object] = None


@dataclass
class _Group:
    candidates: List[CandidateEvent] = field(default_factory=list)
    track_ids: Set[int] = field(default_factory=set)
    last_ts: float = 0.0

    def add(self, candidate: CandidateEvent, ts: float) -> None:
        self.candidates.append(candidate)
        if candidate.primary_track_id is not None:
            self.track_ids.add(candidate.primary_track_id)
        self.track_ids.update(candidate.track_ids)
        self.last_ts = max(self.last_ts, ts)


class EventManager:
    def __init__(
        self,
        merge_window_seconds: float = 12.0,
        cooldown_seconds: float = 8.0,
        id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self.merge_window = float(merge_window_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self._id_factory = id_factory
        self._pending: List[_Group] = []

    def add(self, candidate: CandidateEvent, ts: Optional[float] = None) -> None:
        """Buffer a candidate, merging it into an open group when appropriate."""
        if ts is None:
            ts = candidate.trigger_ts
        for group in self._pending:
            shares_track = (
                candidate.primary_track_id in group.track_ids
                or bool(set(candidate.track_ids) & group.track_ids)
            )
            if shares_track and ts - group.last_ts <= self.merge_window:
                group.add(candidate, ts)
                return
        new_group = _Group()
        new_group.add(candidate, ts)
        self._pending.append(new_group)

    def flush(self, now: float) -> List[FinalEvent]:
        """Finalize groups whose merge window has closed before ``now``."""
        ready, still_pending = [], []
        for group in self._pending:
            if now - group.last_ts > self.merge_window:
                ready.append(group)
            else:
                still_pending.append(group)
        self._pending = still_pending
        return [self._finalize(g) for g in ready]

    def flush_all(self) -> List[FinalEvent]:
        """Finalize all pending groups (end of stream)."""
        finals = [self._finalize(g) for g in self._pending]
        self._pending = []
        return finals

    def _finalize(self, group: _Group) -> FinalEvent:
        primary = max(group.candidates, key=lambda c: c.score)
        # Preserve first-seen order of event types.
        types: List[str] = []
        for c in group.candidates:
            if c.event_type not in types:
                types.append(c.event_type)
        return FinalEvent(
            event_id=self._id_factory(),
            event_type=primary.event_type,
            event_types=types,
            trigger_ts=primary.trigger_ts,
            score=max(c.score for c in group.candidates),
            primary_track_id=primary.primary_track_id,
            track_ids=sorted(group.track_ids),
            candidates=list(group.candidates),
        )
