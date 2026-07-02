"""Rule base classes.

Defines the common contract every detection rule implements: consume track
state, optionally emit a :class:`CandidateEvent`. The concrete rules
(relative speeding, center-lane pass, loud engine) fill this in across
Milestones 4, 5, and 8.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CandidateEvent:
    """A candidate event emitted by a rule, before the manager decides on it.

    The event manager scores, deduplicates, merges, and (if accepted) exports
    these into the final clip + metadata sidecar.
    """

    event_type: str
    trigger_ts: float
    primary_track_id: Optional[int] = None
    score: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    track_ids: List[int] = field(default_factory=list)


class Rule(ABC):
    """Abstract detection rule.

    ``event_type`` names the rule for folder/metadata routing. ``enabled``
    reflects config. ``update`` is called as tracks evolve and returns any
    candidate events triggered this step.
    """

    event_type: str = "base"

    def __init__(self, config: Dict[str, Any], enabled: bool = True) -> None:
        self.config = config
        self.enabled = enabled

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> List[CandidateEvent]:
        """Process the latest track state and return candidate events.

        Signatures are rule-specific (each rule needs different inputs); the
        analyzer knows how to call each rule. Returns a (possibly empty) list.
        """
        raise NotImplementedError
