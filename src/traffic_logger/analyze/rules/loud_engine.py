"""Loud engine rule (Milestone 8, optional).

Triggers a standalone loud_engine event and/or boosts an overlapping visual
event's score when audio loudness exceeds the rolling baseline. Stub for now.
"""

from __future__ import annotations

from .base import CandidateEvent, Rule


class LoudEngineRule(Rule):
    event_type = "loud_engine"

    def evaluate(self, *args, **kwargs) -> list[CandidateEvent]:
        raise NotImplementedError("Loud engine rule lands in Milestone 8 (optional audio)")
