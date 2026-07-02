"""Aggressiveness knob -> concrete thresholds.

The config exposes a single ``events.aggressiveness`` value in [0.0, 1.0]:

    0.0 = strict   / fewer clips
    1.0 = sensitive / more clips

Each rule declares paired ``*_strict`` / ``*_sensitive`` thresholds. This
module interpolates between them so that one knob tunes the whole system
(spec section "Aggressiveness Knob").
"""

from __future__ import annotations

from typing import Any, Dict

_STRICT_SUFFIX = "_strict"
_SENSITIVE_SUFFIX = "_sensitive"


def clamp01(value: float) -> float:
    """Clamp a value to the inclusive range [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def lerp(strict: float, sensitive: float, aggressiveness: float) -> float:
    """Interpolate from ``strict`` (a=0) to ``sensitive`` (a=1).

    ``aggressiveness`` is clamped to [0, 1] first.
    """
    a = clamp01(aggressiveness)
    return strict + (sensitive - strict) * a


def resolve_thresholds(section: Dict[str, Any], aggressiveness: float) -> Dict[str, Any]:
    """Collapse every ``*_strict`` / ``*_sensitive`` pair into a single value.

    For a section like::

        {"percentile_threshold_strict": 0.97,
         "percentile_threshold_sensitive": 0.90,
         "enabled": True}

    returns::

        {"percentile_threshold": <lerp(0.97, 0.90, a)>, "enabled": True}

    Keys without a matching pair are passed through unchanged. A ``*_strict``
    key missing its ``*_sensitive`` partner (or vice versa) is left as-is so
    nothing is silently dropped.
    """
    a = clamp01(aggressiveness)
    resolved: Dict[str, Any] = {}
    consumed: set[str] = set()

    for key, value in section.items():
        if key.endswith(_STRICT_SUFFIX):
            base = key[: -len(_STRICT_SUFFIX)]
            sensitive_key = base + _SENSITIVE_SUFFIX
            if sensitive_key in section:
                resolved[base] = lerp(value, section[sensitive_key], a)
                consumed.add(key)
                consumed.add(sensitive_key)

    for key, value in section.items():
        if key in consumed or key in resolved:
            continue
        resolved[key] = value

    return resolved
