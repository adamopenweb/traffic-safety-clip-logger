"""A tiny manual-exclude store for the Top Speeds page.

The page surfaces the fastest clips on record, but a few are not the kind of
entry it is meant to highlight -- e.g. a marked police car running an
emergency call with its lights on. Rather than build a fragile auto-detector,
this is a deliberately simple, reversible manual override: a JSON list of
``event_id`` strings that the Hall endpoint filters out.

It only affects the Top Speeds page -- excluded events still appear in Now,
Browse, and all the stats/denominator counts. Restoring an event is just
removing its id again.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Set


class Exclusions:
    """Thread-safe, file-backed set of excluded ``event_id`` values."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._ids: Set[str] = self._load()

    def _load(self) -> Set[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        if isinstance(data, dict):  # tolerate {"excluded": [...]} shape
            data = data.get("excluded", [])
        return {str(x) for x in data} if isinstance(data, list) else set()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(sorted(self._ids), indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def all(self) -> Set[str]:
        with self._lock:
            return set(self._ids)

    def contains(self, event_id: str) -> bool:
        with self._lock:
            return event_id in self._ids

    def add(self, event_id: str) -> bool:
        """Exclude an id. Returns True if it was newly added."""
        with self._lock:
            if event_id in self._ids:
                return False
            self._ids.add(event_id)
            self._save()
            return True

    def remove(self, event_id: str) -> bool:
        """Restore an id. Returns True if it was present and removed."""
        with self._lock:
            if event_id not in self._ids:
                return False
            self._ids.discard(event_id)
            self._save()
            return True

    def as_list(self) -> List[str]:
        return sorted(self.all())
