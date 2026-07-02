"""Backfill: rename already-saved event clips to the descriptive scheme.

Older clips were named ``<stamp>_<event_type>_<short>.{mp4,jpg,json,...}`` -- the
speed/type/direction weren't in the name, so a folder listing told you nothing about
what each violation was. This walks the events tree, reads each event's metadata
sidecar, and renames the whole file group (``.mp4 .jpg .json _annotated.mp4
_overlay.json``) to the new ``<stamp>_<tokens>_<event_type>_<short>`` form produced by
``events.metadata.event_stem`` -- the same helper the live writer now uses, so old and
new clips match. The sidecar's ``clip_path`` / ``thumbnail_path`` are rewritten too.

DRY-RUN by default (prints the plan); pass --apply to actually rename. Idempotent: a
clip already in the new scheme is left alone, so it's safe to re-run.

Usage:
    python scripts/rename_event_clips.py [--events-dir data/events] [--apply]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow running as a plain script (no install) by adding src/ to the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_logger.events.metadata import event_stem  # noqa: E402

_STAMP = re.compile(r"^(\d{8}_\d{6})_")  # YYYYMMDD_HHMMSS prefix


def _new_stem(meta: dict, old_stem: str) -> str | None:
    """Compute the descriptive stem for this event, or None if unchanged/unparseable."""
    m = _STAMP.match(old_stem)
    if not m:
        return None
    stamp = m.group(1)
    event_type = meta.get("event_type")
    event_id = meta.get("event_id", "")
    short = event_id[:8] if event_id else old_stem.rsplit("_", 1)[-1]
    triggers = (meta.get("evidence") or {}).get("triggers") or []
    evidences = [t.get("evidence") or {} for t in triggers]
    new_stem = event_stem(stamp, event_type, short, evidences)
    return new_stem if new_stem != old_stem else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-dir", default="data/events")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: dry run)")
    args = ap.parse_args()

    root = Path(args.events_dir)
    if not root.exists():
        print(f"events dir not found: {root}")
        return 1

    renamed = skipped = groups = 0
    # Anchor on the metadata sidecar (<stem>.json); skip overlay sidecars.
    for meta_path in sorted(root.rglob("*.json")):
        if meta_path.name.endswith("_overlay.json"):
            continue
        old_stem = meta_path.name[:-len(".json")]
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! skip {meta_path.name}: {exc}")
            continue
        new_stem = _new_stem(meta, old_stem)
        if new_stem is None:
            skipped += 1
            continue
        groups += 1
        # Every sibling file that starts with the old stem (mp4/jpg/json/annotated/overlay).
        siblings = [p for p in meta_path.parent.iterdir()
                    if p.is_file() and p.name.startswith(old_stem)]
        print(f"{old_stem}\n  -> {new_stem}   ({len(siblings)} files)")
        if not args.apply:
            continue
        # Rewrite the metadata's own path fields before moving it.
        for key in ("clip_path", "thumbnail_path"):
            if isinstance(meta.get(key), str):
                meta[key] = meta[key].replace(old_stem, new_stem)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for p in siblings:
            new_name = p.name.replace(old_stem, new_stem, 1)
            target = p.with_name(new_name)
            if target.exists():
                print(f"  ! target exists, skipping {p.name}")
                continue
            p.rename(target)
            renamed += 1

    verb = "renamed" if args.apply else "would rename"
    print(f"\n{groups} event group(s) {verb}; {renamed} file(s) moved; {skipped} already current.")
    if not args.apply and groups:
        print("Dry run -- re-run with --apply to perform the renames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
