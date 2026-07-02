"""Ring orphan check: ring segments that exist on disk but aren't in the segment
index leak disk forever (pruning is index-driven). The 2026-06-28 recorder fix
(probe-before-move, commit 42bbcb0) should keep this at 0; this is the watchdog.

Counts dated-dir segments missing from the index, ignoring the incoming/ dir and
anything written in the last 2 minutes (a just-finalized file may not be indexed
yet). Appends a timestamped PASS/FAIL line to data/logs/orphan_check.log and exits
0 (clean) or 1 (orphans found). Safe to run on a schedule; read-only, deletes
nothing.

Usage: python scripts/check_orphans.py
"""
from __future__ import annotations
import os
import sqlite3
import glob
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "index" / "segments.sqlite"
RING = ROOT / "data" / "ring"
LOG = ROOT / "data" / "logs" / "orphan_check.log"
GRACE_SECONDS = 120  # ignore very recent files (finalize/index race)


def indexed_paths() -> set:
    con = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True, timeout=10.0)
    try:
        return {os.path.normcase(os.path.abspath(p)) for (p,) in
                con.execute("SELECT path FROM segments")}
    finally:
        con.close()


def main() -> int:
    now = time.time()
    try:
        indexed = indexed_paths()
    except sqlite3.OperationalError as exc:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  ERROR  index unreadable: {exc}"
        _append(line)
        print(line)
        return 2
    # dated dirs only (data/ring/YYYY-MM-DD/...), never the incoming/ write dir
    orphans = []
    for p in glob.glob(str(RING / "20*" / "segment_*.mp4")):
        if now - os.path.getmtime(p) < GRACE_SECONDS:
            continue
        if os.path.normcase(os.path.abspath(p)) not in indexed:
            orphans.append(p)
    stamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    if not orphans:
        line = f"{stamp}  PASS  0 orphans (indexed={len(indexed)})"
        _append(line)
        print(line)
        return 0
    mb = sum(os.path.getsize(p) for p in orphans) / 1048576
    line = (f"{stamp}  FAIL  {len(orphans)} orphan(s), {mb:.1f} MB -- recorder fix "
            f"may have regressed; newest: {os.path.basename(max(orphans, key=os.path.getmtime))}")
    _append(line)
    print(line)
    for p in sorted(orphans):
        print("   ", p)
    return 1


def _append(line: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
