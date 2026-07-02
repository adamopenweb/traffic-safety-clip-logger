"""Re-render an annotated event clip from its overlay sidecar.

The live exporter saves a ``<name>_overlay.json`` beside each event clip holding
the per-frame box geometry + render params. This re-renders the annotated clip
from that sidecar, optionally at a different sync offset -- so the offset that
makes the boxes sit on the moving cars can be dialed in on a real clip without
waiting for new live events.

Examples:
    # Re-render at a single offset (writes <name>_annotated.mp4 next to the clip).
    python scripts/reannotate.py data/events/.../EVENT_overlay.json --offset 0.45

    # Sweep several offsets into _off0.30.mp4, _off0.45.mp4, ... to compare.
    python scripts/reannotate.py data/events/.../EVENT_overlay.json --sweep 0.3 0.45 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_logger.events.overlay_render import reannotate_from_sidecar  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Re-render an annotated clip from its overlay sidecar.")
    ap.add_argument("sidecar", help="Path to <name>_overlay.json")
    ap.add_argument("--offset", type=float, default=None,
                    help="Sync offset seconds (default: the value stored in the sidecar).")
    ap.add_argument("--sweep", type=float, nargs="+", default=None,
                    help="Render one clip per offset, suffixed _off<val>.mp4, for comparison.")
    ap.add_argument("--clip", default=None, help="Override the clean clip path from the sidecar.")
    ap.add_argument("-o", "--out", default=None, help="Output path (single-offset mode only).")
    args = ap.parse_args(argv)

    sidecar = Path(args.sidecar)
    base = sidecar.name.replace("_overlay.json", "")
    parent = sidecar.parent

    if args.sweep:
        for off in args.sweep:
            out = parent / f"{base}_off{off:.2f}.mp4"
            res = reannotate_from_sidecar(sidecar, out, sync_offset=off, clean_clip=args.clip)
            print(f"offset {off:+.2f} -> {res}")
        return 0

    out = Path(args.out) if args.out else parent / f"{base}_annotated.mp4"
    res = reannotate_from_sidecar(sidecar, out, sync_offset=args.offset, clean_clip=args.clip)
    print(f"-> {res}")
    return 0 if res else 1


if __name__ == "__main__":
    raise SystemExit(main())
