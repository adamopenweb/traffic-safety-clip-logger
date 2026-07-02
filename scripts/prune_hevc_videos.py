"""Delete unplayable HEVC event videos, keeping each event's still + metadata.

The mid-tier short clips (and a few early full clips) were stream-copied from the HEVC
ring and won't play in any browser -- or even the iOS Files app. This deletes those
HEVC clean-clip .mp4 files, leaving the .jpg + .json (and any H.264 _annotated.mp4) so
the events become image records. One-off: new mid-tier events are image-only already,
and full/annotated clips are H.264.

    python scripts/prune_hevc_videos.py            # dry run (count + size)
    python scripts/prune_hevc_videos.py --apply    # delete them
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_logger.util.ffmpeg import ffmpeg_path  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Delete unplayable HEVC event clips.")
    ap.add_argument("--events-dir", default="data/events")
    ap.add_argument("--apply", action="store_true", help="Delete (default: dry-run).")
    args = ap.parse_args(argv)

    ff = ffmpeg_path() or "ffmpeg"
    ffprobe = ff[:-len("ffmpeg")] + "ffprobe" if ff.endswith("ffmpeg") else "ffprobe"
    root = Path(args.events_dir)

    clips = [p for p in root.rglob("*.mp4") if not p.name.endswith("_annotated.mp4")]
    hevc = []
    for p in clips:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(p)],
            capture_output=True, text=True)
        if out.stdout.strip() == "hevc":
            hevc.append(p)

    gb = sum(p.stat().st_size for p in hevc) / 1024 ** 3
    print(f"scanned {len(clips)} clean clips; {len(hevc)} are HEVC ({gb:.2f} GB)")
    if not args.apply:
        print("dry run -- re-run with --apply to delete.")
        return 0

    deleted = 0
    for p in hevc:
        try:
            p.unlink()
            deleted += 1
        except OSError as exc:
            print(f"  failed: {p.name}  {exc}")
    print(f"deleted {deleted} HEVC clip(s), freed {gb:.2f} GB "
          f"(stills + metadata kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
