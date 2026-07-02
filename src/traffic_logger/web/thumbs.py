"""On-demand thumbnail downscaling for the dashboard grids.

Event thumbnails are full-resolution stills (~270 KB each), so a dozen cards is a
few MB on first paint -- slow over a phone uplink. This downsizes a thumbnail to a
grid-friendly width on first request and caches the result to disk; later requests
serve the small (~20-40 KB) cached file.

It uses OpenCV (already on the analysis box) but degrades gracefully: if ``cv2``
isn't installed or the source isn't a decodable image, the caller falls back to the
original. So the ``web`` extra stays light and the dashboard still works without CV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def downscaled_thumb(src: Path, cache_dir: Path, max_w: int = 480) -> Optional[Path]:
    """Return a cached, width-capped JPEG copy of ``src`` (regenerated if stale), or
    ``None`` if downscaling isn't possible (no cv2 / not an image / any error) so the
    caller can serve the original."""
    try:
        import cv2  # heavy; optional
    except ImportError:
        return None
    try:
        out = cache_dir / f"{src.stem}_{max_w}.jpg"
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            return out
        img = cv2.imread(str(src))
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > max_w:
            scale = max_w / float(w)
            img = cv2.resize(img, (max_w, max(1, int(round(h * scale)))),
                             interpolation=cv2.INTER_AREA)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # write atomically-ish to a temp name then replace, so a concurrent reader
        # never sees a half-written file.
        tmp = out.with_suffix(".tmp.jpg")
        if cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 80]):
            tmp.replace(out)
            return out
        return None
    except Exception:  # noqa: BLE001 - downscaling is best-effort; fall back to original
        return None
