"""Build the Traffic Watch favicon / home-screen icon set.

Source of truth is the Claude Design component "Traffic Watch Icon.dc.html": a
1024x1024 SVG of an eye whose iris is a speedometer with a road receding into the
pupil. That component computes the gauge ticks/needle in a little JS template; here
we resolve that template to a static SVG (so it's a plain asset, no runtime), then
rasterize the PNG sizes iOS/Android need via headless Chrome.

Outputs into src/traffic_logger/web/static/:
  icon.svg            rounded mark, the modern SVG favicon
  apple-touch-icon.png 180, full-bleed square (iOS rounds it itself)
  icon-192.png / icon-512.png  full-bleed square PWA icons (any + maskable;
                       the eye sits well inside the maskable safe zone)
  favicon-32.png       small PNG favicon
  favicon.ico          16+32 legacy favicon
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "traffic_logger" / "web" / "static"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# --- resolve the gauge template (mirrors the .dc.html renderVals()) ----------
CX = CY = 512
START, END, N, AMBER_IDX = 158, 382, 11, 8


def pt(a_deg: float, r: float):
    a = a_deg * math.pi / 180
    return CX + r * math.cos(a), CY + r * math.sin(a)


_ticks = []
_amber = None
for i in range(N):
    deg = START + (END - START) * i / (N - 1)
    major = i % 2 == 0
    if i == AMBER_IDX:
        x1, y1 = pt(deg, 108)
        x2, y2 = pt(deg, 146)
        _amber = (x1, y1, x2, y2)
        continue
    ix, iy = pt(deg, 110 if major else 120)
    ox, oy = pt(deg, 143)
    _ticks.append(f'<line x1="{ix:.2f}" y1="{iy:.2f}" x2="{ox:.2f}" y2="{oy:.2f}" '
                  f'stroke="{"#6FB8FF" if major else "#3D7FB8"}" '
                  f'stroke-width="{6 if major else 3}" stroke-linecap="round"/>')
_ndeg = START + (END - START) * AMBER_IDX / (N - 1)
NX, NY = pt(_ndeg, 116)
AX1, AY1, AX2, AY2 = _amber

DEFS = """
<radialGradient id="tw-bg" cx="50%" cy="40%" r="75%">
  <stop offset="0%" stop-color="#1a2530"/><stop offset="55%" stop-color="#121922"/><stop offset="100%" stop-color="#0F1419"/>
</radialGradient>
<radialGradient id="tw-eyeFill" cx="50%" cy="50%" r="60%">
  <stop offset="0%" stop-color="#15243a"/><stop offset="100%" stop-color="#0e1620"/>
</radialGradient>
<radialGradient id="tw-irisFill" cx="50%" cy="46%" r="62%">
  <stop offset="0%" stop-color="#1a3050"/><stop offset="70%" stop-color="#102136"/><stop offset="100%" stop-color="#0b1622"/>
</radialGradient>
<radialGradient id="tw-irisGlow" cx="50%" cy="50%" r="50%">
  <stop offset="0%" stop-color="#4AA8FF" stop-opacity="0.45"/><stop offset="55%" stop-color="#2b6bb0" stop-opacity="0.15"/><stop offset="100%" stop-color="#4AA8FF" stop-opacity="0"/>
</radialGradient>
<linearGradient id="tw-road" x1="0" y1="1" x2="0" y2="0">
  <stop offset="0%" stop-color="#33506e"/><stop offset="100%" stop-color="#13202e"/>
</linearGradient>
<radialGradient id="tw-pupil" cx="42%" cy="38%" r="70%">
  <stop offset="0%" stop-color="#16202c"/><stop offset="100%" stop-color="#070a0e"/>
</radialGradient>
<filter id="tw-glow" x="-60%" y="-60%" width="220%" height="220%">
  <feGaussianBlur stdDeviation="7" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>
<filter id="tw-glowAmber" x="-120%" y="-120%" width="340%" height="340%">
  <feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>
<clipPath id="tw-irisClip"><circle cx="512" cy="512" r="153"/></clipPath>
"""


def body(square: bool) -> str:
    bg_rx = 0 if square else 232
    border = ("" if square else
              '<rect x="3" y="3" width="1018" height="1018" rx="229" fill="none" '
              'stroke="#4AA8FF" stroke-opacity="0.14" stroke-width="2.5"/>')
    return f"""
<rect x="0" y="0" width="1024" height="1024" rx="{bg_rx}" fill="url(#tw-bg)"/>
{border}
<circle cx="512" cy="506" r="300" fill="url(#tw-irisGlow)"/>
<path d="M 196 512 Q 512 326 828 512 Q 512 698 196 512 Z" fill="url(#tw-eyeFill)"/>
<path d="M 196 512 Q 512 326 828 512 Q 512 698 196 512 Z" fill="none" stroke="#4AA8FF" stroke-width="11" stroke-linejoin="round" filter="url(#tw-glow)"/>
<circle cx="512" cy="512" r="155" fill="url(#tw-irisFill)"/>
<circle cx="512" cy="512" r="155" fill="none" stroke="#4AA8FF" stroke-opacity="0.42" stroke-width="3"/>
<g clip-path="url(#tw-irisClip)">
  <path d="M 446 700 C 466 606 484 558 499 520 L 525 520 C 542 558 562 606 584 700 Z" fill="url(#tw-road)"/>
  <rect x="505" y="612" width="11" height="30" rx="3" fill="#cfe3f7" fill-opacity="0.6"/>
  <rect x="506" y="566" width="8" height="21" rx="3" fill="#cfe3f7" fill-opacity="0.5"/>
  <rect x="507" y="535" width="6" height="14" rx="3" fill="#cfe3f7" fill-opacity="0.42"/>
</g>
<g filter="url(#tw-glow)">
  {chr(10).join(_ticks)}
</g>
<line x1="{AX1:.2f}" y1="{AY1:.2f}" x2="{AX2:.2f}" y2="{AY2:.2f}" stroke="#FFB02E" stroke-width="9" stroke-linecap="round" filter="url(#tw-glowAmber)"/>
<line x1="512" y1="512" x2="{NX:.2f}" y2="{NY:.2f}" stroke="#FFB02E" stroke-width="7" stroke-linecap="round" filter="url(#tw-glowAmber)"/>
<circle cx="512" cy="512" r="35" fill="url(#tw-pupil)"/>
<circle cx="512" cy="512" r="35" fill="none" stroke="#4AA8FF" stroke-opacity="0.5" stroke-width="3"/>
<circle cx="500" cy="500" r="9" fill="#cfe3f7" fill-opacity="0.35"/>
"""


def svg(square: bool, px: int = 1024) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" '
            f'width="{px}" height="{px}"><defs>{DEFS}</defs>{body(square)}</svg>')


def render_base():
    """Render the full-bleed square icon at high resolution on a white page via
    headless Chrome, then trim the white margins back to the icon square -- robust to
    whatever display-DPI / window-size Chrome captures at. Returns a square RGB image."""
    from PIL import Image, ImageChops

    style = "html,body{margin:0;padding:0;background:#fff}svg{display:block}"
    html = (f'<!doctype html><html><head><meta charset="utf-8"><style>{style}</style>'
            f'</head><body>{svg(True, 1024)}</body></html>')
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        page = tdp / "page.html"
        page.write_text(html, encoding="utf-8")
        shot = tdp / "shot.png"
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--force-device-scale-factor=1", "--virtual-time-budget=4000",
             "--window-size=1200,1200", "--no-sandbox", f"--screenshot={shot}",
             page.as_uri()],
            check=True, capture_output=True, timeout=60)
        img = Image.open(shot).convert("RGB")
    # trim everything that matches the white page -> the dark icon square
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bbox = ImageChops.difference(img, bg).getbbox()
    icon = img.crop(bbox) if bbox else img
    # force exactly square (guard against a 1px AA difference between W/H)
    side = min(icon.size)
    return icon.crop((0, 0, side, side))


def main() -> int:
    from PIL import Image

    STATIC.mkdir(parents=True, exist_ok=True)
    rounded = svg(False)
    (STATIC / "icon.svg").write_text(rounded, encoding="utf-8")
    print(f"wrote icon.svg ({len(rounded)} bytes)")

    base = render_base()
    print(f"rendered base icon {base.size}")
    for size, name in [(180, "apple-touch-icon.png"), (192, "icon-192.png"),
                       (512, "icon-512.png"), (32, "favicon-32.png"),
                       (16, "favicon-16.png")]:
        base.resize((size, size), Image.LANCZOS).save(STATIC / name)
        print(f"  {name}  {size}x{size}")

    ico16 = Image.open(STATIC / "favicon-16.png")
    ico32 = Image.open(STATIC / "favicon-32.png")
    ico32.save(STATIC / "favicon.ico", sizes=[(16, 16), (32, 32)],
               append_images=[ico16])
    (STATIC / "favicon-16.png").unlink()
    print("wrote favicon.ico (16+32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
