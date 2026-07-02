# Third-Party Licenses

Project code is MIT-licensed (see [LICENSE](LICENSE)). It depends on
third-party packages that keep their own licenses. None of them are vendored
into this repository; they are installed from PyPI via the extras in
`pyproject.toml`. Licenses below are as observed at the time of writing;
verify upstream before redistribution or commercial use.

## The one that needs your attention: Ultralytics (AGPL-3.0)

The optional `analyze`/`gpu` extras depend on **ultralytics** (YOLOv8), which
is licensed **AGPL-3.0**. AGPL is a strong copyleft license whose obligations
extend to network-accessible use of combined works. For this project's
personal, non-distributed deployment that is unproblematic, and the spec
flagged it from day one (`traffic.md`, "Important licensing note"). If you
build a distributed or commercial product on this code, either obtain an
Ultralytics commercial license or swap the detector; the `Detector` interface
(`src/traffic_logger/analyze/detector.py`) is abstract for exactly that
reason. The pretrained YOLOv8 weight files are also AGPL-licensed and are not
included in this repository.

## Dependency licenses

| Package | Extra | License |
| --- | --- | --- |
| PyYAML | core | MIT |
| tzdata (PyPI package) | core | Apache-2.0 (IANA tz data is public domain) |
| numpy | analyze/gpu | BSD-3-Clause |
| opencv-python | analyze/gpu | MIT wrapper; OpenCV itself Apache-2.0 |
| supervision (Roboflow) | analyze/gpu | MIT |
| ultralytics | analyze/gpu | **AGPL-3.0** (see above) |
| torch | installed alongside the CV stack | BSD-style |
| fastapi | web/dev | MIT |
| uvicorn | web | BSD-3-Clause |
| itsdangerous | web/dev | BSD-3-Clause |
| httpx | dev | BSD-3-Clause |
| pytest | dev | MIT |
| sounddevice | audio (stubbed milestone) | MIT |
| open_clip_torch, Pillow | optional police-recognition experiment (disabled) | permissive (MIT-style / HPND); verify if enabling |

ffmpeg is invoked as an external system binary and is not distributed with
this project; its license depends on your build (LGPL/GPL components).
