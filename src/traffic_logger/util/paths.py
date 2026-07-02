"""Filesystem path helpers.

Resolves project-relative directories from this module's location. Only
``ensure_dir`` touches the filesystem; everything else is pure path math so
importing this module has no side effects.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repository root: four parents up from this file.

    src/traffic_logger/util/paths.py -> util -> traffic_logger -> src -> root
    """
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return project_root() / "config"


def samples_dir() -> Path:
    return project_root() / "samples"


def data_dir() -> Path:
    return project_root() / "data"


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if missing; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
