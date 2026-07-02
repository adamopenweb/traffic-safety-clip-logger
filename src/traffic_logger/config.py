"""Configuration loading.

Loads a YAML config file into a lightweight typed wrapper. The full parsed
document is always preserved on ``Config.raw`` so later milestones can reach
any section without this module needing to model every field up front.

Design notes:
- Only ``app`` is modeled as a typed dataclass for M0; other sections are
  exposed as plain dicts (with defaults) plus the full ``raw`` document.
- No filesystem paths are created here — values like ``/data/ring`` are Linux
  runtime paths stored as strings and acted on by the capture/event milestones.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

# Environment variable that can supply a default config path.
CONFIG_ENV_VAR = "TRAFFIC_CONFIG"

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(Exception):
    """Raised when a config file is missing or malformed."""


@dataclass(frozen=True)
class AppCfg:
    name: str = "traffic-safety-logger"
    timezone: str = "America/Toronto"
    log_level: str = "INFO"


@dataclass(frozen=True)
class Config:
    """Parsed configuration.

    ``raw`` holds the complete YAML document. Section accessors return the
    corresponding sub-dict (empty dict if absent) so callers can read keys
    defensively.
    """

    app: AppCfg
    raw: Dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    # -- Section accessors (return {} when a section is absent) -------------
    @property
    def camera(self) -> Dict[str, Any]:
        return self.raw.get("camera", {})

    @property
    def recording(self) -> Dict[str, Any]:
        return self.raw.get("recording", {})

    @property
    def network(self) -> Dict[str, Any]:
        return self.raw.get("network", {})

    @property
    def analysis(self) -> Dict[str, Any]:
        return self.raw.get("analysis", {})

    @property
    def models(self) -> Dict[str, Any]:
        return self.raw.get("models", {})

    @property
    def tracking(self) -> Dict[str, Any]:
        return self.raw.get("tracking", {})

    @property
    def calibration(self) -> Dict[str, Any]:
        return self.raw.get("calibration", {})

    @property
    def events(self) -> Dict[str, Any]:
        return self.raw.get("events", {})

    @property
    def audio(self) -> Dict[str, Any]:
        return self.raw.get("audio", {})

    @property
    def privacy(self) -> Dict[str, Any]:
        return self.raw.get("privacy", {})

    @property
    def web(self) -> Dict[str, Any]:
        """Settings for the `serve` web dashboard (host/port/auth). Absent in the
        capture/analyze configs; lives only in the gitignored run config."""
        return self.raw.get("web", {})

    @property
    def debug(self) -> Dict[str, Any]:
        return self.raw.get("debug", {})

    @property
    def aggressiveness(self) -> float:
        """Convenience accessor for ``events.aggressiveness`` (default 0.3)."""
        return float(self.events.get("aggressiveness", 0.3))


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a config file.

    Resolution order when ``path`` is None:
    1. ``$TRAFFIC_CONFIG`` if set
    2. ``config/config.mini_pc.yaml`` (the deployment default)

    Raises ``ConfigError`` if the file is missing or not a YAML mapping.
    """
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise ConfigError(f"Config file not found: {resolved}")

    try:
        with resolved.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config {resolved}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config root must be a mapping, got {type(raw).__name__}: {resolved}"
        )

    app = _parse_app(raw.get("app", {}), resolved)
    return Config(app=app, raw=raw, source_path=resolved)


def _resolve_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        return Path(env)
    from .util.paths import config_dir

    return config_dir() / "config.mini_pc.yaml"


def _parse_app(section: Dict[str, Any], resolved: Path) -> AppCfg:
    if not isinstance(section, dict):
        raise ConfigError(f"'app' section must be a mapping in {resolved}")
    log_level = str(section.get("log_level", "INFO")).upper()
    if log_level not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"Invalid app.log_level '{log_level}' in {resolved}; "
            f"expected one of {sorted(_VALID_LOG_LEVELS)}"
        )
    return AppCfg(
        name=str(section.get("name", "traffic-safety-logger")),
        timezone=str(section.get("timezone", "America/Toronto")),
        log_level=log_level,
    )
