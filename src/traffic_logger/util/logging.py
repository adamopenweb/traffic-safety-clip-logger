"""Structured logging setup.

A single place to configure logging for the whole application. Supports a
human-readable formatter (default) and an optional JSON-line formatter for
machine ingestion (spec section "Logging").

``setup_logging`` is idempotent: calling it again reconfigures the root
handler rather than stacking duplicate handlers.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Optional

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Marker so we can find (and replace) our own handler on repeated calls.
_HANDLER_TAG = "traffic_logger_handler"


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Attach any structured extras the caller passed via `extra={...}`.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


# Standard LogRecord attributes we should not echo back as "extras".
_RESERVED_RECORD_KEYS = set(
    vars(logging.makeLogRecord({})).keys()
) | {"message", "asctime"}

# Third-party libraries that emit noisy DEBUG/INFO chatter we never want, even
# when our own level is DEBUG (matplotlib/PIL are pulled in by supervision).
_NOISY_LOGGERS = ("matplotlib", "PIL", "fontTools", "ultralytics")


def normalize_level(level: Optional[str]) -> int:
    """Resolve a textual level (case-insensitive) to a logging constant.

    Falls back to INFO for unknown/empty values.
    """
    if not level:
        return logging.INFO
    name = level.strip().upper()
    if name not in _VALID_LEVELS:
        return logging.INFO
    return getattr(logging, name)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure the root logger with a single stream handler.

    Idempotent — replaces any previously installed traffic_logger handler so
    repeated calls (e.g. CLI + library use) do not duplicate output.
    """
    root = logging.getLogger()
    root.setLevel(normalize_level(level))

    # Remove our previously installed handler(s) if present.
    for handler in list(root.handlers):
        if getattr(handler, "_tag", None) == _HANDLER_TAG:
            root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler._tag = _HANDLER_TAG  # type: ignore[attr-defined]
    if json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(handler)

    # Keep third-party chatter out of our logs regardless of our level.
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use module ``__name__`` at call sites."""
    return logging.getLogger(name)
