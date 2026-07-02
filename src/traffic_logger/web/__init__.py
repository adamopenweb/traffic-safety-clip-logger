"""The ``traffic-log serve`` web dashboard.

A small, decoupled FastAPI app that reads the stores the analyzer already writes
(``speed_log.sqlite`` for stats, ``data/events/`` for clips) and serves a
responsive phone/desktop UI behind a password. It never touches the live capture
or analysis pipeline -- restarting the web process is harmless to recording.

Optional dependency: install with ``pip install -e .[web]`` (or ``.[dev]``).
"""

from __future__ import annotations

__all__ = ["create_app", "WebSettings"]


def __getattr__(name: str):  # lazy so importing the package doesn't require FastAPI
    if name in ("create_app", "WebSettings"):
        from .app import WebSettings, create_app

        return {"create_app": create_app, "WebSettings": WebSettings}[name]
    raise AttributeError(name)
