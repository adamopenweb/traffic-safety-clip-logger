"""``traffic-log`` command-line entry point.

Parses arguments, loads config, configures logging, and dispatches to the
handler for the chosen subcommand. ``--help`` and ``--version`` short-circuit
before any config is loaded so they work in a bare checkout.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, List, Optional

from . import __version__
from . import cli_handlers
from .config import Config, ConfigError, load_config
from .util.logging import get_logger, setup_logging

log = get_logger(__name__)

DEFAULT_CONFIG = "config/config.mini_pc.yaml"

# command name -> handler
Handler = Callable[[argparse.Namespace, Config], int]
_HANDLERS: Dict[str, Handler] = {
    "probe-camera": cli_handlers.handle_probe_camera,
    "capture": cli_handlers.handle_capture,
    "analyze": cli_handlers.handle_analyze,
    "run": cli_handlers.handle_run,
    "calibrate": cli_handlers.handle_calibrate,
    "test": cli_handlers.handle_test,
    "export-event": cli_handlers.handle_export_event,
    "prune-ring": cli_handlers.handle_prune_ring,
    "health": cli_handlers.handle_health,
    "police-report": cli_handlers.handle_police_report,
    "speed-report": cli_handlers.handle_speed_report,
    "supervise": cli_handlers.handle_supervise,
    "serve": cli_handlers.handle_serve,
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="traffic-log",
        description="Traffic Safety Clip Logger - capture, analyze, and export "
        "traffic safety event clips.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Shared options every subcommand accepts.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to the YAML config file (default: {DEFAULT_CONFIG}).",
    )
    common.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the log level from config.",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    sub.add_parser("probe-camera", parents=[common],
                   help="List /dev/video* devices, formats, resolutions, fps.")
    sub.add_parser("capture", parents=[common],
                   help="Capture camera stream to H.264 ring-buffer segments.")
    p_analyze = sub.add_parser("analyze", parents=[common],
                               help="Run analysis on a live RTSP/camera stream (or a file).")
    p_analyze.add_argument("--source", default=None,
                           help="RTSP URL / device / video file. Defaults to analysis.source.")
    p_analyze.add_argument("--max-seconds", type=float, default=None,
                           help="Stop after N seconds (for testing).")
    p_run = sub.add_parser("run", parents=[common],
                           help="Record + analyze together (single-box); events cut ring clips.")
    p_run.add_argument("--max-seconds", type=float, default=None,
                       help="Stop after N seconds (for testing).")
    sub.add_parser("supervise", parents=[common],
                   help="Run only during daylight (civil twilight by lat/long), auto-restart.")
    p_cal = sub.add_parser("calibrate", parents=[common],
                           help="Interactive 4-point road-surface calibration.")
    p_cal.add_argument("--source", required=True,
                       help="Image or video to grab a calibration frame from.")
    p_cal.add_argument("--points", default=None,
                       help="Skip clicking: 'x1,y1 x2,y2 x3,y3 x4,y4' road corners.")
    p_cal.add_argument("--output", default=None,
                       help="Preview image output path (default: data/calibration/).")
    p_cal.add_argument("--write", action="store_true",
                       help="Write source_points back into the --config file.")

    p_test = sub.add_parser("test", parents=[common],
                            help="Run the offline analyzer against a video file.")
    p_test.add_argument("--source", required=True,
                        help="Path to the input video file to analyze.")

    p_export = sub.add_parser("export-event", parents=[common],
                              help="Export an event clip for a timestamp window.")
    p_export.add_argument("--start-ts", required=True, type=float,
                          help="Clip start as a Unix timestamp.")
    p_export.add_argument("--end-ts", required=True, type=float,
                          help="Clip end as a Unix timestamp.")

    sub.add_parser("prune-ring", parents=[common],
                   help="Prune the ring buffer down to its size cap.")

    sub.add_parser("health", parents=[common],
                   help="Exit 0 if recording is healthy (fresh segments), else 1.")

    p_police = sub.add_parser("police-report", parents=[common],
                              help="Report police-vehicle sightings over a time window.")
    p_police.add_argument("--hours", type=float, default=24.0,
                          help="Look back this many hours (default: 24).")
    p_police.add_argument("--speeding-only", action="store_true",
                          help="List only sightings that were also flagged speeding.")

    p_speed = sub.add_parser("speed-report", parents=[common],
                             help="Summarize absolute-speed-gate violations (safety case).")
    p_speed.add_argument("--days", type=float, default=7.0,
                         help="Look back this many days (default: 7).")
    p_speed.add_argument("--limit", type=float, default=50.0,
                         help="Posted speed limit in km/h, for the over-limit figure (default: 50).")
    p_speed.add_argument("--top", type=int, default=10,
                         help="How many fastest violations to list (default: 10).")
    p_speed.add_argument("--csv", default=None,
                         help="Also write the violations to this CSV path (evidence export).")

    p_serve = sub.add_parser("serve", parents=[common],
                             help="Run the web dashboard (FastAPI/uvicorn) behind a secret-link gate.")
    p_serve.add_argument("--new-link", action="store_true",
                         help="Rotate the access token, write it to config, print the unlock link, and exit.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Program entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load config before logging so the config's log_level can be the default.
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        # Logging isn't configured yet; emit a minimal setup then report.
        setup_logging("ERROR")
        log.error("%s", exc)
        return 2

    level = args.log_level or config.app.log_level
    setup_logging(level)
    log.debug("Loaded config from %s", config.source_path)

    handler = _HANDLERS.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces choices
        parser.error(f"Unknown command: {args.command}")
        return 2

    try:
        return handler(args, config)
    except NotImplementedError as exc:
        log.error("Not implemented: %s", exc)
        return 3
    except Exception:  # noqa: BLE001 - top-level guard logs and exits non-zero
        log.exception("Unhandled error while running '%s'", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
