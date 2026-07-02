"""Smoke tests for the traffic-log CLI."""

from __future__ import annotations

import pytest

from traffic_logger.main import build_parser, main


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_all_subcommands_registered():
    parser = build_parser()
    # Find the subparsers action and assert every required command exists.
    choices = {}
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices:
            choices = action.choices
            break
    for cmd in [
        "probe-camera", "capture", "analyze", "run",
        "calibrate", "test", "export-event", "prune-ring",
    ]:
        assert cmd in choices


def test_test_command_runs_stub(tmp_path):
    # Missing source -> dry stub, still exits 0 (acceptance-friendly).
    rc = main([
        "test",
        "--source", str(tmp_path / "nope.mp4"),
        "--config", "config/config.dev.yaml",
    ])
    assert rc == 0


def test_stub_command_exits_zero(tmp_path):
    # A config whose index path is writable everywhere: the mini_pc default
    # (/data/...) needs root on Linux, so point the index at tmp_path instead.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "recording:\n"
        f"  segment_index_path: {(tmp_path / 'segments.sqlite').as_posix()}\n",
        encoding="utf-8",
    )
    rc = main(["prune-ring", "--config", str(cfg)])
    assert rc == 0


def test_missing_config_returns_error_code():
    rc = main(["prune-ring", "--config", "config/nope.yaml"])
    assert rc == 2
