"""The supervised `run` child is launched unbuffered so crashes leave a traceback."""

from __future__ import annotations

from traffic_logger.cli_handlers import _run_child_spec


def test_child_is_unbuffered_and_targets_run():
    argv, env = _run_child_spec("config/config.run.local.yaml")
    # -u right after the interpreter -> unbuffered stdout/stderr
    assert argv[1] == "-u"
    assert argv[2:] == ["-m", "traffic_logger.main", "run",
                        "--config", "config/config.run.local.yaml"]
    # env carries PYTHONUNBUFFERED so multiprocessing children inherit it too,
    # without dropping the rest of the environment.
    assert env["PYTHONUNBUFFERED"] == "1"
    assert len(env) > 1
