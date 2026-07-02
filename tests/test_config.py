"""Tests for config loading across the three shipped config files."""

from __future__ import annotations

from pathlib import Path

import pytest

from traffic_logger.config import Config, ConfigError, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
ALL_CONFIGS = ["config.example.yaml", "config.dev.yaml", "config.mini_pc.yaml"]


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_all_configs_parse(name: str):
    cfg = load_config(CONFIG_DIR / name)
    assert isinstance(cfg, Config)
    assert cfg.app.name == "traffic-safety-logger"
    assert cfg.app.timezone == "America/Toronto"
    # Every config defines the core sections the pipeline relies on.
    for section in ("camera", "recording", "analysis", "events", "calibration"):
        assert section in cfg.raw


def test_example_defaults(loaded_config: Config):
    assert loaded_config.app.log_level == "INFO"
    assert loaded_config.aggressiveness == 0.3
    assert loaded_config.events["clip_total_seconds"] == 30


def test_dev_vs_mini_pc_differences():
    dev = load_config(CONFIG_DIR / "config.dev.yaml")
    mini = load_config(CONFIG_DIR / "config.mini_pc.yaml")

    # Dev: GPU, no recording. Mini-PC: CPU, recording on.
    assert dev.analysis["device"] == "cuda"
    assert dev.recording["enabled"] is False
    assert mini.analysis["device"] == "cpu"
    assert mini.recording["enabled"] is True
    assert mini.recording["ring_max_gb"] == 200


def test_missing_config_raises():
    with pytest.raises(ConfigError):
        load_config(CONFIG_DIR / "does_not_exist.yaml")


def test_invalid_log_level_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("app:\n  log_level: LOUD\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
