"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from traffic_logger.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture
def example_config_path() -> Path:
    return CONFIG_DIR / "config.example.yaml"


@pytest.fixture
def loaded_config():
    return load_config(CONFIG_DIR / "config.example.yaml")
