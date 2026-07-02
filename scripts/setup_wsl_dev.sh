#!/usr/bin/env bash
# Set up the WSL2 / RTX 4080 development environment.
# Milestone 0 skeleton.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "[setup_wsl_dev] Creating virtualenv (.venv)..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup_wsl_dev] Installing package with dev + analyze extras..."
pip install --upgrade pip
pip install -e ".[dev,analyze]"

echo "[setup_wsl_dev] Verifying CLI..."
traffic-log --help

echo "[setup_wsl_dev] Done. Try: traffic-log test --source samples/street-test.mp4 --config config/config.dev.yaml"
