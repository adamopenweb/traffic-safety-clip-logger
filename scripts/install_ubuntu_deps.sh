#!/usr/bin/env bash
# Install OS-level dependencies on an Ubuntu host (mini-PC or WSL2 dev).
# Milestone 0 skeleton — review before running on a real machine.
set -euo pipefail

echo "[install_ubuntu_deps] Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg v4l-utils \
    libgl1 libglib2.0-0

echo "[install_ubuntu_deps] Done. Next: scripts/setup_wsl_dev.sh or setup_mini_pc.sh"
