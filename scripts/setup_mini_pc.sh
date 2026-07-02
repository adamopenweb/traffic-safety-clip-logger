#!/usr/bin/env bash
# Prepare the Ubuntu mini-PC appliance (Docker-based deployment).
# Milestone 0 skeleton.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "[setup_mini_pc] Ensuring data directories exist..."
mkdir -p data/ring data/events data/index

echo "[setup_mini_pc] Building the slim capture image (no torch)..."
docker compose build capture

echo "[setup_mini_pc] Done. Recommended bring-up order (see DEPLOY.md):"
echo "  1) traffic-log probe-camera --config config/config.mini_pc.yaml"
echo "  2) docker compose up -d capture        # unattended recording"
echo "  3) docker compose ps                   # check 'healthy'"
echo "  4) docker compose logs -f capture"
echo
echo "Analysis (YOLO) is heavy; run it on the RTX 4080 box, not here:"
echo "  docker compose build analyze   # only if you really want CPU inference on the mini-PC"
