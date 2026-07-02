#!/usr/bin/env bash
# Quick camera format dump using v4l2-ctl (mini-PC). Milestone 0 skeleton;
# the `traffic-log probe-camera` command supersedes this in Milestone 1.
set -euo pipefail

DEVICE="${1:-/dev/video0}"

if ! command -v v4l2-ctl >/dev/null 2>&1; then
    echo "v4l2-ctl not found. Install with: sudo apt-get install -y v4l-utils" >&2
    exit 1
fi

echo "[test_camera_formats] Devices:"
v4l2-ctl --list-devices || true

echo "[test_camera_formats] Formats for ${DEVICE}:"
v4l2-ctl -d "${DEVICE}" --list-formats-ext
