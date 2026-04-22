#!/usr/bin/env bash
# Phase 3 — replay trajectories through the model and capture probe scores.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

swebench-emotions replay --config config/deepswe.yaml "$@"
