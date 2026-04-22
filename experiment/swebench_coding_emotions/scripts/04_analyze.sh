#!/usr/bin/env bash
# Phase 4 — aggregate + run statistical tests.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

swebench-emotions analyze --config config/deepswe.yaml "$@"
