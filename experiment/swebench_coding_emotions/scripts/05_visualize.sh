#!/usr/bin/env bash
# Phase 5 — generate plots.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

swebench-emotions viz --config config/deepswe.yaml "$@"
