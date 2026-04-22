#!/usr/bin/env bash
# Phase 2 — fetch DeepSWE trajectories + stratified sample.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

N_TASKS="${N_TASKS:-24}"
swebench-emotions fetch --config config/deepswe.yaml --n-tasks "$N_TASKS" "$@"
