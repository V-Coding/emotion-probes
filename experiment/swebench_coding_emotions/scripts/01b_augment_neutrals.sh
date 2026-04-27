#!/usr/bin/env bash
# Phase 1b — build experiment-local augmented neutral set + refit PCA
# denoising. Produces emotion_vectors_augmented_layer_{L}.safetensors which
# can be loaded by `swebench-emotions replay --variant augmented`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

swebench-emotions augment --config config/deepswe.yaml "$@"
