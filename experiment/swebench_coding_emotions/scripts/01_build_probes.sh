#!/usr/bin/env bash
# Phase 1 — build emotion probes on DeepSWE-Preview.
# Runs the upstream `emotion-probes` CLI; outputs land in ./output/probes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

CFG="config/deepswe.yaml"

emotion-probes generate-stories    --config "$CFG"
emotion-probes generate-neutral    --config "$CFG"
emotion-probes extract-activations --config "$CFG"
emotion-probes compute-vectors     --config "$CFG"
emotion-probes analyze             --config "$CFG"
emotion-probes visualize           --config "$CFG"

echo "Probes built. Artefacts under: $HERE/output/probes/"
