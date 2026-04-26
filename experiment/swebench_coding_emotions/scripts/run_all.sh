#!/usr/bin/env bash
# Run the full experiment pipeline end-to-end.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANT="${VARIANT:-augmented}"

bash "$HERE/01_build_probes.sh"
bash "$HERE/01b_augment_neutrals.sh"
bash "$HERE/02_fetch_trajectories.sh"
bash "$HERE/03_replay.sh" --variant "$VARIANT"
bash "$HERE/04_analyze.sh" --variant "$VARIANT"
bash "$HERE/05_visualize.sh" --variant "$VARIANT"
