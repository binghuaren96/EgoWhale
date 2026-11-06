#!/usr/bin/env bash
# Filter the episode with deterministic L1/L2 checks only: frame checks and action statistics.
# Needs gripper, base, IK, approach, and visual/composite.mp4.
# Writes action/quality.json. L3 runs in scripts/vlm_audit.sh.
# Usage: scripts/curate.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale "$@" --steps curate
