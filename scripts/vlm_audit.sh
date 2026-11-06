#!/usr/bin/env bash
# L3: audit the composite against the description, then write the final decision.
# Needs action/quality.json and visual/composite.mp4; annotation/coarse.json is optional.
# Writes action/vlm_audit.json and action/final.json.
# Usage: scripts/vlm_audit.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale "$@" --steps vlm_audit
