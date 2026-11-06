#!/usr/bin/env bash
# Coarse episode annotation from the source video.
# Needs the sibling .mp4 and EGOWHALE_VLM_API_KEY plus EGOWHALE_VLM_MODEL (or a config override).
# Writes annotation/coarse.json. Mode: off, description, subtasks, or both.
# Usage: scripts/annotate.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${EGOWHALE_ANNOTATE_MODE:-both}"
python -m egowhale "$@" --steps annotate --annotate-mode "$mode"
