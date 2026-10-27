#!/usr/bin/env bash
# One "lane" = one GPU working through a list of full-set conditions, one after another.
#   export PY=... HF_HOME=... HF_HUB_OFFLINE=1
#   bash scripts/run_full_set_lane.sh GPU CASE [CASE ...]
# Stops at the first failure. See run_full_set_generation.sh for the CASE names.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:?usage: run_full_set_lane.sh GPU CASE [CASE ...]}"
shift
[ "$#" -ge 1 ] || { echo "ERROR: give at least one CASE" >&2; exit 1; }
for c in "$@"; do
  bash scripts/run_full_set_generation.sh "$c" "$GPU"
done
echo "LANE COMPLETED on GPU $GPU: $*"
