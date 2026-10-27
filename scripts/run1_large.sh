#!/usr/bin/env bash
# Run 1 for the larger backbones (bf16 base weights; one 80-95GB GPU each).
# These are the exact hyper-parameters of the reported 9B / 27B runs.
#
#   scripts/run1_large.sh NAME BASE_MODEL [GPU] [BASE_REVISION]
#   scripts/run1_large.sh run1-q35-9b-base  Qwen/Qwen3.5-9B-Base 0
#   scripts/run1_large.sh run1-q35-27b-base Qwen/Qwen3.5-27B     1
#   scripts/run1_large.sh run1-q38-27b-base Qwen/Qwen3.8-27B     0 <commit-sha>
#
# Pin BASE_REVISION to a commit SHA for anything you intend to report. Offline nodes:
# export HF_HOME=<cache> HF_HUB_OFFLINE=1 beforehand.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?usage: run1_large.sh NAME BASE_MODEL [GPU] [BASE_REVISION]}"
BASE="${2:?usage: run1_large.sh NAME BASE_MODEL [GPU] [BASE_REVISION]}"
GPU="${3:-0}"
REV_ARG=(); [ -n "${4:-}" ] && REV_ARG=(--base_revision "$4")
mkdir -p runs
[ ! -e "runs/$NAME" ] || { echo "runs/$NAME already exists" >&2; exit 1; }
CUDA_VISIBLE_DEVICES="$GPU" nohup python -m trialmatch.train \
  --base "$BASE" "${REV_ARG[@]}" --base_dtype bf16 \
  --data data/tm-v1-b4096/train.jsonl \
  --val_data data/tm-v1-b4096/development.jsonl \
  --out "runs/$NAME" --seed 0 \
  --epochs 3 --lr 1e-4 --batch 2 --accum 8 \
  --max_state 512 --max_branch 4096 \
  --dtype bf16 --checkpointing 1 \
  --log_every 20 --val_every 200 --save_every 200 --val_records 100000 \
  --wandb 0 > "runs/$NAME.log" 2>&1 &
echo "started $NAME on GPU $GPU (pid $!) -> runs/$NAME.log"
