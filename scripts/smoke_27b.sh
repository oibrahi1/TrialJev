#!/usr/bin/env bash
# Memory + speed check for a 27B backbone before committing a GPU for a day:
# 32 pairs, one epoch, logging every step. Read `peak ..GB` and `s/rec` from the
# output; the full run is ~9,300 pairs x 3 epochs at that s/rec.
#
#   scripts/smoke_27b.sh [GPU] [MODEL]      # default MODEL=Qwen/Qwen3.5-27B
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
MODEL="${2:-Qwen/Qwen3.5-27B}"
python -m trialmatch.train --base "$MODEL" --base_dtype bf16 \
  --data data/tm-v1-b4096/smoke/train.jsonl --val_data data/tm-v1-b4096/smoke/development.jsonl \
  --out "runs/smoke27b-$(date +%Y%m%d-%H%M%S)" --epochs 1 --batch 2 --accum 1 \
  --max_state 512 --max_branch 4096 --dtype bf16 --checkpointing 1 \
  --log_every 1 --val_every 16 --val_records 32 --wandb 0
