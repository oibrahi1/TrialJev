#!/usr/bin/env bash
# End-to-end smoke test on one GPU, a few minutes:
#   1. overfit 32 balanced train pairs     -> val accuracy should reach 1.000
#   2. reload the saved checkpoint + score  -> evaluate.py micro_accuracy should be 1.0
#   3. --init_from the released clinical-note checkpoint (init/clinical-0.8b) loads without a mismatch
#
#   scripts/smoke_test.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
DATA=data/tm-v1-b4096
STAMP=$(date +%Y%m%d-%H%M%S)

echo "== 1. overfit 32 pairs =="
python -m trialmatch.train --base Qwen/Qwen3.5-0.8B-Base \
  --data $DATA/smoke/train.jsonl --val_data $DATA/smoke/development.jsonl \
  --out runs/smoke-$STAMP --epochs 15 --batch 4 --accum 1 --lr 3e-4 \
  --max_state 512 --max_branch 4096 --dtype bf16 --checkpointing 1 \
  --log_every 8 --val_every 40 --val_records 32 --wandb 0

echo "== 2. reload + score =="
python -m trialmatch.evaluate --run runs/smoke-$STAMP --data $DATA/smoke --split development

echo "== 3. init from clinical-note checkpoint (init/clinical-0.8b) =="
[ -f init/clinical-0.8b/head.pt ] || python scripts/fetch_init.py
python -m trialmatch.train --base Qwen/Qwen3.5-0.8B-Base --init_from init/clinical-0.8b \
  --data $DATA/smoke/train.jsonl --val_data $DATA/smoke/development.jsonl \
  --out runs/smoke-init-$STAMP --epochs 1 --batch 4 --accum 1 --max_records 8 \
  --max_state 512 --max_branch 4096 --dtype bf16 --checkpointing 1 \
  --log_every 1 --val_every 0 --wandb 0

echo "SMOKE TEST DONE -- check: step 1 last 'val step' acc 1.0000, step 2 micro_accuracy 1.0, step 3 'initialised LoRA adapter'"
