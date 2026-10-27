#!/usr/bin/env bash
# 0.8B experiments (one GPU each, <=48GB is enough). Check `nvidia-smi` first if GPUs are shared.
# The 9B/27B models are launched with scripts/run1_large.sh.
#
#   scripts/run1_small.sh              # GPUs 0 1 2 3
#   GPUS="0 2" scripts/run1_small.sh   # only the first two experiments, on GPUs 0 and 2
#
# Follow progress:  tail -f runs/run1-*.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
[ -f init/clinical-0.8b/head.pt ] || python scripts/fetch_init.py

#          name                        budget init                       seed
EXPS=("run1-q35-0.8b-base          4096 -                          0"
      "run1-q35-0.8b-clinit        4096 init/clinical-0.8b  0"
      "run1-q35-0.8b-base-b2560    2560 -                          0"
      "run1-q35-0.8b-base-s1       4096 -                          1")

for i in "${!GPUS[@]}"; do
  [ "$i" -lt "${#EXPS[@]}" ] || break
  read -r NAME BUDGET INIT SEED <<< "${EXPS[$i]}"
  INIT_ARG=()
  [ "$INIT" != "-" ] && INIT_ARG=(--init_from "$INIT")
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" nohup python -m trialmatch.train \
    --base Qwen/Qwen3.5-0.8B-Base "${INIT_ARG[@]}" \
    --data "data/tm-v1-b$BUDGET/train.jsonl" \
    --val_data "data/tm-v1-b$BUDGET/development.jsonl" \
    --out "runs/$NAME" --seed "$SEED" \
    --epochs 3 --lr 1e-4 --batch 4 --accum 4 \
    --max_state 512 --max_branch "$BUDGET" \
    --dtype bf16 --checkpointing 1 \
    --log_every 20 --val_every 100 --val_records 100000 --save_every 100 \
    --wandb 0 > "runs/$NAME.log" 2>&1 &
  echo "started $NAME on GPU ${GPUS[$i]} (pid $!) -> runs/$NAME.log"
done
