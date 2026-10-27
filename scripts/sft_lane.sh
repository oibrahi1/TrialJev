#!/usr/bin/env bash
# One generative-fine-tuning lane: train the model to WRITE the answer, pick the lowest-development-loss checkpoint,
# and evaluate it once on the full test set (greedy decoding, strict word parsing, time per pair recorded).
#
#   export PY=/path/to/venv/bin/python   (default: python)   HF_HOME=/path/to/hf_cache   (cluster, offline)
#   bash scripts/sft_lane.sh KEY GPU SEED          KEY: q35-0.8b | q35-9b | q35-27b | q38-27b
#   SMOKE=1 bash scripts/sft_lane.sh q35-9b 0 0     # tiny run to check the whole path in minutes
#
# Writes runs/sft-KEY-sSEED (training run) and runs/gen_sft/sft-KEY-sSEED (full-test evaluation).
# Same recipe as the pointer-head and A/B-logit runs of that size; only the trained readout differs.
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo "ERROR: $*" >&2; exit 1; }

KEY="${1:?usage: sft_lane.sh KEY GPU SEED}"
GPU="${2:?usage: sft_lane.sh KEY GPU SEED}"
SEED="${3:?usage: sft_lane.sh KEY GPU SEED}"
PY="${PY:-python}"
RUNS="${RUNS:-runs}"
DATA="${DATA:-data/tm-v1-b4096}"

REV=""
case "$KEY" in
  q35-0.8b) BASE=Qwen/Qwen3.5-0.8B-Base; BASE_DTYPE=fp32; BATCH=4; ACCUM=4; VAL=100; SAVE=100 ;;
  q35-9b)   BASE=Qwen/Qwen3.5-9B-Base;   BASE_DTYPE=bf16; BATCH=2; ACCUM=8; VAL=200; SAVE=200; REV=68c46c4b3498877f3ef123c856ecfde50c39f404 ;;
  q35-27b)  BASE=Qwen/Qwen3.5-27B;       BASE_DTYPE=bf16; BATCH=2; ACCUM=8; VAL=200; SAVE=200; REV=fc05daec18b0a78c049392ed2e771dde82bdf654 ;;
  q38-27b)  BASE=Qwen/Qwen3.8-27B;       BASE_DTYPE=bf16; BATCH=2; ACCUM=8; VAL=200; SAVE=200; REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 ;;
  *) fail "unknown KEY '$KEY' (q35-0.8b | q35-9b | q35-27b | q38-27b)" ;;
esac
[[ "$GPU" =~ ^[0-9]+$ ]] || fail "GPU must be a number, got '$GPU'"
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "SEED must be a number, got '$SEED'"
[ -f scripts/train_lora_generate.py ] && [ -f scripts/generate_baselines.py ] || fail "run from the repository root with the new scripts uploaded"
[ -f "$DATA/train.jsonl" ] && [ -f "$DATA/development.jsonl" ] && [ -f "$DATA/test.jsonl" ] || fail "missing data in $DATA"

NAME="sft-$KEY-s$SEED"
TRAIN_ARGS=(); EVAL_ARGS=()
if [ "${SMOKE:-0}" = 1 ]; then
  NAME="smoke-$NAME"
  TRAIN_ARGS=(--max_records 32 --val_records 16 --epochs 1 --batch 2 --accum 2 --val_every 4 --save_every 4 --log_every 1)
  EVAL_ARGS=(--limit 5)
else
  TRAIN_ARGS=(--epochs 3 --batch "$BATCH" --accum "$ACCUM" --val_every "$VAL" --save_every "$SAVE" --log_every 20)
fi
OUT_TRAIN="$RUNS/$NAME"; OUT_EVAL="$RUNS/gen_sft/$NAME"
[ ! -e "$OUT_TRAIN" ] || fail "$OUT_TRAIN already exists; refusing to overwrite"
[ ! -e "$OUT_EVAL" ] || fail "$OUT_EVAL already exists; refusing to overwrite"
mkdir -p "$RUNS" "$RUNS/gen_sft"

read -r MEM UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
if [ "${FORCE_GPU:-0}" != 1 ] && { [ "$MEM" -ge 2000 ] || [ "$UTIL" -ge 10 ]; }; then
  fail "GPU $GPU is not idle (${MEM} MiB used, ${UTIL}% utilisation); pick another GPU or set FORCE_GPU=1"
fi
export CUDA_VISIBLE_DEVICES="$GPU"
REV_ARG=(); [ -n "$REV" ] && REV_ARG=(--base_revision "$REV")

echo "$(date +%H:%M:%S) train $NAME on GPU $GPU ($BASE)"
"$PY" scripts/train_lora_generate.py train --base "$BASE" ${REV_ARG[@]+"${REV_ARG[@]}"} --base_dtype "$BASE_DTYPE" \
  --data "$DATA/train.jsonl" --val_data "$DATA/development.jsonl" --out "$OUT_TRAIN" --seed "$SEED" \
  --lr 1e-4 --dtype bf16 --checkpointing 1 --max_state 512 --max_branch 4096 ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} \
  > "$OUT_TRAIN.log" 2>&1 || fail "training failed, see $OUT_TRAIN.log"

BEST="$("$PY" scripts/pick_best.py "$OUT_TRAIN" --path)" || fail "pick_best failed for $OUT_TRAIN"
[ -d "$BEST" ] || fail "best checkpoint '$BEST' is not a directory"
echo "$(date +%H:%M:%S) best checkpoint: $BEST"

echo "$(date +%H:%M:%S) evaluate $NAME on the full test set"
"$PY" scripts/generate_baselines.py run --condition sft-generate --base "$BASE" ${REV_ARG[@]+"${REV_ARG[@]}"} --adapter "$BEST" \
  --data "$DATA/test.jsonl" --per_patient 0 --dtype "$BASE_DTYPE" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} --out "$OUT_EVAL" \
  > "$OUT_EVAL.log" 2>&1 || fail "evaluation failed, see $OUT_EVAL.log"
echo "$BEST" > "$OUT_EVAL/selected_checkpoint.txt"
echo "$(date +%H:%M:%S) LANE COMPLETED $NAME"
