#!/usr/bin/env bash
# Generation baselines + pointer-head timing for one model size on one GPU (batch size 1).
#
#   export PY=/path/to/venv/bin/python HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1
#   bash scripts/run_generation_baselines.sh sample          # once: the shared patient-balanced pair list
#   bash scripts/run_generation_baselines.sh 9b 2            # 9B conditions on GPU 2
#   bash scripts/run_generation_baselines.sh 27b 3           # 27B conditions on GPU 3
#
# Smoke test first:  OUT=runs/gen_smoke LIMIT=5 bash scripts/run_generation_baselines.sh sample
#                    OUT=runs/gen_smoke LIMIT=5 bash scripts/run_generation_baselines.sh 9b 2
#
# Every condition runs on the SAME pairs. The script stops at the first problem (nonzero exit, no "completed"
# message), refuses to overwrite existing output, and checks before it starts: the exact training script, the
# adapter and pointer checkpoints, pinned model revisions, and that the GPU is idle.
#
# Notes: "zero-shot chat" uses Qwen3.5-9B (instruct), the LoRA uses Qwen3.5-9B-Base: different checkpoints.
# The LoRA adapters were trained with a loss restricted to the A/B letter logits (not full-vocabulary SFT).
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo "ERROR: $*" >&2; exit 1; }

MODE="${1:?usage: run_generation_baselines.sh sample | 9b GPU | 27b GPU}"
PY="${PY:?set PY to the python of the training venv}"
HF_HOME="${HF_HOME:?set HF_HOME (used to check pinned model revisions)}"
OUT="${OUT:-runs/gen}"
LIMIT="${LIMIT:-0}"
DATA=data/tm-v1-b4096/test.jsonl
IDS="$OUT/sample_ids.json"
EXPECTED_TLL=ae8ee0c73e8443940e9209512d097bcd5104454202b3d0cbee6a36fed30f2801
REV_9B_BASE=68c46c4b3498877f3ef123c856ecfde50c39f404
REV_27B=fc05daec18b0a78c049392ed2e771dde82bdf654

[ -x "$PY" ] || fail "PY=$PY is not an executable python"
[ -f "$DATA" ] || fail "missing $DATA"
[ "$(sha256sum scripts/train_lora_letter.py | cut -d' ' -f1)" = "$EXPECTED_TLL" ] \
  || fail "scripts/train_lora_letter.py differs from the version the adapters were trained with"
"$PY" -c "import torch, transformers, peft" || fail "torch/transformers/peft are not importable with PY=$PY"
mkdir -p "$OUT"

if [ "$MODE" = sample ]; then
  [ ! -e "$IDS" ] || fail "$IDS already exists; refusing to overwrite the shared sample"
  "$PY" scripts/generate_baselines.py sample --data "$DATA" --per_patient 20 --out "$IDS"
  exit 0
fi

SIZE="$MODE"
GPU="${2:?usage: run_generation_baselines.sh 9b|27b GPU}"
case "$SIZE" in 9b|27b) ;; *) fail "size must be exactly 9b or 27b, got '$SIZE'" ;; esac
[[ "$GPU" =~ ^[0-9]+$ ]] || fail "GPU must be a number, got '$GPU'"
[ -f "$IDS" ] || fail "$IDS missing: run 'sample' first (same OUT)"

# the GPU must be idle (memory < 2 GB, utilisation < 10%), otherwise timings are not trustworthy
read -r MEM UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
if [ "${FORCE_GPU:-0}" != 1 ] && { [ "$MEM" -ge 2000 ] || [ "$UTIL" -ge 10 ]; }; then
  fail "GPU $GPU is not idle (${MEM} MiB used, ${UTIL}% utilisation). Pick another GPU or set FORCE_GPU=1"
fi
export CUDA_VISIBLE_DEVICES="$GPU"

resolve_rev() {   # MODEL [EXPECTED]: the cached snapshot hash; with EXPECTED it must be present
  local dir="$HF_HOME/hub/models--${1//\//--}/snapshots" revs
  [ -d "$dir" ] || fail "no cached snapshots for $1 in $dir"
  revs=$(ls "$dir")
  if [ -n "${2:-}" ]; then
    echo "$revs" | grep -qx "$2" || fail "$1: pinned revision $2 not in the cache (found: $revs)"
    echo "$2"
  else
    [ "$(echo "$revs" | wc -l)" -eq 1 ] || fail "$1: several cached revisions, pin one: $revs"
    echo "$revs"
  fi
}

if [ "$SIZE" = 9b ]; then
  TAG=9b; ADAPTER=runs/ctl-q35-9b/step-400; POINTER=runs/run1-q35-9b-base/step-400
  LORA_BASE=Qwen/Qwen3.5-9B-Base; LORA_REV=$(resolve_rev Qwen/Qwen3.5-9B-Base "$REV_9B_BASE")
  ZS_BASE=Qwen/Qwen3.5-9B;        ZS_REV=$(resolve_rev Qwen/Qwen3.5-9B)
  PLAIN_BASE=Qwen/Qwen3.5-9B-Base; PLAIN_REV=$LORA_REV
else
  TAG=27b; ADAPTER=runs/ctl-q35-27b/step-400; POINTER=runs/run1-q35-27b-base/step-400
  LORA_BASE=Qwen/Qwen3.5-27B; LORA_REV=$(resolve_rev Qwen/Qwen3.5-27B "$REV_27B")
  ZS_BASE=$LORA_BASE;         ZS_REV=$LORA_REV
  PLAIN_BASE=; PLAIN_REV=
fi
for f in adapter_config.json adapter_model.safetensors control_meta.json; do
  [ -f "$ADAPTER/$f" ] || fail "missing $ADAPTER/$f"
done
for f in head.pt adapter_config.json adapter_model.safetensors; do
  [ -f "$POINTER/$f" ] || fail "missing $POINTER/$f"
done
for name in zs-chat-$TAG zs-chat-swapped-$TAG zs-plain-$TAG lora-logits-$TAG lora-generate-$TAG \
            lora-generate-swapped-$TAG pointer-timing-$TAG; do
  [ ! -e "$OUT/$name" ] || fail "$OUT/$name already exists; refusing to overwrite"
done

echo "preflight ok: size=$SIZE gpu=$GPU out=$OUT limit=$LIMIT lora_rev=$LORA_REV zs_rev=$ZS_REV"

run() {
  local name="$1"; shift
  nvidia-smi -i "$GPU" --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader \
    | sed "s/^/$name  /" >> "$OUT/gpu-snapshots-$TAG.txt"
  echo "=== $name  $(date +%H:%M:%S)"
  "$PY" scripts/generate_baselines.py run --data "$DATA" --out "$OUT/$name" --ids_file "$IDS" --limit "$LIMIT" "$@" \
    > "$OUT/$name.log" 2>&1 || fail "$name failed, see $OUT/$name.log"
}

run "zs-chat-$TAG"         --condition zeroshot-chat --base "$ZS_BASE" --base_revision "$ZS_REV"
run "zs-chat-swapped-$TAG" --condition zeroshot-chat --base "$ZS_BASE" --base_revision "$ZS_REV" --order eligible_first
if [ -n "$PLAIN_BASE" ]; then
  run "zs-plain-$TAG"      --condition zeroshot-plain --base "$PLAIN_BASE" --base_revision "$PLAIN_REV"
fi
run "lora-logits-$TAG"           --condition lora-logits   --base "$LORA_BASE" --base_revision "$LORA_REV" --adapter "$ADAPTER"
run "lora-generate-$TAG"         --condition lora-generate --base "$LORA_BASE" --base_revision "$LORA_REV" --adapter "$ADAPTER"
run "lora-generate-swapped-$TAG" --condition lora-generate --base "$LORA_BASE" --base_revision "$LORA_REV" --adapter "$ADAPTER" --order eligible_first

echo "=== pointer-timing-$TAG  $(date +%H:%M:%S)"
nvidia-smi -i "$GPU" --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader \
  | sed "s/^/pointer-timing-$TAG  /" >> "$OUT/gpu-snapshots-$TAG.txt"
"$PY" scripts/time_pointer.py --run "$POINTER" --data data/tm-v1-b4096 --split test --allow-test \
  --ids_file "$IDS" --limit "$LIMIT" --dtype bf16 --out "$OUT/pointer-timing-$TAG" \
  > "$OUT/pointer-timing-$TAG.log" 2>&1 || fail "pointer-timing-$TAG failed, see $OUT/pointer-timing-$TAG.log"

echo "ALL CONDITIONS COMPLETED for $SIZE  $(date +%H:%M:%S)"
