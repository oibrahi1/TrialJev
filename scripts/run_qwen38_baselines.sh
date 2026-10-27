#!/usr/bin/env bash
# Qwen3.8-27B: only a pointer-head model exists for it, so it gets (a) zero-shot generation in both option
# orders and (b) pointer-head timing (old scoring path A and one-pass path B) on the SAME shared sample.
#
#   export PY=/path/to/venv/bin/python HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1
#   bash scripts/run_qwen38_baselines.sh GPU          # needs runs/gen/sample_ids.json (from the main launcher)
#
# Same safety rules as run_generation_baselines.sh: stops at the first error, never overwrites, checks the
# exact training script, the pinned revision, the checkpoint and that the GPU is idle.
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo "ERROR: $*" >&2; exit 1; }

GPU="${1:?usage: run_qwen38_baselines.sh GPU}"
PY="${PY:?set PY to the python of the training venv}"
HF_HOME="${HF_HOME:?set HF_HOME}"
OUT="${OUT:-runs/gen}"
LIMIT="${LIMIT:-0}"
DATA=data/tm-v1-b4096/test.jsonl
IDS="$OUT/sample_ids.json"
EXPECTED_TLL=ae8ee0c73e8443940e9209512d097bcd5104454202b3d0cbee6a36fed30f2801
MODEL=Qwen/Qwen3.8-27B
REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
POINTER=runs/run1-q38-27b-base/step-400
TAG=38

[ -x "$PY" ] || fail "PY=$PY is not an executable python"
[[ "$GPU" =~ ^[0-9]+$ ]] || fail "GPU must be a number, got '$GPU'"
[ -f "$DATA" ] || fail "missing $DATA"
[ -f "$IDS" ] || fail "$IDS missing: create the shared sample first (main launcher: 'sample')"
[ "$(sha256sum scripts/train_lora_letter.py | cut -d' ' -f1)" = "$EXPECTED_TLL" ] \
  || fail "scripts/train_lora_letter.py differs from the expected version"
"$PY" -c "import torch, transformers, peft" || fail "torch/transformers/peft are not importable with PY=$PY"
[ -d "$HF_HOME/hub/models--Qwen--Qwen3.8-27B/snapshots/$REV" ] || fail "pinned revision $REV is not in the cache"
for f in head.pt adapter_config.json adapter_model.safetensors; do
  [ -f "$POINTER/$f" ] || fail "missing $POINTER/$f"
done
for name in zs-chat-$TAG zs-chat-swapped-$TAG pointer-timing-$TAG; do
  [ ! -e "$OUT/$name" ] || fail "$OUT/$name already exists; refusing to overwrite"
done

read -r MEM UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
if [ "${FORCE_GPU:-0}" != 1 ] && { [ "$MEM" -ge 2000 ] || [ "$UTIL" -ge 10 ]; }; then
  fail "GPU $GPU is not idle (${MEM} MiB used, ${UTIL}% utilisation). Pick another GPU or set FORCE_GPU=1"
fi
export CUDA_VISIBLE_DEVICES="$GPU"
echo "preflight ok: gpu=$GPU out=$OUT limit=$LIMIT revision=$REV"

snap() { nvidia-smi -i "$GPU" --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader \
           | sed "s/^/$1  /" >> "$OUT/gpu-snapshots-$TAG.txt"; }

run() {
  local name="$1"; shift
  snap "$name"
  echo "=== $name  $(date +%H:%M:%S)"
  "$PY" scripts/generate_baselines.py run --data "$DATA" --out "$OUT/$name" --ids_file "$IDS" --limit "$LIMIT" "$@" \
    > "$OUT/$name.log" 2>&1 || fail "$name failed, see $OUT/$name.log"
}

run "zs-chat-$TAG"         --condition zeroshot-chat --base "$MODEL" --base_revision "$REV"
run "zs-chat-swapped-$TAG" --condition zeroshot-chat --base "$MODEL" --base_revision "$REV" --order eligible_first

echo "=== pointer-timing-$TAG  $(date +%H:%M:%S)"
snap "pointer-timing-$TAG"
"$PY" scripts/time_pointer.py --run "$POINTER" --data data/tm-v1-b4096 --split test --allow-test \
  --ids_file "$IDS" --limit "$LIMIT" --dtype bf16 --out "$OUT/pointer-timing-$TAG" \
  > "$OUT/pointer-timing-$TAG.log" 2>&1 || fail "pointer-timing-$TAG failed, see $OUT/pointer-timing-$TAG.log"

echo "ALL CONDITIONS COMPLETED for qwen3.8-27b  $(date +%H:%M:%S)"
