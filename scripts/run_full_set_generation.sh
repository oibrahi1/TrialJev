#!/usr/bin/env bash
# Accuracy AND per-pair timing of each condition on the FULL test set (all 6,975 eligible/excluded pairs, 50
# patients). One condition per call, each on its own single GPU, batch size 1. Run the conditions in the same regime
# (e.g. four at once, one per GPU) so the timings are comparable with each other.
#
#   export PY=/path/to/venv/bin/python HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1
#   bash scripts/run_full_set_generation.sh CASE GPU
#
# CASE: zs-chat-9b | zs-chat-swapped-9b | zs-plain-9b | lora-logits-9b | lora-generate-9b | lora-generate-swapped-9b | pointer-onepass-9b
#       zs-chat-27b | zs-chat-swapped-27b | lora-logits-27b | lora-generate-27b | lora-generate-swapped-27b | pointer-onepass-27b
#       zs-chat-38 | zs-chat-swapped-38 | pointer-onepass-38
# pointer-onepass-*: the pointer-head checkpoint scored by the old path (A) and the one-pass path (B) on every pair,
# interleaved and timed in the same run: verifies on the full set that the one-pass path reproduces the predictions
# and gives the A-versus-B latency ratio. The cached backbone revision is recorded in backbone_revision.txt.
# Smoke test:  OUT=runs/gen_full_smoke LIMIT=5 bash scripts/run_full_set_generation.sh lora-generate-9b 1
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo "ERROR: $*" >&2; exit 1; }

CASE="${1:?usage: run_full_set_generation.sh CASE GPU}"
GPU="${2:?usage: run_full_set_generation.sh CASE GPU}"
PY="${PY:?set PY to the python of the training venv}"
HF_HOME="${HF_HOME:?set HF_HOME}"
OUT="${OUT:-runs/gen_full}"
LIMIT="${LIMIT:-0}"
DATA=data/tm-v1-b4096/test.jsonl
EXPECTED_TLL=ae8ee0c73e8443940e9209512d097bcd5104454202b3d0cbee6a36fed30f2801
REV_9B_BASE=68c46c4b3498877f3ef123c856ecfde50c39f404
REV_27B=fc05daec18b0a78c049392ed2e771dde82bdf654
REV_38=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

[ -x "$PY" ] || fail "PY=$PY is not an executable python"
[[ "$GPU" =~ ^[0-9]+$ ]] || fail "GPU must be a number, got '$GPU'"
[ -f "$DATA" ] || fail "missing $DATA"
[ "$(sha256sum scripts/train_lora_letter.py | cut -d' ' -f1)" = "$EXPECTED_TLL" ] \
  || fail "scripts/train_lora_letter.py differs from the expected version"
"$PY" -c "import torch, transformers, peft" || fail "torch/transformers/peft are not importable with PY=$PY"

resolve_rev() {   # MODEL [EXPECTED]
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

COND=; BASE=; REV=; ADAPTER=; ORDER=excluded_first; POINTER=
case "$CASE" in
  zs-chat-9b)                COND=zeroshot-chat;  BASE=Qwen/Qwen3.5-9B;      REV=$(resolve_rev "$BASE") ;;
  zs-chat-swapped-9b)        COND=zeroshot-chat;  BASE=Qwen/Qwen3.5-9B;      REV=$(resolve_rev "$BASE"); ORDER=eligible_first ;;
  pointer-onepass-9b)        POINTER=runs/run1-q35-9b-base/step-400; BASE=Qwen/Qwen3.5-9B-Base; REV=$(resolve_rev "$BASE" "$REV_9B_BASE") ;;
  pointer-onepass-27b)       POINTER=runs/run1-q35-27b-base/step-400; BASE=Qwen/Qwen3.5-27B; REV=$(resolve_rev "$BASE" "$REV_27B") ;;
  pointer-onepass-38)        POINTER=runs/run1-q38-27b-base/step-400; BASE=Qwen/Qwen3.8-27B; REV=$(resolve_rev "$BASE" "$REV_38") ;;
  zs-chat-swapped-27b)       COND=zeroshot-chat;  BASE=Qwen/Qwen3.5-27B;     REV=$(resolve_rev "$BASE" "$REV_27B"); ORDER=eligible_first ;;
  zs-chat-swapped-38)        COND=zeroshot-chat;  BASE=Qwen/Qwen3.8-27B;     REV=$(resolve_rev "$BASE" "$REV_38"); ORDER=eligible_first ;;
  zs-plain-9b)               COND=zeroshot-plain; BASE=Qwen/Qwen3.5-9B-Base; REV=$(resolve_rev "$BASE" "$REV_9B_BASE") ;;
  lora-logits-9b)            COND=lora-logits;    BASE=Qwen/Qwen3.5-9B-Base; REV=$(resolve_rev "$BASE" "$REV_9B_BASE"); ADAPTER=runs/ctl-q35-9b/step-400 ;;
  lora-logits-27b)           COND=lora-logits;    BASE=Qwen/Qwen3.5-27B;     REV=$(resolve_rev "$BASE" "$REV_27B"); ADAPTER=runs/ctl-q35-27b/step-400 ;;
  lora-generate-9b)          COND=lora-generate;  BASE=Qwen/Qwen3.5-9B-Base; REV=$(resolve_rev "$BASE" "$REV_9B_BASE"); ADAPTER=runs/ctl-q35-9b/step-400 ;;
  lora-generate-swapped-9b)  COND=lora-generate;  BASE=Qwen/Qwen3.5-9B-Base; REV=$(resolve_rev "$BASE" "$REV_9B_BASE"); ADAPTER=runs/ctl-q35-9b/step-400; ORDER=eligible_first ;;
  zs-chat-27b)               COND=zeroshot-chat;  BASE=Qwen/Qwen3.5-27B;     REV=$(resolve_rev "$BASE" "$REV_27B") ;;
  lora-generate-27b)         COND=lora-generate;  BASE=Qwen/Qwen3.5-27B;     REV=$(resolve_rev "$BASE" "$REV_27B"); ADAPTER=runs/ctl-q35-27b/step-400 ;;
  lora-generate-swapped-27b) COND=lora-generate;  BASE=Qwen/Qwen3.5-27B;     REV=$(resolve_rev "$BASE" "$REV_27B"); ADAPTER=runs/ctl-q35-27b/step-400; ORDER=eligible_first ;;
  zs-chat-38)                COND=zeroshot-chat;  BASE=Qwen/Qwen3.8-27B;     REV=$(resolve_rev "$BASE" "$REV_38") ;;
  *) fail "unknown CASE '$CASE'" ;;
esac
if [ -n "$ADAPTER" ]; then
  for f in adapter_config.json adapter_model.safetensors control_meta.json; do
    [ -f "$ADAPTER/$f" ] || fail "missing $ADAPTER/$f"
  done
fi
if [ -n "$POINTER" ]; then
  for f in head.pt adapter_config.json adapter_model.safetensors; do
    [ -f "$POINTER/$f" ] || fail "missing $POINTER/$f"
  done
fi
mkdir -p "$OUT"
[ ! -e "$OUT/$CASE" ] || fail "$OUT/$CASE already exists; refusing to overwrite"

read -r MEM UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
if [ "${FORCE_GPU:-0}" != 1 ] && { [ "$MEM" -ge 2000 ] || [ "$UTIL" -ge 10 ]; }; then
  fail "GPU $GPU is not idle (${MEM} MiB used, ${UTIL}% utilisation). Pick another GPU or set FORCE_GPU=1"
fi
export CUDA_VISIBLE_DEVICES="$GPU"
nvidia-smi -i "$GPU" --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader \
  | sed "s/^/$CASE  /" >> "$OUT/gpu-snapshots-full.txt"

echo "=== $CASE  gpu=$GPU  full test set  $(date +%H:%M:%S)  revision=$REV"
if [ -n "$POINTER" ]; then
  "$PY" scripts/time_pointer.py --run "$POINTER" --data data/tm-v1-b4096 --split test --allow-test     --per_patient 0 --limit "$LIMIT" --dtype bf16 --out "$OUT/$CASE" > "$OUT/$CASE.log" 2>&1     || fail "$CASE failed, see $OUT/$CASE.log"
  echo "$BASE@$REV (cached snapshot; the checkpoint itself records no revision)" > "$OUT/$CASE/backbone_revision.txt"
else
  ARGS=(--condition "$COND" --base "$BASE" --base_revision "$REV" --order "$ORDER" --data "$DATA"
        --per_patient 0 --limit "$LIMIT" --out "$OUT/$CASE")
  [ -n "$ADAPTER" ] && ARGS+=(--adapter "$ADAPTER")
  "$PY" scripts/generate_baselines.py run "${ARGS[@]}" > "$OUT/$CASE.log" 2>&1 || fail "$CASE failed, see $OUT/$CASE.log"
fi
echo "COMPLETED $CASE  $(date +%H:%M:%S)"
