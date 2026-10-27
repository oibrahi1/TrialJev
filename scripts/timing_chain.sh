#!/usr/bin/env bash
# Time every system of one model size on the SAME 992 pairs, one after another on ONE GPU, so the ratios are
# within-run (absolute latency varies between runs and is never compared across runs):
#   control          the A/B-logit LoRA classifier reading its logits (the reference, ratio 1.00)
#   sft-generate     the fine-tuned model that WRITES the answer (trained by scripts/train_lora_generate.py)
#   pointer          TrialJev, two serving paths timed on the same pairs in one process
#   zeroshot         un-fine-tuned model prompted to write the verdict (instruct model; Base + plain prompt at 0.8B)
#   reason           un-fine-tuned instruct model that explains, then answers (9B, 27B, Qwen3.8; 200-pair subset of the 992)
#
#   export PY=/path/to/venv/bin/python HF_HOME=/path/to/hf_cache     (or activate your virtual environment)
#   bash scripts/timing_chain.sh KEY GPU        KEY: q35-0.8b | q35-9b | q35-27b | q38-27b
#   needs runs/gen/sample_ids.json (the shared 992-pair list) and the finished runs sft-KEY-s0
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo "ERROR: $*" >&2; exit 1; }

KEY="${1:?usage: timing_chain.sh KEY GPU}"
GPU="${2:?usage: timing_chain.sh KEY GPU}"
PY="${PY:-python}"
RUNS="${RUNS:-runs}"
DATA="${DATA:-data/tm-v1-b4096}"
IDS="${IDS:-$RUNS/gen/sample_ids.json}"
OUT="$RUNS/gen_time/$KEY"

REV=""; ZS_REV=""
case "$KEY" in
  q35-0.8b) BASE=Qwen/Qwen3.5-0.8B-Base; DT=fp32; CTL="$RUNS/ctl-q35-0.8b-s0/step-200"; PTR="$RUNS/run1-q35-0.8b-clinit/step-400"
            ZS_COND=zeroshot-plain; ZS_BASE=Qwen/Qwen3.5-0.8B-Base ;;
  q35-9b)   BASE=Qwen/Qwen3.5-9B-Base; DT=bf16; CTL="$RUNS/ctl-q35-9b/step-400"; PTR="$RUNS/run1-q35-9b-base/step-400"
            REV=68c46c4b3498877f3ef123c856ecfde50c39f404; ZS_COND=zeroshot-chat; ZS_BASE=Qwen/Qwen3.5-9B
            ZS_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a ;;
  q35-27b)  BASE=Qwen/Qwen3.5-27B; DT=bf16; CTL="$RUNS/ctl-q35-27b/step-400"; PTR="$RUNS/run1-q35-27b-base/step-400"
            REV=fc05daec18b0a78c049392ed2e771dde82bdf654; ZS_COND=zeroshot-chat; ZS_BASE=Qwen/Qwen3.5-27B; ZS_REV="$REV" ;;
  q38-27b)  BASE=Qwen/Qwen3.8-27B; DT=bf16; CTL="$RUNS/ctl-q38-27b/step-200"; PTR="$RUNS/run1-q38-27b-base/step-400"
            REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; ZS_COND=zeroshot-chat; ZS_BASE=Qwen/Qwen3.8-27B; ZS_REV="$REV" ;;
  *) fail "unknown KEY '$KEY'" ;;
esac
[[ "$GPU" =~ ^[0-9]+$ ]] || fail "GPU must be a number, got '$GPU'"
[ -f "$IDS" ] || fail "$IDS missing (the shared 992-pair id list)"
[ ! -e "$OUT" ] || fail "$OUT already exists; refusing to overwrite"
for d in "$CTL" "$PTR"; do [ -d "$d" ] || fail "missing checkpoint $d"; done
SFT_RUN="$RUNS/sft-$KEY-s0"
[ -d "$SFT_RUN" ] || fail "missing $SFT_RUN (train it first with scripts/sft_lane.sh)"
SFT="$("$PY" scripts/pick_best.py "$SFT_RUN" --path)"; [ -d "$SFT" ] || fail "no best checkpoint in $SFT_RUN"

read -r MEM UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
if [ "${FORCE_GPU:-0}" != 1 ] && { [ "$MEM" -ge 2000 ] || [ "$UTIL" -ge 10 ]; }; then
  fail "GPU $GPU is not idle (${MEM} MiB used, ${UTIL}% utilisation); pick another GPU or set FORCE_GPU=1"
fi
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT"
REV_ARG=(); [ -n "$REV" ] && REV_ARG=(--base_revision "$REV")
ZS_REV_ARG=(); [ -n "$ZS_REV" ] && ZS_REV_ARG=(--base_revision "$ZS_REV")
snap() { nvidia-smi -i "$GPU" --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | sed "s/^/$1  /" >> "$OUT/gpu-snapshots.txt"; }
gb() { "$PY" scripts/generate_baselines.py run --data "$DATA/test.jsonl" --ids_file "$IDS" --dtype "$DT" "$@"; }

echo "$(date +%H:%M:%S) timing chain $KEY on GPU $GPU"
snap control;      gb --condition lora-logits   --base "$BASE" ${REV_ARG[@]+"${REV_ARG[@]}"} --adapter "$CTL" --out "$OUT/control"      > "$OUT/control.log" 2>&1 || fail "control failed, see $OUT/control.log"
snap sft-generate; gb --condition sft-generate  --base "$BASE" ${REV_ARG[@]+"${REV_ARG[@]}"} --adapter "$SFT" --out "$OUT/sft-generate" > "$OUT/sft-generate.log" 2>&1 || fail "sft-generate failed, see $OUT/sft-generate.log"
snap pointer
"$PY" scripts/time_pointer.py --run "$PTR" --data "$DATA" --split test --allow-test --ids_file "$IDS" --dtype "$DT" \
  --out "$OUT/pointer" > "$OUT/pointer.log" 2>&1 || fail "pointer timing failed, see $OUT/pointer.log"
snap zeroshot;     gb --condition "$ZS_COND"    --base "$ZS_BASE" ${ZS_REV_ARG[@]+"${ZS_REV_ARG[@]}"} --out "$OUT/zeroshot"                > "$OUT/zeroshot.log" 2>&1 || fail "zero-shot failed, see $OUT/zeroshot.log"
if [ "$KEY" != q35-0.8b ] && [ "${REASON:-1}" = 1 ]; then
  # reasoning baseline: explain, then "Answer: ...". Long outputs, so a 200-pair subset (4 per patient) of the same 992 pairs.
  "$PY" scripts/generate_baselines.py sample --data "$DATA/test.jsonl" --per_patient "${REASON_PER_PATIENT:-4}" \
    --out "$OUT/reason_sample_ids.json" > "$OUT/reason_sample.log" 2>&1 || fail "reasoning sample failed, see $OUT/reason_sample.log"
  snap reason
  "$PY" scripts/generate_baselines.py run --data "$DATA/test.jsonl" --ids_file "$OUT/reason_sample_ids.json" --dtype "$DT" \
    --condition zeroshot-reason --base "$ZS_BASE" ${ZS_REV_ARG[@]+"${ZS_REV_ARG[@]}"} --out "$OUT/reason" \
    > "$OUT/reason.log" 2>&1 || fail "reasoning baseline failed, see $OUT/reason.log"
fi
echo "$SFT" > "$OUT/sft_checkpoint.txt"; echo "$CTL" > "$OUT/control_checkpoint.txt"; echo "$PTR" > "$OUT/pointer_checkpoint.txt"
echo "$(date +%H:%M:%S) TIMING CHAIN COMPLETED $KEY"
