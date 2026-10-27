#!/usr/bin/env bash
# Overnight jobs (training + scoring). Fail-fast, never overwrites, GPUs must be idle.
#
#   export PY=/path/to/venv/bin/python HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1
#   nohup bash scripts/overnight_seeds.sh all > runs/overnight-master.log 2>&1 &
#
# MODE
#   calibrate  time the same 200 records on each GPU in turn (quiet node needed), print the four medians
#   plain9     GPU 1: plain-LoRA 9B seeds 1 and 2 (A/B letters), each followed by one test scoring
#   ptr9       GPU 3: pointer-head 9B seeds 1 and 2, each followed by one test scoring
#   q38        GPU 0: plain-LoRA Qwen3.8-27B (smoke test first), then test scoring in both letter orders
#   all        calibrate, then start plain9, ptr9 and q38 in the background. GPU 2 is left idle on purpose.
# DRY=1 prints every command instead of running it.
#
# Same recipe as the seed-0 runs: 3 epochs, lr 1e-4, effective batch 16 (2 x 8), max_state 512, validation and
# checkpoints every 200 steps; checkpoint = lowest development loss; the test set is scored once per run.
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo "ERROR: $*" >&2; exit 1; }

MODE="${1:?usage: overnight_seeds.sh all|calibrate|plain9|ptr9|q38}"
PY="${PY:?set PY to the python of the training venv}"
HF_HOME="${HF_HOME:?set HF_HOME}"
DRY="${DRY:-0}"
D=data/tm-v1-b4096
EXPECTED_TLL=ae8ee0c73e8443940e9209512d097bcd5104454202b3d0cbee6a36fed30f2801
REV_9B_BASE=68c46c4b3498877f3ef123c856ecfde50c39f404
REV_38=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
BIN="$(dirname "$PY")"

[ -x "$PY" ] || fail "PY=$PY is not an executable python"
[ "$(sha256sum scripts/train_lora_letter.py | cut -d' ' -f1)" = "$EXPECTED_TLL" ] \
  || fail "scripts/train_lora_letter.py differs from the expected version"
for rev in "models--Qwen--Qwen3.5-9B-Base/snapshots/$REV_9B_BASE" "models--Qwen--Qwen3.8-27B/snapshots/$REV_38"; do
  [ -d "$HF_HOME/hub/$rev" ] || fail "pinned revision missing from the cache: $rev"
done
mkdir -p runs

gpu_idle() {   # memory < 2 GB and utilisation < 10%
  local mem util
  [ "$DRY" = 1 ] && return 0
  read -r mem util < <(nvidia-smi -i "$1" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
  if [ "${FORCE_GPU:-0}" != 1 ] && { [ "$mem" -ge 2000 ] || [ "$util" -ge 10 ]; }; then
    fail "GPU $1 is not idle (${mem} MiB, ${util}%)"
  fi
}

xl() {   # xl LOGFILE command... : run with output to LOGFILE, stop the script if it fails
  local log="$1"; shift
  if [ "$DRY" = 1 ]; then echo "[dry] $* > $log"; return 0; fi
  echo "$(date +%H:%M:%S) start: $log"
  "$@" > "$log" 2>&1 || fail "failed, see $log"
}

no_overwrite() { [ ! -e "$1" ] || fail "$1 already exists; refusing to overwrite"; }

best_of() {    # best_of RUN -> the lowest-development-loss checkpoint directory
  if [ "$DRY" = 1 ]; then echo "<best checkpoint of $1>"; else "$PY" scripts/pick_best.py "$1" --path; fi
}

calibrate() {
  mkdir -p runs/gen_gpucal
  for g in 0 1 2 3; do
    no_overwrite "runs/gen_gpucal/gpu$g"
    gpu_idle "$g"
    xl "runs/gen_gpucal-$g.log" env CUDA_VISIBLE_DEVICES="$g" "$PY" scripts/generate_baselines.py run \
      --condition lora-logits --base Qwen/Qwen3.5-9B-Base --base_revision "$REV_9B_BASE" \
      --adapter runs/ctl-q35-9b/step-400 --data "$D/test.jsonl" --ids_file runs/gen/sample_ids.json \
      --limit 200 --out "runs/gen_gpucal/gpu$g"
  done
  if [ "$DRY" != 1 ]; then
    "$PY" - <<'PY'
import json, statistics as st
for g in range(4):
    r = [json.loads(l) for l in open(f"runs/gen_gpucal/gpu{g}/preds.jsonl")]
    print("gpu", g, "median ms %.1f" % st.median(x["ms"] for x in r), flush=True)
PY
  fi
}

lane_plain9() {
  local g=1 s out best
  for s in 1 2; do
    out=runs/ctl-q35-9b-s$s; no_overwrite "$out"; gpu_idle $g
    xl "$out.log" env CUDA_VISIBLE_DEVICES=$g "$PY" scripts/train_lora_letter.py train \
      --base Qwen/Qwen3.5-9B-Base --base_revision "$REV_9B_BASE" --base_dtype bf16 \
      --data "$D/train.jsonl" --val_data "$D/development.jsonl" --out "$out" --seed $s \
      --epochs 3 --lr 1e-4 --batch 2 --accum 8 --dtype bf16 --checkpointing 1 \
      --log_every 20 --val_every 200 --save_every 200
    best=$(best_of "$out")
    xl "$out-eval.log" env CUDA_VISIBLE_DEVICES=$g "$PY" scripts/train_lora_letter.py eval \
      --run "$best" --data "$D" --split test --allow-test
  done
  echo "LANE COMPLETED plain9"
}

lane_ptr9() {
  local g=3 s out
  for s in 1 2; do
    out=runs/run1-q35-9b-base-s$s; no_overwrite "$out"; gpu_idle $g
    xl "$out.log" env CUDA_VISIBLE_DEVICES=$g "$PY" -m trialmatch.train \
      --base Qwen/Qwen3.5-9B-Base --base_revision "$REV_9B_BASE" --base_dtype bf16 \
      --data "$D/train.jsonl" --val_data "$D/development.jsonl" --out "$out" --seed $s \
      --epochs 3 --lr 1e-4 --batch 2 --accum 8 --max_state 512 --max_branch 4096 --dtype bf16 --checkpointing 1 \
      --log_every 20 --val_every 200 --save_every 200 --val_records 100000 --wandb 0
    xl "$out-eval.log" env PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES=$g bash scripts/eval_run.sh "$out" $g test
  done
  echo "LANE COMPLETED ptr9"
}

lane_q38() {
  local g=0 out=runs/ctl-q38-27b best order sfx
  no_overwrite runs/smoke-ctl-q38; no_overwrite "$out"; gpu_idle $g
  xl runs/smoke-ctl-q38.log env CUDA_VISIBLE_DEVICES=$g "$PY" scripts/train_lora_letter.py train \
    --base Qwen/Qwen3.8-27B --base_revision "$REV_38" --base_dtype bf16 \
    --data "$D/train.jsonl" --val_data "$D/development.jsonl" --out runs/smoke-ctl-q38 \
    --max_records 32 --val_records 16 --epochs 1 --batch 2 --accum 2 --dtype bf16 \
    --log_every 1 --val_every 4 --save_every 4
  xl "$out.log" env CUDA_VISIBLE_DEVICES=$g "$PY" scripts/train_lora_letter.py train \
    --base Qwen/Qwen3.8-27B --base_revision "$REV_38" --base_dtype bf16 \
    --data "$D/train.jsonl" --val_data "$D/development.jsonl" --out "$out" --seed 0 \
    --epochs 3 --lr 1e-4 --batch 2 --accum 8 --dtype bf16 --checkpointing 1 \
    --log_every 20 --val_every 200 --save_every 200
  best=$(best_of "$out")
  for order in excluded_first eligible_first; do
    sfx=""; [ "$order" = eligible_first ] && sfx="-swapped"
    xl "$out-eval$sfx.log" env CUDA_VISIBLE_DEVICES=$g "$PY" scripts/train_lora_letter.py eval \
      --run "$best" --data "$D" --split test --allow-test --order "$order"
  done
  echo "LANE COMPLETED q38"
}

case "$MODE" in
  calibrate) calibrate ;;
  plain9)    lane_plain9 ;;
  ptr9)      lane_ptr9 ;;
  q38)       lane_q38 ;;
  all)
    calibrate
    if [ "$DRY" = 1 ]; then lane_q38; lane_plain9; lane_ptr9; exit 0; fi
    nohup bash "$0" q38    > runs/overnight-q38.log    2>&1 &
    nohup bash "$0" plain9 > runs/overnight-plain9.log 2>&1 &
    nohup bash "$0" ptr9   > runs/overnight-ptr9.log   2>&1 &
    echo "started q38 (GPU 0), plain9 (GPU 1), ptr9 (GPU 3); GPU 2 left idle  $(date +%H:%M:%S)"
    ;;
  *) fail "unknown MODE '$MODE'" ;;
esac
