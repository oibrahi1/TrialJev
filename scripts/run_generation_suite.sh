#!/usr/bin/env bash
# Everything for the natural-language-generation comparison on one machine, unattended, ending in ONE archive.
#
#   cluster (4 x H100):   export PY=/path/to/venv/bin/python HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1
#                         nohup bash scripts/run_generation_suite.sh cluster > runs/gen-suite-cluster.log 2>&1 &
#   second machine:       (activate the venv)    nohup bash scripts/run_generation_suite.sh small 0 > runs/gen-suite-small.log 2>&1 &
#
# cluster: trains the text-writing fine-tune on 27B (GPU 0), Qwen3.8-27B (GPU 1) and 9B seeds 0,1 (GPU 2) and 2 (GPU 3),
#          evaluates each on the full test set, then times 9B, 27B and Qwen3.8-27B against the control, TrialJev and
#          prompted generation on the shared 992-pair sample (one chain per GPU, conditions one after another).
# small:   trains the 0.8B writer, evaluates it, runs the 0.8B prompted-generation baseline on the full test set, and times
#          the 0.8B systems.
# Output: ~/trialmatch_gen_suite_<mode>.tgz (no adapter weights). Check progress: tail -n 3 runs/gen-suite-*.log
# Smoke test first (minutes):  SMOKE=1 bash scripts/sft_lane.sh q35-9b <idle GPU> 0
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: run_generation_suite.sh cluster|small [GPU]}"
PY="${PY:-python}"; export PY
RUNS="${RUNS:-runs}"; export RUNS
mkdir -p "$RUNS"
declare -A STATUS

wait_lane() {   # wait_lane NAME PID
  if wait "$2"; then STATUS["$1"]=ok; else STATUS["$1"]=FAILED; fi
  echo "$(date +%H:%M:%S) lane $1: ${STATUS[$1]}"
}

pack() {
  local tgz="$HOME/trialmatch_gen_suite_$MODE.tgz"
  tar czf "$tgz" --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' \
    "$RUNS"/gen_sft "$RUNS"/gen_time "$RUNS"/gen_sft_extra "$RUNS"/sft-* "$RUNS"/gen-suite-*.log "$RUNS"/sft-lane-*.log 2>/dev/null
  echo "$(date +%H:%M:%S) packed $tgz  ($(tar tzf "$tgz" | wc -l) files)"
  sha256sum "$tgz" | cut -c1-16
}

case "$MODE" in
  cluster)
    echo "$(date +%H:%M:%S) phase 1: training lanes"
    bash scripts/sft_lane.sh q35-27b 0 0 > "$RUNS/sft-lane-q35-27b.log" 2>&1 & P27=$!
    bash scripts/sft_lane.sh q38-27b 1 0 > "$RUNS/sft-lane-q38-27b.log" 2>&1 & P38=$!
    ( bash scripts/sft_lane.sh q35-9b 2 0 && bash scripts/sft_lane.sh q35-9b 2 1 ) > "$RUNS/sft-lane-q35-9b-gpu2.log" 2>&1 & P9A=$!
    bash scripts/sft_lane.sh q35-9b 3 2 > "$RUNS/sft-lane-q35-9b-gpu3.log" 2>&1 & P9B=$!
    wait_lane q35-27b "$P27"; wait_lane q38-27b "$P38"; wait_lane q35-9b-s0-s1 "$P9A"; wait_lane q35-9b-s2 "$P9B"
    echo "$(date +%H:%M:%S) phase 2: timing chains (only sizes whose seed-0 lane succeeded)"
    T=()
    [ "${STATUS[q35-27b]}" = ok ] && { bash scripts/timing_chain.sh q35-27b 0 > "$RUNS/sft-lane-time-q35-27b.log" 2>&1 & T+=("q35-27b:$!"); }
    [ "${STATUS[q38-27b]}" = ok ] && { bash scripts/timing_chain.sh q38-27b 1 > "$RUNS/sft-lane-time-q38-27b.log" 2>&1 & T+=("q38-27b:$!"); }
    [ "${STATUS[q35-9b-s0-s1]}" = ok ] && { bash scripts/timing_chain.sh q35-9b 2 > "$RUNS/sft-lane-time-q35-9b.log" 2>&1 & T+=("q35-9b:$!"); }
    for t in ${T[@]+"${T[@]}"}; do wait_lane "time-${t%%:*}" "${t##*:}"; done
    ;;
  small)
    GPU="${2:?usage: run_generation_suite.sh small GPU}"
    echo "$(date +%H:%M:%S) phase 1: 0.8B writer"
    bash scripts/sft_lane.sh q35-0.8b "$GPU" 0 > "$RUNS/sft-lane-q35-0.8b.log" 2>&1 & PS=$!
    wait_lane q35-0.8b "$PS"
    if [ "${STATUS[q35-0.8b]}" = ok ]; then
      mkdir -p "$RUNS/gen_sft_extra"
      echo "$(date +%H:%M:%S) phase 2: 0.8B prompted generation (full test set)"
      CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/generate_baselines.py run --condition zeroshot-plain --base Qwen/Qwen3.5-0.8B-Base \
        --data data/tm-v1-b4096/test.jsonl --per_patient 0 --dtype fp32 --out "$RUNS/gen_sft_extra/zeroshot-plain-0.8b" \
        > "$RUNS/gen_sft_extra/zeroshot-plain-0.8b.log" 2>&1 && echo "zero-shot done" || echo "zero-shot FAILED, see $RUNS/gen_sft_extra"
      echo "$(date +%H:%M:%S) phase 3: timing chain"
      bash scripts/timing_chain.sh q35-0.8b "$GPU" > "$RUNS/sft-lane-time-q35-0.8b.log" 2>&1 & PT=$!
      wait_lane time-q35-0.8b "$PT"
    fi
    ;;
  *) echo "unknown MODE '$MODE'" >&2; exit 1 ;;
esac

echo "$(date +%H:%M:%S) summary:"; for k in "${!STATUS[@]}"; do echo "  $k: ${STATUS[$k]}"; done
pack
echo "$(date +%H:%M:%S) SUITE FINISHED ($MODE)"
