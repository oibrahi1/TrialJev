#!/usr/bin/env bash
# Score a run's best checkpoint (lowest dev loss) on the development split.
#   scripts/eval_run.sh runs/run1-base-b4096-s0 [GPU]
# Writes <checkpoint>/eval-development.json and eval-development-preds.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?usage: scripts/eval_run.sh RUN_DIR [GPU] [SPLIT]}"
export CUDA_VISIBLE_DEVICES="${2:-0}"
SPLIT="${3:-development}"
python scripts/pick_best.py "$RUN"
BEST=$(python scripts/pick_best.py "$RUN" --path)
read -r DATA DTYPE < <(python -c "import json,os,sys; a=json.load(open(sys.argv[1]))['args']; print(os.path.dirname(a['data']), a.get('base_dtype','fp32'))" "$RUN/training_config.json")
echo "evaluating $BEST on $DATA/$SPLIT.jsonl (dtype $DTYPE)"
ALLOW=(); [ "$SPLIT" = "test" ] && ALLOW=(--allow-test)
python -m trialmatch.evaluate --run "$BEST" --data "$DATA" --split "$SPLIT" --dtype "$DTYPE" "${ALLOW[@]}"
