#!/usr/bin/env python3
"""Print the validation curve of a run and the checkpoint with the lowest dev loss.

    python scripts/pick_best.py runs/run1-base-b4096-s0          # table + best path
    python scripts/pick_best.py runs/run1-base-b4096-s0 --path   # best path only

Selection is on development loss (a proper scoring rule, so it favours calibrated
probabilities); accuracy is shown alongside. The locked test set is never read here.
"""
import json
import os
import sys

run = sys.argv[1].rstrip("/")
path_only = "--path" in sys.argv
vals = [e for e in map(json.loads, open(os.path.join(run, "train_log_rank0.jsonl"))) if "val/loss" in e]
if not vals:
    sys.exit(f"{run}: no validation entries yet")
cfg = json.load(open(os.path.join(run, "training_config.json")))
last = cfg["optimizer_steps"]
best = min(vals, key=lambda e: e["val/loss"])


def ckpt(step):
    d = os.path.join(run, f"step-{step}")
    if os.path.isdir(d):
        return d
    return run if step == last and os.path.exists(os.path.join(run, "head.pt")) else None


if path_only:
    p = ckpt(best["step"])
    if p is None:
        sys.exit(f"best step {best['step']} has no saved checkpoint")
    print(p)
    sys.exit()
print(f"{'step':>6} {'val_loss':>9} {'val_acc':>8}  checkpoint")
for e in vals:
    mark = "  <- best" if e is best else ""
    print(f"{e['step']:>6} {e['val/loss']:>9.4f} {e['val/accuracy']:>8.4f}  {ckpt(e['step']) or '-'}{mark}")
