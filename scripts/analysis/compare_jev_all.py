#!/usr/bin/env python3
"""TrialJev (every backbone, every seed we have) against hosted Jev on the same 6,975 TREC 2022 pairs.

    python compare_jev_all.py --jev_dir runs/jev_api --stats runs/jev_api/jev_api_stats.json \
        --exports DIR1 DIR2 DIR3 [--boot 10000]

Jev: raw (0.5 threshold) and calibrated (temperature and class prior fitted on TREC 2021 development only, parameters read from
the stats file written by score_jev_api.py). TrialJev rows are the canonical two-pass evaluator at the selected checkpoint.
For multi-seed configurations the accuracy is the mean over seeds and the paired difference uses the per-pair mean correctness
over seeds. Intervals resample the 50 test patients, unadjusted.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from score_jev_api import auroc, boot, calibrate, jl

SEED = 20261001
RUNS = {   # label -> list of (run file stem) in seed order
    "0.8B fresh": ["run1-q35-0.8b-base__step-1000", "run1-q35-0.8b-base-s1__step-100", "run1-q35-0.8b-base-s2__step-500"],
    "0.8B clinical init": ["run1-q35-0.8b-clinit__step-400", "run1-q35-0.8b-clinit-s1__step-500", "run1-q35-0.8b-clinit-s2__step-500"],
    "9B": ["run1-q35-9b-base__step-400", "run1-q35-9b-base-s1__step-400", "run1-q35-9b-base-s2__step-1000"],
    "27B": ["run1-q35-27b-base__step-400"],
    "Qwen3.8-27B": ["run1-q38-27b-base__step-400"],
}


def find(exports, stem):
    for e in exports:
        p = Path(e) / f"{stem}__eval-test-preds.jsonl"
        if p.exists():
            return p
    raise SystemExit(f"missing predictions for {stem} in {exports}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev_dir", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--exports", nargs="+", required=True); ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="jev_vs_trialjev_all.json")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    jev = {r["id"]: r for r in jl(Path(a.jev_dir) / "test.jsonl") if r.get("p_true") is not None}
    st = json.load(open(a.stats)); cal = st["calibration"]
    first = jl(find(a.exports, RUNS["27B"][0]))
    ids = sorted(r["meta"]["id"] for r in first)
    assert set(ids) == set(jev) and len(ids) == 6975, "Jev and TrialJev must cover the same 6,975 pairs"
    groups = {r["meta"]["id"]: str(r["meta"]["patient_id"]) for r in first}
    g = [groups[i] for i in ids]
    y = np.array([jev[i]["gold"] for i in ids]); pj = np.array([jev[i]["p_true"] for i in ids])
    pjc = calibrate(pj, cal["T"], cal["w_true"], cal["w_false"])
    jc = {"raw": (pj >= .5) == y, "calibrated": (pjc >= .5) == y}
    jauc = auroc(y, pj)
    print(f"Jev raw {100 * jc['raw'].mean():.2f}, calibrated {100 * jc['calibrated'].mean():.2f}, AUROC {jauc:.3f}")
    out = {"jev_raw": 100 * jc["raw"].mean(), "jev_calibrated": 100 * jc["calibrated"].mean(), "jev_auroc": jauc, "rows": {}}
    hdr = f"{'configuration':20s} {'seeds':>5s} {'TrialJev acc':>13s} {'AUROC':>6s} | {'minus Jev raw':>26s} | {'minus Jev calibrated':>26s} | AUROC diff"
    print(hdr)
    for label, stems in RUNS.items():
        accs, aucs, corr = [], [], []
        for s in stems:
            rows = {r["meta"]["id"]: r for r in jl(find(a.exports, s))}
            pt = np.array([rows[i]["probs"]["true"] for i in ids])
            c = (pt >= .5) == y
            accs.append(100 * c.mean()); aucs.append(auroc(y, pt)); corr.append(c.astype(float))
            ptm = pt if len(stems) == 1 else None
        cm = np.mean(corr, axis=0)
        row = {"n_seeds": len(stems), "accuracy_mean": statistics.mean(accs), "accuracy_sd": statistics.stdev(accs) if len(accs) > 1 else None,
               "auroc_mean": statistics.mean(aucs), "per_seed_accuracy": accs}
        cells = []
        for name in ("raw", "calibrated"):
            d = 100 * (cm.mean() - jc[name].mean())
            lo, hi = boot(lambda ix, name=name: 100 * (cm[ix].mean() - jc[name][ix].mean()), g, a.boot, rng)
            row[f"minus_jev_{name}"] = {"d": d, "lo": lo, "hi": hi}
            cells.append(f"{d:+7.2f} [{lo:+6.2f}, {hi:+6.2f}]")
        dauc = row["auroc_mean"] - jauc
        row["auroc_minus_jev"] = dauc
        out["rows"][label] = row
        sd = f"±{row['accuracy_sd']:.2f}" if row["accuracy_sd"] is not None else ""
        print(f"{label:20s} {len(stems):5d} {row['accuracy_mean']:7.2f}{sd:>6s} {row['auroc_mean']:6.3f} | {cells[0]:>26s} | {cells[1]:>26s} | {dauc:+.3f}")
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
