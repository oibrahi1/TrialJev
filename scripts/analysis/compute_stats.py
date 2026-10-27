#!/usr/bin/env python3
"""Patient-clustered statistics for the paper, computed from the saved per-pair prediction files.

    python compute_stats.py <DIR_SMALL> <DIR_MAIN> <DIR_OVERNIGHT> [--boot 10000]

Every bootstrap draw resamples the 50 test patients (with replacement) and re-weights every system's
pairs by the same patient multiplicities, so differences between systems are *paired*. For a system made of
several seeds, the metric is the mean over its seeds (seed spread is reported separately as a standard
deviation; the bootstrap captures patient-sampling uncertainty only).
Writes results_stats.json and prints the tables used in the paper.
"""
import argparse
import os
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("TRIALMATCH_OUT", "."))
sys.path.insert(0, str(HERE.parent))
import analyze_predictions as ap  # noqa: E402

METRICS = ["accuracy", "macro_f1", "brier_conventional_binary", "ece_10_bin", "auroc", "average_precision_eligible"]
SEED = 20261001

# system name -> (directory key, [file stems])
SYSTEMS = {
    "ptr_0.8B_fresh": ("ai", ["run1-q35-0.8b-base__step-1000", "run1-q35-0.8b-base-s1__step-100", "run1-q35-0.8b-base-s2__step-500"]),
    "ptr_0.8B_clinit": ("ai", ["run1-q35-0.8b-clinit__step-400", "run1-q35-0.8b-clinit-s1__step-500", "run1-q35-0.8b-clinit-s2__step-500"]),
    "ctl_0.8B_fresh": ("ai", ["ctl-q35-0.8b-s0__step-200"]),
    "medjev_zero_shot": ("ai", ["init-clinical-0.8b"]),
    "ptr_9B": ("cl", ["run1-q35-9b-base__step-400"]),
    "ctl_9B": ("cl", ["ctl-q35-9b__step-400"]),
    "ctl_9B_swapped": ("cl", ["ctl-q35-9b__step-400__eligible_first"]),
    "ptr_27B": ("cl", ["run1-q35-27b-base__step-400"]),
    "ctl_27B": ("cl", ["ctl-q35-27b__step-400"]),
    "ctl_27B_swapped": ("cl", ["ctl-q35-27b__step-400__eligible_first"]),
    "ptr_Qwen3.8-27B": ("cl", ["run1-q38-27b-base__step-400"]),
    # overnight jobs (seeds 1 and 2 at 9B; A/B-logit control on Qwen3.8-27B). Checkpoint = lowest development loss.
    "ptr_9B_3seeds": ("cl", ["run1-q35-9b-base__step-400", "ov:run1-q35-9b-base-s1__step-400", "ov:run1-q35-9b-base-s2__step-1000"]),
    "ctl_9B_s2_step200": ("ov", ["ctl-q35-9b-s2__step-200"]),
    "ctl_9B_3seeds": ("cl", ["ctl-q35-9b__step-400", "ov:ctl-q35-9b-s1__step-400", "ov:ctl-q35-9b-s2__step-200"]),
    "ctl_Qwen3.8-27B": ("ov", ["ctl-q38-27b__step-200"]),
    "ctl_Qwen3.8-27B_swapped": ("ov", ["ctl-q38-27b__step-200__eligible_first"]),
    # POST HOC (scored after the main analysis; the loss rule had selected step 200 for these two runs)
    "post_ctl_9B_s2_step400": ("pt", ["ctl-q35-9b-s2/step-400/eval-test-preds.jsonl"]),
    "post_ctl_Qwen3.8-27B_step400": ("pt", ["ctl-q38-27b/step-400/eval-test-preds.jsonl"]),
    "post_ctl_Qwen3.8-27B_step400_swapped": ("pt", ["ctl-q38-27b/step-400/eval-test-eligible_first-preds.jsonl"]),
}
FILE_OF = {"ctl-q38-27b__step-200__eligible_first": "ctl-q38-27b__step-200__eval-test-eligible_first-preds.jsonl",
           "ctl-q35-9b__step-400__eligible_first": "ctl-q35-9b__step-400__eval-test-eligible_first-preds.jsonl",
           "ctl-q35-27b__step-400__eligible_first": "ctl-q35-27b__step-400__eval-test-eligible_first-preds.jsonl"}

COMPARISONS = [  # (label, system A, system B) -> A minus B
    ("9B: A/B-logit control - pointer", "ctl_9B", "ptr_9B"),
    ("27B: A/B-logit control - pointer", "ctl_27B", "ptr_27B"),
    ("27B: A/B-logit control (swapped) - pointer", "ctl_27B_swapped", "ptr_27B"),
    ("9B: A/B-logit control (swapped) - pointer", "ctl_9B_swapped", "ptr_9B"),
    ("pointer: 27B - 9B", "ptr_27B", "ptr_9B"),
    ("pointer: Qwen3.8-27B - Qwen3.5-27B", "ptr_Qwen3.8-27B", "ptr_27B"),
    ("A/B-logit control: 27B - 9B", "ctl_27B", "ctl_9B"),
    ("A/B-logit control 27B vs pointer Qwen3.8-27B", "ctl_27B", "ptr_Qwen3.8-27B"),
    ("9B (3 seeds): A/B-logit control - pointer", "ctl_9B_3seeds", "ptr_9B_3seeds"),
    ("pointer: 27B - 9B (3 seeds)", "ptr_27B", "ptr_9B_3seeds"),
    ("A/B-logit control: 27B - 9B (3 seeds)", "ctl_27B", "ctl_9B_3seeds"),
    ("Qwen3.8-27B: pointer - A/B-logit control", "ptr_Qwen3.8-27B", "ctl_Qwen3.8-27B"),
    ("Qwen3.8-27B: pointer - A/B-logit control (swapped)", "ptr_Qwen3.8-27B", "ctl_Qwen3.8-27B_swapped"),
    ("A/B-logit control: Qwen3.8-27B - Qwen3.5-27B", "ctl_Qwen3.8-27B", "ctl_27B"),
    ("POST HOC Qwen3.8-27B A/B-logit control: step 400 - step 200", "post_ctl_Qwen3.8-27B_step400", "ctl_Qwen3.8-27B"),
    ("POST HOC Qwen3.8-27B A/B-logit control (swapped): step 400 - step 200", "post_ctl_Qwen3.8-27B_step400_swapped", "ctl_Qwen3.8-27B_swapped"),
    ("POST HOC Qwen3.8-27B: pointer - A/B-logit control step 400", "ptr_Qwen3.8-27B", "post_ctl_Qwen3.8-27B_step400"),
    ("POST HOC 9B seed 2 A/B-logit control: step 400 - step 200", "post_ctl_9B_s2_step400", "ctl_9B_s2_step200"),
    ("0.8B pointer: MedJev init - fresh (3 seeds)", "ptr_0.8B_clinit", "ptr_0.8B_fresh"),
    ("0.8B: MedJev-init pointer - fresh A/B-logit control", "ptr_0.8B_clinit", "ctl_0.8B_fresh"),
    ("0.8B fresh: A/B-logit control - pointer", "ctl_0.8B_fresh", "ptr_0.8B_fresh"),
]
ENSEMBLES = {  # fixed in advance (post hoc, labeled as such in the paper): plain average of P(eligible)
    "ens_AB_logit_control_9B+27B": ["ctl_9B", "ctl_27B"],
    "ens_pointer_9B+27B+Qwen3.8": ["ptr_9B", "ptr_27B", "ptr_Qwen3.8-27B"],
    "ens_all_5_models": ["ctl_9B", "ctl_27B", "ptr_9B", "ptr_27B", "ptr_Qwen3.8-27B"],
}


def load(dirs, key, stem):
    if key == "pt":
        return ap.load_predictions(dirs["pt"] / "runs" / stem)
    if stem.startswith("ov:"):
        key, stem = "ov", stem[3:]
    fname = FILE_OF.get(stem, f"{stem}__eval-test-preds.jsonl" if "init-clinical" not in stem else f"{stem}__eval-test-preds.jsonl")
    if stem.startswith("init-clinical"):
        fname = f"{stem}__eval-test-preds.jsonl"
    elif stem not in FILE_OF:
        run, step = stem.split("__")
        fname = f"{run}__{step}__eval-test-preds.jsonl"
    return ap.load_predictions(dirs[key] / fname)


def reorder(d, order):
    idx = np.array([d["_pos"][i] for i in order])
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in d.items() if k != "_pos"}


def main():
    ar = argparse.ArgumentParser()
    ar.add_argument("small_runs")
    ar.add_argument("cluster")
    ar.add_argument("overnight")
    ar.add_argument("post", help="folder with the post-hoc step-400 scoring")
    ar.add_argument("--boot", type=int, default=10000)
    a = ar.parse_args()
    dirs = {"ai": Path(a.small_runs), "cl": Path(a.cluster), "ov": Path(a.overnight), "pt": Path(a.post)}

    runs, order = {}, None
    for sysname, (key, stems) in SYSTEMS.items():
        runs[sysname] = []
        for stem in stems:
            d = load(dirs, key, stem)
            d["_pos"] = {i: k for k, i in enumerate(d["ids"].tolist())}
            if order is None:
                order = sorted(d["ids"].tolist())
            if set(d["_pos"]) != set(order):
                raise SystemExit(f"{stem}: pair ids differ from the reference set")
            runs[sysname].append(reorder(d, order))
    ref = runs["ptr_9B"][0]
    for s, rl in runs.items():
        for r in rl:
            assert (r["gold"] == ref["gold"]).all() and (r["patients"] == ref["patients"]).all(), s
    n, patients = len(order), sorted(set(ref["patients"].tolist()))
    idx_of = {p: np.flatnonzero(ref["patients"] == p) for p in patients}
    print(f"{n} pairs, {len(patients)} patients, systems: {len(runs)}")

    # ensembles as pseudo-systems (average of the member systems' mean probabilities; 0.5 threshold)
    for ename, members in ENSEMBLES.items():
        prob = np.mean([np.mean([r["prob"] for r in runs[m]], axis=0) for m in members], axis=0)
        runs[ename] = [{"gold": ref["gold"], "patients": ref["patients"], "prob": prob,
                        "pred": (prob > 0.5).astype(np.int8), "ids": ref["ids"]}]

    def sys_metrics(name, w=None):
        vals = [ap.metric_values(r, w) for r in runs[name]]
        return {m: float(np.mean([v[m] for v in vals])) for m in METRICS}

    point = {s: sys_metrics(s) for s in runs}
    seed_sd = {s: {m: float(np.std([ap.metric_values(r)[m] for r in runs[s]], ddof=1)) for m in METRICS}
               for s in runs if len(runs[s]) > 1}

    # ---- paired patient bootstrap ----
    rng = np.random.default_rng(SEED)
    boots = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    for b in range(a.boot):
        draw = rng.choice(len(patients), size=len(patients), replace=True)
        w = np.zeros(n)
        for p_i, c in zip(*np.unique(draw, return_counts=True)):
            w[idx_of[patients[p_i]]] = c
        for s in runs:
            m = sys_metrics(s, w)
            for k, v in m.items():
                boots[s][k].append(v)
        if b == 99:
            print(f"  ~{(time.time() - t0) / 100 * a.boot / 60:.1f} min total for {a.boot} resamples", flush=True)
    ci = {s: {m: [float(np.nanquantile(boots[s][m], 0.025)), float(np.nanquantile(boots[s][m], 0.975))] for m in METRICS}
          for s in runs}
    comp = {}
    for label, A, B in COMPARISONS:
        comp[label] = {"A": A, "B": B, "metrics": {}}
        for m in METRICS:
            d = np.array(boots[A][m]) - np.array(boots[B][m])
            est = point[A][m] - point[B][m]
            comp[label]["metrics"][m] = {"diff": est, "ci95": [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))],
                                         "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}

    # ---- extra diagnostics ----
    extra = {}
    for s in ("ptr_9B", "ctl_9B", "ptr_27B", "ctl_27B", "ptr_Qwen3.8-27B", "ctl_Qwen3.8-27B"):
        r = runs[s][0]
        gold, pred = r["gold"], r["pred"]
        extra[s] = {"eligible_recall": float(((pred == 1) & (gold == 1)).sum() / (gold == 1).sum()),
                    "excluded_recall": float(((pred == 0) & (gold == 0)).sum() / (gold == 0).sum()),
                    "predicted_eligible_fraction": float(pred.mean())}
        per_patient = np.array([(pred[idx_of[p]] == gold[idx_of[p]]).mean() for p in patients])
        extra[s]["per_patient_accuracy"] = {"min": float(per_patient.min()), "q25": float(np.quantile(per_patient, .25)),
                                            "median": float(np.median(per_patient)), "q75": float(np.quantile(per_patient, .75)),
                                            "max": float(per_patient.max()), "sd": float(per_patient.std(ddof=1))}
    for a_, b_ in (("ctl_9B", "ctl_9B_swapped"), ("ctl_27B", "ctl_27B_swapped"), ("ctl_Qwen3.8-27B", "ctl_Qwen3.8-27B_swapped")):
        extra[f"order_agreement_{a_}"] = float((runs[a_][0]["pred"] == runs[b_][0]["pred"]).mean())
    # reliability (10 equal-width confidence bins) for the figure
    rel = {}
    for s in ("ptr_9B", "ctl_9B", "ptr_27B", "ctl_27B"):
        r = runs[s][0]
        conf = np.maximum(r["prob"], 1 - r["prob"])
        correct = (r["gold"] == r["pred"]).astype(float)
        edges = np.linspace(0, 1, 11)
        bins = []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (conf >= lo) & ((conf < hi) | (i == 9 and conf <= hi))
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                         "conf": float(conf[m].mean()) if m.any() else None,
                         "acc": float(correct[m].mean()) if m.any() else None})
        rel[s] = bins

    out = {"n_pairs": n, "n_patients": len(patients), "bootstrap": {"resamples": a.boot, "seed": SEED, "unit": "patient"},
           "point": point, "ci95": ci, "seed_sd": seed_sd, "comparisons": comp, "extra": extra, "reliability": rel,
           "ensembles_note": "post hoc: fixed sets, plain average of P(eligible), threshold 0.5; not tuned on the test set"}
    (OUT / "results_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    pct = lambda v: f"{100 * v:.2f}"
    print("\n== point estimates and 95% patient-clustered CIs ==")
    for s in runs:
        p, c = point[s], ci[s]
        print(f"{s:30s} acc {pct(p['accuracy'])} [{pct(c['accuracy'][0])},{pct(c['accuracy'][1])}]  "
              f"F1 {pct(p['macro_f1'])} [{pct(c['macro_f1'][0])},{pct(c['macro_f1'][1])}]  "
              f"AUROC {p['auroc']:.4f} [{c['auroc'][0]:.4f},{c['auroc'][1]:.4f}]  "
              f"Brier {p['brier_conventional_binary']:.4f}  ECE {p['ece_10_bin']:.4f} [{c['ece_10_bin'][0]:.4f},{c['ece_10_bin'][1]:.4f}]")
    print("\n== paired differences (A - B), accuracy and ECE ==")
    for label, v in comp.items():
        ac, ec, au = v["metrics"]["accuracy"], v["metrics"]["ece_10_bin"], v["metrics"]["auroc"]
        print(f"{label:52s} acc {100*ac['diff']:+.2f} [{100*ac['ci95'][0]:+.2f},{100*ac['ci95'][1]:+.2f}] p={ac['p_two_sided']:.3f} | "
              f"ECE {ec['diff']:+.4f} [{ec['ci95'][0]:+.4f},{ec['ci95'][1]:+.4f}] | AUROC {au['diff']:+.4f} [{au['ci95'][0]:+.4f},{au['ci95'][1]:+.4f}]")
    print("\n== extra ==")
    print(json.dumps(extra, indent=1))


if __name__ == "__main__":
    main()
