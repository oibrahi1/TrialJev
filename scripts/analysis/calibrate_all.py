#!/usr/bin/env python3
"""Development-fitted decision correction for every system that returns a score, applied identically to all of them.

    python calibrate_all.py --jev_dir runs/jev_api --dev_dirs DEV1 [DEV2 ...] --test_dirs T1 [T2 ...] --suite_dirs S1 [S2 ...] \
        [--boot 10000] [--out calibrate_all.json]

For each system (TrialJev, the A/B-logit control, the fine-tuned LLM's first-token score, and hosted Jev) a temperature and a
class-prior weight are fitted on the TREC 2021 DEVELOPMENT scores only, by MedJev's own recipe (temperature 0.5 to 3.0, prior
mix 0 to 1, chosen by development accuracy), frozen, and applied once to the saved TEST scores. Multi-seed configurations are
fitted per seed. The fine-tuned LLM's headline accuracy is its written verdict; its corrected row thresholds the first-token
score, a different decision rule. Prompted and explain-then-answer LLMs return no score and cannot be corrected.
Raw accuracy for TrialJev and the control is the evaluator's own saved decision (an exact 0.5 tie counts as excluded; the
control has such ties), for the fine-tuned LLM the first-token score at 0.5, and for Jev the probability at least 0.5 (as in the
paper). Intervals resample the 50 test patients (paired across systems); unadjusted.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

SEED = 20261001
GRID_T = np.arange(0.5, 3.05, 0.1)
MIX = (0.0, 0.25, 0.5, 0.75, 1.0)

# (label, size, [(test file stem, dev run dir, dev step)]) ; test stems are the saved test predictions
POINTER = {
    "0.8B fresh": ["run1-q35-0.8b-base__step-1000", "run1-q35-0.8b-base-s1__step-100", "run1-q35-0.8b-base-s2__step-500"],
    "0.8B clinical init": ["run1-q35-0.8b-clinit__step-400", "run1-q35-0.8b-clinit-s1__step-500", "run1-q35-0.8b-clinit-s2__step-500"],
    "9B": ["run1-q35-9b-base__step-400", "run1-q35-9b-base-s1__step-400", "run1-q35-9b-base-s2__step-1000"],
    "27B": ["run1-q35-27b-base__step-400"],
    "Qwen3.8-27B": ["run1-q38-27b-base__step-400"],
}
CONTROL = {
    "0.8B fresh": ["ctl-q35-0.8b-s0__step-200"],
    "9B": ["ctl-q35-9b__step-400", "ctl-q35-9b-s1__step-400", "ctl-q35-9b-s2__step-200"],
    "27B": ["ctl-q35-27b__step-400"],
    "Qwen3.8-27B": ["ctl-q38-27b__step-200"],
}
LLM = {"0.8B fresh": ["sft-q35-0.8b-s0"], "9B": ["sft-q35-9b-s0", "sft-q35-9b-s1", "sft-q35-9b-s2"],
       "27B": ["sft-q35-27b-s0"], "Qwen3.8-27B": ["sft-q38-27b-s0"]}


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    order = s.argsort(kind="mergesort"); ranks = np.empty(len(s)); sv = s[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def apply(P, T, w):                      # MedJev calibrate_jev.apply
    Q = np.clip(P, 1e-6, 1) ** (1.0 / T) * w
    return Q / Q.sum(1, keepdims=True)


def fit(P, y):                           # MedJev calibrate_jev.fit (temperature grid x prior mix, by development accuracy)
    best = (None, None, -1)
    prior = np.bincount(y, minlength=2) / len(y)
    pred_prior = P.mean(0)
    for T in GRID_T:
        for mix in MIX:
            w = (prior / np.maximum(pred_prior, 1e-6)) ** mix
            acc = (apply(P, T, w).argmax(1) == y).mean()
            if acc > best[2]:
                best = (T, w, acc)
    return best


def correct(p_dev, y_dev, p_test):
    """-> (corrected P(eligible), eligible decision, fit info). The decision is the argmax of the corrected two-way scores, exactly
    as in the development fit, so an exact tie counts as excluded in both places (a tie survives the correction when the fitted
    prior weights are 1, e.g. the 9B control seed 2)."""
    T, w, dev_acc = fit(np.stack([1 - p_dev, p_dev], 1), y_dev.astype(int))
    Q = apply(np.stack([1 - p_test, p_test], 1), T, w)
    return Q[:, 1], Q.argmax(1) == 1, {"T": float(T), "w_false": float(w[0]), "w_true": float(w[1]), "dev_accuracy_corrected": 100 * float(dev_acc)}


def find(dirs, rel):
    for d in dirs:
        p = Path(d) / rel
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev_dir", required=True)
    ap.add_argument("--dev_dirs", nargs="+", required=True, help="extracted development-score archives (hold runs/...)")
    ap.add_argument("--test_dirs", nargs="+", required=True, help="folders holding <run>__step-N__eval-test-preds.jsonl")
    ap.add_argument("--suite_dirs", nargs="+", required=True, help="folders holding runs/gen_sft/<run>/preds.jsonl (fine-tuned LLM test)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="calibrate_all.json")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    # hosted Jev on the same recipe
    jt = {r["id"]: r for r in jl(Path(a.jev_dir) / "test.jsonl") if r.get("p_true") is not None}
    jd = {r["id"]: r for r in jl(Path(a.jev_dir) / "development.jsonl") if r.get("p_true") is not None}
    ids = sorted(jt)
    y = np.array([jt[i]["gold"] for i in ids]); pj = np.array([jt[i]["p_true"] for i in ids])
    dids = sorted(jd); yd = np.array([jd[i]["gold"] for i in dids]); pjd = np.array([jd[i]["p_true"] for i in dids])
    qj, jdec, jinfo = correct(pjd, yd, pj)
    jev_raw = (pj >= 0.5) == y
    jev_cor = jdec == y
    print(f"Jev raw {100 * jev_raw.mean():.2f}  corrected {100 * jev_cor.mean():.2f}  (T {jinfo['T']:.1f}, dev {jinfo['dev_accuracy_corrected']:.2f})")

    groups = None
    out = {"jev": {"raw": 100 * jev_raw.mean(), "corrected": 100 * jev_cor.mean(), **jinfo, "auroc": auroc(y, pj)}, "rows": []}

    def load_test_pointer_like(stem):
        p = find(a.test_dirs, f"{stem}__eval-test-preds.jsonl")
        if p is None:
            return None
        rows = {r["meta"]["id"]: r for r in jl(p)}
        return rows

    def run_group(kind, label, stems):
        nonlocal groups
        seeds = []
        for stem in stems:
            if kind == "llm":
                tp = find(a.suite_dirs, f"runs/gen_sft/{stem}/preds.jsonl")
                dp = find(a.dev_dirs, f"runs/gen_sft_dev/{stem}/preds.jsonl")
                if tp is None or dp is None:
                    return None
                T = {r["meta"]["id"]: r for r in jl(tp)}; D = {r["meta"]["id"]: r for r in jl(dp)}
                pt = np.array([T[i]["p_true"] for i in ids]); written = np.array([bool(T[i]["valid"]) and T[i]["pred"] == T[i]["gold"] for i in ids])
                saved = None
                pdv = np.array([D[i]["p_true"] for i in sorted(D)]); ydv = np.array([D[i]["gold"] for i in sorted(D)])
                if groups is None:
                    groups = [str(T[i]["meta"]["patient_id"]) for i in ids]
            else:
                T = load_test_pointer_like(stem)
                run, step = stem.split("__")
                dp = None
                for d in a.dev_dirs:
                    cand = Path(d) / "runs" / run / step / "eval-development-preds.jsonl"
                    if cand.exists():
                        dp = cand
                if T is None or dp is None:
                    return None
                D = {r["meta"]["id"]: r for r in jl(dp)}
                pt = np.array([T[i]["probs"]["true"] for i in ids]); written = None
                saved = np.array([T[i]["pred"] in (True, "true") for i in ids]) == y   # the evaluator's own decision (ties count as excluded)
                pdv = np.array([D[i]["probs"]["true"] for i in sorted(D)]); ydv = np.array([D[i]["gold"] in (True, "true") for i in sorted(D)])
                if groups is None:
                    groups = [str(T[i]["meta"]["patient_id"]) for i in ids]
            q, dec, info = correct(pdv, ydv, pt)
            seeds.append({"raw": saved if saved is not None else (pt >= .5) == y, "cor": dec == y, "written": written, "auroc": auroc(y, pt), "info": info})
        return seeds

    def boot_diff(ca, cb):
        ug = sorted(set(groups)); idx = {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in ug}
        d = []
        for _ in range(a.boot):
            pick = rng.integers(0, len(ug), len(ug))
            ix = np.concatenate([idx[ug[k]] for k in pick])
            d.append(ca[ix].mean() - cb[ix].mean())
        return 100 * (ca.mean() - cb.mean()), 100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5)

    print(f"{'system':32s} {'n':>2s} {'raw':>12s} {'corrected':>12s} | corrected minus Jev corrected | raw minus Jev raw")
    for kind, name, table in (("pointer", "TrialJev", POINTER), ("control", "A/B-logit control", CONTROL), ("llm", "Fine-tuned LLM score", LLM)):
        for label, stems in table.items():
            seeds = run_group(kind, label, stems)
            if seeds is None:
                print(f"{name + ' ' + label:32s} -- missing files, skipped")
                continue
            raw = [100 * s["raw"].mean() for s in seeds]; cor = [100 * s["cor"].mean() for s in seeds]
            cm = np.mean([s["cor"].astype(float) for s in seeds], axis=0); rm = np.mean([s["raw"].astype(float) for s in seeds], axis=0)
            dc = boot_diff(cm, jev_cor.astype(float)); dr = boot_diff(rm, jev_raw.astype(float))
            row = {"system": name, "config": label, "n_seeds": len(seeds), "raw_mean": statistics.mean(raw), "corrected_mean": statistics.mean(cor),
                   "raw_sd": statistics.stdev(raw) if len(raw) > 1 else None, "corrected_sd": statistics.stdev(cor) if len(cor) > 1 else None,
                   "per_seed_raw": raw, "per_seed_corrected": cor, "fits": [s["info"] for s in seeds], "auroc_mean": statistics.mean(s["auroc"] for s in seeds),
                   "corrected_minus_jev_corrected": dict(zip(("d", "lo", "hi"), dc)), "raw_minus_jev_raw": dict(zip(("d", "lo", "hi"), dr))}
            if kind == "llm":
                wr = [100 * s["written"].mean() for s in seeds]
                row["written_verdict_mean"] = statistics.mean(wr); row["per_seed_written_verdict"] = wr
            out["rows"].append(row)
            sd = lambda v: f"±{statistics.stdev(v):.2f}" if len(v) > 1 else ""
            print(f"{name + ' ' + label:32s} {len(seeds):2d} {statistics.mean(raw):6.2f}{sd(raw):>6s} {statistics.mean(cor):6.2f}{sd(cor):>6s} | "
                  f"{dc[0]:+6.2f} [{dc[1]:+6.2f}, {dc[2]:+6.2f}] | {dr[0]:+6.2f} [{dr[1]:+6.2f}, {dr[2]:+6.2f}]")
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
