#!/usr/bin/env python3
"""Score the hosted-Jev API answers (scripts/run_jev_trial.py) against TREC 2022, and compare with TrialJev and the writer.

    python score_jev_api.py --jev_dir runs/jev_api --data DIR --pointer_dir DIR1 \
        --suite_dir DIR2 [--boot 10000] [--out jev_api_stats.json]

Raw Jev: P(eligible) thresholded at 0.5. Calibrated Jev follows MedJev's recipe for the hosted model: a temperature T and a
class-prior weight fitted on the DEVELOPMENT split only (maximising development accuracy) and frozen before the test split
is scored. Intervals resample the 50 test patients (paired across systems; unadjusted). Hosted-API latency is client-side
wall time (network plus the hosted service) and is never put in a ratio with local GPU latency.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

SEED = 20261001
T_GRID = [round(0.5 + 0.1 * i, 1) for i in range(26)]
MIX_GRID = [0, 0.25, 0.5, 0.75, 1.0]


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


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    conf = np.maximum(p, 1 - p); pred = (p >= .5).astype(float); acc = (pred == y).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def macro_f1(y, pred):
    f = []
    for c in (True, False):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def metrics(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, float); pred = p >= 0.5
    return {"accuracy": 100 * float((pred == y).mean()), "macro_f1": 100 * macro_f1(y, pred), "auroc": auroc(y, p),
            "brier_binary": float(np.mean((p - y) ** 2)), "ece": ece(y, p), "predicted_eligible": 100 * float(pred.mean()),
            "eligible_recall": 100 * float(pred[y].mean()), "excluded_recall": 100 * float((~pred[~y]).mean())}


def calibrate(p, T, w_true, w_false):
    a = np.clip(p, 1e-6, 1 - 1e-6) ** (1 / T) * w_true
    b = (1 - np.clip(p, 1e-6, 1 - 1e-6)) ** (1 / T) * w_false
    return a / (a + b)


def fit_calibration(y_dev, p_dev, f_true):
    best, mean_p = None, float(np.mean(p_dev))
    for mix in MIX_GRID:
        wt, wf = (f_true / mean_p) ** mix, ((1 - f_true) / (1 - mean_p)) ** mix
        for T in T_GRID:
            acc = float(((calibrate(p_dev, T, wt, wf) >= .5) == y_dev).mean())
            if best is None or acc > best[0]:
                best = (acc, T, mix, wt, wf)
    return {"dev_accuracy": 100 * best[0], "T": best[1], "mix": best[2], "w_true": best[3], "w_false": best[4]}


def boot(fn, groups, n, rng):
    """fn(index array) -> statistic difference; resample patients."""
    ug = sorted(set(groups)); idx = {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in ug}
    d = []
    for _ in range(n):
        pick = rng.integers(0, len(ug), len(ug))
        d.append(fn(np.concatenate([idx[ug[k]] for k in pick])))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def load_jev(path):
    rows = {}
    for r in jl(path):
        if r.get("p_true") is not None and "error" not in r:
            rows[r["id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev_dir", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--pointer_dir", required=True); ap.add_argument("--suite_dir", required=True)
    ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--out", default="jev_api_stats.json")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    jd = Path(a.jev_dir); out = {}

    test = load_jev(jd / "test.jsonl"); dev = load_jev(jd / "development.jsonl")
    ids = sorted(test); y = np.array([test[i]["gold"] for i in ids]); p = np.array([test[i]["p_true"] for i in ids])
    models = {r["model"] for r in test.values()} | {r["model"] for r in dev.values()}
    print(f"test pairs scored {len(ids)}; dev pairs {len(dev)}; model strings {sorted(models)}; eligible fraction {y.mean():.4f}")
    out["n_test"], out["n_dev"], out["models"] = len(ids), len(dev), sorted(models)

    raw = metrics(y, p); out["jev_raw"] = raw
    print("Jev raw (0.5 threshold):", {k: round(v, 3) for k, v in raw.items()})

    # calibration fitted on development only
    dids = sorted(dev); yd = np.array([dev[i]["gold"] for i in dids]); pd_ = np.array([dev[i]["p_true"] for i in dids])
    train = jl(Path(a.data) / "train.jsonl")
    f_true = float(np.mean([str(next(iter(r["questions"].values()))["label"]).lower() in ("true", "1") for r in train]))
    cal = fit_calibration(yd, pd_, f_true); out["calibration"] = {**cal, "train_eligible_fraction": f_true}
    pc = calibrate(p, cal["T"], cal["w_true"], cal["w_false"])
    craw = metrics(y, pc); out["jev_calibrated"] = craw
    print(f"calibration (dev only): T={cal['T']} mix={cal['mix']} dev accuracy {cal['dev_accuracy']:.2f}")
    print("Jev calibrated:", {k: round(v, 3) for k, v in craw.items()})

    # paired comparison with TrialJev (seed 0, two-pass evaluator) and the fine-tuned writer on the same pairs
    pmap = {"9B": "run1-q35-9b-base", "27B": "run1-q35-27b-base", "Qwen3.8-27B": "run1-q38-27b-base"}
    wmap = {"9B": ("cluster", "sft-q35-9b-s0"), "27B": ("cluster", "sft-q35-27b-s0"), "Qwen3.8-27B": ("cluster", "sft-q38-27b-s0")}
    meta = {i: None for i in ids}
    first = {json.loads(l)["meta"]["id"]: json.loads(l)["meta"]["patient_id"]
             for l in open(Path(a.pointer_dir) / "run1-q35-9b-base__step-400__eval-test-preds.jsonl", encoding="utf-8")}
    groups = [first[i] for i in ids]
    out["paired"] = {}
    for size, run in pmap.items():
        pr = {json.loads(l)["meta"]["id"]: json.loads(l) for l in open(Path(a.pointer_dir) / f"{run}__step-400__eval-test-preds.jsonl", encoding="utf-8")}
        pt = np.array([pr[i]["probs"]["true"] for i in ids]); ptc = (pt >= .5) == y
        wr = {json.loads(l)["meta"]["id"]: json.loads(l) for l in open(Path(a.suite_dir) / wmap[size][0] / "runs" / "gen_sft" / wmap[size][1] / "preds.jsonl", encoding="utf-8")}
        wc = np.array([bool(wr[i]["valid"]) and wr[i]["pred"] == wr[i]["gold"] for i in ids])
        res = {"trialjev_accuracy": 100 * ptc.mean(), "trialjev_auroc": auroc(y, pt), "writer_accuracy": 100 * wc.mean()}
        for name, jc in (("raw", (p >= .5) == y), ("calibrated", (pc >= .5) == y)):
            d = 100 * (ptc.mean() - jc.mean())
            lo, hi = boot(lambda ix, jc=jc: 100 * (ptc[ix].mean() - jc[ix].mean()), groups, a.boot, rng)
            res[f"trialjev_minus_jev_{name}_accuracy"] = {"d": d, "lo": lo, "hi": hi}
        ja = auroc(y, p); d = res["trialjev_auroc"] - ja
        lo, hi = boot(lambda ix: auroc(y[ix], pt[ix]) - auroc(y[ix], p[ix]), groups, min(a.boot, 2000), rng)
        res["trialjev_minus_jev_auroc"] = {"d": d, "lo": lo, "hi": hi}
        out["paired"][size] = res
        print(f"{size}: TrialJev {res['trialjev_accuracy']:.2f} writer {res['writer_accuracy']:.2f} | minus Jev raw "
              f"{res['trialjev_minus_jev_raw_accuracy']['d']:+.2f} [{res['trialjev_minus_jev_raw_accuracy']['lo']:+.2f}, "
              f"{res['trialjev_minus_jev_raw_accuracy']['hi']:+.2f}]; minus Jev calibrated "
              f"{res['trialjev_minus_jev_calibrated_accuracy']['d']:+.2f} [{res['trialjev_minus_jev_calibrated_accuracy']['lo']:+.2f}, "
              f"{res['trialjev_minus_jev_calibrated_accuracy']['hi']:+.2f}]; AUROC TrialJev {res['trialjev_auroc']:.3f} vs Jev {ja:.3f} "
              f"({d:+.3f} [{lo:+.3f}, {hi:+.3f}])")

    # hosted latency on the timing sample
    tim = load_jev(jd / "timing992.jsonl")
    if tim:
        lat = [r["latency_ms"] for r in tim.values()]; el = [r["elapsed_ms"] for r in tim.values()]
        q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]
        ty = np.array([r["gold"] for r in tim.values()]); tp = np.array([r["p_true"] for r in tim.values()])
        out["timing992"] = {"n": len(tim), "latency_ms": {"p50": q(lat, .5), "p90": q(lat, .9), "p99": q(lat, .99), "mean": statistics.mean(lat)},
                            "elapsed_incl_retries_ms": {"p50": q(el, .5), "p99": q(el, .99)}, "accuracy_raw": 100 * float(((tp >= .5) == ty).mean()),
                            "retries_total": sum(r["retries"] for r in tim.values())}
        print("hosted-API latency, 992 pairs, one request at a time:", out["timing992"])
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
