#!/usr/bin/env python3
"""Candidate-pool ranking metrics from the saved per-pair predictions.

    python compute_rank_stats.py <DIR_SMALL> <DIR_MAIN> \
        <DIR_OVERNIGHT> <DIR_POST> [--boot 10000]

For each test patient, the judged-relevant trials (eligible and excluded; irrelevant trials are absent from this
binary dataset) are ranked by P(eligible), with eligible as the relevant class. Metrics are computed per patient and
averaged over the 50 patients: precision@10, nDCG@10 (binary gains), mean reciprocal rank of the first eligible trial,
and average precision. Intervals resample patients (paired across systems). This is reranking inside the judged pool,
not full-collection TREC retrieval. Writes rank_stats.json.
"""
import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("TRIALMATCH_OUT", "."))
sys.path.insert(0, str(HERE.parent))
import analyze_predictions as ap  # noqa: E402

SEED = 20261005
K = 10
# system -> prediction files identified by (directory key, file name relative to that directory).
# Multi-seed probabilities are averaged before ranking.
SYSTEMS = {
    "TrialJev 9B (3 seeds)": [("cl", "run1-q35-9b-base__step-400__eval-test-preds.jsonl"),
                                ("ov", "run1-q35-9b-base-s1__step-400__eval-test-preds.jsonl"),
                                ("ov", "run1-q35-9b-base-s2__step-1000__eval-test-preds.jsonl")],
    "Control 9B (3 seeds)": [("cl", "ctl-q35-9b__step-400__eval-test-preds.jsonl"),
                             ("ov", "ctl-q35-9b-s1__step-400__eval-test-preds.jsonl"),
                             ("ov", "ctl-q35-9b-s2__step-200__eval-test-preds.jsonl")],
    "TrialJev 27B": [("cl", "run1-q35-27b-base__step-400__eval-test-preds.jsonl")],
    "Control 27B": [("cl", "ctl-q35-27b__step-400__eval-test-preds.jsonl")],
    "TrialJev Qwen3.8-27B": [("cl", "run1-q38-27b-base__step-400__eval-test-preds.jsonl")],
    "Control Qwen3.8-27B": [("ov", "ctl-q38-27b__step-200__eval-test-preds.jsonl")],
    "Zero-shot letters 27B (no adapter)": [("pz", "runs/gen_zsl/zsl-27b/preds.jsonl")],
}
PAIRS = [("TrialJev 9B (3 seeds)", "Control 9B (3 seeds)"), ("TrialJev 27B", "Control 27B"),
         ("TrialJev Qwen3.8-27B", "Control Qwen3.8-27B"), ("Control 27B", "Zero-shot letters 27B (no adapter)")]
METRICS = ["P@10", "nDCG@10", "MRR", "MAP"]


def load_probs(dirs, key, name):
    path = dirs[key] / name
    if key == "pz":   # generation-style rows with p_true
        rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        found = [r["meta"]["id"] for r in rows]
        if len(set(found)) != len(found):
            raise ValueError(f"duplicate pair ids in {path}")
        return {r["meta"]["id"]: (int(r["gold"]), float(r["p_true"]), str(r["meta"]["patient_id"])) for r in rows}
    d = ap.load_predictions(path)
    return {i: (int(g), float(p), str(pt)) for i, g, p, pt in zip(d["ids"], d["gold"], d["prob"], d["patients"])}


def check_same_ids(reference_ids, found_ids, name):
    """Every system must score exactly the same pairs; reject missing or extra ids."""
    missing, extra = set(reference_ids) - set(found_ids), set(found_ids) - set(reference_ids)
    if missing or extra:
        raise ValueError(f"{name}: {len(missing)} pair ids missing and {len(extra)} unexpected")


def patient_metrics(gold, score):
    order = np.argsort(-score, kind="mergesort")
    rel = gold[order].astype(float)
    n_rel = rel.sum()
    k = min(K, len(rel))
    p_at_k = rel[:k].mean()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (rel[:k] * disc).sum()
    ideal = np.sort(gold.astype(float))[::-1][:k]
    idcg = (ideal * disc[:len(ideal)]).sum()
    ndcg = dcg / idcg if idcg > 0 else 0.0
    first = np.flatnonzero(rel)
    mrr = 1.0 / (first[0] + 1) if len(first) else 0.0
    prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    ap_ = (prec * rel).sum() / n_rel if n_rel > 0 else 0.0
    return [p_at_k, ndcg, mrr, ap_]


def main():
    ar = argparse.ArgumentParser()
    for n in ("small_runs", "cluster", "overnight", "post"):
        ar.add_argument(n)
    ar.add_argument("--boot", type=int, default=10000)
    a = ar.parse_args()
    dirs = {"ai": Path(a.small_runs), "cl": Path(a.cluster), "ov": Path(a.overnight), "pz": Path(a.post)}

    ids = None
    mean_probs, gold, pat = {}, None, None
    for sysname, files in SYSTEMS.items():
        loaded = [load_probs(dirs, k, f) for k, f in files]
        if ids is None:
            ids = sorted(loaded[0])
            gold = np.array([loaded[0][i][0] for i in ids])
            pat = np.array([loaded[0][i][2] for i in ids])
        for ld in loaded:
            check_same_ids(ids, ld, sysname)
            assert (np.array([ld[i][0] for i in ids]) == gold).all(), sysname
            assert (np.array([ld[i][2] for i in ids]) == pat).all(), sysname
        mean_probs[sysname] = np.mean([[ld[i][1] for i in ids] for ld in loaded], axis=0)
    patients = sorted(set(pat))
    idx = {p: np.flatnonzero(pat == p) for p in patients}

    per = {}
    for sysname, prob in mean_probs.items():
        per[sysname] = np.array([patient_metrics(gold[idx[p]], prob[idx[p]]) for p in patients])
    # random ordering reference (expected value by Monte Carlo, 200 shuffles per patient)
    rng = np.random.default_rng(SEED)
    rand = np.zeros((len(patients), 4))
    for j, p in enumerate(patients):
        g = gold[idx[p]]
        rand[j] = np.mean([patient_metrics(g, rng.random(len(g))) for _ in range(200)], axis=0)
    per["Random order (expected)"] = rand

    boots = rng.integers(0, len(patients), size=(a.boot, len(patients)))
    out = {"n_patients": len(patients), "K": K, "bootstrap": {"resamples": a.boot, "seed": SEED, "unit": "patient"},
           "note": ("ranking of eligible vs excluded trials inside the judged-relevant pool; not full-collection "
                    "retrieval; multi-seed probabilities are averaged before ranking"),
           "systems": {}, "paired": {}}
    print(f"{len(patients)} patients, {len(gold)} pairs; mean pool {len(gold) / len(patients):.0f}; mean eligible fraction per patient {np.mean([gold[idx[p]].mean() for p in patients]):.3f}")
    print(f"{'system':40s} " + "  ".join(f"{m:>22s}" for m in METRICS))
    for sysname, m in per.items():
        mean = m.mean(0)
        bs = m[boots].mean(1)
        lo, hi = np.quantile(bs, [0.025, 0.975], axis=0)
        out["systems"][sysname] = {METRICS[i]: {"mean": float(mean[i]), "ci95": [float(lo[i]), float(hi[i])]} for i in range(4)}
        print(f"{sysname:40s} " + "  ".join(f"{mean[i]:.3f} [{lo[i]:.3f},{hi[i]:.3f}]" for i in range(4)))
    print("\npaired differences (A - B), patient bootstrap")
    for A, B in PAIRS:
        d = per[A] - per[B]
        mean = d.mean(0)
        bs = d[boots].mean(1)
        lo, hi = np.quantile(bs, [0.025, 0.975], axis=0)
        out["paired"][f"{A} - {B}"] = {METRICS[i]: {"diff": float(mean[i]), "ci95": [float(lo[i]), float(hi[i])]} for i in range(4)}
        print(f"{A} - {B}: " + "  ".join(f"{METRICS[i]} {mean[i]:+.3f} [{lo[i]:+.3f},{hi[i]:+.3f}]" for i in range(4)))
    (OUT / "rank_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote rank_stats.json")


if __name__ == "__main__":
    main()
