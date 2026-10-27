#!/usr/bin/env python3
"""Per-checkpoint metrics for the Hugging Face model cards, computed from the saved per-pair test predictions (no model run).

    python compute_release_metrics.py --src CKPT_DIR --suite_dirs DIR1 DIR2 --out release_metrics.json

TrialJev and A/B-logit checkpoints: the evaluator's own saved decision for accuracy and macro-F1 (an exact 0.5 tie counts as excluded),
AUROC from P(eligible), binary Brier score and 10-bin ECE as in the manuscript. Fine-tuned LLM checkpoints: accuracy and macro-F1 of the
written verdict (strict: an unparsable answer counts as wrong), AUROC from the first-token two-way score; Brier and ECE are not reported
because the written verdict has no probability. All on the 6,975 TREC 2022 test pairs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from prepare_hf_release import REG, family_of  # noqa: E402


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


def ece(y, p, pred, bins=10):
    conf = np.maximum(p, 1 - p); acc = (pred == y).astype(float)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--suite_dirs", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for run, (sub, base, step, desc, paper_acc) in REG.items():
        fam = family_of(run)
        if fam == "TrialJev-FineTunedLLM":
            p = next((Path(d) / "runs" / "gen_sft" / run / "preds.jsonl" for d in a.suite_dirs if (Path(d) / "runs" / "gen_sft" / run / "preds.jsonl").exists()), None)
            assert p, f"no fine-tuned LLM predictions for {run}"
            rows = jl(p)
            y = np.array([r["gold"] for r in rows], bool)
            pred_ok = np.array([bool(r["valid"]) and r["pred"] == r["gold"] for r in rows])
            pred = np.where([bool(r["valid"]) for r in rows], np.array([bool(r["pred"]) for r in rows]), ~y)   # strict: invalid counts as the opposite class
            m = {"n": len(rows), "accuracy": 100 * float(pred_ok.mean()), "macro_f1": 100 * macro_f1(y, pred),
                 "auroc": auroc(y, np.array([r["p_true"] for r in rows])), "brier": None, "ece": None,
                 "invalid_rate": float(np.mean([not r["valid"] for r in rows]))}
        else:
            rows = jl(Path(a.src) / run / f"step-{step}" / "eval-test-preds.jsonl")
            y = np.array([r["gold"] in (True, "true") for r in rows], bool)
            p = np.array([r["probs"]["true"] for r in rows], float)
            pred = np.array([r["pred"] in (True, "true") for r in rows], bool)
            m = {"n": len(rows), "accuracy": 100 * float((pred == y).mean()), "macro_f1": 100 * macro_f1(y, pred), "auroc": auroc(y, p),
                 "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p, pred), "invalid_rate": 0.0}
        assert m["n"] == 6975, (run, m["n"])
        assert abs(m["accuracy"] - paper_acc) < 0.011, (run, m["accuracy"], paper_acc)
        out[run] = m
        print(f"{run:26s} acc {m['accuracy']:.2f} F1 {m['macro_f1']:.1f} AUROC {m['auroc']:.3f} Brier {m['brier'] if m['brier'] is None else round(m['brier'], 3)} ECE {m['ece'] if m['ece'] is None else round(m['ece'], 3)}")
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
