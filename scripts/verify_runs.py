#!/usr/bin/env python3
"""Read-only audit of finished runs: no GPU, no model loading, never re-scores anything.

    python scripts/verify_runs.py [runs_dir] [--out runs/verification.json]

For every run it (1) re-derives the best checkpoint from the training log, (2) reads the
checkpoint's recorded base model / revision / dtype, (3) recomputes accuracy, macro-F1,
Brier, ECE and class balance from the per-pair prediction file and compares them with what
the evaluator wrote in its report, and (4) checks the prediction file itself (row count,
duplicates, patients, gold counts, pred == argmax of probs). It also hashes the code and
data files. Zero-shot probes (runs/prefinetune-*/rows.jsonl) are recomputed the same way.
Standard library only; torch is used if importable to read head.pt.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

EXPECTED = {   # SHA-256 of the files behind every reported number
    "trialmatch/train.py": "ebe0d4553bbf4059681f8b41c0d4ebe9f4f07f05c7cf43174e0d0a944b260c6a",
    "trialmatch/model.py": "9de045f3c4f6425832b4394f91e40e9c8c2895e21a5e935d1df9bdeb726a844f",
    "data/tm-v1-b4096/train.jsonl": "fe0bcb356fcdf0a3e54e9727f62456d163dfa62b46c87da21e8e0a7674934411",
    "data/tm-v1-b4096/development.jsonl": "222d12f11aa089ae305b75d8b02cb8280158e425ade91868517d49a7f12fe3cc",
    "data/tm-v1-b4096/test.jsonl": "77644c6d88864a434e30d62a439c8c172ddef16ead5fbb880263dccff5e9f984",
    "data/splits.json": "35767cdb7b0ac2c6f7e2fbfc1b759700fc202218e980e88620390f6af831e575",
}
TEST_GOLD = {"true": 3939, "false": 3036}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def metrics(gold, pred, p_true):
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    f1s = []
    for c in ("true", "false"):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    y = [1.0 if g == "true" else 0.0 for g in gold]
    brier_two_class = sum(2 * (p - t) ** 2 for p, t in zip(p_true, y)) / n   # evaluator's convention
    conf = [max(p, 1 - p) for p in p_true]
    correct = [g == q for g, q in zip(gold, pred)]
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10      # same bins as trialmatch.evaluate.ece: [lo, hi), last bin closed
        idx = [i for i, c in enumerate(conf) if c >= lo and (c < hi or (b == 9 and c <= hi))]
        if idx:
            ece += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return {"n": n, "accuracy": acc, "macro_f1": sum(f1s) / 2, "brier_two_class": brier_two_class,
            "ece_10bin": ece, "pred_eligible_fraction": sum(p == "true" for p in pred) / n}


def check_preds(preds_path, report_path, expect_test):
    rows = jl(preds_path)
    out, problems = {"preds_file": preds_path, "rows": len(rows)}, []
    ids = [(r.get("meta") or {}).get("id") for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    p_true = [float(r["probs"]["true"]) for r in rows]
    if any((p > 0.5) != (q == "true") and abs(p - 0.5) > 1e-9 for p, q in zip(p_true, pred)):
        problems.append("pred is not argmax of probs")
    out["patients"] = len({(r.get("meta") or {}).get("patient_id") for r in rows})
    out["gold_counts"] = {k: gold.count(k) for k in ("true", "false")}
    m = metrics(gold, pred, p_true)
    out["recomputed"] = m
    if expect_test:
        if len(rows) != 6975:
            problems.append(f"expected 6975 rows, found {len(rows)}")
        if out["gold_counts"] != TEST_GOLD:
            problems.append(f"gold counts {out['gold_counts']} != {TEST_GOLD}")
        if out["patients"] != 50:
            problems.append(f"expected 50 patients, found {out['patients']}")
    if os.path.exists(report_path):
        rep = json.load(open(report_path, encoding="utf-8"))
        out["report"] = {k: rep.get(k) for k in ("run", "split", "dtype", "records", "micro_accuracy", "brier",
                                                 "states_truncated", "latency_ms_per_record")}
        if abs(rep["micro_accuracy"] - m["accuracy"]) > 6e-5:
            problems.append(f"report accuracy {rep['micro_accuracy']} != recomputed {m['accuracy']:.4f}")
        if abs(rep["brier"] - m["brier_two_class"]) > 6e-5:
            problems.append(f"report brier {rep['brier']} != recomputed {m['brier_two_class']:.4f}")
        pq = next(iter(rep["per_question"].values()))
        if abs(pq["macro_f1"] - m["macro_f1"]) > 6e-5:
            problems.append(f"report macro_f1 {pq['macro_f1']} != recomputed {m['macro_f1']:.4f}")
        if abs(pq["ece"] - m["ece_10bin"]) > 6e-5:
            problems.append(f"report ece {pq['ece']} != recomputed {m['ece_10bin']:.4f}")
    else:
        problems.append("no report json next to preds")
    out["problems"] = problems
    return out


def head_meta(path):
    try:
        import torch
        m = torch.load(path, map_location="cpu", weights_only=False)
        keep = ("base", "base_revision", "weights_dtype", "lora", "head_dim", "max_state", "max_branch")
        meta = {k: m.get(k) for k in keep}
        a = m.get("args") or {}
        meta["args"] = {k: a.get(k) for k in ("lr", "epochs", "batch", "accum", "seed", "init_from", "base_dtype",
                                              "dtype", "data", "val_data") if k in a}
        return meta
    except Exception as e:   # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def audit_run(run):
    res = {"run": run, "problems": []}
    cfg_path = os.path.join(run, "training_config.json")
    log_path = os.path.join(run, "train_log_rank0.jsonl")
    if not (os.path.exists(cfg_path) and os.path.exists(log_path)):
        res["problems"].append("no training_config/train_log (not a training run)")
        return res
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    res["config"] = {k: cfg["args"].get(k) for k in ("base", "base_revision", "base_dtype", "init_from", "lr",
                                                       "epochs", "batch", "accum", "seed", "max_state",
                                                       "max_branch", "data", "val_data")}
    res["optimizer_steps"] = cfg.get("optimizer_steps")
    vals = [e for e in jl(log_path) if "val/loss" in e]
    res["validation"] = [{"step": e["step"], "val_loss": round(e["val/loss"], 4),
                          "val_acc": round(e.get("val/accuracy", float("nan")), 4)} for e in vals]
    if not vals:
        res["problems"].append("no validation entries")
        return res
    best = min(vals, key=lambda e: e["val/loss"])
    res["best_step_by_dev_loss"] = best["step"]
    ckpt = os.path.join(run, f"step-{best['step']}")
    if not os.path.isdir(ckpt):
        res["problems"].append(f"best checkpoint {ckpt} missing")
        return res
    res["checkpoint"] = head_meta(os.path.join(ckpt, "head.pt"))
    res["finished"] = os.path.exists(os.path.join(run, "training_metrics.json"))
    for split in ("development", "test"):
        rp, pp = (os.path.join(ckpt, f"eval-{split}{s}") for s in (".json", "-preds.jsonl"))
        if os.path.exists(pp):
            res[f"eval_{split}"] = check_preds(pp, rp, split == "test")
            res["problems"] += [f"[{split}] {p}" for p in res[f"eval_{split}"]["problems"]]
        else:
            res[f"eval_{split}"] = "missing"
    other = sorted(glob.glob(os.path.join(run, "step-*", "eval-test-preds.jsonl")))
    others = [p for p in other if os.path.dirname(p) != ckpt]
    if others:
        res["problems"].append("test predictions exist for a NON-selected checkpoint: " + ", ".join(others))
    return res


def audit_probe(d):
    rep_p, rows_p = os.path.join(d, "report.json"), os.path.join(d, "rows.jsonl")
    if not (os.path.exists(rep_p) and os.path.exists(rows_p)):
        return None
    rep, rows = json.load(open(rep_p, encoding="utf-8")), jl(rows_p)
    gold = ["true" if r["label"] else "false" for r in rows]
    pred = ["true" if r["pred"] else "false" for r in rows]
    m = metrics(gold, pred, [r["p_eligible"] for r in rows])
    probs = []
    if len(rows) != rep.get("n"):
        probs.append(f"rows {len(rows)} != report n {rep.get('n')}")
    if abs(rep["accuracy"] - m["accuracy"]) > 6e-5:
        probs.append(f"report accuracy {rep['accuracy']} != recomputed {m['accuracy']:.4f}")
    if m["pred_eligible_fraction"] > 0.97 or m["pred_eligible_fraction"] < 0.03:
        probs.append(f"degenerate: predicts eligible {m['pred_eligible_fraction']:.3f} of the time (position bias)")
    return {"dir": d, "model": rep.get("model"), "prompt": rep.get("prompt_mode"),
            "option_order": rep.get("option_order", "excluded_first (default, before --order existed)"),
            "n": len(rows), "recomputed": m, "problems": probs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", nargs="?", default="runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = {}
    for rel, want in EXPECTED.items():
        got = sha256(rel) if os.path.exists(rel) else None
        files[rel] = "OK" if got == want else ("MISSING" if got is None else f"MISMATCH {got}")
    runs = [audit_run(d) for d in sorted(glob.glob(os.path.join(a.runs_dir, "*"))) if os.path.isdir(d)
            and not os.path.basename(d).startswith(("prefinetune", "smoke"))]
    runs = [r for r in runs if "config" in r]
    probes = [p for p in (audit_probe(d) for d in sorted(glob.glob(os.path.join(a.runs_dir, "prefinetune*")))) if p]

    print("== file hashes ==")
    for k, v in files.items():
        print(f"  {v:9s} {k}")
    print("\n== fine-tuned runs ==")
    print(f"{'run':28s} {'base':26s} {'best':>5s} {'dev_acc':>8s} {'TEST acc':>9s} {'macroF1':>8s} {'brier':>6s} {'ECE':>6s} {'n':>5s} {'pat':>4s}  problems")
    for r in runs:
        t, d = r.get("eval_test"), r.get("eval_development")
        tm = t["recomputed"] if isinstance(t, dict) else None
        dv = next((v["val_acc"] for v in r.get("validation", []) if v["step"] == r.get("best_step_by_dev_loss")), None)
        base = f"{os.path.basename(str(r['config']['base']))}@{str((r.get('checkpoint') or {}).get('base_revision'))[:7]}"
        print(f"{os.path.basename(r['run']):28s} {base:26s} {str(r.get('best_step_by_dev_loss')):>5s} "
              f"{dv if dv is not None else float('nan'):8.4f} "
              + (f"{tm['accuracy']:9.4f} {tm['macro_f1']:8.4f} {tm['brier_two_class']:6.3f} {tm['ece_10bin']:6.4f} {tm['n']:5d} {t['patients']:4d}"
                 if tm else f"{'-':>9s} {'-':>8s} {'-':>6s} {'-':>6s} {'-':>5s} {'-':>4s}")
              + f"  {len(r['problems']) or 'none'}")
    print("\n== zero-shot probes ==")
    for p in probes:
        m = p["recomputed"]
        print(f"  {os.path.basename(p['dir']):34s} {p['prompt']:5s} acc={m['accuracy']:.4f} predElig={m['pred_eligible_fraction']:.3f} "
              f"brier={m['brier_two_class'] / 2:.4f}(conventional) n={p['n']}  problems={len(p['problems'])}")
    print("\n== problems ==")
    n_prob = 0
    for r in runs:
        for p in r["problems"]:
            n_prob += 1
            print(f"  {os.path.basename(r['run'])}: {p}")
    for p in probes:
        for x in p["problems"]:
            n_prob += 1
            print(f"  {os.path.basename(p['dir'])}: {x}")
    for k, v in files.items():
        if v != "OK":
            n_prob += 1
            print(f"  file {k}: {v}")
    print("  none" if not n_prob else f"  ({n_prob} total)")
    out = a.out or os.path.join(a.runs_dir, "verification.json")
    json.dump({"files": files, "runs": runs, "probes": probes}, open(out, "w", encoding="utf-8"), indent=2, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
