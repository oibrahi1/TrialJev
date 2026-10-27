"""verify_runs.py must pass a self-consistent run and flag a doctored report."""
import json
import random

import numpy as np
import pytest

import verify_runs as vr


def evaluator_ece(conf, correct, bins=10):          # verbatim logic of trialmatch.evaluate.ece
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def make_run(tmp_path, tamper=False):
    rnd = random.Random(3)
    run = tmp_path / "runs" / "run1-fake"
    ck = run / "step-400"
    ck.mkdir(parents=True)
    (run / "training_config.json").write_text(json.dumps({"args": {"base": "Qwen/Fake"}, "optimizer_steps": 600}))
    (run / "train_log_rank0.jsonl").write_text("\n".join(
        json.dumps({"step": s, "val/loss": l, "val/accuracy": 0.7}) for s, l in ((200, .6), (400, .5), (600, .55))))
    gold = ["true"] * 3939 + ["false"] * 3036
    rnd.shuffle(gold)
    rows = []
    for i, g in enumerate(gold):
        p = min(max((0.7 if g == "true" else 0.3) + rnd.gauss(0, .25), .01), .99)
        rows.append({"qid": "eligible", "gold": g, "pred": "true" if p > .5 else "false",
                     "probs": {"false": 1 - p, "true": p}, "meta": {"id": f"t/{i}", "patient_id": str(i % 50)}})
    (ck / "eval-test-preds.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    m = vr.metrics([r["gold"] for r in rows], [r["pred"] for r in rows], [r["probs"]["true"] for r in rows])
    ece = evaluator_ece([max(r["probs"].values()) for r in rows], [r["gold"] == r["pred"] for r in rows])
    rep = {"run": str(ck), "split": "test", "dtype": "bf16", "records": len(rows),
           "micro_accuracy": round(m["accuracy"] + (0.02 if tamper else 0), 4), "brier": round(m["brier_two_class"], 4),
           "states_truncated": 0, "latency_ms_per_record": {"median": 1.0},
           "per_question": {"eligible": {"macro_f1": round(m["macro_f1"], 4), "ece": round(ece, 4)}}}
    (ck / "eval-test.json").write_text(json.dumps(rep))
    return run


def test_consistent_run_has_no_problems(tmp_path):
    res = vr.audit_run(str(make_run(tmp_path)))
    assert res["best_step_by_dev_loss"] == 400
    assert res["problems"] == []
    assert res["eval_test"]["patients"] == 50 and res["eval_test"]["rows"] == 6975


def test_doctored_report_is_flagged(tmp_path):
    res = vr.audit_run(str(make_run(tmp_path, tamper=True)))
    assert any("accuracy" in p for p in res["problems"])


def test_test_predictions_for_unselected_checkpoint_are_flagged(tmp_path):
    run = make_run(tmp_path)
    other = run / "step-600"
    other.mkdir()
    (other / "eval-test-preds.jsonl").write_text((run / "step-400" / "eval-test-preds.jsonl").read_text())
    assert any("NON-selected" in p for p in vr.audit_run(str(run))["problems"])


def test_degenerate_probe_is_flagged(tmp_path):
    d = tmp_path / "prefinetune-x"
    d.mkdir()
    rows = [{"label": i % 2, "pred": 0, "p_eligible": 0.01} for i in range(100)]
    (d / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (d / "report.json").write_text(json.dumps({"n": 100, "accuracy": 0.5, "model": "m", "prompt_mode": "plain"}))
    res = vr.audit_probe(str(d))
    assert any("degenerate" in p for p in res["problems"])
