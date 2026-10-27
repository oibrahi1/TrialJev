"""The analysis script is checked against brute-force definitions on a small synthetic set."""
import itertools
import json

import numpy as np
import pytest

import analyze_predictions as ap


def make(path, gold, prob, patients):
    with open(path, "w", encoding="utf-8") as f:
        for i, (g, p, pt) in enumerate(zip(gold, prob, patients)):
            f.write(json.dumps({"gold": bool(g), "pred": bool(p >= 0.5),
                                "probs": {"false": 1 - p, "true": p},
                                "meta": {"id": f"r{i}", "patient_id": pt}}) + "\n")


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    gold = rng.integers(0, 2, n)
    prob = np.clip(0.5 + (gold - 0.5) * 0.5 + rng.normal(0, 0.25, n), 0.01, 0.99)
    patients = [f"p{i % 6}" for i in range(n)]
    path = tmp_path / "preds.jsonl"
    make(path, gold, prob, patients)
    return ap.load_predictions(path)


def brute_auroc(g, p):
    pos, neg = p[g == 1], p[g == 0]
    wins = sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def brute_ap(g, p):
    order = np.argsort(-p, kind="stable")
    g = g[order]
    hits = np.cumsum(g)
    precision = hits / (np.arange(len(g)) + 1)
    return float(precision[g == 1].mean())


def test_auroc_matches_brute_force(data):
    assert ap.metric_values(data)["auroc"] == pytest.approx(brute_auroc(data["gold"], data["prob"]))


def test_average_precision_matches_brute_force(data):
    got = ap.metric_values(data)["average_precision_eligible"]
    assert got == pytest.approx(brute_ap(data["gold"], data["prob"]), abs=1e-9)


def test_accuracy_and_brier_conventions(data):
    m = ap.metric_values(data)
    assert m["accuracy"] == pytest.approx(np.mean(data["gold"] == data["pred"]))
    assert m["brier_evaluator_two_class_sum"] == pytest.approx(2 * m["brier_conventional_binary"])


def test_bootstrap_is_deterministic_and_brackets_the_estimate(data):
    a = ap.cluster_bootstrap(data, 200, seed=1)
    b = ap.cluster_bootstrap(data, 200, seed=1)
    assert a == b
    acc = a["metrics"]["accuracy"]
    assert acc["ci95_percentile"][0] <= acc["estimate"] <= acc["ci95_percentile"][1]
    assert a["patients"] == 6                      # the resampling unit is the patient, not the pair


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "d.jsonl"
    row = {"gold": True, "pred": True, "probs": {"true": 0.9, "false": 0.1}, "meta": {"id": "x", "patient_id": "p"}}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        ap.load_predictions(path)
