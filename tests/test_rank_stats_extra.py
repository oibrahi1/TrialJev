"""Extra ranking-analysis checks: a second hand-calculated ranking, no relevant items, and rejection of bad pair ids."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "analysis"))

import compute_rank_stats as rs  # noqa: E402


def test_relevant_items_ranked_last_by_hand():
    p_at_k, ndcg, mrr, ap = rs.patient_metrics(np.array([0, 0, 1, 1]), np.array([4.0, 3.0, 2.0, 1.0]))
    assert p_at_k == pytest.approx(0.5)
    assert ndcg == pytest.approx((1 / math.log2(4) + 1 / math.log2(5)) / (1 + 1 / math.log2(3)))
    assert mrr == pytest.approx(1 / 3)
    assert ap == pytest.approx((1 / 3 + 2 / 4) / 2)


def test_no_relevant_items_gives_zero_not_nan():
    assert rs.patient_metrics(np.array([0, 0, 0]), np.array([0.3, 0.2, 0.1])) == [0.0, 0.0, 0.0, 0.0]


def test_check_same_ids_rejects_missing_and_extra_pairs():
    rs.check_same_ids(["a", "b"], {"b": 1, "a": 2}, "same set in another order")
    with pytest.raises(ValueError):
        rs.check_same_ids(["a", "b"], {"a": 1}, "b missing")
    with pytest.raises(ValueError):
        rs.check_same_ids(["a", "b"], {"a": 1, "b": 2, "c": 3}, "c unexpected")


def test_generation_rows_with_duplicate_pair_ids_are_rejected(tmp_path):
    row = {"gold": True, "p_true": 0.7, "meta": {"id": "t/1/N1", "patient_id": "p1"}}
    (tmp_path / "dup.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        rs.load_probs({"pz": tmp_path}, "pz", "dup.jsonl")
    (tmp_path / "ok.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert rs.load_probs({"pz": tmp_path}, "pz", "ok.jsonl") == {"t/1/N1": (1, 0.7, "p1")}
