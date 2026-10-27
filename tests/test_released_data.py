"""Frozen-data regression tests (qrel counts, label mapping, split leakage, test-set construction).

They need the processed datasets, which are not in the repository (see DATA.md); set
TRIALMATCH_DATA to the directory holding them. Without it they are skipped.
"""
import hashlib
import json
from pathlib import Path

import pytest
from conftest import DATA_DIR, ROOT

B = DATA_DIR / "tm-v1-b4096"
need = pytest.mark.skipif(not (B / "test.jsonl").exists(), reason=f"processed data not found in {DATA_DIR}")


def load(split):
    return [json.loads(l) for l in (B / f"{split}.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]


@pytest.fixture(scope="module")
def splits():
    return {s: load(s) for s in ("train", "development", "test")}


@need
def test_manifest_hashes_match():
    """The datasets behind every reported number are byte-identical to the frozen ones."""
    bad, checked = [], 0
    for line in (ROOT / "data" / "DATA_MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, rel = line.split(None, 1)
        f = DATA_DIR / Path(rel.strip()).relative_to("data")
        if f.exists():
            checked += 1
            if hashlib.sha256(f.read_bytes()).hexdigest() != digest:
                bad.append(rel)
    assert checked and not bad, bad


@need
@pytest.mark.parametrize("split,n,elig,excl,patients", [
    ("train", 9302, 4348, 4954, 60), ("development", 2233, 1168, 1065, 15), ("test", 6975, 3939, 3036, 50)])
def test_counts(splits, split, n, elig, excl, patients):
    recs = splits[split]
    labels = [r["questions"]["eligible"]["label"] for r in recs]
    assert (len(recs), sum(labels), len(labels) - sum(labels)) == (n, elig, excl)
    assert len({r["_meta"]["patient_id"] for r in recs}) == patients


@need
def test_labels_follow_trec_grades_and_not_relevant_is_absent(splits):
    for recs in splits.values():
        for r in recs:
            m = r["_meta"]
            assert m["trec_label"] in (1, 2)
            assert r["questions"]["eligible"]["label"] is (m["trec_label"] == 2)


@need
def test_no_patient_appears_in_two_splits(splits):
    ids = {s: {(r["_meta"]["year"], r["_meta"]["patient_id"]) for r in recs} for s, recs in splits.items()}
    assert not ids["train"] & ids["development"]
    assert not ids["train"] & ids["test"] and not ids["development"] & ids["test"]
    manifest = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))
    assert {p for _, p in ids["train"]} == set(manifest["train_patients"])
    assert {p for _, p in ids["development"]} == set(manifest["development_patients"])


@need
def test_record_ids_unique_and_years_correct(splits):
    for s, recs in splits.items():
        assert len({r["_meta"]["id"] for r in recs}) == len(recs)
        assert {r["_meta"]["year"] for r in recs} == {2022 if s == "test" else 2021}


@need
def test_quarantined_pairs_never_reach_train_or_dev(splits):
    for s in ("train", "development"):
        assert not [r for r in splits[s] if r["_meta"]["suspect"]]


@need
def test_trial_overlap_between_train_dev_and_test_is_small_and_text_identical(splits):
    seen = {r["_meta"]["nct_id"]: r["questions"]["eligible"]["instructions"]
            for s in ("train", "development") for r in splits[s]}
    shared = [r for r in splits["test"] if r["_meta"]["nct_id"] in seen]
    assert len(shared) / len(splits["test"]) < 0.04       # measured: 246/6975 = 3.53%
    assert all(seen[r["_meta"]["nct_id"]] == r["questions"]["eligible"]["instructions"] for r in shared)
