"""CPU-only: dataset-construction logic (label mapping, patient split, trimming, sanity flags)."""
import json
from pathlib import Path

import build_data as bd


class WordCounter:
    """Stand-in tokenizer: 1 token per whitespace-separated word (no network needed)."""

    def n(self, text):
        return len(text.split())

    def truncate(self, text, n):
        w = text.split()
        return text if len(w) <= n else " ".join(w[:n]) + bd.MARK


def row(patient, nct, label, vignette="55 year old man with HTN.", trial=None):
    trial = trial or ("TITLE:\nT\n\nDEMOGRAPHICS:\nSex: All\nMinimum age: 18 Years\n\n"
                      "ELIGIBILITY:\nInclusion: adults")
    return {"patient_id": patient, "nct_id": nct, "patient_vignette": vignette, "trial_text": trial,
            "label": label, "label_name": {0: "not_relevant", 1: "excluded", 2: "eligible"}[label]}


def test_run1_labels_eligible_true_excluded_false_not_relevant_absent():
    assert bd.RUN1_LABELS == {2: True, 1: False}
    assert 0 not in bd.RUN1_LABELS            # not-relevant is never turned into a negative
    rec = bd.make_record(row("1", "NCT1", 2), "t", [], 2021, [])
    assert rec["questions"]["eligible"]["label"] is True
    assert bd.make_record(row("1", "NCT1", 1), "t", [], 2021, [])["questions"]["eligible"]["label"] is False


def test_split_is_by_patient_deterministic_and_complete():
    rows = [row(str(p), f"NCT{i}", 2 if i % 2 else 1) for p in range(1, 76) for i in range(p % 7 + 1)]
    tr, dv = bd.split_patients(rows, 5, 13)
    assert not set(tr) & set(dv)
    assert set(tr) | set(dv) == {r["patient_id"] for r in rows}
    assert (tr, dv) == bd.split_patients(rows, 5, 13)
    assert 10 <= len(dv) <= 20                 # about 1 in 5 patients


def test_suspect_flags_only_eligible_labels_that_violate_own_limits():
    young = row("1", "N", 2, vignette="10 year old boy with asthma.",
                trial="DEMOGRAPHICS:\nSex: All\nMinimum age: 18 Years\n\nELIGIBILITY:\nx")
    assert any("age" in r for r in bd.suspect_reasons(young))
    excluded = dict(young, label=1, label_name="excluded")
    assert bd.suspect_reasons(excluded) == []   # excluded labels are never second-guessed
    wrong_sex = row("1", "N", 2, vignette="A 40 year old woman.",
                    trial="DEMOGRAPHICS:\nSex: Male\n\nELIGIBILITY:\nx")
    assert any("sex" in r for r in bd.suspect_reasons(wrong_sex))
    assert bd.suspect_reasons(row("1", "N", 2)) == []


def test_fit_trial_noop_when_it_fits():
    t = "TITLE:\nShort\n\nELIGIBILITY:\nInclusion: adults"
    assert bd.fit_trial(t, 10_000, WordCounter()) == (t, [])


def test_fit_trial_drops_low_value_sections_before_touching_eligibility():
    secs = [("TITLE", "Trial title"), ("OFFICIAL TITLE", "long " * 60), ("KEYWORDS", "k " * 60),
            ("SUMMARY", "s " * 200), ("DEMOGRAPHICS", "Sex: All"), ("ELIGIBILITY", "Inclusion: adults over 18")]
    text = bd.join_sections(secs)
    tc = WordCounter()
    budget = tc.n(bd.instructions(text)) - 100
    out, steps = bd.fit_trial(text, budget, tc)
    assert tc.n(bd.instructions(out)) <= budget
    assert "drop_KEYWORDS" in steps and "cut_ELIGIBILITY" not in steps
    parsed = dict(bd.parse_sections(out))
    assert parsed["TITLE"] == "Trial title" and parsed["DEMOGRAPHICS"] == "Sex: All"
    assert parsed["ELIGIBILITY"] == "Inclusion: adults over 18"


def test_written_jsonl_has_one_record_per_line_even_with_exotic_separators(tmp_path):
    p = Path(tmp_path) / "x.jsonl"
    bd.write_jsonl(p, [{"state": "a b\x85c", "n": 1}, {"state": "d", "n": 2}])
    raw = p.read_bytes().decode("utf-8")
    assert raw.count("\n") == len(raw.splitlines()) == 2
    assert json.loads(raw.split("\n")[0])["state"] == "a b\x85c"
