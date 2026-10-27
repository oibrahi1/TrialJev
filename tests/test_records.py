"""CPU-only: the JSONL reader and the label -> option-index contract. No GPU, no model."""
import json

import pytest

from trialmatch.records import load_records, materialize

NOUL = {"type": "noul", "instructions": "Eligible?", "criteria": {"true": "Eligible", "false": "Excluded"}}


def write(path, rows):
    path.write_bytes(("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8"))


def test_reader_is_utf8_regardless_of_platform_codec(tmp_path, monkeypatch):
    """Windows' default codec is cp1252; non-ASCII trial text (>=, mu, accents) must still load."""
    p = tmp_path / "x.jsonl"
    write(p, [{"state": "Patient ≥ 18 y, µ-opioid, naïve", "questions": {"eligible": {**NOUL, "label": True}}}])
    assert load_records(p)[0]["state"] == "Patient ≥ 18 y, µ-opioid, naïve"


def test_reader_splits_on_newline_only(tmp_path):
    """U+2028/U+0085 inside a JSON string must not cut a record in half."""
    p = tmp_path / "x.jsonl"
    rows = [{"state": f"a b\x85c{i}", "questions": {"eligible": {**NOUL, "label": bool(i % 2)}}} for i in range(3)]
    write(p, rows)
    got = load_records(p)
    assert [r["state"] for r in got] == [r["state"] for r in rows]


def test_record_without_label_is_rejected(tmp_path):
    p = tmp_path / "x.jsonl"
    write(p, [{"state": "s", "questions": {"eligible": dict(NOUL)}}])
    with pytest.raises(ValueError, match="no label"):
        load_records(p)


@pytest.mark.parametrize("label,index", [(True, 1), (False, 0)])
def test_noul_label_maps_to_option_index(tmp_path, label, index):
    """eligible=True must land on option 1 ('yes'), excluded=False on option 0: the trainer and
    scorer both rely on this, and the reported probabilities are P(option 1)."""
    p = tmp_path / "x.jsonl"
    write(p, [{"state": "s", "questions": {"eligible": {**NOUL, "label": label}}}])
    q = materialize(load_records(p)[0])["questions"][0]
    assert q["label"] == index
    assert q["keys"] == ["false", "true"]
    assert q["qtype"] == "noul"
