"""The generation baselines must parse strictly, sample deterministically and present options in both orders."""
import pytest

import generate_baselines as gb


@pytest.mark.parametrize("text,expected", [
    ("Eligible", True), (" Excluded.", False), ("eligible\n", True), ("**Excluded**", False),
    ("Not eligible", None), ("Ineligible", None), ("Eligible, because the patient...", None), ("", None), ("Maybe", None),
])
def test_strict_word_parser(text, expected):
    assert gb.strict_answer(text, "word") is expected


def test_strict_letter_parser_follows_order():
    assert gb.strict_answer(" A", "letter", "excluded_first") is False      # A = excluded
    assert gb.strict_answer(" B", "letter", "excluded_first") is True
    assert gb.strict_answer(" A", "letter", "eligible_first") is True       # letters swapped
    assert gb.strict_answer("C", "letter") is None
    assert gb.strict_answer("A and B", "letter") is None


def test_lenient_metric_is_separate_and_labelled():
    assert gb.lenient_first_word("Eligible, because") is True
    assert gb.lenient_first_word("The patient is eligible") is None


def make_req(i, patient, label=True):
    return {"state": "note", "questions": {"eligible": {"instructions": "Q?\n\nTRIAL", "label": label,
            "criteria": {"true": "Eligible: yes", "false": "Excluded: no"}}},
            "_meta": {"id": f"t/{patient}/{i}", "patient_id": patient}}


def test_sample_is_deterministic_capped_and_order_independent():
    recs = [make_req(i, f"p{i % 4}") for i in range(80)]
    a = gb.pick_sample(recs, 5)
    b = gb.pick_sample(list(reversed(recs)), 5)
    assert len(a) == 20 and {r["_meta"]["id"] for r in a} == {r["_meta"]["id"] for r in b}
    assert all(sum(r["_meta"]["patient_id"] == p for r in a) == 5 for p in ("p0", "p1", "p2", "p3"))


def test_options_swap_between_orders():
    req = make_req(0, "p0")
    first = gb.user_text(req, "excluded_first")
    second = gb.user_text(req, "eligible_first")
    assert first.index("Excluded: no") < first.index("Eligible: yes")
    assert second.index("Eligible: yes") < second.index("Excluded: no")
    assert "Reply with exactly one word" in first


def test_plain_prompt_ends_with_answer_marker():
    assert gb.render_zeroshot(make_req(0, "p0"), "plain", "excluded_first").endswith("Answer:")


def test_strict_macro_f1_counts_invalid_as_wrong():
    rows = [{"gold": True, "pred": True, "valid": True}, {"gold": False, "pred": False, "valid": True},
            {"gold": True, "pred": None, "valid": False}, {"gold": False, "pred": None, "valid": False}]
    perfect_on_valid = gb.macro_f1([True, False], [True, False])
    assert perfect_on_valid == 1.0
    assert gb.strict_macro_f1(rows) < 1.0                    # the two invalid answers count as errors
    assert gb.strict_macro_f1(rows[:2]) == 1.0


def test_adapter_check_rejects_wrong_base_and_budget(tmp_path):
    d = tmp_path / "ck"
    d.mkdir()
    for f in ("adapter_model.safetensors",):
        (d / f).write_bytes(b"x")
    (d / "adapter_config.json").write_text('{"base_model_name_or_path": "Qwen/Qwen3.5-9B-Base"}')
    (d / "control_meta.json").write_text('{"base": "Qwen/Qwen3.5-9B-Base", "max_state": 512, "order": "excluded_first"}')
    assert gb.check_adapter(str(d), "Qwen/Qwen3.5-9B-Base", 512)["order"] == "excluded_first"
    with pytest.raises(SystemExit, match="trained on"):
        gb.check_adapter(str(d), "Qwen/Qwen3.5-27B", 512)
    with pytest.raises(SystemExit, match="max_state"):
        gb.check_adapter(str(d), "Qwen/Qwen3.5-9B-Base", 256)
    (d / "control_meta.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        gb.check_adapter(str(d), "Qwen/Qwen3.5-9B-Base", 512)


def test_zeroshot_letters_condition_is_registered_and_needs_no_adapter():
    # no adapter, scored from the A/B logits with the fine-tuning prompt
    assert "zeroshot-letters" in gb.CONDITIONS
    assert not "zeroshot-letters".startswith("lora")
