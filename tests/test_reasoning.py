"""The reasoning-generation baseline: prompt wording and strict reading of the final verdict line."""
import generate_baselines as gb

REQ = {"state": "NOTE", "questions": {"eligible": {"instructions": "Q?",
                                                   "criteria": {"false": "Excluded: x", "true": "Eligible: y"}}}}


def test_reasoning_is_a_registered_condition():
    assert "zeroshot-reason" in gb.CONDITIONS


def test_reasoning_prompt_asks_for_reasons_then_a_final_answer_line():
    prompt = gb.render_zeroshot(REQ, "reason", "excluded_first")
    assert gb.SYSTEM_REASON in prompt
    assert "Options:\nExcluded: x\nEligible: y" in prompt
    assert prompt.rstrip().endswith("<|im_start|>assistant")
    assert "Explain your reasoning briefly, then end with exactly one line: Answer: Eligible or Answer: Excluded." in prompt
    assert "Options:\nEligible: y\nExcluded: x" in gb.render_zeroshot(REQ, "reason", "eligible_first")
    assert "Reply with exactly one word" in gb.render_zeroshot(REQ, "chat", "excluded_first")   # one-word prompt unchanged


def test_verdict_is_the_last_answer_line_and_missing_is_invalid():
    assert gb.reason_answer("Criteria are met.\nAnswer: Eligible") is True
    assert gb.reason_answer("Answer: Eligible? No, age is out of range.\nAnswer: Excluded") is False
    assert gb.reason_answer("**Answer:** Excluded") is False
    assert gb.reason_answer("answer - ELIGIBLE") is True
    assert gb.reason_answer("The patient is probably eligible.") is None
    assert gb.reason_answer("") is None
    assert gb.reason_answer(None) is None
