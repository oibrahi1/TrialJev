"""Conventional generative fine-tuning: prompt, target, loss alignment, and the `sft-generate` evaluation condition.

Stand-in modules are injected through ``sys.modules`` so these tests need torch but no model weights, peft or GPU.
"""
import json
import sys
import types

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import generate_baselines as gb  # noqa: E402

ELIG, EXCL, EOS = [201, 202], [203], 1          # token ids of " Eligible", " Excluded" and end of sequence


class FakeTok:
    pad_token_id = 0
    eos_token_id = EOS
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return {" Eligible": ELIG, " Excluded": EXCL}[text]

    def decode(self, ids, skip_special_tokens=False):
        ids = [int(i) for i in ids if not (skip_special_tokens and int(i) in (0, EOS))]
        return {tuple(ELIG): " Eligible", tuple(EXCL): " Excluded"}.get(tuple(ids), "?")


def fake_user_tokens(tok, text):
    return list(text.encode("ascii", "replace"))      # one token per character, easy to read back


@pytest.fixture
def tg(monkeypatch):
    pkg = types.ModuleType("trialmatch")
    pkg.__path__ = []
    model = types.ModuleType("trialmatch.model")
    model.user_tokens = fake_user_tokens
    monkeypatch.setitem(sys.modules, "trialmatch", pkg)
    monkeypatch.setitem(sys.modules, "trialmatch.model", model)
    import train_lora_generate
    return train_lora_generate


def _req(label, state="the patient note"):
    return {"_meta": {"id": "t/1/N1", "patient_id": "p1", "group_id": "p1", "nct_id": "N1"}, "state": state,
            "questions": {"eligible": {"label": label, "instructions": "Is this patient eligible?",
                                       "criteria": {"false": "Excluded: reasons", "true": "Eligible: reasons"}}}}


def test_prompt_uses_the_zero_shot_words_and_both_option_orders(tg):
    ids, truncated = tg.build_prompt(FakeTok(), _req(True), 512, "excluded_first")
    text = bytes(ids).decode("ascii")
    assert not truncated
    assert text.startswith("Patient note:\nthe patient note")
    assert "Options:\nExcluded: reasons\nEligible: reasons" in text
    assert text.endswith("Reply with exactly one word: Eligible or Excluded.\nAnswer:")
    swapped = bytes(tg.build_prompt(FakeTok(), _req(True), 512, "eligible_first")[0]).decode("ascii")
    assert "Options:\nEligible: reasons\nExcluded: reasons" in swapped


def test_long_state_is_cut_from_the_head_and_flagged(tg):
    ids, truncated = tg.build_prompt(FakeTok(), _req(True, state="x" * 1000), 100, "excluded_first")
    assert truncated
    head = len("Patient note:\n")
    assert bytes(ids[head: head + 99]).decode("ascii") == "x" * 99      # max_state - 1 state tokens, then the tail
    assert bytes(ids[head + 99: head + 101]).decode("ascii") == "\n\n"


def test_target_is_the_word_then_end_of_sequence(tg):
    assert tg.answer_ids(FakeTok(), True) == ELIG + [EOS]
    assert tg.answer_ids(FakeTok(), False) == EXCL + [EOS]
    assert tg.word_first_ids(FakeTok()) == [ELIG[0], EXCL[0]]
    tok = FakeTok()
    tok.eos_token_id = None
    with pytest.raises(ValueError):
        tg.answer_ids(tok, True)


def test_first_tokens_must_differ(tg):
    tok = FakeTok()
    tok.encode = lambda text, add_special_tokens=False: [7, 8]
    with pytest.raises(ValueError):
        tg.word_first_ids(tok)


class ToyCausal(torch.nn.Module):
    """hidden state at a position = embedding of the token there, so logits depend only on that token."""

    def __init__(self, vocab=300, dim=16):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, dim)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)
        self.model = self

    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(last_hidden_state=self.embed(input_ids))


def test_loss_is_taken_on_the_answer_positions_only(tg):
    causal = ToyCausal()
    batch = [([10, 11, 12, 13], ELIG + [EOS]), ([20, 21], EXCL + [EOS])]       # different prompt lengths -> padding
    logits = tg.answer_logits(causal, batch, 0, "cpu")
    assert [z.shape[0] for z in logits] == [3, 2]                                # one row per answer token
    for z, (prompt, answer) in zip(logits, batch):
        full = prompt + answer
        for j, target in enumerate(answer):
            predicting_position = len(prompt) - 1 + j                            # the token before the one being predicted
            expected = causal.lm_head(causal.embed(torch.tensor(full[predicting_position]))).float()
            assert torch.allclose(z[j], expected, atol=1e-6)
    manual = torch.stack([F.cross_entropy(z, torch.tensor(a)) for z, (_, a) in zip(logits, batch)]).mean()
    assert float(tg.batch_loss(logits, batch, "cpu")) == pytest.approx(float(manual))


def test_changing_a_prompt_token_does_not_change_the_target_loss_terms_it_does_not_feed(tg):
    causal = ToyCausal()
    a = tg.answer_logits(causal, [([10, 11, 12, 13], EXCL + [EOS])], 0, "cpu")[0]
    b = tg.answer_logits(causal, [([99, 98, 97, 13], EXCL + [EOS])], 0, "cpu")[0]   # same last prompt token
    assert torch.allclose(a, b, atol=1e-6)                                      # with this toy model only the last prompt token feeds the first target


# ---- the sft-generate condition of generate_baselines.run --------------------------------------------------------

VOCAB = 300


class FakeCausal(torch.nn.Module):
    def __init__(self, log, logit_eligible, logit_excluded):
        super().__init__()
        self.config = types.SimpleNamespace(use_cache=False)
        self.log = log
        self.scores = torch.zeros(1, VOCAB)
        self.scores[0, ELIG[0]], self.scores[0, EXCL[0]] = logit_eligible, logit_excluded

    def generate(self, **kwargs):
        self.log["generate"].append({k: v for k, v in kwargs.items() if k != "input_ids" and k != "attention_mask"})
        self.log["prompts"].append(kwargs["input_ids"][0].tolist())
        prompt = kwargs["input_ids"]
        seq = torch.cat([prompt, torch.tensor([ELIG + [EOS]])], dim=1)
        return types.SimpleNamespace(sequences=seq, scores=(self.scores,))


def _install(monkeypatch, tmp_path, log, logit_eligible, logit_excluded):
    fake_tl = types.ModuleType("train_lora_letter")
    fake_tl.load_backbone = lambda base, revision, dtype: torch.nn.Identity()
    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "0.0-test"
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda base, revision=None: FakeTok())
    causal = FakeCausal(log, logit_eligible, logit_excluded)

    class PeftModel:
        @staticmethod
        def from_pretrained(lm, adapter):
            log["adapter_loaded"].append(adapter)
            return types.SimpleNamespace(to=lambda device: types.SimpleNamespace(get_base_model=lambda: causal))

    fake_peft = types.ModuleType("peft")
    fake_peft.__version__ = "0.0-test"
    fake_peft.PeftModel = PeftModel
    pkg = types.ModuleType("trialmatch")
    pkg.__path__ = []
    model = types.ModuleType("trialmatch.model")
    model.user_tokens = fake_user_tokens
    for name, module in (("train_lora_letter", fake_tl), ("transformers", fake_tf), ("peft", fake_peft),
                         ("trialmatch", pkg), ("trialmatch.model", model)):
        monkeypatch.setitem(sys.modules, name, module)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "fake/base"}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    (adapter / "control_meta.json").write_text(json.dumps({"base": "fake/base", "max_state": 512, "order": "excluded_first"}),
                                               encoding="utf-8")
    return adapter


def _args(tmp_path, adapter, order="excluded_first"):
    data = tmp_path / "test.jsonl"
    data.write_text("\n".join(json.dumps(_req(lab)) .replace('"t/1/N1"', f'"t/{i}/N{i}"').replace('"p1"', f'"p{i}"')
                              for i, lab in enumerate((True, False))) + "\n", encoding="utf-8")
    return types.SimpleNamespace(condition="sft-generate", base="fake/base", base_revision="", adapter=str(adapter) if adapter else "",
                                 data=str(data), order=order, per_patient=0, ids_file="", limit=0, max_new_tokens=0,
                                 max_state=512, warmup=0, dtype="fp32", device="cpu", out=str(tmp_path / "out"))


@pytest.mark.parametrize("order", ["excluded_first", "eligible_first"])
def test_sft_generate_writes_text_stops_at_eos_and_scores_the_two_words(tmp_path, monkeypatch, order):
    log = {"generate": [], "prompts": [], "adapter_loaded": []}
    adapter = _install(monkeypatch, tmp_path, log, logit_eligible=2.0, logit_excluded=0.5)
    gb.run(_args(tmp_path, adapter, order))

    assert log["adapter_loaded"] == [str(adapter)]
    assert all(g["eos_token_id"] == EOS and g["return_dict_in_generate"] and g["output_scores"] and g["max_new_tokens"] == 8
               for g in log["generate"])
    assert len(log["generate"]) == 2
    first = bytes(log["prompts"][0]).decode("ascii")
    assert first.startswith("Patient note:\n") and first.endswith("Reply with exactly one word: Eligible or Excluded.\nAnswer:")
    expect_first = "Options:\nExcluded: reasons\nEligible: reasons" if order == "excluded_first" else "Options:\nEligible: reasons\nExcluded: reasons"
    assert expect_first in first

    rows = [json.loads(x) for x in (tmp_path / "out" / "preds.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_p = float(torch.softmax(torch.tensor([2.0, 0.5]), -1)[0])
    for r in rows:
        assert r["text"] == " Eligible" and r["valid"] and r["pred"] is True
        assert r["n_new_tokens"] == 3                                  # two word tokens plus the end-of-sequence token
        assert r["p_true"] == pytest.approx(expected_p)
    assert [r["correct"] for r in rows] == [True, False]               # gold True, False
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["prompt_format"].startswith("fine-tuned writer prompt")
    assert "conventional generative fine-tuning" in report["adapter_note"]
    assert report["max_new_tokens"] == 8


def test_sft_generate_requires_an_adapter(tmp_path, monkeypatch):
    log = {"generate": [], "prompts": [], "adapter_loaded": []}
    _install(monkeypatch, tmp_path, log, 1.0, 0.0)
    with pytest.raises(SystemExit):
        gb.run(_args(tmp_path, None))
    assert log["adapter_loaded"] == []
