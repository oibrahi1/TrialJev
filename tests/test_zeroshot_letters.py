"""The zero-shot-letters condition must load no adapter, use the training-prompt builder, and map A/B correctly.

These tests run the real ``generate_baselines.run`` on two synthetic records with a stand-in backbone and tokenizer
(injected through ``sys.modules``), so they need torch but no model weights, peft or GPU.
"""
import json
import sys
import types

import pytest

torch = pytest.importorskip("torch")

import generate_baselines as gb  # noqa: E402

LOGITS_A_HIGH = [3.0, 1.0]            # the model prefers the letter A


class FakeTok:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return {" A": [11], " B": [12]}[text]


class FakeLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(use_cache=False)


def _records(path):
    rows = []
    for i, label in enumerate((True, False)):
        rows.append({"_meta": {"id": f"t/{i}/N{i}", "patient_id": f"p{i}", "group_id": f"p{i}", "nct_id": f"N{i}"},
                     "state": "patient note",
                     "questions": {"eligible": {"label": label, "instructions": "Is this patient eligible?",
                                                "criteria": {"false": "Excluded: ...", "true": "Eligible: ..."}}}})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _install(monkeypatch, log, logits):
    fake_tl = types.ModuleType("train_lora_letter")

    def build_item(tok, req, max_state, order):
        log["build_item"].append((max_state, order))
        return [5, 6, 7, 8 if order == "excluded_first" else 9], 0, False

    def load_backbone(base, revision, dtype):
        log["load_backbone"].append((base, revision, dtype))
        return FakeLM()

    def letter_logits(causal, batch_ids, pad_id, device, ids_ab):
        log["letter_logits"].append((batch_ids, ids_ab))
        return torch.tensor([logits])

    fake_tl.build_item = build_item
    fake_tl.letter_ids = lambda tok: [11, 12]
    fake_tl.load_backbone = load_backbone
    fake_tl.letter_logits = letter_logits

    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "0.0-test"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(base, revision=None):
            return FakeTok()

    fake_tf.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "train_lora_letter", fake_tl)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "peft", None)       # any `import peft` now raises ImportError
    monkeypatch.setattr(gb, "check_adapter", lambda *a, **k: pytest.fail("no adapter exists, so none may be checked"))


def _args(tmp_path, condition, order, adapter=""):
    data = tmp_path / "test.jsonl"
    _records(data)
    return types.SimpleNamespace(condition=condition, base="fake/base", base_revision="", adapter=adapter, data=str(data),
                                 order=order, per_patient=0, ids_file="", limit=0, max_new_tokens=0, max_state=512,
                                 warmup=0, dtype="fp32", device="cpu", out=str(tmp_path / "out"))


@pytest.mark.parametrize("order,code,expect_pred,eligible_index", [
    ("excluded_first", 8, False, 1),     # A = Excluded, so a preference for A means "excluded"; eligible is letter B
    ("eligible_first", 9, True, 0),      # A = Eligible
])
def test_zeroshot_letters_no_adapter_exact_prompt_and_ab_mapping(tmp_path, monkeypatch, order, code, expect_pred, eligible_index):
    log = {"build_item": [], "load_backbone": [], "letter_logits": []}
    _install(monkeypatch, log, LOGITS_A_HIGH)
    gb.run(_args(tmp_path, "zeroshot-letters", order))

    # no adapter: the backbone is loaded once and peft was never imported
    assert log["load_backbone"] == [("fake/base", "", "fp32")]
    assert sys.modules["peft"] is None
    # exact training prompt: built by the trainer's own build_item with the requested order and token budget,
    # and exactly those ids reach the model
    assert log["build_item"] == [(512, order)] * 2
    assert len(log["letter_logits"]) == 2
    assert all(batch == [[5, 6, 7, code]] and ids_ab == [11, 12] for batch, ids_ab in log["letter_logits"])

    rows = [json.loads(x) for x in (tmp_path / "out" / "preds.jsonl").read_text(encoding="utf-8").splitlines()]
    probs = torch.softmax(torch.tensor(LOGITS_A_HIGH), -1)
    for r in rows:
        assert r["valid"] and r["text"] == " A" and r["n_new_tokens"] == 0
        assert r["pred"] is expect_pred
        assert r["p_true"] == pytest.approx(float(probs[eligible_index]))
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["adapter"] is None and report["max_new_tokens"] == 0
    assert "NO adapter" in report["prompt_format"]


def test_lora_conditions_require_an_adapter(tmp_path, monkeypatch):
    log = {"build_item": [], "load_backbone": [], "letter_logits": []}
    _install(monkeypatch, log, LOGITS_A_HIGH)
    with pytest.raises(SystemExit):
        gb.run(_args(tmp_path, "lora-logits", "excluded_first", adapter=""))
    assert log["load_backbone"] == []
