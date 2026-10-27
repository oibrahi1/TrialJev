#!/usr/bin/env python3
"""Generation baselines: does letting the model WRITE its answer cost time, and what does it do to accuracy?

Four conditions, all on the same sampled test pairs, one pair at a time (batch size 1), greedy decoding,
timed with GPU synchronization. Each condition gets the prompt format it was built for:

  zeroshot-chat   un-fine-tuned instruct model, chat template, writes "Eligible" or "Excluded"
  zeroshot-plain  un-fine-tuned Base model, plain text ending "Answer:", writes the word
  lora-generate   plain-LoRA model (adapter loaded), its EXACT training prompt (A./B. options, "Answer:"),
                  generate() for exactly one token, which must be the letter A or B
  lora-logits     the same adapter and prompt, one forward pass, read the A/B logits (the readout we reported)
  zeroshot-letters  NO adapter: the fine-tuning prompt (A./B. options, "Answer:") scored from the A/B logits.
                  Separates the effect of fine-tuning from the effect of the prompt and answer format.
  sft-generate    a model fine-tuned to WRITE the answer by conventional generative fine-tuning
                  (scripts/train_lora_generate.py): its training prompt (word options, "Answer:"), greedy decoding that
                  stops at end of sequence, strict word parsing; the first-step scores of " Eligible" / " Excluded"
                  give a two-way score p_true for AUROC and calibration
  zeroshot-reason un-fine-tuned instruct model asked to explain its reasoning briefly and then give the verdict on a final
                  line "Answer: Eligible" or "Answer: Excluded" (up to 512 new tokens by default). The verdict is read from
                  the LAST such line; no such line = invalid. This is the reasoning-generation baseline.

Parsing is strict: the generated text, stripped of whitespace, punctuation and special tokens, must be exactly
Eligible/Excluded (zero-shot) or A/B (LoRA). Anything else is INVALID and counts as wrong. There is no
fallback to logits and no calibration metric here (a two-token renormalized probability is not a valid
generation probability).

Look at the prompts first (loads no model):
  python scripts/generate_baselines.py show --data data/tm-v1-b4096/test.jsonl --shortest --base Qwen/Qwen3.5-27B

Run (GPU), the same sample for every condition:
  python scripts/generate_baselines.py run --condition zeroshot-chat --base Qwen/Qwen3.5-27B \
      --data data/tm-v1-b4096/test.jsonl --per_patient 20 --out runs/gen/zs-chat-27b
  python scripts/generate_baselines.py run --condition lora-generate --base Qwen/Qwen3.5-27B \
      --adapter runs/ctl-q35-27b/step-400 --data ... --ids_file runs/gen/zs-chat-27b/sample_ids.json \
      --order excluded_first --out runs/gen/lora-gen-27b
Repeat LoRA conditions with --order eligible_first (letters swapped) to separate letter bias from skill.
"""
import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

QID = "eligible"
SYSTEM = ("You are a clinical trial screening assistant. Decide whether the patient is eligible for the "
          "clinical trial. Reply with exactly one word: Eligible or Excluded.")
SYSTEM_REASON = ("You are a clinical trial screening assistant. Decide whether the patient is eligible for the "
                 "clinical trial. Reason briefly about the inclusion and exclusion criteria, then give the verdict on the "
                 "last line as 'Answer: Eligible' or 'Answer: Excluded'.")
CONDITIONS = ("zeroshot-chat", "zeroshot-plain", "lora-generate", "lora-logits", "zeroshot-letters", "sft-generate",
              "zeroshot-reason")
ORDERS = {"excluded_first": (False, True), "eligible_first": (True, False)}   # label shown first, second
LETTER_FORMAT = ("zero-shot conditions use a natural instruction prompt; LoRA conditions use the exact prompt "
                 "the adapter was trained on (scripts/train_lora_letter.py: build_item)")


# --------------------------------------------------------------------------- #
# prompts (zero-shot) and strict parsing
# --------------------------------------------------------------------------- #

def option_lines(req, order):
    c = req["questions"][QID]["criteria"]
    return [c["true" if lab else "false"] for lab in ORDERS[order]]


def user_text(req, order):
    q = req["questions"][QID]
    return (f"Patient note:\n{req['state']}\n\n{q['instructions']}\n\n"
            f"Options:\n" + "\n".join(option_lines(req, order)) + "\n\n"
            "Reply with exactly one word: Eligible or Excluded.")


def user_text_reason(req, order):
    q = req["questions"][QID]
    return (f"Patient note:\n{req['state']}\n\n{q['instructions']}\n\n"
            f"Options:\n" + "\n".join(option_lines(req, order)) + "\n\n"
            "Explain your reasoning briefly, then end with exactly one line: Answer: Eligible or Answer: Excluded.")


def render_zeroshot(req, mode, order, tok=None):
    if mode == "plain":
        return user_text(req, order) + "\nAnswer:"
    system, user = (SYSTEM_REASON, user_text_reason(req, order)) if mode == "reason" else (SYSTEM, user_text(req, order))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if tok is not None and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")


def reason_answer(text):
    """Verdict of a reasoning output: the LAST 'Answer: Eligible/Excluded' line; None (invalid) if there is none."""
    found = re.findall(r"answer\s*[:\-]\s*\**\s*(eligible|excluded)\b", text or "", flags=re.IGNORECASE)
    return None if not found else found[-1].lower() == "eligible"


def _clean(text):
    return re.sub(r"^[\W_]+|[\W_]+$", "", (text or "").strip()).lower()


def strict_answer(text, kind, order="excluded_first"):
    """True = eligible, False = excluded, None = invalid. kind: 'word' (Eligible/Excluded) or 'letter' (A/B)."""
    t = _clean(text)
    if kind == "word":
        return {"eligible": True, "excluded": False}.get(t)
    first, second = ORDERS[order]
    return {"a": first, "b": second}.get(t)


def lenient_first_word(text):
    """Secondary, clearly labelled metric: the first word of the output, if it is Eligible/Excluded."""
    m = re.match(r"[\W_]*(\w+)", text or "")
    return {"eligible": True, "excluded": False}.get(m.group(1).lower()) if m else None


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def read_records(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def pick_sample(recs, per_patient=0, ids_file=None):
    if ids_file:
        ids = set(json.loads(Path(ids_file).read_text(encoding="utf-8")))
        return [r for r in recs if r["_meta"]["id"] in ids]
    if not per_patient:
        return recs
    by = defaultdict(list)
    for r in recs:
        by[r["_meta"]["patient_id"]].append(r)
    out = []
    for p in sorted(by, key=lambda x: (len(x), x)):
        rs = sorted(by[p], key=lambda r: hashlib.sha256(r["_meta"]["id"].encode()).hexdigest())
        out += rs[:per_patient]
    return out


def macro_f1(gold, pred):
    f1s = []
    for c in (True, False):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1s) / 2


def strict_macro_f1(rows):
    """Macro-F1 over ALL sampled pairs; an invalid answer is scored as the wrong class (opposite of gold)."""
    return macro_f1([r["gold"] for r in rows], [r["pred"] if r["valid"] else (not r["gold"]) for r in rows])


def check_adapter(adapter, base, max_state):
    """Refuse to evaluate an adapter with the wrong base model or token budget; return its training metadata."""
    d = Path(adapter)
    for f in ("adapter_config.json", "adapter_model.safetensors", "control_meta.json"):
        if not (d / f).is_file():
            raise SystemExit(f"adapter {adapter}: missing {f}")
    meta = json.loads((d / "control_meta.json").read_text(encoding="utf-8"))
    cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
    for trained_on in (meta.get("base"), cfg.get("base_model_name_or_path")):
        if trained_on and Path(str(trained_on)).name != Path(base).name:
            raise SystemExit(f"adapter {adapter} was trained on {trained_on}, not {base}")
    if meta.get("max_state") != max_state:
        raise SystemExit(f"adapter was trained with max_state={meta.get('max_state')}, not {max_state}")
    return meta


def env_info(torch, with_peft):
    import transformers
    info = {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if with_peft:
        import peft
        info["peft"] = peft.__version__
    return info


def cmd_sample(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to overwrite the shared sample")
    recs = pick_sample(read_records(a.data), a.per_patient)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r["_meta"]["id"] for r in recs]), encoding="utf-8")
    pats = {r["_meta"]["patient_id"] for r in recs}
    gold = sum(bool(r["questions"][QID]["label"]) for r in recs)
    print(f"sample of {len(recs)} pairs from {len(pats)} patients ({gold} eligible / {len(recs) - gold} excluded) -> {out}")


# --------------------------------------------------------------------------- #
# show
# --------------------------------------------------------------------------- #

def show(a):
    recs = read_records(a.data)
    if a.shortest:
        req = min(recs, key=lambda r: len(user_text(r, "excluded_first")))
        idx = recs.index(req)
    else:
        idx = a.index
        req = recs[idx]
    tok = None
    if a.base:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.base)
    print(f"record #{idx}  id={req['_meta']['id']}  gold={'Eligible' if req['questions'][QID]['label'] else 'Excluded'}")
    blocks = [("zeroshot-chat  (un-fine-tuned instruct model)", "chat", "excluded_first"),
              ("zeroshot-chat  (same, options in the other order)", "chat", "eligible_first"),
              ("zeroshot-plain (un-fine-tuned Base model)", "plain", "excluded_first")]
    for title, mode, order in blocks:
        _banner(f"{title}   [model WRITES the word]")
        print(render_zeroshot(req, mode, order, tok if mode == "chat" else None))
        _end()
    if tok is not None:
        import train_lora_letter as tl
        for order in ("excluded_first", "eligible_first"):
            ids, _, _ = tl.build_item(tok, req, 512, order)
            _banner(f"lora-generate / lora-logits  (EXACT training prompt, order={order})   "
                    f"[generate ONE token: ' A' or ' B']")
            print(tok.decode(ids))
            _end()


def _banner(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def _end():
    print("=" * 100)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def run(a):
    import torch
    import train_lora_letter as tl
    from transformers import AutoTokenizer

    cond = a.condition
    lora = cond.startswith("lora")
    sft = cond == "sft-generate"                              # fine-tuned to write the word (conventional generative SFT)
    adapted = lora or sft                                     # an adapter is loaded
    scored = cond in ("lora-logits", "zeroshot-letters")      # answer read from one pass over the A/B logits
    letters = lora or cond == "zeroshot-letters"              # uses the lettered fine-tuning prompt
    if adapted and not a.adapter:
        raise SystemExit(f"--adapter is required for {cond}")
    out_dir = Path(a.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already exists and is not empty; refusing to overwrite evidence")
    adapter_meta = check_adapter(a.adapter, a.base, a.max_state) if adapted else None
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sync = (lambda: torch.cuda.synchronize()) if device.startswith("cuda") else (lambda: None)

    tok = AutoTokenizer.from_pretrained(a.base, revision=a.base_revision or None)
    lm = tl.load_backbone(a.base, a.base_revision, a.dtype).to(device).eval()
    if adapted:
        from peft import PeftModel
        causal = PeftModel.from_pretrained(lm, a.adapter).to(device).get_base_model()
    else:
        causal = lm
    causal.eval()
    causal.config.use_cache = True
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids_ab = tl.letter_ids(tok) if letters else None
    kind = "letter" if letters else "word"
    tg, first_ids = None, None
    if sft:
        import train_lora_generate as tg
        first_ids = tg.word_first_ids(tok)
    max_new = a.max_new_tokens or (1 if lora else 512 if cond == "zeroshot-reason" else 8)
    if cond == "lora-generate" and max_new != 1:
        raise SystemExit("lora-generate must generate exactly one token (--max_new_tokens 1)")

    recs = pick_sample(read_records(a.data), a.per_patient, a.ids_file)
    n_requested = len(set(json.loads(Path(a.ids_file).read_text(encoding="utf-8")))) if a.ids_file else None
    if n_requested is not None and len(recs) != n_requested:
        raise SystemExit(f"{n_requested - len(recs)} of the {n_requested} requested pair ids were not found in {a.data}")
    n_sample = len(recs)
    if a.limit:
        recs = recs[: a.limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_ids.json").write_text(json.dumps([r["_meta"]["id"] for r in recs]), encoding="utf-8")
    print(f"{cond}: {len(recs)} pairs, {len({r['_meta']['patient_id'] for r in recs})} patients; "
          f"model {a.base}; adapter {a.adapter or '-'}; order {a.order}", flush=True)

    def prepare(req):
        if sft:
            ids, _ = tg.build_prompt(tok, req, a.max_state, a.order)
            return torch.tensor([ids], device=device)
        if letters:
            ids, _, _ = tl.build_item(tok, req, a.max_state, a.order)
            return torch.tensor([ids], device=device)
        mode = {"zeroshot-chat": "chat", "zeroshot-reason": "reason"}.get(cond, "plain")
        prompt = render_zeroshot(req, mode, a.order, tok)
        return tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    def step(input_ids):
        """-> (generated text, n new tokens, ms, p_true or None)"""
        att = torch.ones_like(input_ids)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            if scored:
                z = tl.letter_logits(causal, [input_ids[0].tolist()], pad, device, ids_ab)[0]
                sync()
                ms = 1000 * (time.perf_counter() - t0)
                return (" A" if z[0] >= z[1] else " B"), 0, ms, float(torch.softmax(z, -1)[
                    0 if ORDERS[a.order][0] else 1])
            gen = dict(input_ids=input_ids, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                       temperature=None, top_p=None, top_k=None, pad_token_id=pad)
            if sft:    # stops at end of sequence; the first-step scores give a two-way score for AUROC and calibration
                gen.update(eos_token_id=tok.eos_token_id, return_dict_in_generate=True, output_scores=True)
            out = causal.generate(**gen)
        sync()
        ms = 1000 * (time.perf_counter() - t0)
        if sft:
            new = out.sequences[0, input_ids.shape[1]:]
            two_way = out.scores[0][0].float()[first_ids]                # [' Eligible', ' Excluded'] first-token scores
            return (tok.decode(new, skip_special_tokens=True), int(new.shape[0]), ms,
                    float(torch.softmax(two_way, -1)[0]))
        new = out[0, input_ids.shape[1]:]
        return tok.decode(new, skip_special_tokens=True), int(new.shape[0]), ms, None

    for req in recs[: a.warmup]:
        step(prepare(req))

    rows, t_all = [], time.time()
    for i, req in enumerate(recs):
        input_ids = prepare(req)
        text, n_new, ms, p_true = step(input_ids)
        gold = bool(req["questions"][QID]["label"])
        if scored:
            pred = strict_answer(text, "letter", a.order)
        elif cond == "zeroshot-reason":
            pred = reason_answer(text)
        else:
            pred = strict_answer(text, kind, a.order)
        rows.append({"gold": gold, "pred": pred, "valid": pred is not None, "correct": pred is not None and pred == gold,
                     "lenient": lenient_first_word(text) if kind == "word" else None, "text": text,
                     "n_new_tokens": n_new, "ms": ms, "n_prompt_tokens": int(input_ids.shape[1]), "p_true": p_true,
                     "meta": req["_meta"]})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(recs)}  median {statistics.median(r['ms'] for r in rows):.0f} ms/pair", flush=True)

    n = len(rows)
    valid = [r for r in rows if r["valid"]]
    gold_n = [r for r in rows if r["gold"]]
    excl_n = [r for r in rows if not r["gold"]]
    ms_sorted = sorted(r["ms"] for r in rows)
    report = {
        "condition": cond, "model": a.base, "adapter": a.adapter or None, "order": a.order,
        "prompt_format": ("exact LoRA training prompt (A./B. options, 'Answer:')" if lora else
                          "fine-tuned writer prompt (word options, 'Answer:'), greedy decoding to end of sequence"
                          if sft else
                          "fine-tuning prompt (A./B. options, 'Answer:'), NO adapter, A/B logits scored"
                          if cond == "zeroshot-letters" else
                         ("chat template + system line" if cond == "zeroshot-chat" else
                          "chat template, reason briefly then 'Answer: Eligible/Excluded' on the last line"
                          if cond == "zeroshot-reason" else "plain text ending 'Answer:'")),
        "base_revision": a.base_revision or "UNPINNED",
        "adapter_training_metadata": adapter_meta,
        "order_vs_adapter_training": (None if not adapted else
                                      "as_trained" if a.order == adapter_meta.get("order") else "swapped_diagnostic"),
        "adapter_note": ("the adapter was trained with a loss restricted to the two letter logits (A/B), "
                         "not conventional full-vocabulary generative fine-tuning; invalid outputs are possible"
                         if lora else
                         "the adapter was trained with conventional generative fine-tuning: causal-LM cross-entropy over the "
                         "whole vocabulary on the answer tokens (' Eligible'/' Excluded' + end of sequence)"
                         if sft else None),
        "backbone_note": ("un-fine-tuned model; compare with a LoRA condition only if the backbone is the same "
                          "(e.g. Qwen3.5-9B chat vs Qwen3.5-9B-Base are different checkpoints)" if not adapted else None),
        "environment": env_info(torch, adapted),
        "dtype": a.dtype, "device": device, "pairs": n, "pairs_in_full_sample": n_sample,
        "pair_ids_requested": n_requested, "patients": len({r["meta"]["patient_id"] for r in rows}),
        "max_new_tokens": 0 if scored else max_new, "batch_size": 1,
        "invalid_rate": 1 - len(valid) / n,
        "accuracy_strict_invalid_counts_wrong": sum(r["correct"] for r in rows) / n,
        "macro_f1_strict_invalid_counts_wrong": strict_macro_f1(rows),
        "eligible_recall_invalid_is_miss": sum(r["correct"] for r in gold_n) / max(len(gold_n), 1),
        "excluded_recall_invalid_is_miss": sum(r["correct"] for r in excl_n) / max(len(excl_n), 1),
        "macro_f1_valid_answers_only": macro_f1([r["gold"] for r in valid], [r["pred"] for r in valid]) if valid else None,
        "predicted_eligible_fraction_of_valid": (sum(r["pred"] for r in valid) / len(valid)) if valid else None,
        "majority_floor_on_sample": max(len(gold_n), len(excl_n)) / n,
        "secondary_accuracy_first_word_lenient": (sum(r["lenient"] == r["gold"] for r in rows) / n) if kind == "word" else None,
        "ms_per_pair": {"min": ms_sorted[0], "p05": ms_sorted[int(0.05 * (n - 1))], "median": statistics.median(ms_sorted),
                        "p95": ms_sorted[int(0.95 * (n - 1))], "max": ms_sorted[-1], "mean": statistics.mean(ms_sorted)},
        "new_tokens": {"mean": statistics.mean(r["n_new_tokens"] for r in rows), "max": max(r["n_new_tokens"] for r in rows)},
        "prompt_tokens_median": statistics.median(r["n_prompt_tokens"] for r in rows),
        "outputs_seen": Counter(r["text"].strip() or "<empty>" for r in rows).most_common(8),
        "wall_seconds": time.time() - t_all,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out_dir / "preds.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out_dir}/report.json, preds.jsonl, sample_ids.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--data", required=True)
    s.add_argument("--index", type=int, default=0)
    s.add_argument("--shortest", action="store_true")
    s.add_argument("--base", default="", help="load only this tokenizer, to render the real chat template / training prompt")
    m = sub.add_parser("sample", help="write the shared, patient-balanced list of pair ids (no model, no GPU)")
    m.add_argument("--data", required=True)
    m.add_argument("--per_patient", type=int, default=20)
    m.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--condition", choices=CONDITIONS, required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--base_revision", default="")
    r.add_argument("--adapter", default="", help="plain-LoRA checkpoint dir (lora-* conditions)")
    r.add_argument("--data", required=True)
    r.add_argument("--order", choices=list(ORDERS), default="excluded_first")
    r.add_argument("--per_patient", type=int, default=20)
    r.add_argument("--ids_file", default="")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--max_new_tokens", type=int, default=0, help="default 8 (zero-shot) or 1 (LoRA)")
    r.add_argument("--max_state", type=int, default=512)
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    r.add_argument("--device", default=None)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"show": show, "sample": cmd_sample, "run": run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
