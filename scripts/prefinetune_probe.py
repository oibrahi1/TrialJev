"""How well does a model answer eligible/excluded with ZERO fine-tuning -- before
any of our LoRA/pointer-head training ever touches it -- read out through its own
next-token letter logits. Adapted from MedJev's own base_probe.py (same technique
MedJev used to floor-test itself): build
    <patient note>
    <trial criteria>
    A. Excluded: ...
    B. Eligible: ...
    Answer:
and softmax the logits of " A"/" B" (or "A"/"B" for chat models) at the last
position. No LoRA, no pointer head, no fine-tuning of any kind -- this is the model
exactly as released, before our work starts.

    python scripts/prefinetune_probe.py --model Qwen/Qwen3.5-9B-Base --prompt plain \
      --data data/tm-v1-b4096/test.jsonl --out runs/prefinetune-q35-9b

    python scripts/prefinetune_probe.py --model Qwen/Qwen3.5-27B --prompt chat \
      --data data/tm-v1-b4096/test.jsonl --out runs/prefinetune-q35-27b

Run it twice (--order excluded_first and --order eligible_first). A model whose two
accuracies differ a lot, or that predicts one class almost always, is showing letter
position bias, not task knowledge.

--prompt plain: for pretraining-only checkpoints (e.g. Qwen3.5-9B-Base) -- raw
completion prompt, no chat template.
--prompt chat: for post-trained/instruction-tuned checkpoints (e.g. Qwen3.5-27B,
which has no Base variant) -- uses the tokenizer's chat template.
"""
import argparse
import json
import os
import time

LETTERS = "AB"
EXCLUDED = "Excluded: an inclusion criterion is not met or an exclusion criterion applies"
ELIGIBLE = "Eligible: meets the inclusion criteria and no exclusion criterion applies"
# `excluded_first` (A=excluded, B=eligible) is the order the reported zero-shot numbers used.
# `eligible_first` swaps it; run both and compare to expose letter position bias.
ORDERS = {"excluded_first": ("excluded", "eligible"), "eligible_first": ("eligible", "excluded")}
DESC = {"excluded": EXCLUDED, "eligible": ELIGIBLE}
SYSTEM = ("Decide whether the patient described is eligible for the clinical trial, "
          "based on its inclusion and exclusion criteria. Choose exactly one listed "
          "option. Respond with only its uppercase letter, with no explanation.")


def build_prompt(state, instructions, tok, mode, order="excluded_first"):
    options = [(l, DESC[name]) for l, name in zip(LETTERS, ORDERS[order])]
    if mode == "chat":
        payload = {"patient": state, "trial": instructions,
                   "options": [{"letter": l, "description": d} for l, d in options]}
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    body = "\n".join(f"{l}. {d}" for l, d in options)
    return f"{state}\n\n{instructions}\n\n{body}\nAnswer:"


def main():
    import torch  # heavy imports live here so build_prompt() can be unit-tested without a GPU stack
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local path, exactly as released")
    ap.add_argument("--data", required=True, help="trec pairs jsonl, e.g. data/tm-v1-b4096/test.jsonl")
    ap.add_argument("--prompt", choices=["plain", "chat"], required=True)
    ap.add_argument("--order", choices=list(ORDERS), default="excluded_first",
                    help="which label gets letter A (default: the order used for the reported numbers)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = [tok.encode((" " if a.prompt == "plain" else "") + L, add_special_tokens=False)[0] for L in LETTERS]
    if len(set(letter_ids)) != len(letter_ids):
        raise ValueError("A/B token ids collide -- pick different marker tokens")

    dtype = torch.bfloat16 if a.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()
    t_loaded = time.time()

    items = []
    with open(a.data, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if a.limit and n >= a.limit:
                break
            r = json.loads(line)
            q = r["questions"]["eligible"]
            prompt = build_prompt(r["state"], q["instructions"], tok, a.prompt, a.order)
            items.append({"patient_id": r["_meta"]["patient_id"] if "_meta" in r else None,
                         "nct_id": r["_meta"]["nct_id"] if "_meta" in r else None,
                         "label": int(bool(q["label"])), "prompt": prompt})
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
    t_prepped = time.time()

    with torch.no_grad():
        for b in range(0, len(order), a.batch_size):
            chunk = [items[i] for i in order[b: b + a.batch_size]]
            enc = tok([c["prompt"] for c in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=a.prompt == "plain").to(a.device)
            last = enc["attention_mask"].sum(1) - 1
            hidden = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            logits = model.lm_head(hidden[torch.arange(len(chunk), device=a.device), last]).float()
            for c, row in zip(chunk, logits):
                p = torch.softmax(row[letter_ids], -1).tolist()
                by_label = dict(zip(ORDERS[a.order], p))
                c["p_excluded"], c["p_eligible"] = by_label["excluded"], by_label["eligible"]
                c["pred"] = int(c["p_eligible"] > c["p_excluded"])
            if b % (a.batch_size * 20) == 0:
                print(f"{b + len(chunk)}/{len(order)}", flush=True)

    hits = sum(c["pred"] == c["label"] for c in items)
    brier = sum((c["p_eligible"] - c["label"]) ** 2 for c in items) / len(items)
    majority = max(sum(c["label"] == 1 for c in items), sum(c["label"] == 0 for c in items)) / len(items)
    report = {
        "model": a.model, "prompt_mode": a.prompt, "option_order": a.order, "readout": "zero-shot next-token letter logits, no fine-tuning",
        "data": a.data, "n": len(items),
        "accuracy": round(hits / len(items), 4),
        "majority_floor": round(majority, 4),
        "brier": round(brier, 4),
        "predicted_eligible_fraction": round(sum(c["pred"] for c in items) / len(items), 4),
        "runtime_s": round(time.time() - t0, 1),
        "model_load_s": round(t_loaded - t0, 1),
        "data_prep_s": round(t_prepped - t_loaded, 1),
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "rows.jsonl"), "w", encoding="utf-8") as f:
        for c in items:
            f.write(json.dumps({k: v for k, v in c.items() if k != "prompt"}) + "\n")
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
