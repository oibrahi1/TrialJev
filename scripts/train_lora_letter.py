#!/usr/bin/env python3
"""Plain-LoRA control: fine-tune the SAME backbone on the SAME data with NO pointer head.

The model answers through its own language-model head: the prompt ends with
    A. <first option>
    B. <second option>
    Answer:
and the answer is the softmax of the " A" / " B" next-token logits at the last position -- the same
readout as `scripts/prefinetune_probe.py`, so zero-shot -> fine-tuned is a like-for-like comparison.

Everything else mirrors `trialmatch.train` so that the pointer head is the only difference:
LoRA r16 / alpha 32 / dropout 0.05 on the same projection groups, AdamW (wd 0.01), one-cycle LR
(5% warm-up), clip 1.0, the same record order for a given seed, effective batch = batch x accum,
state truncated from the head at max_state, validation on the whole development split, and the same
`train_log_rank0.jsonl` / `training_config.json` / `step-N/` layout, so `scripts/pick_best.py` and
`scripts/analyze_predictions.py` work unchanged. Single GPU.

    # train (example: 9B, the settings of the pointer-head run)
    python scripts/train_lora_letter.py train --base Qwen/Qwen3.5-9B-Base --base_dtype bf16 \
      --data data/tm-v1-b4096/train.jsonl --val_data data/tm-v1-b4096/development.jsonl \
      --out runs/ctl-q35-9b --seed 0 --epochs 3 --lr 1e-4 --batch 2 --accum 8 \
      --max_state 512 --max_branch 4096 --dtype bf16 --checkpointing 1 \
      --log_every 20 --val_every 200 --save_every 200

    python scripts/pick_best.py runs/ctl-q35-9b
    # score a checkpoint (writes eval-<split>.json and eval-<split>-preds.jsonl next to it)
    python scripts/train_lora_letter.py eval --run runs/ctl-q35-9b/step-400 --data data/tm-v1-b4096 \
      --split test --allow-test

`--order eligible_first` swaps which option gets the letter A (use it at eval time to check letter bias).
"""
import argparse
import contextlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from trialmatch.model import load_tokenizer, user_tokens  # noqa: E402
from trialmatch.records import load_records  # noqa: E402

QID = "eligible"
ORDERS = {"excluded_first": ("false", "true"), "eligible_first": ("true", "false")}   # letters A, B
SOURCE = "medjev"      # same record-source tag as trialmatch.train (only affects the `src` field)


# --------------------------------------------------------------------------- #
# prompt and model
# --------------------------------------------------------------------------- #

def build_item(tok, req, max_state, order):
    """-> (token ids, index of the gold letter in (A, B), state_truncated)."""
    q = req["questions"][QID]
    state = user_tokens(tok, req["state"])
    truncated = len(state) + 1 > max_state                 # same accounting as trialmatch.model.encode
    first, second = ORDERS[order]
    tail = f"\n\n{q['instructions']}\n\nA. {q['criteria'][first]}\nB. {q['criteria'][second]}\nAnswer:"
    ids = state[: max_state - 1] + user_tokens(tok, tail)
    gold = str(bool(q["label"])).lower()
    return ids, (0 if gold == first else 1), truncated


def letter_ids(tok):
    ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in "AB"]
    if len(set(ids)) != 2:
        raise ValueError("A/B token ids collide")
    return ids


def lora_targets(backbone_config):
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "linear_attention" in set(getattr(backbone_config, "layer_types", None) or []):
        # Gated DeltaNet projections of hybrid Qwen3.5 backbones (same list as trialmatch.model)
        targets += ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
    return targets


def load_backbone(base, revision, base_dtype):
    dtype = torch.bfloat16 if base_dtype == "bf16" else torch.float32
    return AutoModelForCausalLM.from_pretrained(base, revision=revision or None, dtype=dtype,
                                                attn_implementation="sdpa")


def letter_logits(causal, batch_ids, pad_id, device, ids_ab):
    """[B, 2] logits of the letters A and B at the last real position (lm_head only on that position)."""
    length = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), length), pad_id, dtype=torch.long, device=device)
    att = torch.zeros((len(batch_ids), length), dtype=torch.long, device=device)
    for i, x in enumerate(batch_ids):
        ids[i, : len(x)] = torch.tensor(x, device=device)
        att[i, : len(x)] = 1
    hidden = causal.model(input_ids=ids, attention_mask=att).last_hidden_state
    last = att.sum(1) - 1
    h_last = hidden[torch.arange(len(batch_ids), device=device), last]
    return causal.lm_head(h_last).float()[:, ids_ab]


def save_checkpoint(peft_model, path, meta):
    os.makedirs(path, exist_ok=True)
    peft_model.save_pretrained(path)
    with open(os.path.join(path, "control_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

@torch.no_grad()
def validate(causal, tok, reqs, a, device, ids_ab, autocast):
    causal.eval()
    use_cache = causal.config.use_cache
    causal.config.use_cache = False
    loss_sum, hits, n, truncated = 0.0, 0, 0, 0
    items = []
    for req in reqs:
        ids, y, tr = build_item(tok, req, a.max_state, a.order)
        items.append((ids, y))
        truncated += int(tr)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))      # length-sorted: less padding
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for b in range(0, len(order), a.eval_batch):
        chunk = [items[i] for i in order[b: b + a.eval_batch]]
        with autocast:
            z = letter_logits(causal, [c[0] for c in chunk], pad, device, ids_ab)
        y = torch.tensor([c[1] for c in chunk], device=device)
        loss_sum += float(F.cross_entropy(z, y, reduction="sum"))
        hits += int((z.argmax(-1) == y).sum())
        n += len(chunk)
    causal.config.use_cache = use_cache
    causal.train()
    return {"val/loss": loss_sum / n, "val/accuracy": hits / n, "val/questions": float(n),
            "val/acc_noul": hits / n, f"val_q/{QID}": hits / n, "val/truncated": truncated}


def train(a):
    out_dir = Path(a.out)
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite an existing run: {a.out}")
    out_dir.mkdir(parents=True)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.dtype == "bf16" and not device.startswith("cuda"):
        raise SystemExit("--dtype bf16 requires CUDA")
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.dtype == "bf16" else contextlib.nullcontext()

    tok = load_tokenizer(a.base, revision=a.base_revision or None)
    ids_ab = letter_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lm = load_backbone(a.base, a.base_revision, a.base_dtype)
    if a.checkpointing:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm.config.use_cache = False
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(task_type="CAUSAL_LM", r=a.lora, lora_alpha=2 * a.lora, lora_dropout=0.05,
                     target_modules=lora_targets(lm.model.config))
    peft_model = get_peft_model(lm, cfg).to(device)
    causal = peft_model.get_base_model()
    peft_model.train()
    trainable = [p for p in peft_model.parameters() if p.requires_grad]

    reqs = load_records(a.data, source=SOURCE)
    if a.max_records:
        reqs = reqs[: a.max_records]
    val_reqs = load_records(a.val_data, source=SOURCE)[: a.val_records] if a.val_every and os.path.exists(a.val_data) else []
    rng.shuffle(reqs)                                   # same first-epoch order as trialmatch.train for a given seed
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.weight_decay)
    micro_per_epoch = math.ceil(len(reqs) / a.batch)
    steps = a.epochs * math.ceil(micro_per_epoch / a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.05)

    meta = {"base": os.path.abspath(a.base) if os.path.isdir(a.base) else a.base,
            "base_revision": a.base_revision or None, "base_dtype": a.base_dtype, "lora": a.lora,
            "order": a.order, "letters": [" A", " B"], "max_state": a.max_state, "max_branch": a.max_branch,
            "readout": "lm_head next-token logits for ' A' / ' B' (no pointer head)"}
    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "device": device, "world_size": 1, "records": len(reqs),
                   "optimizer_steps": steps, "effective_batch": a.batch * a.accum,
                   "trainable_params": sum(p.numel() for p in trainable)}, f, indent=2)
    print(f"device={device} trainable={sum(p.numel() for p in trainable)/1e6:.1f}M records={len(reqs)} "
          f"steps={steps} val={len(val_reqs)}", flush=True)

    log = open(out_dir / "train_log_rank0.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    acc, step, seen, truncated, peak = Counter(), 0, 0, 0, 0
    for ep in range(a.epochs):
        if ep:
            order = list(range(len(reqs)))
            random.Random(f"{a.seed}:epoch{ep}").shuffle(order)
            shard = [reqs[i] for i in order]
        else:
            shard = reqs
        for mb in range(micro_per_epoch):
            chunk = shard[mb * a.batch: (mb + 1) * a.batch]
            items = [build_item(tok, r, a.max_state, a.order) for r in chunk]
            truncated += sum(int(i[2]) for i in items)
            with autocast:
                z = letter_logits(causal, [i[0] for i in items], pad, device, ids_ab)
            y = torch.tensor([i[1] for i in items], device=device)
            loss = F.cross_entropy(z, y)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            acc["ce"] += float(loss) * len(chunk)
            acc["n"] += len(chunk)
            (loss * len(chunk) / (len(chunk) * a.accum)).backward()
            seen += len(chunk)
            if device.startswith("cuda"):
                peak = max(peak, torch.cuda.max_memory_allocated())
            if (mb + 1) % a.accum == 0 or mb + 1 == micro_per_epoch:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0:
                    entry = {"epoch": ep, "step": step, "of": steps, "train/loss": acc["ce"] / max(acc["n"], 1),
                             "lr": sched.get_last_lr()[0], "records": seen,
                             "s_per_record": (time.time() - t0) / max(seen, 1),
                             "eta_hours": (time.time() - t0) / max(seen, 1) * (a.epochs * len(reqs) - seen) / 3600,
                             "peak_gb": peak / 2**30}
                    print(f"ep{ep} step {step}/{steps} loss {entry['train/loss']:.4f} "
                          f"{entry['s_per_record']:.2f}s/rec eta {entry['eta_hours']:.1f}h peak {entry['peak_gb']:.1f}GB", flush=True)
                    log.write(json.dumps(entry) + "\n")
                    log.flush()
                    acc = Counter()
                if a.val_every and val_reqs and (step % a.val_every == 0 or step == steps):
                    vt = time.time()
                    metrics = validate(causal, tok, val_reqs, a, device, ids_ab, autocast)
                    print(f"  val step {step}: loss {metrics['val/loss']:.4f} acc {metrics['val/accuracy']:.4f} "
                          f"in {time.time()-vt:.0f}s", flush=True)
                    log.write(json.dumps({"step": step, **metrics}) + "\n")
                    log.flush()
                if a.save_every and step % a.save_every == 0:
                    save_checkpoint(peft_model, str(out_dir / f"step-{step}"), {**meta, "step": step, "epoch": ep})
                    print(f"checkpoint step-{step}", flush=True)
    save_checkpoint(peft_model, str(out_dir), {**meta, "step": step})
    with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"wall_seconds": time.time() - t0, "records_seen": seen, "truncated_states": truncated,
                   "optimizer_steps": step, "peak_device_bytes": peak, "device": device, "dtype": a.dtype}, f, indent=2)
    print(f"saved {a.out} in {(time.time()-t0)/3600:.2f}h", flush=True)
    log.close()


# --------------------------------------------------------------------------- #
# evaluation (same output schema as trialmatch.evaluate)
# --------------------------------------------------------------------------- #

def ece(conf, correct, bins=10):
    e, n = 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if c >= lo and (c < hi or (b == bins - 1 and c <= hi))]
        if idx:
            e += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return e


def macro_f1(gold, pred):
    f1s = []
    for c in ("true", "false"):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1s) / 2


def evaluate(a):
    if a.split == "test" and not a.allow_test:
        raise SystemExit("test split is locked: pass --allow-test to read it")
    run = Path(a.run)
    meta = json.load(open(run / "control_meta.json", encoding="utf-8"))
    order = a.order or meta["order"]
    max_state = a.max_state or meta["max_state"]
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_dtype = a.dtype or meta["base_dtype"]
    if device == "cuda" and base_dtype == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tok = load_tokenizer(meta["base"], revision=meta.get("base_revision"))
    ids_ab = letter_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lm = load_backbone(meta["base"], meta.get("base_revision"), base_dtype)
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(lm, str(run)).to(device)
    if base_dtype == "fp32":
        causal = peft_model.merge_and_unload()           # exact in fp32; bf16 keeps the adapter unmerged
    else:
        causal = peft_model.get_base_model()
    causal.eval()
    causal.config.use_cache = False

    reqs = load_records(os.path.join(a.data, f"{a.split}.jsonl"), source=SOURCE)
    if a.limit:
        reqs = reqs[: a.limit]
    first, second = ORDERS[order]
    rows, latencies, truncated = [], [], 0
    t0 = time.time()
    for i, req in enumerate(reqs):
        ids, y, tr = build_item(tok, req, max_state, order)
        truncated += int(tr)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            z = letter_logits(causal, [ids], pad, device, ids_ab)[0]
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append(1000 * (time.perf_counter() - start))
        p = F.softmax(z, -1).tolist()
        by_key = {first: p[0], second: p[1]}
        gold = first if y == 0 else second
        pred = max(by_key, key=by_key.get)
        p_true, p_false = by_key["true"], by_key["false"]
        rows.append({"qid": QID, "type": "noul", "gold": gold, "pred": pred, "correct": pred == gold,
                     "p_gold": by_key[gold], "conf": by_key[pred],
                     "brier": (p_false - float(gold == "false")) ** 2 + (p_true - float(gold == "true")) ** 2,
                     "level_err": None, "probs": {"false": p_false, "true": p_true}, "meta": req["_meta"]})
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(reqs)} records, {statistics.median(latencies):.0f} ms/record median", flush=True)

    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    correct = [r["correct"] for r in rows]
    majority = max(Counter(gold).values()) / len(rows)
    n = len(rows)
    entry = {"type": "noul", "n": n, "accuracy": round(sum(correct) / n, 4), "majority_class": round(majority, 4),
             "lift_over_majority": round(sum(correct) / n - majority, 4),
             "macro_f1": round(macro_f1(gold, pred), 4), "brier": round(sum(r["brier"] for r in rows) / n, 4),
             "ece": round(ece([r["conf"] for r in rows], correct), 4),
             "mean_confidence": round(sum(r["conf"] for r in rows) / n, 4)}
    report = {"run": a.run, "split": a.split, "dtype": base_dtype, "device": device, "readout": "lm_head A/B logits",
              "order": order, "records": n, "questions": n, "max_state": max_state, "states_truncated": truncated,
              "micro_accuracy": entry["accuracy"], "macro_accuracy_over_questions": entry["accuracy"],
              "brier": entry["brier"], "majority_micro": entry["majority_class"],
              "latency_ms_per_record": {"median": round(statistics.median(latencies), 2),
                                        "p95": round(sorted(latencies)[int(0.95 * (n - 1))], 2),
                                        "mean": round(statistics.mean(latencies), 2)},
              "questions_per_second": round(n / (sum(latencies) / 1000), 1), "wall_seconds": round(time.time() - t0, 1),
              "by_type": {"noul": {"questions": 1, "n": n, "micro_accuracy": entry["accuracy"],
                                   "macro_f1_mean": entry["macro_f1"], "brier": entry["brier"],
                                   "majority_micro": entry["majority_class"]}},
              "per_question": {QID: entry}}
    suffix = "" if order == meta["order"] else f"-{order}"
    out = a.out or str(run / f"eval-{a.split}{suffix}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    preds_path = os.path.splitext(out)[0] + "-preds.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {preds_path}")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=2))
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--base", required=True, help="local dir or Hub id of the base model")
    t.add_argument("--base_revision", default="")
    t.add_argument("--base_dtype", choices=["fp32", "bf16"], default="fp32")
    t.add_argument("--data", required=True)
    t.add_argument("--val_data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight_decay", type=float, default=0.01)
    t.add_argument("--lora", type=int, default=16)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--accum", type=int, default=8)
    t.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16", help="bf16 autocast (CUDA only)")
    t.add_argument("--checkpointing", type=int, choices=[0, 1], default=1)
    t.add_argument("--max_state", type=int, default=512)
    t.add_argument("--max_branch", type=int, default=4096)
    t.add_argument("--max_records", type=int, default=0)
    t.add_argument("--order", choices=list(ORDERS), default="excluded_first")
    t.add_argument("--save_every", type=int, default=0)
    t.add_argument("--log_every", type=int, default=20)
    t.add_argument("--val_every", type=int, default=200)
    t.add_argument("--val_records", type=int, default=100000)
    t.add_argument("--eval_batch", type=int, default=4, help="validation batch size (does not change results)")
    t.add_argument("--device", default=None)
    t.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("eval")
    e.add_argument("--run", required=True, help="a step-N checkpoint directory written by `train`")
    e.add_argument("--data", required=True, help="dataset directory holding <split>.jsonl")
    e.add_argument("--split", default="development")
    e.add_argument("--allow-test", action="store_true")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--order", choices=list(ORDERS), default=None, help="default: the order used in training")
    e.add_argument("--dtype", choices=["fp32", "bf16"], default=None, help="backbone dtype; default: as trained")
    e.add_argument("--max_state", type=int, default=None)
    e.add_argument("--device", default=None)
    e.add_argument("--out", default=None)

    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
