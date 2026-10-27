#!/usr/bin/env python3
"""Conventional generative LoRA fine-tuning: the model learns to WRITE the answer as text.

This is the natural-language-generation counterpart of the pointer-head model (`trialmatch.train`) and of the
A/B-logit control (`scripts/train_lora_letter.py`). Same backbone, data, LoRA configuration, optimizer, schedule,
record order and checkpoint selection; the only difference is what is trained and read out:

  prompt   Patient note, question, the two option texts, "Reply with exactly one word: Eligible or Excluded.",
           "Answer:"   (the same words as the zero-shot word prompt of `generate_baselines.py`)
  target   " Eligible" or " Excluded" followed by the end-of-sequence token, so the model learns to stop
  loss     ordinary causal-language-model cross-entropy over the whole vocabulary at every target position
           (the prompt is not in the loss), averaged over the target tokens of each example

At test time the model decodes greedily (`scripts/generate_baselines.py run --condition sft-generate`), the text is
parsed strictly, and the time per pair includes the decoding steps. No rationale is trained: TREC provides labels only.

    python scripts/train_lora_generate.py train --base Qwen/Qwen3.5-9B-Base --base_dtype bf16 \
      --data data/tm-v1-b4096/train.jsonl --val_data data/tm-v1-b4096/development.jsonl \
      --out runs/sft-q35-9b --seed 0 --epochs 3 --lr 1e-4 --batch 2 --accum 8 \
      --max_state 512 --max_branch 4096 --dtype bf16 --checkpointing 1 \
      --log_every 20 --val_every 200 --save_every 200
    python scripts/pick_best.py runs/sft-q35-9b --path
"""
import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

QID = "eligible"
ORDERS = {"excluded_first": ("false", "true"), "eligible_first": ("true", "false")}   # option order in the prompt
WORDS = {"true": " Eligible", "false": " Excluded"}
SOURCE = "medjev"      # same record-source tag as trialmatch.train (only affects the `src` field)


# --------------------------------------------------------------------------- #
# prompt, target and loss
# --------------------------------------------------------------------------- #

def _user_tokens(tok, text):
    from trialmatch.model import user_tokens        # imported late so the pure helpers below stay testable
    return user_tokens(tok, text)


def build_prompt(tok, req, max_state, order):
    """-> (prompt token ids, state_truncated). Same words as the zero-shot word prompt, state truncated from the head."""
    q = req["questions"][QID]
    head = _user_tokens(tok, "Patient note:\n")
    state = _user_tokens(tok, req["state"])
    truncated = len(state) + 1 > max_state                 # same accounting as trialmatch.model.encode
    first, second = ORDERS[order]
    tail = (f"\n\n{q['instructions']}\n\nOptions:\n{q['criteria'][first]}\n{q['criteria'][second]}\n\n"
            "Reply with exactly one word: Eligible or Excluded.\nAnswer:")
    return head + state[: max_state - 1] + _user_tokens(tok, tail), truncated


def answer_ids(tok, label):
    """Target tokens: the answer word followed by end-of-sequence, so the model learns to stop."""
    if tok.eos_token_id is None:
        raise ValueError("the tokenizer has no end-of-sequence token")
    return tok.encode(WORDS[str(bool(label)).lower()], add_special_tokens=False) + [tok.eos_token_id]


def word_first_ids(tok):
    """First token of ' Eligible' and ' Excluded' (in that order); they must differ to give a two-way score."""
    ids = [tok.encode(WORDS[k], add_special_tokens=False)[0] for k in ("true", "false")]
    if ids[0] == ids[1]:
        raise ValueError("the first tokens of ' Eligible' and ' Excluded' collide")
    return ids


def answer_logits(causal, batch, pad_id, device):
    """batch: list of (prompt ids, answer ids). Returns one [len(answer), vocab] float tensor per example: the logits at
    the positions that predict each answer token (lm_head applied only there)."""
    seqs = [p + a for p, a in batch]
    length = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), length), pad_id, dtype=torch.long, device=device)
    att = torch.zeros((len(seqs), length), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=device)
        att[i, : len(s)] = 1
    hidden = causal.model(input_ids=ids, attention_mask=att).last_hidden_state
    out = []
    for i, (p, a) in enumerate(batch):
        pos = torch.arange(len(p) - 1, len(p) + len(a) - 1, device=device)
        out.append(causal.lm_head(hidden[i, pos]).float())
    return out


def batch_loss(logits_list, batch, device):
    """Mean over examples of the mean token cross-entropy over the full vocabulary (prompt tokens are not in the loss)."""
    losses = [F.cross_entropy(z, torch.tensor(a, device=device)) for z, (_, a) in zip(logits_list, batch)]
    return torch.stack(losses).mean()


def lora_targets(backbone_config):
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "linear_attention" in set(getattr(backbone_config, "layer_types", None) or []):
        targets += ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
    return targets


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

@torch.no_grad()
def validate(causal, tok, reqs, a, device, first_ids, autocast):
    causal.eval()
    use_cache = causal.config.use_cache
    causal.config.use_cache = False
    items, truncated = [], 0
    for req in reqs:
        prompt, tr = build_prompt(tok, req, a.max_state, a.order)
        items.append((prompt, answer_ids(tok, req["questions"][QID]["label"]), bool(req["questions"][QID]["label"])))
        truncated += int(tr)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))      # length-sorted: less padding
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    loss_sum, hits, n = 0.0, 0, 0
    for b in range(0, len(order), a.eval_batch):
        chunk = [items[i] for i in order[b: b + a.eval_batch]]
        batch = [(c[0], c[1]) for c in chunk]
        with autocast:
            logits = answer_logits(causal, batch, pad, device)
        for z, (_, ans, gold) in zip(logits, chunk):
            loss_sum += float(F.cross_entropy(z, torch.tensor(ans, device=device)))
            two_way = z[0][first_ids]                                      # [eligible, excluded] first-token logits
            hits += int((int(two_way.argmax()) == 0) == gold)
            n += 1
    causal.config.use_cache = use_cache
    causal.train()
    return {"val/loss": loss_sum / n, "val/accuracy": hits / n, "val/questions": float(n),
            "val/acc_noul": hits / n, f"val_q/{QID}": hits / n, "val/truncated": truncated}


def train(a):
    import train_lora_letter as tl
    from trialmatch.model import load_tokenizer
    from trialmatch.records import load_records

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
    first_ids = word_first_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lm = tl.load_backbone(a.base, a.base_revision, a.base_dtype)
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
            "order": a.order, "words": [WORDS["true"], WORDS["false"]], "max_state": a.max_state,
            "max_branch": a.max_branch,
            "readout": "generative: greedy decoding of ' Eligible' / ' Excluded' + end of sequence; trained with "
                       "causal-LM cross-entropy over the whole vocabulary on the answer tokens (no pointer head, no letters)"}
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
            built = [build_prompt(tok, r, a.max_state, a.order) for r in chunk]
            truncated += sum(int(b[1]) for b in built)
            batch = [(b[0], answer_ids(tok, r["questions"][QID]["label"])) for b, r in zip(built, chunk)]
            with autocast:
                logits = answer_logits(causal, batch, pad, device)
            loss = batch_loss(logits, batch, device)
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
                    metrics = validate(causal, tok, val_reqs, a, device, first_ids, autocast)
                    print(f"  val step {step}: loss {metrics['val/loss']:.4f} first-token acc {metrics['val/accuracy']:.4f} "
                          f"in {time.time()-vt:.0f}s", flush=True)
                    log.write(json.dumps({"step": step, **metrics}) + "\n")
                    log.flush()
                if a.save_every and step % a.save_every == 0:
                    tl.save_checkpoint(peft_model, str(out_dir / f"step-{step}"), {**meta, "step": step, "epoch": ep})
                    print(f"checkpoint step-{step}", flush=True)
    tl.save_checkpoint(peft_model, str(out_dir), {**meta, "step": step})
    with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"wall_seconds": time.time() - t0, "records_seen": seen, "truncated_states": truncated,
                   "optimizer_steps": step, "peak_device_bytes": peak, "device": device, "dtype": a.dtype}, f, indent=2)
    print(f"saved {a.out} in {(time.time()-t0)/3600:.2f}h", flush=True)
    log.close()


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
    a = ap.parse_args()
    train(a)


if __name__ == "__main__":
    main()
