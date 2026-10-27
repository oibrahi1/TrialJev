"""Load a trained TrialJev checkpoint for evaluation or serving.

Derived from `kev/kev/evaluate.py` (Jared Palmer, Apache-2.0) — see NOTICE. MedJev
vendors the loader and drops upstream's `KEV_*` environment knobs in favour of
explicit arguments.

A checkpoint directory holds the LoRA adapter, `head.pt` (the pointer head plus the
base model id, rank, head dimension and the `max_state` it was trained at) and the
tokenizer. The base weights are never modified, so the artifact is ~45 MB.
"""
import json
import os

import torch

from trialmatch.model import DecisionModel, load_tokenizer


def resolve_run(run):
    """A local run directory, or a Hub repo id optionally pinned with `@revision`,
    downloaded to the HF cache."""
    if os.path.isdir(run):
        return run
    from huggingface_hub import snapshot_download
    repo, _, revision = run.partition("@")
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        # snapshot_download needs the network even for a cached repo; read the cache directly.
        from huggingface_hub.constants import HF_HUB_CACHE
        base = os.path.join(HF_HUB_CACHE, f"models--{repo.replace('/', '--')}")
        with open(os.path.join(base, "refs", "main"), encoding="utf-8") as f:
            commit = revision or f.read().strip()
        snap = os.path.join(base, "snapshots", commit)
        if os.path.isdir(snap):
            return snap
        raise FileNotFoundError(f"{repo}@{commit} is not in the offline cache {HF_HUB_CACHE}")
    return snapshot_download(repo, revision=revision or None,
                             allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])


def load(run, device, dtype=torch.float32, merge=True, attn=None):
    """-> (tokenizer, model), ready for `probs_and_prefix`.

    dtype: fp32 reproduces published probabilities exactly; bf16 is the serving
    default — measured at no accuracy cost and substantially faster.

    merge: fold the LoRA into the base weights in fp32 before any cast. Exact in
    fp32, and in bf16 both faster and closer to the fp32 numbers than running the
    adapter unmerged. Adapters that carry trainable token embeddings stay unmerged.
    """
    run = resolve_run(run)
    meta = torch.load(f"{run}/head.pt", map_location="cpu", weights_only=False)
    with open(f"{run}/adapter_config.json", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    merge = merge and not adapter_cfg.get("trainable_token_indices")
    tok = load_tokenizer(meta["base"], revision=meta.get("base_revision"))
    m = DecisionModel(meta["base"], tok, device, lora=None, revision=meta.get("base_revision"),
                      head_dim=meta.get("head_dim", 256),
                      option_isolation=meta.get("option_isolation", False),
                      dtype=torch.float32 if merge else dtype, attn=attn)
    from peft import PeftModel
    m.lm = PeftModel.from_pretrained(m.lm, run).to(device)
    if merge:
        m.lm = m.lm.merge_and_unload()          # in fp32: exact
    if dtype != torch.float32:
        m.lm = m.lm.to(dtype)
    m.head.load_state_dict(meta["head"])
    m.eval()
    return tok, m


def checkpoint_max_state(run, default=None):
    """The `max_state` a checkpoint was trained at, so scoring can match training
    rather than silently using the current module default."""
    meta = torch.load(f"{resolve_run(run)}/head.pt", map_location="cpu", weights_only=False)
    return meta.get("max_state", default)
