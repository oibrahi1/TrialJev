#!/usr/bin/env python3
"""Verify the selected checkpoints, assemble three Hugging Face model repositories, and (only with --push) upload them.

    # 1. verify and assemble, no network, nothing uploaded
    python prepare_hf_release.py --src CKPT_DIR --out hf_release --user YOUR_HF_NAME --meta_dirs DIR1 DIR2 ... --dry_run
    # 2. upload as PRIVATE repositories (to test)
    python prepare_hf_release.py --src CKPT_DIR --out hf_release --user YOUR_HF_NAME --meta_dirs DIR1 DIR2 ... --push
    # 3. publish (make them public) when everything has been checked on the Hub
    python prepare_hf_release.py --src CKPT_DIR --out hf_release --user YOUR_HF_NAME --meta_dirs DIR1 DIR2 ... --push --public

Three repositories are built (the layout the manuscript promises): TrialJev, TrialJev-FineTunedLLM, TrialJev-ABLogit, each with one
sub-folder per backbone / initialization / seed, plus a model card and a SHA-256 manifest, and one Hub collection.
Copied per checkpoint: adapter weights, the pointer head, small metadata, the tokenizer files and the run's training config, metrics
and log (found through --meta_dirs). The per-pair evaluation files that sit next to the checkpoints (eval-*.json, eval-*-preds.jsonl,
which contain TREC judgments) and the auto-generated PEFT README are never copied. The Hub token is read from HF_TOKEN or the cached
login; it is never printed or written.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REVISIONS = {"Qwen/Qwen3.5-0.8B-Base": "not recorded at training time", "Qwen/Qwen3.5-9B-Base": "68c46c4b3498877f3ef123c856ecfde50c39f404",
             "Qwen/Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654", "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}
META_FILES = ("training_config.json", "training_metrics.json", "train_log_rank0.jsonl")
COPY = ("adapter_model.safetensors", "adapter_config.json", "head.pt", "control_meta.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
FAMILIES = {
    "TrialJev": dict(prefix="run1-", title="TrialJev (pointer-head eligibility scorer)", need=("adapter_model.safetensors", "adapter_config.json", "head.pt")),
    "TrialJev-FineTunedLLM": dict(prefix="sft-", title="TrialJev comparison: LLM fine-tuned to write the verdict", need=("adapter_model.safetensors", "adapter_config.json", "control_meta.json")),
    "TrialJev-ABLogit": dict(prefix="ctl-", title="TrialJev ablation: A/B-logit LoRA classifier", need=("adapter_model.safetensors", "adapter_config.json", "control_meta.json")),
}
# run folder -> (sub-folder name, backbone, selected step, description, full-test accuracy %). Accuracies are from the manuscript (6,975 pairs).
REG = {
    "run1-q35-0.8b-base": ("qwen3.5-0.8b-fresh-seed0", "Qwen/Qwen3.5-0.8B-Base", 1000, "0.8B, fresh initialization, seed 0", 54.27),
    "run1-q35-0.8b-base-s1": ("qwen3.5-0.8b-fresh-seed1", "Qwen/Qwen3.5-0.8B-Base", 100, "0.8B, fresh initialization, seed 1", 55.23),
    "run1-q35-0.8b-base-s2": ("qwen3.5-0.8b-fresh-seed2", "Qwen/Qwen3.5-0.8B-Base", 500, "0.8B, fresh initialization, seed 2", 53.32),
    "run1-q35-0.8b-clinit": ("qwen3.5-0.8b-medjev-init-seed0", "Qwen/Qwen3.5-0.8B-Base", 400, "0.8B, MedJev clinical initialization, seed 0", 73.48),
    "run1-q35-0.8b-clinit-s1": ("qwen3.5-0.8b-medjev-init-seed1", "Qwen/Qwen3.5-0.8B-Base", 500, "0.8B, MedJev clinical initialization, seed 1", 68.22),
    "run1-q35-0.8b-clinit-s2": ("qwen3.5-0.8b-medjev-init-seed2", "Qwen/Qwen3.5-0.8B-Base", 500, "0.8B, MedJev clinical initialization, seed 2", 70.44),
    "run1-q35-9b-base": ("qwen3.5-9b-seed0", "Qwen/Qwen3.5-9B-Base", 400, "9B, seed 0", 78.52),
    "run1-q35-9b-base-s1": ("qwen3.5-9b-seed1", "Qwen/Qwen3.5-9B-Base", 400, "9B, seed 1", 77.65),
    "run1-q35-9b-base-s2": ("qwen3.5-9b-seed2", "Qwen/Qwen3.5-9B-Base", 1000, "9B, seed 2", 78.65),
    "run1-q35-27b-base": ("qwen3.5-27b", "Qwen/Qwen3.5-27B", 400, "Qwen3.5-27B, seed 0", 80.97),
    "run1-q38-27b-base": ("qwen3.8-27b", "Qwen/Qwen3.8-27B", 400, "Qwen3.8-27B, seed 0", 81.84),
    "sft-q35-0.8b-s0": ("qwen3.5-0.8b-fresh-seed0", "Qwen/Qwen3.5-0.8B-Base", 800, "0.8B, fresh, seed 0", 74.05),
    "sft-q35-9b-s0": ("qwen3.5-9b-seed0", "Qwen/Qwen3.5-9B-Base", 400, "9B, seed 0", 79.50),
    "sft-q35-9b-s1": ("qwen3.5-9b-seed1", "Qwen/Qwen3.5-9B-Base", 400, "9B, seed 1", 79.84),
    "sft-q35-9b-s2": ("qwen3.5-9b-seed2", "Qwen/Qwen3.5-9B-Base", 200, "9B, seed 2", 75.68),
    "sft-q35-27b-s0": ("qwen3.5-27b", "Qwen/Qwen3.5-27B", 200, "Qwen3.5-27B, seed 0", 79.87),
    "sft-q38-27b-s0": ("qwen3.8-27b", "Qwen/Qwen3.8-27B", 200, "Qwen3.8-27B, seed 0", 79.93),
    "ctl-q35-0.8b-s0": ("qwen3.5-0.8b-fresh-seed0", "Qwen/Qwen3.5-0.8B-Base", 200, "0.8B, fresh, seed 0", 52.85),
    "ctl-q35-9b": ("qwen3.5-9b-seed0", "Qwen/Qwen3.5-9B-Base", 400, "9B, seed 0", 80.50),
    "ctl-q35-9b-s1": ("qwen3.5-9b-seed1", "Qwen/Qwen3.5-9B-Base", 400, "9B, seed 1", 79.00),
    "ctl-q35-9b-s2": ("qwen3.5-9b-seed2", "Qwen/Qwen3.5-9B-Base", 200, "9B, seed 2", 75.18),
    "ctl-q35-27b": ("qwen3.5-27b", "Qwen/Qwen3.5-27B", 400, "Qwen3.5-27B, seed 0", 81.25),
    "ctl-q38-27b": ("qwen3.8-27b", "Qwen/Qwen3.8-27B", 200, "Qwen3.8-27B, seed 0", 78.25),
}
FRAMEWORK = {"small": "torch 2.14.0, transformers 5.17.0, peft 0.21.0, accelerate 1.15.0, flash-linear-attention 0.5.2 (RTX 6000 Ada, fp32 backbone)",
             "large": "torch 2.14.0+cu130, transformers 5.17.0, peft 0.21.0, accelerate 1.15.0, flash-linear-attention 0.5.2 (H100 NVL, bf16 backbone)"}
CITATION = """```bibtex
@misc{ibrahim2026trialjev,
  title  = {TrialJev: Open-Weight Single-Pass Eligibility Scoring for Patient--Trial Screening},
  author = {Ibrahim, Omar and Hamblin, Grant and Haibe-Kains, Benjamin},
  year   = {2026},
  url    = {https://github.com/oibrahi1/TrialJev},
  note   = {Preprint; arXiv identifier to be added at posting}
}
```
Please also cite the work this builds on: MedJev (https://github.com/JunMa11/MedJev) and kev (https://github.com/jaredpalmer/kev)."""
TRAINING = """| Setting | Value |
|---|---|
| Training data | TREC Clinical Trials 2021: 60 training patients, 9,302 eligible-or-excluded pairs (development: 15 patients, 2,233 pairs) |
| Adapter | LoRA rank 16, alpha 32, dropout 0.05 on the attention, MLP and Gated DeltaNet projections |
| Optimization | AdamW, learning rate 1e-4 (one-cycle, 5% warm-up), weight decay 0.01, gradient clipping 1.0, 3 epochs (1,746 steps), effective batch 16 |
| Inputs | patient note capped at 512 tokens, state plus trial text capped at 4,096 tokens |
| Selection | lowest development loss (checked every 100 steps at 0.8B, every 200 at 9B and above); the test set was scored once |
| Precision | fp32 backbone at 0.8B, bf16 at 9B and above |"""
WHAT = {"TrialJev": "a LoRA adapter (rank 16) and a bilinear pointer head (dimension 256) that score the two answer options in one forward pass and return P(eligible), without generating text",
        "TrialJev-FineTunedLLM": "a LoRA adapter trained the conventional generative way (full-vocabulary cross-entropy on the answer tokens) to write \"Eligible\" or \"Excluded\" and stop; this is a comparison model, not the proposed method. Accuracy below is the written verdict",
        "TrialJev-ABLogit": "a LoRA adapter whose loss and readout use only the language model's scores for the letters A and B (no pointer head, no text generation); this is the ablation of the pointer readout"}
HOW = {"TrialJev": "The patient note, the question with the trial text and the two natural-language answer options are serialized once. A frozen Qwen backbone with a LoRA adapter encodes the sequence; a bilinear pointer head compares the hidden state at a decision token with the hidden state at the end of each option, and a softmax over the two scores gives the probabilities. No text is generated. This follows the typed-decision design of the open-weight MedJev and kev releases.",
       "TrialJev-FineTunedLLM": "The same backbone, LoRA settings, data, optimizer schedule and development-loss checkpoint selection as TrialJev, but the training target is the answer word followed by the end-of-sequence token and the loss covers the whole vocabulary on the answer tokens. At inference the model decodes greedily until the end token.",
       "TrialJev-ABLogit": "The same recipe as TrialJev with lettered options (A and B). The loss is a cross-entropy over the two letter logits at the last position, and inference reads those two logits in one forward pass. It adds no parameters beyond the LoRA adapter."}

USAGE_TRIALJEV = """```python
from huggingface_hub import snapshot_download
local = snapshot_download("omareng/TrialJev", allow_patterns=["SUB/*"])      # one sub-folder, not the whole repository
path = f"{local}/SUB"

import torch
from trialmatch.checkpoint import load            # code repository: https://github.com/oibrahi1/TrialJev
from trialmatch.records import materialize

tok, model = load(path, "cuda", dtype=torch.bfloat16, merge=False)     # the backbone is fetched from its own Hub repository
model.lm.config.use_cache = True
request = {"state": "<patient note>",
           "questions": {"eligible": {"type": "noul", "instructions": "<question and trial text>",
                                       "criteria": {"false": "<excluded option>", "true": "<eligible option>"},
                                       "label": True, "src": "x"}}}      # the label is ignored when scoring
enc = model.encode(tok, materialize(request), max_state=512, max_branch=4096)
with torch.no_grad():
    probs = model.probs(enc)                       # one forward pass; probs[0] = [P(excluded), P(eligible)]
    # probs, _ = model.probs_and_prefix(enc)       # two-pass path used for the paper's accuracies (accuracy differs by at most 0.09 points)
print(float(probs[0][1]))
```
The exact prompt serialization used in the manuscript (question wording, option text, trial sections) is in `docs/REPRODUCE.md` and `build_data.py` of the code repository.
To reproduce a reported number, score a split with `python -m trialmatch.evaluate --run <path> --data <dir> --split <split> --dtype bf16` (use `--dtype fp32` for the 0.8B models, `bf16` for 9B and above)."""
USAGE_SFT = """```python
from huggingface_hub import snapshot_download
local = snapshot_download("omareng/TrialJev-FineTunedLLM", allow_patterns=["SUB/*"])      # one sub-folder, not the whole repository
path = f"{local}/SUB"
```
Evaluate with the code repository (https://github.com/oibrahi1/TrialJev):
`python scripts/generate_baselines.py run --condition sft-generate --base <backbone> --base_revision <revision> --adapter <path> --data <dir>/test.jsonl --per_patient 0`.
The prompt is the zero-shot word prompt of the manuscript (Appendix D); the model writes " Eligible" or " Excluded" and stops."""
USAGE_CTL = """```python
from huggingface_hub import snapshot_download
local = snapshot_download("omareng/TrialJev-ABLogit", allow_patterns=["SUB/*"])      # one sub-folder, not the whole repository
path = f"{local}/SUB"
```
Evaluate with the code repository (https://github.com/oibrahi1/TrialJev):
`python scripts/train_lora_letter.py eval --run <path> --data <dir> --split <split>`.
The answer is read from the logits of " A" and " B" at the last position of the lettered prompt (Appendix D)."""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def family_of(run):
    return next(k for k, v in FAMILIES.items() if run.startswith(v["prefix"]))


def find_meta(meta_dirs, run, name):
    for d in meta_dirs:
        for cand in (Path(d) / f"{run}__{name}", Path(d) / "runs" / run / name, Path(d) / run / name):
            if cand.exists():
                return cand
    return None


def verify(src, allow_missing):
    problems, ok = [], []
    for run, (sub, base, step, desc, acc) in REG.items():
        d = src / run / f"step-{step}"
        fam = family_of(run)
        if not d.exists() or not any(d.iterdir()):
            if allow_missing:
                print(f"  skipped (not present): {run}/step-{step}")
            else:
                problems.append(f"{run}/step-{step}: missing or empty")
            continue
        names = {p.name for p in d.iterdir() if p.is_file()}
        miss = [n for n in FAMILIES[fam]["need"] if n not in names]
        if miss:
            problems.append(f"{run}/step-{step}: missing {miss}")
            continue
        if any(p.is_dir() for p in d.iterdir()):
            problems.append(f"{run}/step-{step}: contains a nested folder (copied one level too deep?)")
            continue
        cfg = json.load(open(d / "adapter_config.json", encoding="utf-8"))
        if cfg.get("base_model_name_or_path") != base:
            problems.append(f"{run}: adapter base {cfg.get('base_model_name_or_path')} != expected {base}")
        if "control_meta.json" in names:
            m = json.load(open(d / "control_meta.json", encoding="utf-8"))
            if m.get("step") != step or m.get("base") != base:
                problems.append(f"{run}: control_meta step/base {m.get('step')}/{m.get('base')} != {step}/{base}")
        if "head.pt" in names:
            try:
                import torch
                m = torch.load(d / "head.pt", map_location="cpu", weights_only=False)
                if m.get("step") != step or m.get("base") != base or m.get("head_dim") != 256:
                    problems.append(f"{run}: head.pt step/base/dim {m.get('step')}/{m.get('base')}/{m.get('head_dim')} != {step}/{base}/256")
                if any(k not in m["head"] for k in ("q.weight", "k.weight")):
                    problems.append(f"{run}: head.pt lacks the pointer projections")
            except ImportError:
                print("  (torch not installed: head.pt contents not inspected)")
        ok.append(run)
    return ok, problems


def card(fam, runs, metrics=None):
    rows = []
    for run in runs:
        sub, base, step, desc, acc = REG[run]
        rev = REVISIONS[base]
        m = (metrics or {}).get(run)
        fmt = lambda v, d: "n/a" if v is None else f"{v:.{d}f}"
        extra = f" {fmt(m['macro_f1'], 1)}% | {fmt(m['auroc'], 3)} | {fmt(m['brier'], 3)} | {fmt(m['ece'], 3)} |" if m else ""
        rows.append(f"| `{sub}` | {desc} | {base} ({('revision `' + rev + '`') if len(rev) > 30 else 'revision ' + rev}) | {step} | {(m['accuracy'] if m else acc):.2f}% |{extra}")
    head = ("| Sub-folder | Configuration | Backbone (revision) | Step | Accuracy | Macro-F1 | AUROC | Brier | ECE |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n"
            if metrics else "| Sub-folder | Configuration | Backbone (revision) | Step | Full-test accuracy |\n|---|---|---|---:|---:|\n")
    base_models = sorted({REG[r][1] for r in runs})
    front = "---\nlicense: apache-2.0\nlibrary_name: peft\nbase_model:\n" + "".join(f"- {b}\n" for b in base_models) + \
            "tags:\n- clinical-trials\n- patient-trial-matching\n- eligibility-screening\n- lora\n- research-preview\n---\n"
    sub0 = REG[next((r for r in runs if "q38" in r), runs[0])][0]
    usage = {"TrialJev": USAGE_TRIALJEV, "TrialJev-FineTunedLLM": USAGE_SFT, "TrialJev-ABLogit": USAGE_CTL}[fam].replace("SUB", sub0)
    fw = "Large models: " + FRAMEWORK["large"] + (". 0.8B models: " + FRAMEWORK["small"] if any("0.8b" in r for r in runs) else "")
    text = front + f"""# {FAMILIES[fam]['title']}

Research preview. **Not for clinical decisions, trial enrollment or patient care.** TrialJev is independent research that adopts the design of the
open-weight MedJev and kev releases (Apache-2.0). It is not affiliated with or endorsed by TypeSafe AI, the developer of Jev.

This repository holds {WHAT[fam]}. Each sub-folder is one selected checkpoint (lowest development loss), trained on TREC Clinical Trials 2021 and evaluated on
6,975 eligible-or-excluded pairs from 50 unseen TREC 2022 patients (always predicting "eligible" scores 56.47%).

## Checkpoints

""" + head + "\n".join(rows) + f"""

All metrics are on the 6,975 eligible-or-excluded TREC 2022 test pairs and are the numbers reported in the manuscript for that exact checkpoint: accuracy and macro-F1 of the model's decision,
AUROC from P(eligible), the binary Brier score and a 10-bin expected calibration error (lower is better for both). For the fine-tuned LLM the accuracy and macro-F1 are those of the written verdict
(an unparsable answer counts as wrong) and the AUROC comes from the first-token two-way score; Brier and ECE are not reported because a written verdict carries no probability.
Higher AUROC with a higher ECE, as for TrialJev against the A/B-logit control, means the scores rank pairs equally well but their scale is less calibrated: choose a cutoff on development data.
(The TrialJev numbers use the canonical two-pass evaluator; 27B and
Qwen3.8-27B are single seeds, and 95% intervals are about plus or minus 3 points because the 50 test patients are the independent units). The backbone
weights are not included: the adapter loads on the upstream model at the listed revision. Seed-0 checkpoints of the 9B and 27B models did not record their
backbone revision; the revision shown is the only snapshot in the training cache. Every sub-folder also holds `training_config.json`, `training_metrics.json` and the training
log, and `SHA256SUMS.txt` lists the checksum of every file.

## How it works

{HOW[fam]}

## Usage

{usage}

## Training

{TRAINING}

## Training data

TREC Clinical Trials 2021 patient cases and ClinicalTrials.gov records. The processed data are not redistributed because they derive from TREC topics and registry text; the code
repository explains how to rebuild them and lists SHA-256 hashes. No institutional patient records were used.

## Limitations and intended use

Research use only. One benchmark of 50 synthetic patients; pairs that assessors had already judged relevant (not retrieval); synthetic, short patient notes; no explanation or rationale
is produced; the scores are not calibrated probabilities and a cutoff should be chosen on development data; one seed at 27B. `head.pt` is a PyTorch pickle holding the pointer-head
tensors and small metadata; load it only from this repository. See the manuscript and `docs/RESULTS.md` in the code repository.

## Citation

{CITATION}

## License

Apache-2.0 for the adapters and heads. The backbones are released by Alibaba Cloud under their own licenses; the TREC data belong to NIST/TREC and are not included.

## Framework versions

{fw}
"""
    if fam == "TrialJev-FineTunedLLM":     # the calibration remark is about models that return probabilities
        remark = "Higher AUROC with a higher ECE, as for TrialJev against the A/B-logit control, means the scores rank pairs equally well but their scale is less calibrated: choose a cutoff on development data."
        text = text.replace(remark + chr(10), "")
    return text


def build(src, out, present_runs, meta_dirs=(), metrics=None):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    summary = {}
    for fam in FAMILIES:
        runs = [r for r in REG if family_of(r) == fam and r in present_runs]
        if not runs:
            continue
        repo = out / fam
        manifest = []
        for run in runs:
            sub, base, step, desc, acc = REG[run]
            dst = repo / sub
            dst.mkdir(parents=True)
            for name in COPY:
                s = src / run / f"step-{step}" / name
                if s.exists():
                    shutil.copy2(s, dst / name)
                    manifest.append(f"{sha256(dst / name)}  {sub}/{name}")
            for name in META_FILES:
                m = find_meta(meta_dirs, run, name)
                if m is not None:
                    shutil.copy2(m, dst / name)
                    manifest.append(f"{sha256(dst / name)}  {sub}/{name}")
                else:
                    print(f"  note: {name} not found for {run}")
        (repo / "README.md").write_text(card(fam, runs, metrics), encoding="utf-8", newline="\n")
        (repo / "SHA256SUMS.txt").write_text("\n".join(sorted(manifest, key=lambda x: x.split("  ")[1])) + "\n", encoding="utf-8", newline="\n")
        size = sum(p.stat().st_size for p in repo.rglob("*") if p.is_file())
        summary[fam] = (len(runs), size / 1e9)
    return summary


def push(out, user, public):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub")
    api = HfApi()
    who = api.whoami()["name"]
    print(f"logged in as {who} (token not shown); uploading to {user}/ as {'PUBLIC' if public else 'PRIVATE'} repositories")
    ids = []
    for fam in FAMILIES:
        folder = out / fam
        if not folder.exists():
            continue
        rid = f"{user}/{fam}"
        api.create_repo(rid, repo_type="model", private=not public, exist_ok=True)
        api.upload_folder(folder_path=str(folder), repo_id=rid, repo_type="model", commit_message=f"Update {fam}: model card, usage, training files")
        if public:
            api.update_repo_settings(rid, private=False)
        ids.append(rid)
        print("uploaded", rid)
    try:
        col = api.create_collection(title="TrialJev", namespace=user, description="TrialJev eligibility scorer, the fine-tuned LLM comparison and the A/B-logit ablation", private=not public, exists_ok=True)
        for rid in ids:
            api.add_collection_item(col.slug, item_id=rid, item_type="model", exists_ok=True)
        print("collection:", f"https://huggingface.co/collections/{col.slug}")
    except Exception as e:  # collections need a recent hub client and a token with collection permission
        print("collection step skipped (create it by hand in the browser):", type(e).__name__)
    print("links:", [f"https://huggingface.co/{r}" for r in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with <run>/step-N checkpoints (CKPT_DIR)")
    ap.add_argument("--out", required=True, help="where to assemble the three repositories")
    ap.add_argument("--user", required=True, help="Hugging Face user or organisation name")
    ap.add_argument("--meta_dirs", nargs="*", default=[], help="folders holding <run>__training_config.json etc. or runs/<run>/training_config.json")
    ap.add_argument("--metrics", default=None, help="release_metrics.json from compute_release_metrics.py (adds macro-F1, AUROC, Brier, ECE to the cards)")
    ap.add_argument("--allow_missing", action="store_true", help="assemble only the checkpoints that are present")
    ap.add_argument("--dry_run", action="store_true", help="verify and assemble only; no network")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--public", action="store_true", help="make repositories public (default: private)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    ok, problems = verify(src, a.allow_missing)
    print(f"verified {len(ok)} of {len(REG)} checkpoints")
    if problems:
        print("PROBLEMS:")
        [print("  -", p) for p in problems]
        sys.exit(1)
    metrics = json.load(open(a.metrics, encoding="utf-8")) if a.metrics else None
    summary = build(src, out, set(ok), a.meta_dirs, metrics)
    for fam, (n, gb) in summary.items():
        print(f"assembled {fam}: {n} checkpoints, {gb:.2f} GB -> {out / fam}")
    leaked = [p for p in out.rglob("*") if p.is_file() and ("eval-" in p.name or p.name.endswith("preds.jsonl"))]
    assert not leaked, f"evaluation files leaked into the release: {leaked[:3]}"
    print("no per-pair or evaluation files in the release folders")
    if a.push and not a.dry_run:
        if len(ok) < len(REG) and not a.allow_missing:
            sys.exit("refusing to push an incomplete release")
        push(out, a.user, a.public)
    else:
        print("dry run: nothing uploaded")


if __name__ == "__main__":
    main()
