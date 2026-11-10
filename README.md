<h1 align="center">TrialJev</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2611.10145"><img src="https://img.shields.io/badge/paper-arXiv-blue?logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/omareng/TrialJev"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-omareng-yellow" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License: Apache-2.0"></a>
</p>

*Python package: `trialmatch`. Trained weights: [Hugging Face](#pretrained-weights).*

**Fast, single-pass patient-to-clinical-trial eligibility screening with a pointer-head LLM.**

Given a patient description and a clinical-trial record, TrialJev returns the probability
that the patient is **eligible** versus **excluded**. It does not generate text: frozen Qwen
base weights carry trainable LoRA adapters, and a small pointer head scores the two answer options.
Scoring is one forward pass per pair with no decoding.
It is meant as a first-pass screening score in a staged workflow, with explanations and the final
decision left to a clinician or a downstream reasoning model.

Built on [MedJev](https://github.com/JunMa11/MedJev) and [kev](https://github.com/jaredpalmer/kev)
— see [NOTICE](NOTICE) and [Citing](#citing). TrialJev is independent research that adopts the
design of those open-weight releases. It is not affiliated with or endorsed by TypeSafe AI, the
developer of Jev.

> **Research preview. Research use only.** Not validated for clinical decisions,
> trial enrollment, or patient care.

## What it does and does not do

| Does | Does not |
|---|---|
| Score eligible-vs-excluded for patient–trial pairs that TREC assessors already judged relevant | Retrieve trials from a registry |
| Train on TREC 2021 patients, test on 50 different TREC 2022 patients | Handle the full 3-class TREC task (`not relevant` is left out here) |
| Output pair-level probabilities in one forward pass | Explain its decision per criterion or write any text |

Unjudged pairs are never treated as negatives.

## What is compared

1. **TrialJev** — the pointer-head model (MedJev/kev design). One pass, a probability, no text.
2. **LLM without fine-tuning** — the un-adapted Qwen model asked in the prompt to answer with one word, "Eligible" or "Excluded".
3. **Fine-tuned LLM** — the conventional generative alternative: the same Qwen backbone, LoRA recipe, data and checkpoint
   selection, but trained with full-vocabulary cross-entropy to write "Eligible" or "Excluded" and stop
   (`scripts/train_lora_generate.py`; in code and JSON files it is called `sft-generate`, and some keys say "writer").
4. **Further zero-shot baselines** — an explain-then-answer LLM (512-token cap, 200-pair subset); untrained base models
   scored on lettered options; and the released MedJev adapter as shipped.
5. **Hosted Jev** — TypeSafe's Jev API (`jev-1.13.0`), used without adaptation by us, raw and after a cutoff fitted
   on TREC 2021 development data only (the calibration MedJev applied to Jev).
6. **A/B-logit control** — an ablation: the same LoRA recipe, but the answer is read from the language model's own
   scores for `A` and `B` and the loss uses only those two scores. It generates no text and is *not* conventional
   generative fine-tuning. (Older logs and file names call it `ctl-*`, `lora-logits` or "plain LoRA".)

## Results

TREC 2022 test set: 6,975 pairs from 50 patients (3,939 eligible / 3,036 excluded). Always predicting
"eligible" gives **56.47% accuracy**. Checkpoints are chosen on development loss. Accuracy, mean ± sd over seeds
where three seeds were run; intervals and all other metrics are in [docs/RESULTS.md](docs/RESULTS.md).

**TrialJev against LLMs with and without fine-tuning** (main comparison; same data and recipe for the fine-tuned LLM).
Each cell is accuracy and, in parentheses, median latency per pair as a multiple of TrialJev's one-pass path (batch size 1,
the same 992 pairs, one GPU, run one after another).

| Backbone | LLM, no fine-tuning (prompted for one word) | Fine-tuned LLM (writes the word) | TrialJev |
|---|---:|---:|---:|
| Qwen3.5-0.8B-Base (TrialJev: MedJev init; fine-tuned LLM: fresh; prompted: Base, plain prompt) | 56.01% (1.47×) | 74.05% (1.85×) | 70.71 ± 2.64% (1×, 54 ms) |
| Qwen3.5-9B (prompted: instruct checkpoint; others: Base) | 77.35% (2.30×) | 78.34 ± 2.31% (2.45×) | 78.27 ± 0.55% (1×, 106 ms) |
| Qwen3.5-27B (post-trained) | 72.89% (1.96×) | 79.87% (2.62×) | 80.97% (1×, 178 ms) |
| Qwen3.8-27B (post-trained) | 61.66% (1.90×) | 79.93% (2.70×) | **81.84%** (1×, 178 ms) |

Seeds: TrialJev and the fine-tuned LLM are three-seed means at 9B; TrialJev at 0.8B is a three-seed mean; everything else
is one seed. Fresh 0.8B TrialJev scores 54.27 ± 0.95%. Always predicting "eligible" gives 56.47%.

**Other comparisons** (full test set unless marked).

| System | 0.8B | 9B | 27B | Qwen3.8-27B |
|---|---:|---:|---:|---:|
| Untrained model, letter scoring | 55.70% | 43.68% | 64.00% | — |
| Explain-then-answer LLM (200 pairs; TrialJev on the same pairs) | — | 65.5% (83.0%) | 61.5% (79.5%) | 52.5% (82.0%) |
| Hosted Jev, raw / development-calibrated | 63.89% / 81.28% | same | same | same |
| A/B-logit control (ablation) | 52.85% | 78.23 ± 2.74% | 81.25% | 78.25% |

What the numbers say:

- TrialJev reaches about 81–82% accuracy at 27B on unseen TREC 2022 patients (AUROC 0.89).
- Against the fine-tuned LLM the accuracy is similar (paired differences −0.97 to +1.91 points; only Qwen3.8-27B excludes
  zero, in unadjusted intervals), but the fine-tuned LLM takes 1.85–2.70× as long per pair, and its accuracy varied more
  across three 9B seeds (sd 2.31 against 0.55). Similar observed accuracy is not proven equivalence.
- Zero-shot prompted and explain-then-answer LLMs are less accurate and slower: explanation is 17.5–29.5 points below
  TrialJev on the same 200 pairs and 68–139× slower. That is a zero-shot, token-capped baseline, not evidence that
  reasoning hurts.
- Hosted Jev ranks about as well as TrialJev 27B (AUROC 0.882) but is over-cautious at its default cutoff (63.89%);
  with a cutoff fitted on development data it reaches 81.28%, level with TrialJev at 27B and above the smaller sizes.
  Its latency is 325 ms per request over the network, which is not comparable with local GPU times. When the same
  development-fitted correction is applied to every system, TrialJev is level with corrected Jev at 27B (−0.29 points
  [−1.16, +0.62]) and on Qwen3.8-27B (+0.60 [−0.30, +1.49]), and below it at 9B (−2.34 [−3.60, −0.95]) and 0.8B (−10.47);
  TrialJev itself moves by 0.0 to 0.7 points under the correction, so its native cutoff is already close to the fitted one.
  We do not claim TrialJev is more accurate than corrected Jev; the fit uses 15 development patients and is noisy.
- On Qwen3.5 the A/B-logit control matches TrialJev in accuracy and ranking and is better calibrated, so the pointer head
  is not the source of accuracy there.
- At 0.8B the released MedJev adapter and head are what make TrialJev work (fresh: 54.27%), but a fresh fine-tuned LLM
  reaches 74.05%, so at this size TrialJev's case is speed, not accuracy.

Please read the limits before citing a number ([docs/RESULTS.md](docs/RESULTS.md)): the 50 test patients, not the
6,975 pairs, are the independent units (intervals are about ±3 points); 27B and Qwen3.8 are single seeds; the table is
not a clean parameter-scaling experiment because starting checkpoints differ in kind; latency is batch size 1 in one
implementation and varied between runs, so only within-chain ratios are used; and TREC 2022 was examined repeatedly while
developing the recipe, so it is not an untouched test.

## Pretrained weights

The trained adapters and pointer heads for every configuration and seed in the paper (23 checkpoints) are on Hugging Face.
They are not standalone `transformers` models: the pointer head is loaded by this repository's code
(`trialmatch.checkpoint.load`), and the Qwen backbone is fetched from its own Hub repository (each model card names the revision used; for the
Qwen3.5-0.8B-Base backbone it was not recorded at training time). The weights contain no TREC or ClinicalTrials.gov data.

| Hugging Face repository | What it holds | Checkpoints |
|---|---|---:|
| [`omareng/TrialJev`](https://huggingface.co/omareng/TrialJev) | TrialJev pointer-head models: 0.8B (fresh and MedJev-initialized, three seeds each), 9B (three seeds), 27B, Qwen3.8-27B | 11 |
| [`omareng/TrialJev-FineTunedLLM`](https://huggingface.co/omareng/TrialJev-FineTunedLLM) | the conventionally fine-tuned LLM comparison (writes the verdict) | 6 |
| [`omareng/TrialJev-ABLogit`](https://huggingface.co/omareng/TrialJev-ABLogit) | the A/B-logit control ablation | 6 |

Each repository has one sub-folder per checkpoint (for example `qwen3.5-0.8b-medjev-init-seed0`), and each model card gives that
checkpoint's accuracy, macro-F1, AUROC, Brier score and ECE on the 6,975 TREC 2022 test pairs, its training settings and a
SHA-256 manifest. To score a split with a downloaded TrialJev checkpoint (you need the processed data, see [DATA.md](DATA.md)):

```bash
pip install -e ".[cuda]"
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("omareng/TrialJev", allow_patterns=["<sub-folder>/*"]))   # prints the local folder
PY
python -m trialmatch.evaluate --run <that folder>/<sub-folder> --data data/tm-v1-b4096 --split development --dtype fp32
```

`head.pt` is a PyTorch pickle: load it only from the official repositories above. The scripts that verified and uploaded the
checkpoints are in `scripts/release/`.

## Quick start

```bash
pip install -e ".[dev]"            # CPU: data building, tests, analysis
pip install -e ".[cuda,train]"     # GPU training and evaluation
python check_env.py                # checks Python/CUDA and runs a real forward pass
pytest -q                          # CPU-only; data-dependent tests skip without the data
```

**Data.** The processed datasets are not included (they contain TREC topics and
ClinicalTrials.gov text). [DATA.md](DATA.md) explains how to rebuild them and verify the
result against `data/DATA_MANIFEST.sha256`.

**Train and evaluate TrialJev.**

```bash
bash scripts/smoke_test.sh 0                                         # minutes, one GPU
bash scripts/run1_small.sh                                           # the 0.8B experiments
bash scripts/run1_large.sh run1-q35-9b-base Qwen/Qwen3.5-9B-Base 0   # 9B / 27B
python scripts/pick_best.py runs/<run>                               # lowest dev-loss checkpoint
bash scripts/eval_run.sh runs/<run> 0 development
bash scripts/eval_run.sh runs/<run> 0 test                           # final report only
python scripts/analyze_predictions.py --preds <ckpt>/eval-test-preds.jsonl --out report.json
```

**Control, fine-tuned LLM, baselines and timing.** `scripts/train_lora_letter.py` trains and scores the A/B-logit control;
`scripts/train_lora_generate.py` trains the fine-tuned LLM; `scripts/sft_lane.sh KEY GPU SEED` trains, selects and
evaluates one fine-tuned LLM on one GPU; `scripts/generate_baselines.py` runs prompted generation, the explain-then-answer
baseline, the matched-prompt zero-shot condition and the fine-tuned LLM's evaluation; `scripts/time_pointer.py` times
TrialJev's two serving paths; `scripts/timing_chain.sh KEY GPU` times every system of one size in one chain.

**Hosted Jev.** `scripts/run_jev_trial.py` sends the processed pairs (label removed) to the Jev API and saves validated,
resumable answers. The API key is read from the `TYPESAFE_API_KEY` environment variable or a text file you pass with
`--key_file`; keep that file outside the repository. Never commit a key.

Commands and launchers: [docs/REPRODUCE.md](docs/REPRODUCE.md).

**Statistics.** `scripts/analysis/` recomputes every table from the saved per-pair predictions with patient-clustered,
paired bootstrap intervals (`compute_stats.py`), the generation and timing tables (`compute_gen_stats.py`),
candidate-pool ranking metrics (`compute_rank_stats.py`), the fine-tuned LLM, timing-chain and explanation tables
(`compute_suite_stats.py`), and the hosted-Jev scoring and comparison (`score_jev_api.py`, `compare_jev_all.py`). `analyze_predictions.py` reports AUROC, average precision,
ECE and confusion matrices for one run or a pair of runs. The summary statistics behind the reported tables are committed in
`results/` with SHA-256 hashes (aggregated metrics only; the per-pair predictions are not redistributed).

**Zero-shot letter probe.** `scripts/prefinetune_probe.py` reads a model's own A/B next-token logits with no fine-tuning.
Run both `--order excluded_first` and `--order eligible_first`: a large gap, or near-constant predictions, indicates
letter-position bias rather than skill.

## Repository layout

| Path | Contents |
|---|---|
| `trialmatch/` | decision model, trainer, evaluator, checkpoint loader |
| `build_data.py`, `build_trial_text.py` | TREC judgments → training records; trial XML → text |
| `scripts/` | launch scripts, checkpoint picker, A/B-logit control and fine-tuned LLM trainers, generation, timing and hosted-Jev scripts, zero-shot probe, prediction analysis, run audit |
| `scripts/release/` | checks the released checkpoints and builds the Hugging Face repositories and model cards |
| `scripts/analysis/` | statistics from saved predictions: paired bootstrap, generation/timing tables, ranking metrics, fine-tuned LLM and hosted-Jev comparisons |
| `results/` | summary statistics (JSON) behind the reported tables, with `SHA256SUMS`; no per-pair data or gold labels |
| `tests/` | CPU unit tests; frozen-data regression tests (need the data) |
| `data/` | split manifest, statistics and SHA-256 manifest only |
| `docs/` | [results](docs/RESULTS.md), [reproduction](docs/REPRODUCE.md), [model card](docs/MODEL_CARD.md) |

## License and credits

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Built on MedJev and kev; backbones from
Qwen (Alibaba Cloud). TREC Clinical Trials track data belong to NIST/TREC and are not
redistributed.

## Citing

If you use TrialJev, please cite:

> Ibrahim O, Hamblin G, Haibe-Kains B. *TrialJev: Open-Weight Single-Pass Eligibility Scoring for
> Patient–Trial Screening*. arXiv:2611.10145, 2026. https://arxiv.org/abs/2611.10145

```bibtex
@misc{ibrahim2026trialjev,
  title         = {TrialJev: Open-Weight Single-Pass Eligibility Scoring for Patient--Trial Screening},
  author        = {Ibrahim, Omar and Hamblin, Grant and Haibe-Kains, Benjamin},
  year          = {2026},
  eprint        = {2611.10145},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.10145},
  howpublished  = {Code: \url{https://github.com/oibrahi1/TrialJev}}
}
```

Please also cite the projects it builds on:

- **MedJev** — *MedJev: Ultra-fast Clinical Variable Extraction from Free-text Notes*, software v0.1.0, 2026.
  https://github.com/JunMa11/MedJev
- **kev** — the typed decision-model architecture MedJev derives from. https://github.com/jaredpalmer/kev
- **Jev** — Almeida D. *Introducing System One Models & Jev*. TypeSafe AI blog, 15 September 2026.
  https://typesafe.ai/blog/introducing-system-one-models-and-jev (vendor description; its performance claims were not verified here).
- **TREC Clinical Trials track** (2021, 2022). https://www.trec-cds.org/
- Roberts K, Demner-Fushman D, Voorhees EM, Bedrick S, Hersh WR. *Overview of the TREC 2022 Clinical Trials Track*.
  https://trec.nist.gov/pubs/trec31/papers/Overview_trials.pdf
