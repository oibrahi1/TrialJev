# Reproducing the reported runs

All runs: `--epochs 3 --lr 1e-4 --max_state 512 --max_branch 4096 --dtype bf16
--checkpointing 1 --seed 0` (seeds 1 and 2 were added at 0.8B and 9B), data `data/tm-v1-b4096`. Defaults that also matter:
LoRA r/alpha/dropout 16/32/0.05, all supported projection groups, pointer-head dim 256,
AdamW, weight decay 0.01, OneCycleLR (`pct_start` 0.05), no class weighting, one question per
record. Effective batch is 16 records (0.8B: `--batch 4 --accum 4`; 9B/27B: `--batch 2 --accum 8`).

## Code and data identity

Run these on the machine that trains. They must match for the numbers in
[RESULTS.md](RESULTS.md) to apply.

| File | SHA-256 |
|---|---|
| `trialmatch/train.py` | `ebe0d4553bbf4059681f8b41c0d4ebe9f4f07f05c7cf43174e0d0a944b260c6a` |
| `trialmatch/model.py` | `9de045f3c4f6425832b4394f91e40e9c8c2895e21a5e935d1df9bdeb726a844f` |
| `scripts/train_lora_letter.py` (A/B-logit control trainer and scorer) | `ae8ee0c73e8443940e9209512d097bcd5104454202b3d0cbee6a36fed30f2801` |
| `scripts/generate_baselines.py` (generation, matched-prompt, one-token, fine-tuned-LLM and explanation conditions; LF line endings as stored in the repository) | `7be88127920cb487851ffc055228598ec9c9d93cdaf8167552a8115ecee682f1` |
| `scripts/train_lora_generate.py` (fine-tuned LLM trainer) | `c81ca9bf15cc1627dbc5b475e777177f54be987f15af439d73d061ba2f251a0c` |
| `scripts/sft_lane.sh` | `8d747e6ebcc067ae777ac14f8cd5036d81a2958bc15b7395a41e8e4e1faa2d3b` |
| `scripts/timing_chain.sh` | `01289f28adc5e57f9ade03c2651379cb1f2d510ef05bcab865c6f4b55465ca86` |
| `scripts/run_jev_trial.py` (hosted Jev API runner) | `36aba8c7409464b7f236713a32e365920724b02797cb5024ff421dd25d8852ee` |
| `scripts/time_pointer.py` (TrialJev two-path timing) | `ad52cdc5cb62b698a06f2459f55ca7145b69933d7eb88302bced3e3e3200ad4b` |
| `scripts/analysis/compute_stats.py` | `b4097a5fb818566b05079bbfcbbf9202ea83ef5d3de96aa05fd97c75e88c2619` |
| `scripts/analysis/compute_gen_stats.py` | `fa74fd116b197e3a24c612d8c6352d13b833d35faadd4a2b3098ec61f649670f` |
| `scripts/analysis/compute_rank_stats.py` | `a8dee903027f4db31fc03e045bc7053007d5db7c392915538720117264e68f27` |
| `scripts/analysis/compute_suite_stats.py` | `b4bf9ccb82e31863024c1b6eb44b1aaaccea5fb217afdcff3e0df67bc5edd62c` |
| `scripts/analysis/score_jev_api.py` | `9e19e20087c77e4818811b44a0545bd924f1d44fdb769adb5e7ef319c31fd67a` |
| `scripts/analysis/compare_jev_all.py` | `5e3a9ac4c37e7566f4fdd157273127633bb8a67b05125ab3c4b80cacaf26a14d` |
| `scripts/analysis/calibrate_all.py` | `bf2b35a75f149194365e63f131595a98545ece9495bea27072a9d35b2b6cb4fd` |
| datasets | `data/DATA_MANIFEST.sha256` |

Hashes of the released files. The training, scoring and timing code listed above is byte-for-byte the code that produced
the reported runs, with one exception: `scripts/timing_chain.sh` was run as `e6089490261a8d2c7e698d9c8cc155034a505be25eb9a686a5af49ff2ef51318`
and differs in the released file only by one comment line (a machine-specific hint was removed). The analysis scripts under
`scripts/analysis/` were run from a Windows checkout with CRLF line endings; their hashes above are of the files as released
(LF line endings, identical content apart from docstring and help-text wording in `compute_stats.py`, `score_jev_api.py` and
`compare_jev_all.py`, where machine-specific folder names were replaced by `DIR1`, `DIR2`). No executable statement in any file
was changed.

The 2026-10-01 to 02 generation runs used a copy of an earlier `generate_baselines.py` with CRLF line endings (SHA-256 prefix
`a0f957df409c20ac`, content equal to the earlier repository version). The fine-tuned-LLM, explanation and timing-chain runs
of 2026-10-05 to 06 used the LF copy with the hash above, verified by hash on both machines before launch. The first generation runs (992-pair
sample and full-set word generation) used an earlier copy without the `zeroshot-letters` condition (prefix `ac2eb793b89b5f75`),
which is otherwise identical.

`records.py`, `checkpoint.py` and `evaluate.py` were later edited for packaging (explicit
UTF-8, offline Hub-cache lookup, per-pair prediction output); none of those edits change
training behaviour.

## Commands

0.8B (also `scripts/run1_small.sh`):

```bash
# fresh LoRA + head
python -m trialmatch.train --base Qwen/Qwen3.5-0.8B-Base \
  --data data/tm-v1-b4096/train.jsonl --val_data data/tm-v1-b4096/development.jsonl \
  --out runs/run1-q35-0.8b-base --seed 0 --epochs 3 --lr 1e-4 --batch 4 --accum 4 \
  --max_state 512 --max_branch 4096 --dtype bf16 --checkpointing 1 \
  --log_every 20 --val_every 100 --val_records 100000 --save_every 100 --wandb 0

# MedJev clinical initialization: add  --init_from init/clinical-0.8b
#   (python scripts/fetch_init.py downloads junma/MedJev-Qwen3.5-0.8B)
```

9B / 27B / Qwen3.8-27B (bf16 frozen backbone):

```bash
scripts/run1_large.sh run1-q35-9b-base  Qwen/Qwen3.5-9B-Base 0
scripts/run1_large.sh run1-q35-27b-base Qwen/Qwen3.5-27B     1
scripts/run1_large.sh run1-q38-27b-base Qwen/Qwen3.8-27B     0 <pinned-commit-sha>
```

Selection and scoring:

```bash
python scripts/pick_best.py runs/<run>                    # best = lowest dev loss
bash scripts/eval_run.sh runs/<run> 0 development
bash scripts/eval_run.sh runs/<run> 0 test                # final report only
```

For the 9B/27B evaluation use bf16 (`evaluate.py --dtype bf16`; `eval_run.sh` reads it from the
run's `training_config.json`). `evaluate.py` writes `eval-<split>.json` and
`eval-<split>-preds.jsonl` next to the checkpoint.

## Zero-shot probes

```bash
python scripts/prefinetune_probe.py --model Qwen/Qwen3.5-9B-Base --prompt plain \
  --data data/tm-v1-b4096/test.jsonl --out runs/prefinetune-q35-9b            # A=excluded
python scripts/prefinetune_probe.py --model Qwen/Qwen3.5-9B-Base --prompt plain --order eligible_first \
  --data data/tm-v1-b4096/test.jsonl --out runs/prefinetune-q35-9b-swapped
```

Use `--prompt chat` for post-trained checkpoints.

## A/B-logit control (same recipe, answer read from the logits of " A" and " B")

Train (9B shown; seeds 1 and 2 used `--seed 1` / `--seed 2`; the 0.8B control used `--base_dtype fp32 --batch 4 --accum 4
--val_every 100 --save_every 100`; 27B and Qwen3.8-27B use the 9B settings with their model ids and pinned revisions):

```bash
python scripts/train_lora_letter.py train --base Qwen/Qwen3.5-9B-Base --base_revision <sha> --base_dtype bf16 \
  --data data/tm-v1-b4096/train.jsonl --val_data data/tm-v1-b4096/development.jsonl --out runs/ctl-q35-9b --seed 0 \
  --epochs 3 --lr 1e-4 --batch 2 --accum 8 --dtype bf16 --checkpointing 1 --log_every 20 --val_every 200 --save_every 200
python scripts/pick_best.py runs/ctl-q35-9b --path                 # lowest development loss
python scripts/train_lora_letter.py eval --run <best checkpoint> --data data/tm-v1-b4096 --split test --allow-test
python scripts/train_lora_letter.py eval --run <best checkpoint> --data data/tm-v1-b4096 --split test --allow-test --order eligible_first
```

`scripts/overnight_seeds.sh` launches the seed runs and the Qwen3.8-27B control (it checks the trainer hash, the pinned
revisions and that each GPU is idle, and never overwrites an output).

## Generation baselines, matched prompt and timing

`generate_baselines.py` conditions: `zeroshot-chat` (un-adapted model, chat template, writes the word), `zeroshot-plain`
(Base model, plain prompt, writes the word), `zeroshot-letters` (no adapter, the control's lettered prompt, A/B logits),
`lora-logits` (the control, A/B logits) and `lora-generate` (the control emitting its one label token). Parsing is strict:
anything other than exactly one of the two permitted words or letters is invalid and counts as wrong.

```bash
python scripts/generate_baselines.py show   --data data/tm-v1-b4096/test.jsonl --shortest --base Qwen/Qwen3.5-27B   # prompts, no model
python scripts/generate_baselines.py sample --data data/tm-v1-b4096/test.jsonl --per_patient 20 --out runs/gen/sample_ids.json
python scripts/generate_baselines.py run --condition zeroshot-chat --base Qwen/Qwen3.5-27B --base_revision <sha> \
  --data data/tm-v1-b4096/test.jsonl --ids_file runs/gen/sample_ids.json --out runs/gen/zs-chat-27b       # 992-pair timing sample
python scripts/generate_baselines.py run --condition zeroshot-letters --base Qwen/Qwen3.5-27B --base_revision <sha> \
  --data data/tm-v1-b4096/test.jsonl --per_patient 0 --out runs/gen_zsl/zsl-27b                           # full test set
python scripts/generate_baselines.py run --condition lora-generate --base Qwen/Qwen3.5-27B --base_revision <sha> \
  --adapter runs/ctl-q35-27b/step-400 --data data/tm-v1-b4096/test.jsonl --ids_file runs/gen/sample_ids.json --out runs/gen/lora-gen-27b
python scripts/time_pointer.py --run runs/run1-q35-9b-base/step-400 --data data/tm-v1-b4096 --split test --allow-test \
  --ids_file runs/gen/sample_ids.json --dtype bf16 --out runs/gen/pointer-timing-9b
```

The 992-pair timing sample is up to 20 pairs per patient chosen by the hash order of the pair id; the same id list was used
for every condition and model size. Run all conditions of one model size one after another on one GPU, so timings can be
compared as ratios within that run. Launchers: `scripts/run_generation_baselines.sh` (992-pair runs),
`scripts/run_full_set_generation.sh CASE GPU` and `scripts/run_full_set_lane.sh` (full test set, accuracy; also times the
pointer-head paths on every pair). Set `PY` and `HF_HOME` first; the launchers stop at the first failure and check the GPU is idle.

## Fine-tuned LLM, explanation baseline and same-chain timing (2026-10-05 to 06)

```bash
export PY=<venv>/bin/python HF_HOME=<cache>          # cluster: also HF_HUB_OFFLINE=1
# smoke test first: SMOKE=1 bash scripts/sft_lane.sh q35-9b 3 0
bash scripts/sft_lane.sh q35-27b 0 0     # KEY GPU SEED: trains the fine-tuned LLM (3 epochs, same recipe), picks the
bash scripts/sft_lane.sh q38-27b 1 0     # lowest-development-loss checkpoint, evaluates once on the full test set
bash scripts/sft_lane.sh q35-9b  2 0 && bash scripts/sft_lane.sh q35-9b 2 1
bash scripts/sft_lane.sh q35-9b  3 2
bash scripts/sft_lane.sh q35-0.8b 0 0    # second machine, fp32
# when every lane prints LANE COMPLETED, time every system of one size in one chain on one GPU:
bash scripts/timing_chain.sh q35-27b 0 ; bash scripts/timing_chain.sh q38-27b 1 ; bash scripts/timing_chain.sh q35-9b 2
bash scripts/timing_chain.sh q35-0.8b 0  # second machine
```

Each lane writes `runs/sft-KEY-sSEED` and `runs/gen_sft/sft-KEY-sSEED`; each chain writes `runs/gen_time/KEY/{control,sft-generate,
pointer,zeroshot,reason}` (control, fine-tuned LLM, TrialJev both paths, prompted one-word model, then the explanation baseline
on a 200-pair subset). The scripts refuse to overwrite an output and check that the GPU is idle (`FORCE_GPU=1` overrides).
Selected checkpoints: 0.8B step 800; 9B seeds 0/1/2 steps 400/400/200; 27B and Qwen3.8-27B step 200. On the cluster, two
unrelated jobs held GPUs 0 and 1 and were stopped before the 27B lanes and the timing chains ran; the three cluster chains then
ran at the same time on GPUs 0, 1 and 2, and the 0.8B chain ran alone on the second machine.

## Hosted Jev API (run from a workstation with network access)

Needs an API key. Put it in a text file outside the repository (or export `TYPESAFE_API_KEY`); it is never printed or saved.

```bash
D=<folder holding the processed development.jsonl and test.jsonl>; K=<path to key file>
python scripts/run_jev_trial.py --data $D --split test --limit 20 --workers 1 --key_file $K --out runs/jev_api/pilot.jsonl   # pilot
python scripts/run_jev_trial.py --data $D --split test --ids_file runs/gen/sample_ids.json --workers 1 --expect_model jev-1.13.0 \
  --key_file $K --out runs/jev_api/timing992.jsonl                                                                # latency
python scripts/run_jev_trial.py --data $D --split test --workers 6 --expect_model jev-1.13.0 --key_file $K --out runs/jev_api/test.jsonl
python scripts/run_jev_trial.py --data $D --split development --workers 6 --expect_model jev-1.13.0 --key_file $K --out runs/jev_api/development.jsonl
```

`--dry_run` prints the pair count and an estimated cost without a key or network. Run one copy at a time: a second copy writing
the same file produces duplicate and interleaved rows (this happened once; see RESULTS.md). The runner checks every answer,
pins the returned model string, and verifies a resumed file. Total cost of our runs was about $0.45 at $0.042 per million input
tokens. Tests with a mock server: `pytest tests/test_run_jev_trial.py`.

## Development-split scores and the common correction (2026-10-07)

Score every checkpoint on the 2,233 development pairs (no test data), then fit and apply the correction. Set `PY` to the venv python first.

```bash
python -m trialmatch.evaluate --run "$(python scripts/pick_best.py runs/<pointer-run> --path)" --data data/tm-v1-b4096 --split development --dtype bf16   # fp32 at 0.8B
python scripts/train_lora_letter.py eval --run "$(python scripts/pick_best.py runs/<control-run> --path)" --data data/tm-v1-b4096 --split development
python scripts/generate_baselines.py run --condition sft-generate --base <model> --base_revision <sha> --adapter "$(python scripts/pick_best.py runs/<sft-run> --path)"   --data data/tm-v1-b4096/development.jsonl --per_patient 0 --dtype bf16 --out runs/gen_sft_dev/<sft-run>
python scripts/analysis/calibrate_all.py --jev_dir runs/jev_api --dev_dirs <dev-score-dirs> --test_dirs <saved-test-pred-dirs> --suite_dirs <suite-dirs> --out calibrate_all.json
```

On the cluster, plain `python` inside `nohup bash -c` did not find the venv; call `$PY` explicitly as in the earlier commands.

## Statistics from saved predictions

```bash
export TRIALMATCH_OUT=out        # where the json files are written (default: current directory)
python scripts/analysis/compute_stats.py     <0.8B-runs-dir> <main-runs-dir> <seed-runs-dir> <post-hoc-dir>
python scripts/analysis/compute_gen_stats.py <generation-dir> <main-runs-dir> <post-hoc-dir> <seed-runs-dir> \
  --e08b <0.8B-timing-dir> --e08b_zsl <0.8B-matched-prompt-dir> --small_runs <0.8B-runs-dir>
python scripts/analysis/compute_rank_stats.py <0.8B-runs-dir> <main-runs-dir> <seed-runs-dir> <post-hoc-dir>
python scripts/analysis/compute_suite_stats.py <cluster-suite-runs> <second-machine-suite-runs> --pointer_dir <main-runs-dir> --out suite_stats.json
python scripts/analysis/score_jev_api.py --jev_dir runs/jev_api --data <processed-data-dir> --pointer_dir <main-runs-dir> \
  --suite_dir <suite-evidence-dir> --out jev_api_stats.json
python scripts/analysis/compare_jev_all.py --jev_dir runs/jev_api --stats runs/jev_api/jev_api_stats.json \
  --exports <main-runs-dir> <seed-runs-dir> <0.8B-runs-dir> --out jev_vs_trialjev_all.json
```

They read the per-pair prediction files that `evaluate.py`, `train_lora_letter.py eval` and `generate_baselines.py` write,
check each condition against its own `report.json`, and resample the 50 test patients (10,000 resamples, fixed seed).
The prediction files themselves are not in this repository. The summary statistics they produce are committed in `results/`
(`results_stats.json`, `gen_stats.json`, `rank_stats.json`, `suite_stats.json`, `jev_api_stats.json`, `jev_vs_trialjev_all.json`, `calibrate_all.json`, with `results/SHA256SUMS`; verify with `sha256sum -c SHA256SUMS`).
They hold aggregated metrics only, with no per-pair rows and no TREC gold labels, because redistributing the judgments needs
terms that have not been confirmed. Run-to-run statistics in these files can be compared with a re-run of the scripts above;
the bootstrap uses a fixed seed.

## Offline compute nodes

Cache models on a machine with internet, then on the node:
`export HF_HOME=<cache> HF_HUB_OFFLINE=1`. With `HF_HUB_OFFLINE=1`, `trialmatch.checkpoint`
resolves Hub ids from the local cache directly (`snapshot_download` needs the network).

## Statistics

```bash
python scripts/analyze_predictions.py --preds A/eval-test-preds.jsonl \
  --compare B/eval-test-preds.jsonl --bootstrap 10000 --out comparison.json
```

Record `python --version`, `pip freeze`, the model commit SHA, GPU type, and the literal
command with every result.

## Environment and model revisions of the reported 9B / 27B runs

Checked read-only on the training cluster, 2026-10-01.

| Item | Value |
|---|---|
| GPUs | 4 x NVIDIA H100 NVL, 95,830 MiB each, driver 610.43.02 |
| Python | 3.12.9 |
| torch | 2.14.0+cu130 |
| transformers / peft / accelerate | 5.17.0 / 0.21.0 / 1.15.0 |
| flash-linear-attention (fla-core) | 0.5.2 |
| huggingface_hub / numpy | 1.33.0 / 2.5.3 |

| Base model | Hub revision | How known |
|---|---|---|
| Qwen/Qwen3.5-9B-Base | `68c46c4b3498877f3ef123c856ecfde50c39f404` | only snapshot in the training cache; the checkpoint's `base_revision` is `None` |
| Qwen/Qwen3.5-27B | `fc05daec18b0a78c049392ed2e771dde82bdf654` | only snapshot in the training cache; `base_revision` is `None` |
| Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | recorded in the checkpoint and used by the zero-shot probe |

## Environment of the reported 0.8B runs (second machine)

Checked read-only, 2026-10-01.

| Item | Value |
|---|---|
| GPUs | 4 x NVIDIA RTX 6000 Ada, 49,140 MiB each, driver 590.48.01 |
| Python | 3.12.14 |
| torch / transformers / peft / accelerate | 2.14.0 / 5.17.0 / 0.21.0 / 1.15.0 |
| flash-linear-attention (fla-core) | 0.5.2 |
| huggingface_hub / numpy | 1.33.0 / 2.5.3 |
| Model cache | `~/.cache/huggingface/hub` (Qwen3.5-0.8B-Base, MedJev-Qwen3.5-0.8B) |

Both 0.8B runs: seed 0, lr 1e-4, 3 epochs, batch 4 x accum 4, fp32 backbone; selected steps: fresh 1000, clinical-init 400.
The checkpoints record `base_revision: None`.

## Environments of the later runs (seeds, control, generation, timing)

The 9B, 27B and Qwen3.8-27B seed, control, generation and timing runs used the cluster environment above. The 0.8B timing
and matched-prompt runs used the second machine above, in fp32; there the optimized `causal_conv1d` kernel was not
installed and transformers used its reference implementation (slower, same results). Latencies are per-pair medians at batch
size 1 and are only comparable within one run. The hosted-Jev API runs used a Windows workstation (Python 3.12.7, numpy 2.5.3);
the latency they report is network plus service time.
