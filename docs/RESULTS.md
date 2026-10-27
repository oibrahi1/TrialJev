# Results and how far to trust them

Task: eligible vs excluded on TREC 2022 pairs that assessors judged relevant
(6,975 pairs, 50 patients; 3,939 eligible / 3,036 excluded; majority-class floor 56.47%).
Training/development: TREC 2021 patients only. Checkpoint = lowest development loss.
Intervals are 95% patient-clustered bootstrap intervals (10,000 resamples of the 50 test patients). Brier is the
conventional binary form (half of the evaluator's two-class sum). ECE uses 10 equal-width bins.

Names: **TrialJev** is the pointer-head model. The **A/B-logit control** is the same LoRA recipe read out from the
language model's own scores for the letters `A` and `B`, trained with a loss on those two scores only (not conventional
generative fine-tuning). Older run folders and logs call it `ctl-*`, `lora-logits` or "plain LoRA". The **fine-tuned LLM**
is the same LoRA recipe trained with full-vocabulary cross-entropy to write "Eligible" or "Excluded" and stop (code and
JSON names: `sft-generate`, keys such as `writer`). **Hosted Jev** is TypeSafe's Jev API (`jev-1.13.0`), used without
adaptation by us.

## 1. TrialJev against LLMs with and without fine-tuning

Full-test columns: 6,975 pairs, canonical two-pass evaluator for TrialJev. Timing columns: 992 identical pairs per size,
seed-0 checkpoints, batch size 1, one GPU, systems one after another (H100 NVL/bf16 for 9B to 27B, RTX 6000 Ada/fp32 for
0.8B); ratio = median ms divided by TrialJev's one-pass median in the same chain. "3 seeds": mean ± sd, seed 0 in
parentheses. The 0.8B TrialJev starts from the MedJev clinical initialization; the 0.8B fine-tuned LLM starts fresh. The prompted
LLM is the un-adapted post-trained model asked for one word (9B: the instruct checkpoint, not the Base model the adapted
systems start from; 0.8B: the Base model with a plain prompt).

| Size | System | Full-test accuracy | Full-test AUROC | Accuracy, n = 992 | Median ms | Ratio |
|---|---|---:|---:|---:|---:|---:|
| 0.8B | TrialJev, one-pass | 70.71 ± 2.64 (seed 0: 73.48) | 0.786 | 74.80% | 54.0 | 1.00 |
| 0.8B | TrialJev, two-pass | same | same | 74.80% | 69.2 | 1.28 |
| 0.8B | Fine-tuned LLM | 74.05% | 0.802 | 72.68% | 99.9 | 1.85 |
| 0.8B | LLM, no fine-tuning (prompted) | 56.01% | — | 56.35% | 79.2 | 1.47 |
| 9B | TrialJev, one-pass | 78.27 ± 0.55 (seed 0: 78.52) | 0.862 | 80.85% | 106.2 | 1.00 |
| 9B | TrialJev, two-pass | same | same | 80.34% | 183.2 | 1.73 |
| 9B | Fine-tuned LLM | 78.34 ± 2.31 (seed 0: 79.50) | 0.862 | 80.65% | 259.9 | 2.45 |
| 9B | LLM, no fine-tuning (prompted) (instruct checkpoint) | 77.35% | — | 77.82% | 243.9 | 2.30 |
| 27B | TrialJev, one-pass | 80.97% | 0.885 | 81.25% | 177.6 | 1.00 |
| 27B | TrialJev, two-pass | same | same | 81.25% | 328.3 | 1.85 |
| 27B | Fine-tuned LLM | 79.87% | 0.879 | 80.14% | 464.4 | 2.62 |
| 27B | LLM, no fine-tuning (prompted) | 72.89% | — | 74.40% | 347.1 | 1.96 |
| Qwen3.8-27B | TrialJev, one-pass | 81.84% | 0.888 | 82.66% | 178.2 | 1.00 |
| Qwen3.8-27B | TrialJev, two-pass | same | same | 82.66% | 330.1 | 1.85 |
| Qwen3.8-27B | Fine-tuned LLM | 79.93% | 0.888 | 80.54% | 480.3 | 2.70 |
| Qwen3.8-27B | LLM, no fine-tuning (prompted) | 61.66% | — | 65.52% | 339.1 | 1.90 |

Paired differences, TrialJev (seed 0) minus fine-tuned LLM, unadjusted patient-bootstrap intervals. Full test: 9B −0.97
[−4.56, +2.47] (+0.18 [−2.22, +2.39] against the mean of three LLM seeds), 27B +1.10 [−0.57, +3.03], Qwen3.8-27B +1.91
[+0.80, +3.06]. Timing sample: 0.8B +2.12 [−1.32, +5.84], 9B +0.20 [−2.43, +2.84], 27B +1.11 [−1.01, +3.33], Qwen3.8-27B
+2.12 [+0.60, +3.78]. Only Qwen3.8-27B excludes zero; similar observed accuracy is not equivalence.

Per-run fine-tuned LLM results (full test; no invalid answers in any run): 0.8B 74.05%; 9B 79.50 / 79.84 / 75.68%; 27B
79.87%; Qwen3.8-27B 79.93%. The third 9B seed predicts "eligible" for 43.6% of pairs against 65.3% and 63.5% for the first
two, with a similar AUROC (0.860 against 0.864 and 0.862); the cause was not tested. The written verdict differs from
thresholding the first-token score at 0.5 for 146 / 149 / 211 pairs (9B seeds), 139 (27B) and 144 (Qwen3.8-27B). Both
are descriptive (three seeds at 9B, one elsewhere). Timing notes: batch size 1; the fine-tuned LLM emits three tokens
including the end token and its timed call also collects output scores; Hugging Face generation with the model's own
cache; ratios are not engine-independent and not end-to-end workflow times.

## 2. Other comparisons

*Prompted and untrained models* (full test; timing ratio to TrialJev one-pass in the same chain):

| System | 0.8B | 9B | 27B | Qwen3.8-27B |
|---|---:|---:|---:|---:|
| Prompted LLM, one word: accuracy | 56.01% | 77.35% (instruct) | 72.89% | 61.66% |
| Prompted LLM, one word: median ms (ratio) | 79.2 (1.47) | 243.9 (2.30) | 347.1 (1.96) | 339.1 (1.90) |
| Untrained model, letter scoring | 55.70% | 43.68% | 64.00% | not run |
| MedJev as released, no TREC training | 56.59% | — | — | — |

*Explain-then-answer LLM* (instruct model, native thinking off, at most 512 new tokens; 200 pairs = 4 per patient, a subset
of the timing sample; invalid output counts as wrong, and every invalid output reached the cap):

| Size | Explanation, strict | Valid answers only | Invalid / 200 | TrialJev | Fine-tuned LLM | TrialJev − explanation | Median s | Ratio |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 9B | 65.5% | 72.8% (n = 180) | 20 | 83.0% | 81.0% | +17.5 [+11.5, +24.0] | 7.3 | 68 |
| 27B | 61.5% | 71.1% (n = 173) | 27 | 79.5% | 83.0% | +18.0 [+10.0, +26.0] | 19.2 | 104 |
| Qwen3.8-27B | 52.5% | 75.5% (n = 139) | 61 | 82.0% | 81.0% | +29.5 [+22.0, +37.5] | 25.4 | 139 |

This is a zero-shot baseline with a token cap, not an upper bound on reasoning: a larger budget, few-shot prompts and a
fine-tuned reasoner were not tested, and explanation quality was not evaluated. A strict final-line parser gave the same
verdict for all 600 saved outputs.

*Hosted Jev* (`jev-1.13.0`; one model, so one result for every row). Default cutoff 0.5: accuracy 63.89%, macro-F1 62.4%,
AUROC 0.882, predicts "eligible" for 23.7% of pairs (gold 56.5%; eligible recall 39.0%, excluded recall 96.2%). Calibrated
on TREC 2021 development data only (temperature 0.5 to 3.0 and class-prior weight, as MedJev did for Jev), frozen before
the test split was scored: accuracy 81.28%, macro-F1 81.1%, 53.5% predicted eligible. MedJev's own calibration code gives
the same 81.28% on our answers, and a development-chosen cutoff of 0.12 alone also gives 81.28%. Test accuracy by fixed
cutoff (diagnostic, not used for selection): 0.50 → 63.89%, 0.30 → 73.43%, 0.20 → 78.02%, 0.15 → 79.94%, 0.10 → 81.43%.

| TrialJev model | Accuracy | AUROC | minus Jev raw (63.89%) | minus Jev calibrated (81.28%) | AUROC minus Jev (0.882) |
|---|---:|---:|---|---|---:|
| 0.8B, fresh (3 seeds) | 54.27 ± 0.95% | 0.582 | −9.62 [−14.10, −4.74] | −27.01 [−30.66, −23.25] | −0.300 |
| 0.8B, MedJev init (3 seeds) | 70.71 ± 2.64% | 0.786 | +6.82 [+1.36, +12.86] | −10.57 [−13.34, −7.84] | −0.095 |
| 9B (3 seeds) | 78.27 ± 0.55% | 0.862 | +14.39 [+8.93, +20.30] | −3.00 [−4.58, −1.44] | −0.020 |
| 27B | 80.97% | 0.885 | +17.09 [+10.83, +23.91] | −0.30 [−1.17, +0.61] | +0.003 |
| Qwen3.8-27B | 81.84% | 0.888 | +17.95 [+11.88, +24.61] | +0.56 [−0.44, +1.51] | +0.006 |

Hosted-API latency (one request at a time, 992 pairs): median 325 ms, p90 360 ms, p99 455 ms, no retries; about 18 pairs
per second with six parallel requests. It includes the network and the vendor's servers and is not comparable with local GPU
latency (54 to 178 ms for TrialJev's one-pass path). Caveats: Jev's size, training data and hardware are unknown, so it is
"without adaptation by us" and not shown to be untrained on TREC; its calibration used TREC 2021 labels; the same
correction applied to every system is in section 2b;
the API is not fully deterministic (of 2,638 test pairs requested twice, half returned a slightly different probability,
mean difference 0.018, 23 decisions crossed 0.5) and returns probabilities rounded to two decimals. One valid answer per
pair is used. An overlap of two copies of the job caused the duplicate requests; the test file was cleaned (12 corrupted
lines and duplicate rows dropped, raw file kept) and the timing sample was re-run alone.

## 2b. The same development-fitted correction for every system

Each system that returns a score gets the recipe MedJev used for Jev: a temperature (0.5 to 3.0) and a class-prior weight fitted
on the TREC 2021 development scores only (2,233 pairs, 15 patients), frozen, then applied once to the saved test scores
(`scripts/analysis/calibrate_all.py`, `results/calibrate_all.json`). Multi-seed rows are fitted per seed. Raw is the model's own
saved decision (the fine-tuned LLM row is its first-token score at 0.5, not its written verdict, which is 74.05 / 78.34 / 79.87 /
79.93%). Differences are corrected minus corrected Jev (81.28%), unadjusted patient-bootstrap intervals; prompted and
explain-then-answer LLMs return no score and cannot be corrected. An exact tie in a corrected score counts as excluded, in the
development fit and in the test decision alike (the 9B control seed 2 has 181 such ties; counting them as eligible would give 78.07%).

| System | Raw | Corrected | Corrected minus corrected Jev |
|---|---:|---:|---|
| TrialJev 0.8B, fresh (3 seeds) | 54.27 ± 0.95 | 53.90 ± 1.47 | −27.38 [−31.28, −23.39] |
| TrialJev 0.8B, MedJev init (3 seeds) | 70.71 ± 2.64 | 70.81 ± 2.32 | −10.47 [−13.22, −7.71] |
| TrialJev 9B (3 seeds) | 78.27 ± 0.55 | 78.94 ± 1.21 | −2.34 [−3.60, −0.95] |
| TrialJev 27B | 80.97 | 80.99 | −0.29 [−1.16, +0.62] |
| TrialJev Qwen3.8-27B | 81.84 | 81.88 | +0.60 [−0.30, +1.49] |
| A/B-logit control 0.8B, 9B (3 seeds), 27B, Qwen3.8-27B | 52.85, 78.23 ± 2.74, 81.25, 78.25 | 52.86, 77.68 ± 2.23, 79.54, 79.23 | −28.42, −3.60, −1.73, −2.05 |
| Fine-tuned LLM score 0.8B, 9B (3 seeds), 27B, Qwen3.8-27B | 74.05, 78.41 ± 1.51, 80.34, 80.62 | 72.89, 78.06 ± 0.65, 80.34, 80.62 | −8.39, −3.22, −0.93, −0.66 |
| Hosted Jev | 63.89 | 81.28 | — |

The correction moves TrialJev by 0.0 to +0.7 points (its native cutoff is already close), whereas hosted Jev gains 17 points.
TrialJev's corrected accuracy is the highest of our three scoring systems at every size ≥ 9B. At 27B and on Qwen3.8-27B it is
statistically level with corrected Jev; at 9B and 0.8B corrected Jev is ahead. The fit uses 15 development patients and is
noisy (the control moves by up to −1.7 points), and results at 27B are single seeds.

## 3. TrialJev and the control across sizes

"3 seeds" rows: mean ± sd of seeds 0, 1, 2, with the interval for the seed mean. Other rows are seed 0.

| Backbone | System | Accuracy [95% CI] | Macro-F1 | AUROC | Brier | ECE |
|---|---|---|---:|---:|---:|---:|
| 0.8B, fresh | TrialJev (3 seeds) | 54.27 ± 0.95 [51.75, 57.11] | 51.2% | 0.582 | 0.254 | 0.077 |
| 0.8B, fresh | control | 52.85 [47.91, 57.99] | 52.5% | 0.567 | 0.252 | 0.070 |
| 0.8B, MedJev init | TrialJev (3 seeds) | 70.71 ± 2.64 [68.12, 73.36] | 70.4% | 0.786 | 0.196 | 0.044 |
| 9B | TrialJev (3 seeds) | 78.27 ± 0.55 [75.80, 80.75] | 78.1% | 0.862 | 0.158 | 0.063 |
| 9B | control (3 seeds) | 78.23 ± 2.74 [75.43, 80.92] | 77.9% | 0.861 | 0.155 | 0.024 |
| 27B | TrialJev | 80.97 [78.14, 83.50] | 80.8% | 0.885 | 0.145 | 0.089 |
| 27B | control | 81.25 [78.36, 83.81] | 81.0% | 0.884 | 0.136 | 0.028 |
| Qwen3.8-27B | TrialJev | 81.84 [79.04, 84.28] | 81.7% | 0.888 | 0.145 | 0.098 |
| Qwen3.8-27B | control | 78.25 [74.89, 81.34] | 78.2% | 0.887 | 0.146 | 0.028 |

Per-seed values: 0.8B TrialJev fresh 54.27 / 55.23 / 53.32%, MedJev init 73.48 / 68.22 / 70.44%; 9B TrialJev 78.52 /
77.65 / 78.65%, control 80.50 / 79.00 / 75.18%. 0.8B and 9B/27B differ in precision, batch layout, validation schedule and
hardware (0.8B: fp32 backbone, batch 4 x accum 4, validation every 100 steps; 9B/27B: bf16 backbone, batch 2 x accum 8,
every 200 steps). 27B has no Base release, so the 27B runs start from post-trained models.

## 4. Is the pointer head needed? (paired, patient bootstrap)

| Comparison (A − B) | Accuracy, points [95% CI] | ECE | AUROC |
|---|---|---|---|
| 9B, 3 seeds: control − TrialJev | −0.05 [−1.34, +1.16] | −0.039 [−0.045, −0.019] | −0.001 [−0.009, +0.006] |
| 27B: control − TrialJev | +0.27 [−0.35, +0.87] | −0.061 [−0.077, −0.036] | −0.001 [−0.005, +0.002] |
| Qwen3.8-27B: control − TrialJev | −3.58 [−5.32, −1.97] | −0.070 [−0.083, −0.040] | −0.001 [−0.008, +0.005] |
| TrialJev: 27B − 9B (3 seeds) | +2.70 [+1.26, +4.25] | +0.027 [+0.006, +0.044] | +0.023 [+0.013, +0.034] |
| TrialJev: Qwen3.8-27B − Qwen3.5-27B | +0.86 [+0.05, +1.61] | +0.008 [−0.001, +0.017] | +0.003 [−0.003, +0.009] |
| 0.8B TrialJev: MedJev init − fresh (3 seeds) | +16.44 [+13.49, +19.31] | −0.033 [−0.069, +0.005] | +0.205 [+0.163, +0.247] |

On Qwen3.5 the control matches TrialJev in accuracy and ranking and is better calibrated; TrialJev's reliability curve is mostly
below the accuracy line at 9B and 27B (Qwen3.8-27B was not inspected bin by bin); less well calibrated (ECE 0.063 against 0.024 at 9B, 0.089 against 0.028 at 27B). On Qwen3.8-27B TrialJev is more accurate at the 0.5 threshold with the same AUROC; the control leans
toward "excluded" there (eligible recall 0.712, excluded recall 0.874). The control's step-400 checkpoint (scored after the
main analysis, see section 8) did not remove the gap. The control's three 9B seeds span 5.3 points, TrialJev's 1.0.

## 5. Prompted generation, matched prompt and earlier timing runs

Full test set, strict scoring (an answer that is not exactly one of the two permitted words or letters counts as wrong).

| Condition | Model | Accuracy [95% CI] | AUROC |
|---|---|---|---:|
| Zero-shot chat, writes "Eligible"/"Excluded" | Qwen3.5-27B | 72.89% [66.77, 77.83] (71.54% options swapped) | — |
| Zero-shot chat | Qwen3.5-9B (instruct, a different checkpoint from the 9B Base) | 77.35% [74.41, 80.31] | — |
| Zero-shot chat | Qwen3.8-27B | 61.66% [54.78, 68.37] | — |
| Zero-shot plain prompt, writes the word | Qwen3.5-9B-Base | 44.24% (16.5% invalid answers) | — |
| Zero-shot letters, matched prompt, no adapter | Qwen3.5-27B | 64.00% [57.05, 70.34] (61.52% swapped) | 0.860 |
| Zero-shot letters, matched prompt, no adapter | Qwen3.5-9B-Base | 43.68% (56.89% swapped; one letter for nearly all pairs) | 0.559 |
| Zero-shot letters, matched prompt, no adapter | Qwen3.5-0.8B-Base | 55.70% (46.72% swapped) | 0.505 |

At 27B the gap between the control and prompted word writing is 8.36 points [+3.87, +14.40], but it combines adaptation
with a change of prompt, readout and decoding. The matched-prompt comparison isolates adaptation: 64.00% without the
adapter against 81.25% with it (+17.25 points [+10.77, +24.41]), while AUROC only rises from 0.860 to 0.884
(+0.024 [+0.006, +0.043]); much of the accuracy gap is the decision threshold and letter bias. No threshold was tuned for
the un-adapted model.

Earlier timing runs (superseded by sections 1 and 2): time per pair, batch size 1, 992-pair sample, ratio to the control reading its A/B logits in the same run (one H100 GPU
per model size, conditions run one after another). 0.8B ratios come from a separate run on a different GPU in fp32.

| System | 0.8B | 9B | 27B |
|---|---:|---:|---:|
| Control, reads the A/B logits | 1.00 | 1.00 | 1.00 |
| Control, emits its one label token | 1.01 (54% empty outputs) | 0.99 | 1.00 |
| TrialJev, one-pass path | 0.82 | 0.90 | 0.82 |
| TrialJev, two-pass path (used for all accuracy results) | 1.06 | 1.54 | 1.49 |
| Zero-shot chat, writes the word | — | 2.13 | 1.63 |

Absolute latency was not stable between runs of identical code (114.9 vs 77.8 ms for the same 9B model on the same
pairs), so only within-run ratios are used. The one-pass path is not identical to the two-pass path: over all 6,975 pairs
it changes 12 to 20 predictions (0.17 to 0.29%) and accuracy by at most 0.09 points. The control emitting a label token
gives exactly the same predictions as reading its logits at 9B and 27B; at 0.8B the greedy output was an empty string
(after special tokens were removed) for 534 of 992 pairs, so the one-token readout does not work there. Only one-word
verdicts were measured in these runs; the explain-then-answer baseline was timed separately (section 2).

## 6. Ranking inside the judged pool

For each patient, the judged-relevant trials (eligible and excluded) are ranked by P(eligible). This is reranking within
the judged pool, not full-collection retrieval: irrelevant trials are absent from this dataset.

| System | P@10 | nDCG@10 | MRR | MAP |
|---|---|---|---|---|
| TrialJev 27B | 0.940 [0.908, 0.966] | 0.945 [0.914, 0.970] | 0.970 | 0.894 |
| Control 27B | 0.936 [0.904, 0.964] | 0.945 [0.919, 0.969] | 0.990 | 0.897 |
| TrialJev Qwen3.8-27B | 0.936 [0.902, 0.964] | 0.941 [0.909, 0.968] | 0.960 | 0.897 |
| Control Qwen3.8-27B | 0.944 [0.910, 0.972] | 0.954 [0.926, 0.978] | 1.000 | 0.901 |
| Random order (expected) | 0.590 | 0.589 | 0.735 | 0.609 |

TrialJev and the control do not differ beyond noise, except the mean reciprocal rank on Qwen3.8-27B (TrialJev 0.04
lower, [−0.08, −0.01]). P@10 and nDCG@10 are near their ceiling because 59% of the pool is eligible. Full table with 9B and
the zero-shot condition: `scripts/analysis/compute_rank_stats.py`.

## 7. Zero-shot letter-logit probes (older, cruder)

| Model | Prompt | Accuracy |
|---|---|---:|
| Qwen3.5-9B-Base | plain | 43.66% |
| Qwen3.5-9B (instruct) | chat | 68.66% |
| Qwen3.5-27B | chat | 63.90% |
| Qwen3.8-27B | chat | 57.63% |
| MedJev-0.8B as shipped (note-extraction model, not trained for this task) | — | 56.59% (predicts eligible for 99.7% of pairs; AUROC 0.562) |

One option order only. The matched-prompt rows in section 5 supersede these for 9B-Base and 27B (43.68% and 64.00%).

## 8. Checkpoint selection

Development loss (15 development patients) was lowest at step 400 in 7 of 10 runs at 9B and above, at step 200 for the 9B
control seed 2 and the Qwen3.8-27B control, and at step 1000 for TrialJev 9B seed 2. It is a noisy rule with so few
development patients. After the main analysis, the step-400 checkpoints of the two control runs selected at step 200 were
scored on the test set as a labeled post-hoc check (not used for selection): 9B seed 2 75.48% vs 75.18% at step 200;
Qwen3.8-27B 78.11% vs 78.25% (80.29% vs 80.06% with letters swapped). Other checkpoints and selection rules were not tested.
Fresh 0.8B runs select checkpoints at steps 100 to 1000 where development loss is near chance (ln 2 = 0.693).

## 9. What the evidence does and does not support

Supported:
- Cross-year transfer: train/dev on TREC 2021 patients, test on 50 different 2022 patients, 81–82% at 27B.
- On Qwen3.5 the pointer head is not what makes the model accurate: the A/B-logit control matches it and is better
  calibrated. On Qwen3.8-27B TrialJev is more accurate at the default threshold (one seed each).
- TrialJev's speed depends on its serving path (one pass about as fast as the control, two passes 1.3 to 1.9× slower).
- Against a conventionally fine-tuned LLM trained on the same data, TrialJev shows similar observed accuracy with the LLM's median batch-1 latency 1.85 to 2.70× TrialJev's one-pass path.
- Zero-shot prompted and explain-then-answer LLMs are less accurate and slower.
- Hosted Jev, used without adaptation by us, ranks about as well as TrialJev 27B; with a development-fitted cutoff it is level at 27B.
- At 0.8B, a clinical-note starting point is what makes the model work; fresh models do not exceed the majority accuracy.

Not supported:
- **A scaling curve.** Parameter count, post-training and initialization are confounded, and the 27B cells are single seeds.
- **Attribution of the Qwen3.8 gap** to the backbone rather than seed variation (the control's seed spread at 9B is 5.3 points).
- **Splitting the 0.8B gain** between the clinical adapter and the pointer head (the control has no matching start).
- **That TrialJev is more accurate than calibrated hosted Jev**, or anything about Jev's size, training data or serving cost; Jev's latency is network latency.
- **Speed of written rationales from a fine-tuned model** (none was trained; TREC has labels only).
- **That reasoning in general hurts**: the explain-then-answer baseline is zero-shot with a 512-token cap on 200 pairs.
- **Absolute throughput** (single stream, batch 1, latency varied between runs; batching not evaluated).
- **An untouched test.** TREC 2022 was examined repeatedly while the recipe was developed.
- **Comparison with published systems.** VERDICT (arXiv 2609.03366) reports F1 0.828 (GPT-5-mini) / 0.829 (distilled Qwen2.5-7B)
  on 363 selected TREC-2021 pairs; LLM-Match reports AUROC 0.8102 on TREC 2022 under a different protocol and a relevance task.
  Different year, sample, class balance and metric: contextual only.
- **Retrieval, 3-class matching, clinical use.**

## 10. How the numbers were produced

Every number above is recomputed from per-pair prediction files by `scripts/analysis/` (`compute_stats.py`,
`compute_gen_stats.py`, `compute_rank_stats.py`, `compute_suite_stats.py`, `score_jev_api.py`, `compare_jev_all.py`) and `scripts/analyze_predictions.py`; the evaluator reports were
cross-checked against them. The prediction files and run reports are not in this repository. Environments, model
revisions and script hashes: [REPRODUCE.md](REPRODUCE.md).

## Still open

1. A frozen patient-held-out evaluation set that was not used for any recipe decision.
2. More seeds at 27B and on Qwen3.8; other checkpoints and selection rules.
3. A fine-tuned or longer-budget reasoning baseline; optimized serving engines and batched throughput; more seeds and more training patients (the development fit uses only 15 patients).
4. The released clinical adapter inside the A/B-logit control at 0.8B, to split the adapter from the pointer head.
5. External evaluation on SIGIR 2016, and graded or per-criterion labels where the pointer design should matter more.
