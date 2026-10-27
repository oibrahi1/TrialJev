# Model card — TrialJev eligibility scorer, research preview

**Status: research preview. Weights are published** on Hugging Face (adapters and pointer heads, no data):
[TrialJev](https://huggingface.co/omareng/TrialJev),
[TrialJev-FineTunedLLM](https://huggingface.co/omareng/TrialJev-FineTunedLLM) and
[TrialJev-ABLogit](https://huggingface.co/omareng/TrialJev-ABLogit). Each checkpoint has its own sub-folder and model card
with its seed, backbone, test metrics and a SHA-256 manifest. See "Pretrained weights" in the [README](../README.md).

## What it is
A LoRA adapter (rank 16) and a pointer head (dimension 256) over frozen Qwen base weights, following the
open-weight MedJev/kev decision-model design. Given a patient description and a clinical-trial record it outputs
P(eligible) and P(excluded) in one forward pass. It does not generate text and gives no reasons.
Backbones studied: Qwen3.5-0.8B-Base, Qwen3.5-9B-Base, Qwen3.5-27B, Qwen3.8-27B.

TrialJev is independent research. It is not affiliated with or endorsed by TypeSafe AI, the developer of Jev.

## Intended use
Research on patient–trial eligibility screening. A fast first-pass score over already-retrieved candidate trials in a
staged workflow, with explanations and the final eligibility decision left to a clinician or a downstream reasoning model.

## Out of scope
Clinical decision-making, enrollment decisions, or any use where an error harms a patient;
retrieval over a full trial registry; patients unlike the TREC synthetic cases (real EHR
notes, other languages); location/recruitment-status reasoning; generating explanations.

## Training data
TREC 2021 Clinical Trials track: 60 train + 15 development patients, 11,535 judged
eligible/excluded pairs (grade 0 left out; 54 label-inconsistent "eligible" pairs removed from train/dev).
Patient descriptions are synthetic cases. See [DATA.md](../DATA.md).

## Evaluation
TREC 2022, 50 patients, 6,975 pairs, majority floor 56.47%: see [RESULTS.md](RESULTS.md). Headline: 80.97% (Qwen3.5-27B)
and 81.84% (Qwen3.8-27B) accuracy, AUROC 0.89; 78.27 ± 0.55% at 9B (three seeds); 70.71 ± 2.64% at 0.8B with the
MedJev clinical initialization (three seeds) and 54.27 ± 0.95% for fresh 0.8B. Intervals are about ±3 points. The test
set was examined repeatedly during development, and 27B and Qwen3.8 are single seeds.

Against a conventionally fine-tuned LLM trained on the same data (it writes "Eligible"/"Excluded"), TrialJev shows similar
observed accuracy (79.87% and 79.93% for the LLM at the two 27B backbones; paired differences −0.97 to +1.91 points, only
Qwen3.8-27B excluding zero), while the LLM's median batch-1 latency is 1.85 to 2.70 times TrialJev's one-pass path. Zero-shot
prompted LLMs are less accurate and slower; a zero-shot explain-then-answer LLM (512-token cap, 200 pairs) is 17.5 to 29.5
points less accurate and 68 to 139 times slower. The hosted Jev API (`jev-1.13.0`), used without adaptation by us, reaches 63.89%
at its default cutoff and 81.28% after a cutoff fitted on TREC 2021 development data, which is level with TrialJev at 27B; its
latency (325 ms) includes the network. With the same development-fitted correction applied to every system, TrialJev is level with corrected Jev at 27B and on Qwen3.8-27B and below it at 9B and 0.8B; TrialJev's own accuracy changes by at most 0.7 points under the correction.

An A/B-logit control (same recipe, answer read from the language model's scores for two letters, not conventional
generative fine-tuning) matches TrialJev in accuracy on Qwen3.5 and is better calibrated; on Qwen3.8-27B TrialJev is
3.6 points more accurate at the 0.5 threshold with the same AUROC.

## Known limitations
- Binary task on pre-filtered relevant pairs; real use also needs retrieval and a relevance decision.
- Synthetic, short patient descriptions; behaviour on long real records is untested.
- Accuracy on ~7k pairs reflects 50 patients; variance across patients is large.
- TrialJev's ECE is 0.06–0.10 at 9B and above, higher than the A/B-logit control's, and its reliability curve is mostly below
  the accuracy line at 9B and 27B; calibrate its scores and choose a threshold on development data before using them as
  probabilities. No subgroup calibration was checked.
- Trial text may be truncated to the 4,096-token branch budget (34 trials in train/dev).
- Base-model pretraining may have seen TREC data; the effect is unmeasured.
- No explanation or rationale is produced.
- No fairness or subgroup analysis has been done.

## Attribution
Built on MedJev and kev, Apache-2.0; Qwen base models (Alibaba Cloud).
See [NOTICE](../NOTICE).
