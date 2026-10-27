# Data

Nothing derived from TREC topics or ClinicalTrials.gov is committed here
(`.gitignore` blocks `data/**/*.jsonl`). This repository ships only the patient-id split
(`data/splits.json`), per-dataset statistics (`data/*/stats.json`) and
`data/DATA_MANIFEST.sha256`. Check the terms of use of each source before sharing any
processed file.

## Sources (obtain them yourself)

| Source | What | Where |
|---|---|---|
| TREC 2021 Clinical Trials track | 75 topics (synthetic patient cases), judgments (0 not relevant / 1 excluded / 2 eligible) | https://www.trec-cds.org/2021.html |
| TREC 2022 Clinical Trials track | 50 topics, judgments | https://www.trec-cds.org/2022.html |
| ClinicalTrials.gov snapshot used by the track | trial XML records | linked from the track pages |

The track topics are synthetic patient descriptions written to resemble an admission note.
A few 2021 topics contain de-identification placeholders (e.g. `[**2148-10-1**]`); they are
not real identifiers.

## Rebuild

1. Build the per-pair files (schema: `patient_id, nct_id, patient_vignette, trial_text,
   label, label_name`). `scripts/build_trec2022_pairs.py` does this for 2022 from the raw
   topics, qrels and trial XML; 2021 pairs use the same `build_trial_text()` recipe.
   `scripts/validate_trial_text.py` checks the recipe against an existing 2021 pairs file
   (it matched 26,162 / 26,162 distinct trial texts).

   ```bash
   python scripts/build_trec2022_pairs.py --topics topics2022.xml --qrels qrels2022.txt \
       --trials_root <dir with trial XML> --out trec2022_patient_trial_pairs.jsonl
   ```

2. Build the model-ready datasets:

   ```bash
   python build_data.py --pairs trec2021_patient_trial_pairs.jsonl \
       --test_pairs trec2022_patient_trial_pairs.jsonl --out data
   ```

3. Verify against the frozen hashes: `sha256sum -c --ignore-missing data/DATA_MANIFEST.sha256`
   (run from the repository root; it reports the files you have and fails on mismatches),
   or run `TRIALMATCH_DATA=<data dir> pytest tests/test_released_data.py`.

## What the datasets contain

| Split | Source | Pairs | Patients | Eligible / Excluded |
|---|---|---:|---:|---:|
| train | TREC 2021 | 9,302 | 60 | 4,348 / 4,954 |
| development | TREC 2021 | 2,233 | 15 | 1,168 / 1,065 |
| test | TREC 2022 | 6,975 | 50 | 3,939 / 3,036 |

- Patient-level split (never pair-level), exact ids in `data/splits.json` (seed 13).
- Run 1 keeps only grades 2 (eligible → yes) and 1 (excluded → no). Grade 0 pairs
  (24,243 in 2021) are left out; a trial the assessors never judged is never a negative.
- 54 "eligible" 2021 pairs whose patient age/sex contradict the trial's own structured
  limits were removed from train/dev. In the 2022 test set suspect pairs are **flagged in
  `_meta.suspect` but kept** (82 of them), so the test set is not hand-cleaned.
- `tm-v1-b4096` (main) trims trial text to a 4,096-token branch budget (34 trials needed
  trimming; the eligibility section is cut last). `tm-v1-b2560` is a shorter-budget ablation.

## Leakage checks

- Patient overlap between train/dev (2021) and test (2022): 0 of 50.
- Trial overlap: 246 of 6,975 test pairs (3.53%, 211 distinct trials) use a trial that also
  occurs in train/dev, with identical text. Their labels agree with the train-majority label
  for that trial only about 62% of the time, so this is not a usable shortcut.
- The base models were pretrained after TREC 2021/2022 were published; contamination of the
  pretraining corpora cannot be excluded or measured here.
- TREC 2022 was examined many times while developing the recipe. Treat it as a
  development-adjacent benchmark, not an untouched test.
