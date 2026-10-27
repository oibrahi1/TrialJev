#!/usr/bin/env python3
"""TREC Clinical Trials patient-trial pairs -> trialmatch (decision-model format) training data.

    python build_data.py --pairs ../data/trec2021_patient_trial_pairs.jsonl --out data

Run 1 task: one `noul` question per judged pair, "is this patient eligible for this
trial?"  eligible (2) -> true, excluded (1) -> false. not_relevant (0) pairs are left
out; they return in Run 2 as a separate relevance question. Nothing unjudged is ever
added.

Writes, per --max_branch budget, data/tm-v1-b<budget>/:
    train.jsonl, development.jsonl   patient-level split of --pairs (never pair-level)
    test.jsonl                       only with --test_pairs (e.g. TREC 2022); unfiltered,
                                     suspect rows are flagged in _meta, never dropped
    smoke/{train,development}.jsonl  32 balanced train pairs for the overfit smoke test
    quarantine.jsonl                 suspect train/dev pairs removed (review these)
    stats.json
and data/splits.json with the exact patient ids per split.

Trial text is trimmed to fit MedJev's branch budget in a fixed order that protects
eligibility criteria: drop KEYWORDS, drop OFFICIAL TITLE, shorten SUMMARY, shorten
INTERVENTIONS, cap an oversized CONDITIONS list at 256 tokens, and only then cut the
tail of ELIGIBILITY. TITLE and DEMOGRAPHICS are never cut. Every trim is recorded in
the record's _meta.
"""
import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trialmatch.records import materialize  # noqa: E402

QUESTION_ID = "eligible"
QUESTION_TEXT = ("Is this patient eligible for the clinical trial below? Eligible means every "
                 "inclusion criterion is met and no exclusion criterion applies.")
CRITERIA = {
    "true": "Eligible: meets the inclusion criteria and no exclusion criterion applies",
    "false": "Excluded: an inclusion criterion is not met or an exclusion criterion applies",
}
RUN1_LABELS = {2: True, 1: False}
SECTIONS = ("TITLE", "OFFICIAL TITLE", "CONDITIONS", "INTERVENTIONS", "KEYWORDS", "SUMMARY",
            "DEMOGRAPHICS", "ELIGIBILITY")
HEADER = re.compile(r"^(" + "|".join(SECTIONS) + r"):\s*$", re.M)
SHRINK_ORDER = [("drop", "KEYWORDS"), ("drop", "OFFICIAL TITLE"), ("shrink", "SUMMARY"),
                ("shrink", "INTERVENTIONS"), ("cap", "CONDITIONS"), ("shrink", "ELIGIBILITY")]
CONDITIONS_CAP = 256
MARK = " [truncated]"
# mirrors trialmatch.model.user_tokens, so token counts here equal the model's
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
LINE_BREAKS = str.maketrans({c: "\\u%04x" % ord(c) for c in "\x0b\x0c\x1c\x1d\x1e\x85  "})


class Counter_:
    def __init__(self, base):
        self.tok = Tokenizer.from_file(hf_hub_download(base, "tokenizer.json"))

    def ids(self, text):
        return self.tok.encode(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False)

    def n(self, text):
        return len(self.ids(text).ids)

    def truncate(self, text, n):
        enc = self.ids(text)
        if len(enc.ids) <= n:
            return text
        return text[: enc.offsets[max(n - 1, 0)][1]].rstrip() + MARK


def parse_sections(text):
    parts = HEADER.split(text)
    if len(parts) < 3:
        return None
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def join_sections(secs):
    return "\n\n".join(f"{h}:\n{c}" for h, c in secs if c)


def instructions(trial_text):
    return f"{QUESTION_TEXT}\n\n{trial_text}"


def fit_trial(trial_text, budget, tc):
    """-> (trial text whose instructions fit `budget` tokens, list of trim steps)."""
    if tc.n(instructions(trial_text)) <= budget:
        return trial_text, []
    secs = parse_sections(trial_text)
    if secs is None:
        return tc.truncate(trial_text, budget - tc.n(instructions("")) - 8), ["whole_text_cut"]
    secs, steps = [list(s) for s in secs], []
    for action, name in SHRINK_ORDER:
        idx = next((i for i, (h, _) in enumerate(secs) if h == name), None)
        if idx is None or not secs[idx][1]:
            continue
        if action == "drop":
            secs[idx][1] = ""
            steps.append(f"drop_{name}")
        elif action == "cap":
            if tc.n(secs[idx][1]) > CONDITIONS_CAP:
                secs[idx][1] = tc.truncate(secs[idx][1], CONDITIONS_CAP)
                steps.append(f"cap_{name}")
        else:
            without = [s if i != idx else [s[0], ""] for i, s in enumerate(secs)]
            room = budget - tc.n(instructions(join_sections(without))) - tc.n(f"\n\n{name}:\n") - 8
            if room >= 32:
                secs[idx][1] = tc.truncate(secs[idx][1], room)
                steps.append(f"cut_{name}")
            else:
                secs[idx][1] = ""
                steps.append(f"drop_{name}")
        text = join_sections(secs)
        while tc.n(instructions(text)) > budget and name == "ELIGIBILITY" and secs[idx][1]:
            body = secs[idx][1].removesuffix(MARK)
            secs[idx][1] = tc.truncate(body, max(tc.n(body) - 32, 1))
            text = join_sections(secs)
        if tc.n(instructions(text)) <= budget:
            return text, steps
    raise ValueError("trial cannot fit budget even after cutting eligibility")


AGE_PATTERNS = [r"(\d{1,3})[- ]?(?:year|yr)s?[- ]?old", r"\b(\d{1,3})\s?(?:yo|y/o|y\.o\.)\b",
                r"^\s*(\d{1,3})\s?[MF]\b"]
MALE = re.compile(r"\b(man|male|boy|gentleman|he|his|him)\b|^\s*\d+\s?M\b", re.I)
FEMALE = re.compile(r"\b(woman|female|girl|lady|she|her|pregnan\w*)\b|^\s*\d+\s?F\b", re.I)


def patient_demographics(text):
    m = re.search(r"(\d+)[- ]?month[- ]?old", text, re.I)
    age = float(m.group(1)) / 12 if m else None
    if age is None:
        for pat in AGE_PATTERNS:
            m = re.search(pat, text, re.I if "[MF]" not in pat else 0)
            if m:
                age = float(m.group(1))
                break
    male, female = bool(MALE.search(text)), bool(FEMALE.search(text))
    sex = "male" if male and not female else "female" if female and not male else None
    return age, sex


def years(s):
    m = re.match(r"\s*(\d+)\s*(Year|Month|Week|Day|Hour|Minute)", s or "", re.I)
    if not m:
        return None
    div = {"year": 1, "month": 12, "week": 52, "day": 365, "hour": 8760, "minute": 525600}[m.group(2).lower()]
    return float(m.group(1)) / div


def trial_demographics(trial_text):
    secs = dict(parse_sections(trial_text) or [])
    d = secs.get("DEMOGRAPHICS", "")
    get = lambda k: (re.search(rf"^{k}:\s*(.+)$", d, re.M | re.I) or [None, ""])[1].strip()
    return get("Sex").lower(), years(get("Minimum age")), years(get("Maximum age"))


def suspect_reasons(row):
    """Only 'eligible' labels are checked: eligible while violating the trial's own
    structured age/sex limits contradicts the label definition."""
    if row["label"] != 2:
        return []
    age, sex = patient_demographics(row["patient_vignette"])
    t_sex, t_min, t_max = trial_demographics(row["trial_text"])
    reasons = []
    if age is not None and t_min is not None and age < t_min:
        reasons.append(f"age {age:g} < min {t_min:g}")
    if age is not None and t_max is not None and age > t_max:
        reasons.append(f"age {age:g} > max {t_max:g}")
    if sex and t_sex in ("male", "female") and sex != t_sex:
        reasons.append(f"sex {sex} vs trial {t_sex}")
    return reasons


def make_record(row, trial_text, steps, year, reasons):
    return {
        "state": row["patient_vignette"],
        "questions": {QUESTION_ID: {
            "type": "noul",
            "instructions": instructions(trial_text),
            "criteria": CRITERIA,
            "label": RUN1_LABELS[row["label"]],
            "src": f"trec{year}",
        }},
        "_meta": {"id": f"trec{year}/{row['patient_id']}/{row['nct_id']}",
                  "group_id": f"trec{year}/{row['patient_id']}",
                  "patient_id": row["patient_id"], "nct_id": row["nct_id"], "year": year,
                  "trec_label": row["label"], "trec_label_name": row["label_name"],
                  "trim_steps": steps, "suspect": reasons},
    }


def branch_len(rec, tc):
    """Exactly the branch length trialmatch.model.encode computes for this record."""
    q = rec["questions"][0]
    return 1 + tc.n(q["instr"]) + sum(2 + tc.n(o) for o in q["options"]) + 1


def load_pairs(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    need = {"patient_id", "nct_id", "patient_vignette", "trial_text", "label", "label_name"}
    bad = [r for r in rows if not need <= set(r)]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} rows missing fields {need - set(bad[0])}")
    if len({(r["patient_id"], r["nct_id"]) for r in rows}) != len(rows):
        raise SystemExit(f"{path}: duplicate (patient_id, nct_id) pairs")
    return rows


def split_patients(rows, dev_every, seed):
    """Deterministic patient-level split. Patients are ordered by their number of Run-1
    pairs and every `dev_every`-th goes to development, so dev spans small and large
    patients instead of being dominated by a few big ones."""
    counts = Counter(r["patient_id"] for r in rows if r["label"] in RUN1_LABELS)
    order = sorted(counts, key=lambda p: (-counts[p], int(p) if p.isdigit() else p))
    offset = random.Random(seed).randrange(dev_every)
    dev = {p for i, p in enumerate(order) if (i + offset) % dev_every == 0}
    return sorted(set(order) - dev, key=lambda p: (len(p), p)), sorted(dev, key=lambda p: (len(p), p))


def write_jsonl(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False).translate(LINE_BREAKS) + "\n")


def describe(xs):
    xs = sorted(xs)
    return {"n": len(xs), "min": xs[0], "median": statistics.median(xs),
            "p95": xs[int(0.95 * (len(xs) - 1))], "max": xs[-1]} if xs else {"n": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="train/dev source, e.g. trec2021_patient_trial_pairs.jsonl")
    ap.add_argument("--pairs_year", type=int, default=2021)
    ap.add_argument("--test_pairs", default="", help="optional locked test set in the same format (e.g. TREC 2022)")
    ap.add_argument("--test_year", type=int, default=2022)
    ap.add_argument("--out", default="data")
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B-Base", help="tokenizer used for all token budgets")
    ap.add_argument("--max_state", type=int, default=512)
    ap.add_argument("--max_branch", type=int, nargs="+", default=[4096, 2560],
                    help="one dataset per value; MedJev's max_branch is state + branch total")
    ap.add_argument("--dev_every", type=int, default=5, help="1 in N patients goes to development")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--keep_suspect", action="store_true", help="keep suspect eligible labels in train/dev")
    a = ap.parse_args()

    tc = Counter_(a.base)
    out = Path(a.out)
    rows = load_pairs(a.pairs)
    test_rows = load_pairs(a.test_pairs) if a.test_pairs else []

    for vig in {r["patient_vignette"] for r in rows + test_rows}:
        if tc.n(vig) + 1 > a.max_state:
            raise SystemExit(f"a patient vignette has {tc.n(vig) + 1} tokens > --max_state {a.max_state}")

    train_p, dev_p = split_patients(rows, a.dev_every, a.seed)
    (out / "splits.json").parent.mkdir(parents=True, exist_ok=True)
    with open(out / "splits.json", "w", encoding="utf-8") as f:
        json.dump({"source": a.pairs, "seed": a.seed, "dev_every": a.dev_every,
                   "train_patients": train_p, "development_patients": dev_p,
                   "test_source": a.test_pairs or None}, f, indent=2)

    opt_tokens = sum(2 + tc.n(f"{k}: {v}") for k, v in (("no", CRITERIA["false"]), ("yes", CRITERIA["true"])))
    for max_branch in a.max_branch:
        budget = max_branch - a.max_state - 2 - opt_tokens
        name = f"tm-v1-b{max_branch}"
        d = out / name
        print(f"\n== {name}: instruction budget {budget} tokens ==")
        fitted = {}
        splits = {"train": [], "development": [], "test": []}
        quarantine, stats, skipped = [], defaultdict(Counter), Counter()
        branch_lens = defaultdict(list)
        for split, src_rows, year in (("train", rows, a.pairs_year), ("test", test_rows, a.test_year)):
            for r in src_rows:
                if r["label"] not in RUN1_LABELS:
                    skipped[year] += 1
                    continue
                part = split if split == "test" else ("development" if r["patient_id"] in dev_p else "train")
                key = (year, r["nct_id"])
                if key not in fitted:
                    fitted[key] = fit_trial(r["trial_text"], budget, tc)
                trial_text, steps = fitted[key]
                reasons = suspect_reasons(r)
                rec = make_record(r, trial_text, steps, year, reasons)
                mat = materialize(rec)
                bl = branch_len(mat, tc)
                if bl > max_branch - a.max_state:
                    raise SystemExit(f"{rec['_meta']['id']}: branch {bl} exceeds {max_branch - a.max_state}")
                if reasons and part != "test" and not a.keep_suspect:
                    quarantine.append({**rec["_meta"], "split": part, "state": r["patient_vignette"][:300]})
                    stats[part]["quarantined"] += 1
                    continue
                splits[part].append(rec)
                branch_lens[part].append(bl)
                stats[part][r["label_name"]] += 1
                stats[part]["trimmed"] += bool(steps)
                for s in steps:
                    stats[part][f"trim:{s}"] += 1
                if reasons:
                    stats[part]["suspect_flagged"] += 1

        rng = random.Random(a.seed)
        for part, recs in splits.items():
            if not recs:
                continue
            rng.shuffle(recs)
            write_jsonl(d / f"{part}.jsonl", recs)
        write_jsonl(d / "quarantine.jsonl", quarantine)
        smoke = ([r for r in splits["train"] if r["questions"][QUESTION_ID]["label"]][:16]
                 + [r for r in splits["train"] if not r["questions"][QUESTION_ID]["label"]][:16])
        write_jsonl(d / "smoke" / "train.jsonl", smoke)
        write_jsonl(d / "smoke" / "development.jsonl", smoke)

        summary = {"max_state": a.max_state, "max_branch": max_branch, "instruction_budget": budget,
                   "question": {"id": QUESTION_ID, "type": "noul", "text": QUESTION_TEXT, "criteria": CRITERIA},
                   "not_relevant_pairs_left_out_by_year": dict(skipped),
                   "splits": {p: {"records": len(splits[p]), **dict(stats[p]),
                                  "patients": len({r["_meta"]["patient_id"] for r in splits[p]}),
                                  "branch_tokens": describe(branch_lens[p])}
                              for p in splits if splits[p] or stats[p]}}
        with open(d / "stats.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        for p, s in summary["splits"].items():
            print(f"  {p:12s} records {s['records']:>6}  patients {s['patients']:>3}  "
                  f"eligible {s.get('eligible', 0):>5}  excluded {s.get('excluded', 0):>5}  "
                  f"trimmed {s.get('trimmed', 0):>5}  quarantined {s.get('quarantined', 0):>3}  "
                  f"branch max {s['branch_tokens'].get('max')}")
        print(f"  wrote {d}")


if __name__ == "__main__":
    main()
