"""Build trec2022_patient_trial_pairs.jsonl in the exact schema of the user's
trec2021_patient_trial_pairs.jsonl, using the validated build_trial_text() recipe
(scripts/validate_trial_text.py: 26,162/26,162 exact match on 2021).

Only judged pairs (from the official qrels2022.txt) are emitted -- never a fabricated
negative for an unjudged trial.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_trial_text import build_trial_text, find_xml  # noqa: E402

LABEL_NAME = {0: "not_relevant", 1: "excluded", 2: "eligible"}


def load_topics(path):
    root = ET.parse(str(path)).getroot()
    return {t.get("number").strip(): " ".join((t.text or "").split()) for t in root.findall("topic")}


def load_qrels(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append((parts[0], parts[2], int(parts[3])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topics", type=Path, required=True, help="topics2022.xml")
    ap.add_argument("--qrels", type=Path, required=True, help="qrels2022.txt")
    ap.add_argument("--trials_root", type=Path, required=True,
                    help="directory holding the ClinicalTrials.gov XML (the 2021 snapshot is the corpus for 2022 too)")
    ap.add_argument("--out", type=Path, required=True, help="trec2022_patient_trial_pairs.jsonl")
    a = ap.parse_args()
    TRIALS_ROOT, OUT = a.trials_root, a.out
    topics = load_topics(a.topics)
    qrels = load_qrels(a.qrels)
    print(f"topics: {len(topics)}  qrels rows: {len(qrels)}")

    text_cache, missing_xml, missing_topic = {}, set(), set()
    written = 0
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for patient_id, nct_id, rel in qrels:
            vignette = topics.get(patient_id)
            if vignette is None:
                missing_topic.add(patient_id)
                continue
            if nct_id not in text_cache:
                xml_path = find_xml(nct_id, TRIALS_ROOT)
                text_cache[nct_id] = build_trial_text(xml_path) if xml_path else None
            trial_text = text_cache[nct_id]
            if trial_text is None:
                missing_xml.add(nct_id)
                continue
            row = {"patient_id": patient_id, "nct_id": nct_id, "patient_vignette": vignette,
                   "trial_text": trial_text, "label": rel, "label_name": LABEL_NAME[rel]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"written: {written}  missing_xml (trial not in corpus): {len(missing_xml)}  "
          f"missing_topic (patient not in topics2022.xml): {len(missing_topic)}")
    if missing_xml:
        print("  sample missing NCT ids:", sorted(missing_xml)[:10])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
