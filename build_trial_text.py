"""Reconstruct the exact `trial_text` format used in trec2021_patient_trial_pairs.jsonl
from raw ClinicalTrials.gov XML, so the same recipe can be applied to TREC 2022 trials
(same corpus) without introducing a formatting shift between train and test.

Derived empirically from diffing processed trial_text against raw XML for known NCT ids
-- see HISTORY.md. Validated by regenerating all 26,162 distinct 2021 trials and diffing
against the source file (scripts/validate_trial_text.py).
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

_PARTS = [f"ClinicalTrials.2021-04-27.part{i}" for i in range(1, 6)]


def find_xml(nct_id, trials_root):
    folder = "NCT" + nct_id[3:7] + "xxxx"
    root = Path(trials_root)
    for part in _PARTS:
        p = root / part / part / folder / f"{nct_id}.xml"
        if p.exists():
            return p
    return None


def clean_block(text):
    """XML entity decoding is already done by ElementTree; this collapses each line's
    internal whitespace to single spaces (source hard-wraps criteria/summary text with
    variable indentation), keeps line breaks and blank lines (paragraph breaks) as-is,
    and strips the whole block."""
    if not text:
        return ""
    lines = re.split(r"\r\n|\r|\n", text)
    lines = [re.sub(r"[ \t]+", " ", ln.strip()) for ln in lines]
    return "\n".join(lines).strip()


def clean_line(text):
    return re.sub(r"[ \t]+", " ", (text or "").strip())


_NCT_RE = re.compile(r"\bNCT\d{8}\b")


def _redact_nct_ids(text):
    """Body text sometimes names other trials by NCT id (merged trials, prior-phase
    references); the source data redacts these to keep the model from keying off a raw
    trial id rather than the trial's actual content."""
    return _NCT_RE.sub("[TRIAL_ID]", text)


def build_trial_text(xml_path):
    root = ET.parse(str(xml_path)).getroot()
    parts = []

    def add(header, body):
        if body:
            parts.append(f"{header}:\n{body}")

    title = clean_line(root.findtext("brief_title"))
    official = clean_line(root.findtext("official_title"))
    add("TITLE", title)
    add("OFFICIAL TITLE", official if official.casefold() != title.casefold() else "")
    conditions = "\n".join(clean_line(e.text) for e in root.findall("condition") if e.text and e.text.strip())
    add("CONDITIONS", conditions)

    ivs, seen_iv = [], set()
    for iv in root.findall("intervention"):
        itype = clean_line(iv.findtext("intervention_type"))
        name = clean_line(iv.findtext("intervention_name"))
        desc = clean_block(iv.findtext("description"))
        line = f"{itype} - {name}" + (f": {desc}" if desc else "")
        if line not in seen_iv:      # the same intervention can repeat once per arm_group
            seen_iv.add(line)
            ivs.append(line)
    add("INTERVENTIONS", "\n".join(ivs))

    add("SUMMARY", clean_block(root.findtext("brief_summary/textblock")))
    add("ELIGIBILITY", clean_block(root.findtext("eligibility/criteria/textblock")))

    demo = []
    elig = root.find("eligibility")
    if elig is not None:
        gender = clean_line(elig.findtext("gender"))
        if gender:
            demo.append(f"Sex: {gender}")
        min_age = clean_line(elig.findtext("minimum_age"))
        if min_age:
            demo.append(f"Minimum age: {min_age}")
        max_age = clean_line(elig.findtext("maximum_age"))
        if max_age:
            demo.append(f"Maximum age: {max_age}")
        hv = elig.find("healthy_volunteers")
        if hv is not None:
            demo.append(f"Healthy volunteers: {clean_line(hv.text)}")
    add("DEMOGRAPHICS", "\n".join(demo))

    keywords = "; ".join(clean_line(e.text) for e in root.findall("keyword") if e.text and e.text.strip())
    add("KEYWORDS", keywords)

    return _redact_nct_ids("\n\n".join(parts))
