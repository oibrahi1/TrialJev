"""Validate build_trial_text.py by regenerating every distinct trial_text in the 2021
pairs file from raw XML and diffing against the source. Must be ~100% exact match before
trusting the same recipe on TREC 2022."""
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_trial_text import build_trial_text, find_xml  # noqa: E402

if len(sys.argv) != 3:
    sys.exit("usage: validate_trial_text.py <trec2021_patient_trial_pairs.jsonl> <TREC_2021 trial-XML root>")
SRC, TRIALS_ROOT = sys.argv[1], sys.argv[2]

trials = {}
for line in open(SRC, encoding="utf-8"):
    r = json.loads(line)
    trials.setdefault(r["nct_id"], r["trial_text"])

n = match = missing_xml = 0
mismatches = []
for nct, expected in trials.items():
    n += 1
    xml_path = find_xml(nct, TRIALS_ROOT)
    if xml_path is None:
        missing_xml += 1
        continue
    got = build_trial_text(xml_path)
    if got == expected:
        match += 1
    else:
        mismatches.append((nct, expected, got))

print(f"trials checked: {n}  exact match: {match}  missing_xml: {missing_xml}  mismatched: {len(mismatches)}")
for nct, expected, got in mismatches[:8]:
    print(f"\n{'='*20} MISMATCH {nct} {'='*20}")
    diff = list(difflib.unified_diff(expected.splitlines(), got.splitlines(),
                                     fromfile="source(2021 jsonl)", tofile="rebuilt(xml)", lineterm=""))
    if diff:
        print("\n".join(diff))
    else:
        print("(same lines; byte-level difference)")
        print("expected repr tail:", repr(expected[-60:]))
        print("got      repr tail:", repr(got[-60:]))
        print("expected repr head:", repr(expected[:60]))
        print("got      repr head:", repr(got[:60]))
        for i, (a, b) in enumerate(zip(expected, got)):
            if a != b:
                print(f"first char diff at {i}: {a!r} vs {b!r}  context: {expected[max(0,i-30):i+30]!r}")
                break
        if len(expected) != len(got):
            print(f"length differs: expected {len(expected)} vs got {len(got)}")
