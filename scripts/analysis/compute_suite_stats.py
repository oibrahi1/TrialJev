#!/usr/bin/env python3
"""Statistics for the generation suite (fine-tuned writer, reasoning baseline, same-run timing).

    python compute_suite_stats.py <CLUSTER_RUNS_DIR> <SMALL_RUNS_DIR> [--boot 10000] [--out suite_stats.json]

Inputs are the saved report.json / preds.jsonl files only (runs/gen_sft, runs/gen_time, runs/gen_sft_extra).
Strict rule: an answer that is not exactly one permitted word counts as wrong. Intervals resample the 50 test patients,
paired across systems. Latency is compared only within one timing chain (same GPU, same 992 pairs, batch 1).
"""
import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

SEED = 20261001
SIZES = [("0.8B", "q35-0.8b", "small"), ("9B", "q35-9b", "cluster"), ("27B", "q35-27b", "cluster"), ("Qwen3.8-27B", "q38-27b", "cluster")]
WRITERS = {"q35-0.8b": ["sft-q35-0.8b-s0"], "q35-9b": ["sft-q35-9b-s0", "sft-q35-9b-s1", "sft-q35-9b-s2"],
           "q35-27b": ["sft-q35-27b-s0"], "q38-27b": ["sft-q38-27b-s0"]}


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); i = 0
    sv = allv[order]
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def by_id(rows):
    return {r["meta"]["id"]: r for r in rows}


def correct_writer(r):
    return bool(r.get("valid")) and r["pred"] == r["gold"]


def boot_diff(a, b, groups, n, rng):
    """a, b: 0/1 arrays over the same pairs; groups: patient id per pair. Returns diff in points and 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ug = sorted(set(groups)); idx = {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in ug}
    d = []
    for _ in range(n):
        pick = rng.integers(0, len(ug), len(ug))
        ii = np.concatenate([idx[ug[k]] for k in pick])
        d.append(a[ii].mean() - b[ii].mean())
    return 100 * (a.mean() - b.mean()), 100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster"); ap.add_argument("small")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="suite_stats.json")
    ap.add_argument("--pointer_dir", default=None, help="folder with run1-*__step-400__eval-test-preds.jsonl (full-test pointer predictions)")
    a = ap.parse_args()
    root = {"cluster": Path(a.cluster), "small": Path(a.small)}
    rng = np.random.default_rng(SEED)
    out = {"full_test_writer": {}, "timing": {}, "paired_992": {}, "reason_200": {}}

    print("== Fine-tuned writer, full test set (6,975 pairs), strict")
    for label, key, where in SIZES:
        accs = []
        for w in WRITERS[key]:
            rows = jl(root[where] / "gen_sft" / w / "preds.jsonl")
            rep = json.load(open(root[where] / "gen_sft" / w / "report.json"))
            y = [r["gold"] for r in rows]; p = [r.get("p_true", 0.5) for r in rows]
            acc = 100 * float(np.mean([correct_writer(r) for r in rows]))
            accs.append(acc)
            out["full_test_writer"][w] = {"n": len(rows), "accuracy": acc, "invalid_rate": rep["invalid_rate"],
                                          "auroc_first_token": auroc(y, p), "macro_f1": 100 * rep["macro_f1_strict_invalid_counts_wrong"],
                                          "median_ms": rep["ms_per_pair"]["median"]}
            print(f"  {w:18s} n={len(rows)} acc {acc:.2f} AUROC {auroc(y, p):.4f} invalid {rep['invalid_rate']:.4f} "
                  f"elig-recall {100 * rep['eligible_recall_invalid_is_miss']:.1f} excl-recall {100 * rep['excluded_recall_invalid_is_miss']:.1f}")
        if len(accs) > 1:
            print(f"  -> {label} mean {st.mean(accs):.2f} sd {st.stdev(accs):.2f}")
            out["full_test_writer"][f"{key}-mean"] = {"mean": st.mean(accs), "sd": st.stdev(accs)}

    print("\n== Same-run timing chain, 992 pairs, batch 1 (ratio = median ms / TrialJev one-pass median)")
    hdr = f"{'size':12s} {'system':28s} {'acc%':>6s} {'ms':>8s} {'x TrialJev':>10s}"
    print(hdr)
    for label, key, where in SIZES:
        d = root[where] / "gen_time" / key
        ptr_rows = jl(d / "pointer" / "preds.jsonl"); ptr = json.load(open(d / "pointer" / "report.json"))
        ms_B = ptr["path_B_one_pass_ms"]["median"]; ms_A = ptr["path_A_probs_and_prefix_ms"]["median"]
        systems = {}
        for name in ["control", "sft-generate", "zeroshot"]:
            rep = json.load(open(d / name / "report.json"))
            systems[name] = (100 * rep["accuracy_strict_invalid_counts_wrong"], rep["ms_per_pair"]["median"], rep["invalid_rate"])
        rows_t = {
            "TrialJev one-pass": (100 * ptr["accuracy_B"], ms_B, 0.0),
            "TrialJev two-pass": (100 * ptr["accuracy_A"], ms_A, 0.0),
            "A/B-logit LoRA (ablation)": systems["control"],
            "Fine-tuned writer (seed 0)": systems["sft-generate"],
            "Prompted one-word": systems["zeroshot"],
        }
        out["timing"][key] = {}
        for n, (acc, ms, inv) in rows_t.items():
            print(f"{label:12s} {n:28s} {acc:6.2f} {ms:8.1f} {ms / ms_B:10.2f}" + (f"   invalid {100 * inv:.1f}%" if inv else ""))
            out["timing"][key][n] = {"accuracy": acc, "median_ms": ms, "ratio_vs_trialjev_onepass": ms / ms_B, "invalid_rate": inv}

        # paired accuracy: TrialJev one-pass vs writer vs control on the same 992 pairs
        pid = {r["meta"]["id"]: r for r in ptr_rows}
        wr = by_id(jl(d / "sft-generate" / "preds.jsonl")); ct = by_id(jl(d / "control" / "preds.jsonl"))
        ids = sorted(pid); groups = [pid[i]["meta"]["patient_id"] for i in ids]
        c_ptr = [int((pid[i]["p_B"] >= 0.5) == pid[i]["gold"]) for i in ids]
        c_wr = [int(correct_writer(wr[i])) for i in ids]
        c_ct = [int(correct_writer(ct[i])) for i in ids]
        out["paired_992"][key] = {}
        for nm, other in (("writer", c_wr), ("A/B-logit", c_ct)):
            dlt, lo, hi = boot_diff(c_ptr, other, groups, a.boot, rng)
            print(f"{'':12s}   TrialJev one-pass - {nm}: {dlt:+.2f} pts [{lo:+.2f}, {hi:+.2f}]")
            out["paired_992"][key][nm] = {"diff": dlt, "lo": lo, "hi": hi}

        # reasoning on the 200-pair subset, all systems on the same pairs
        rd = d / "reason"
        if rd.exists():
            rr = by_id(jl(rd / "preds.jsonl")); rep = json.load(open(rd / "report.json"))
            sub = sorted(rr); g2 = [rr[i]["meta"]["patient_id"] for i in sub]
            c_r = [int(correct_writer(rr[i])) for i in sub]
            c_p = [int((pid[i]["p_B"] >= 0.5) == pid[i]["gold"]) for i in sub]
            c_w = [int(correct_writer(wr[i])) for i in sub]
            ptr_sub_ms = float(np.median([pid[i]["ms_B"] for i in sub]))
            dlt, lo, hi = boot_diff(c_p, c_r, g2, a.boot, rng)
            r_ms = rep["ms_per_pair"]["median"]
            print(f"{label:12s} reasoning, {len(sub)} pairs: acc {100 * np.mean(c_r):.1f} (TrialJev {100 * np.mean(c_p):.1f}, writer {100 * np.mean(c_w):.1f}) "
                  f"invalid {100 * rep['invalid_rate']:.1f}% tokens mean {rep['new_tokens']['mean']:.0f} median {r_ms:.0f} ms "
                  f"= {r_ms / ptr_sub_ms:.0f}x TrialJev on the same pairs; TrialJev-reasoning {dlt:+.1f} [{lo:+.1f}, {hi:+.1f}]")
            out["reason_200"][key] = {"n": len(sub), "reason_acc": 100 * np.mean(c_r), "trialjev_acc": 100 * np.mean(c_p), "writer_acc": 100 * np.mean(c_w),
                                      "invalid_rate": rep["invalid_rate"], "mean_new_tokens": rep["new_tokens"]["mean"], "reason_median_ms": r_ms,
                                      "trialjev_onepass_median_ms_on_subset": ptr_sub_ms, "ratio": r_ms / ptr_sub_ms,
                                      "diff_trialjev_minus_reason": {"d": dlt, "lo": lo, "hi": hi}}
        print()

    if a.pointer_dir:
        print("== Full test (6,975 pairs): TrialJev two-pass seed 0 minus fine-tuned writer (paired, patient bootstrap, unadjusted)")
        out["paired_full"] = {}
        pmap = {"q35-9b": "run1-q35-9b-base", "q35-27b": "run1-q35-27b-base", "q38-27b": "run1-q38-27b-base"}
        for label, key, where in SIZES:
            if key not in pmap:
                continue
            pr = {r["meta"]["id"]: r for r in jl(Path(a.pointer_dir) / f"{pmap[key]}__step-400__eval-test-preds.jsonl")}
            ids = sorted(pr); groups = [pr[i]["meta"]["patient_id"] for i in ids]
            c_ptr = np.array([int((pr[i]["probs"]["true"] >= 0.5) == (pr[i]["gold"] in (True, "true"))) for i in ids])
            ws = []
            for w in WRITERS[key]:
                wr = by_id(jl(root[where] / "gen_sft" / w / "preds.jsonl"))
                ws.append(np.array([int(correct_writer(wr[i])) for i in ids]))
            res = {"pointer_seed0_acc": 100 * c_ptr.mean()}
            dlt, lo, hi = boot_diff(c_ptr, ws[0], groups, a.boot, rng)
            print(f"  {label:12s} pointer s0 {100 * c_ptr.mean():.2f} vs writer s0 {100 * ws[0].mean():.2f}: {dlt:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            res["vs_writer_s0"] = {"d": dlt, "lo": lo, "hi": hi, "writer_acc": 100 * ws[0].mean()}
            if len(ws) > 1:
                m = np.mean(ws, axis=0)
                dlt, lo, hi = boot_diff(c_ptr, m, groups, a.boot, rng)
                print(f"  {'':12s} pointer s0 vs mean of {len(ws)} writer seeds ({100 * m.mean():.2f}): {dlt:+.2f} [{lo:+.2f}, {hi:+.2f}]")
                res["vs_writer_seedmean"] = {"d": dlt, "lo": lo, "hi": hi, "writer_acc": 100 * m.mean()}
            out["paired_full"][key] = res

    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
