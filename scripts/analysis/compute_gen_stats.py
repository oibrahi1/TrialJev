#!/usr/bin/env python3
"""Generation-baseline, readout-agreement and latency statistics for the paper (full test set + 992-pair timing sample).

    python compute_gen_stats.py <DIR_GEN> <DIR_MAIN> <DIR_POST> [--boot 10000]

Inputs are the saved per-pair files only. Accuracy and macro-F1 use the strict rule of generate_baselines.py (an
answer that is not exactly one of the two permitted words/letters counts as wrong; for macro-F1 it is counted as the
opposite of the gold class). Intervals resample the 50 test patients (paired across systems). Latency is reported
only within one run/GPU regime. Writes gen_stats.json and prints every table used in the paper.
"""
import argparse
import os
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("TRIALMATCH_OUT", "."))
sys.path.insert(0, str(HERE.parent))
import analyze_predictions as ap  # noqa: E402
SEED = 20261001
FULL = ["zs-chat-9b", "zs-chat-swapped-9b", "zs-plain-9b", "zs-chat-27b", "zs-chat-swapped-27b", "zs-chat-38", "zs-chat-swapped-38",
        "lora-logits-9b", "lora-generate-9b", "lora-generate-swapped-9b",
        "lora-logits-27b", "lora-generate-27b", "lora-generate-swapped-27b"]
ZSL = ["zsl-9b", "zsl-swapped-9b", "zsl-27b", "zsl-swapped-27b"]   # matched-prompt zero-shot (post-hoc export)
ZSL08 = ["zsl-0.8b", "zsl-swapped-0.8b"]                           # same, 0.8B (second machine export)
POINTER = ["pointer-onepass-9b", "pointer-onepass-27b", "pointer-onepass-38"]
EARLIER_PLAIN = {"lora-logits-9b": "ctl-q35-9b__step-400", "lora-logits-27b": "ctl-q35-27b__step-400"}
EARLIER_PTR = {"pointer-onepass-9b": "run1-q35-9b-base__step-400", "pointer-onepass-27b": "run1-q35-27b-base__step-400",
               "pointer-onepass-38": "run1-q38-27b-base__step-400"}
PAIRED = [  # (label, A, B) -> A minus B
    ("27B: LoRA (written answer) - zero-shot chat (written answer)", "lora-generate-27b", "zs-chat-27b"),
    ("27B: LoRA (written, swapped) - zero-shot chat (swapped)", "lora-generate-swapped-27b", "zs-chat-swapped-27b"),
    ("9B: LoRA 9B-Base (written) - zero-shot chat 9B instruct [different checkpoints]", "lora-generate-9b", "zs-chat-9b"),
    ("9B: LoRA 9B-Base (written, swapped) - zero-shot chat 9B instruct (swapped) [different checkpoints]", "lora-generate-swapped-9b", "zs-chat-swapped-9b"),
    ("zero-shot chat: 27B - 9B instruct [different sizes]", "zs-chat-27b", "zs-chat-9b"),
    ("zero-shot chat: Qwen3.5-27B - Qwen3.8-27B", "zs-chat-27b", "zs-chat-38"),
    ("zero-shot chat 27B: excluded-first - eligible-first", "zs-chat-27b", "zs-chat-swapped-27b"),
    ("zero-shot chat 9B: excluded-first - eligible-first", "zs-chat-9b", "zs-chat-swapped-9b"),
    ("zero-shot chat Qwen3.8: excluded-first - eligible-first", "zs-chat-38", "zs-chat-swapped-38"),
    ("27B LoRA written: excluded-first - eligible-first", "lora-generate-27b", "lora-generate-swapped-27b"),
    ("27B: LoRA (written) - zero-shot, same lettered prompt, no adapter", "lora-generate-27b", "zsl-27b"),
    ("27B: LoRA (written, swapped) - zero-shot letters (swapped)", "lora-generate-swapped-27b", "zsl-swapped-27b"),
    ("9B-Base: LoRA (written) - zero-shot letters, same checkpoint", "lora-generate-9b", "zsl-9b"),
    ("9B-Base: LoRA (written, swapped) - zero-shot letters (swapped)", "lora-generate-swapped-9b", "zsl-swapped-9b"),
    ("27B zero-shot: chat word - lettered prompt", "zs-chat-27b", "zsl-27b"),
    ("0.8B: fresh A/B-logit control - zero-shot letters (same prompt, no adapter)", "ctl-0.8b-fresh", "zsl-0.8b"),
]


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def pct(v):
    return f"{100 * v:.2f}"


def main():
    ar = argparse.ArgumentParser()
    ar.add_argument("gen")
    ar.add_argument("cluster")
    ar.add_argument("post")
    ar.add_argument("overnight", help="directory with gpucal_gpu{0-3}__preds.jsonl")
    ar.add_argument("--boot", type=int, default=10000)
    ar.add_argument("--e08b", required=True, help="directory holding runs/gen/{pointer-timing,lora-logits,lora-generate}-0.8b (0.8B timing)")
    ar.add_argument("--e08b_zsl", required=True, help="directory holding runs/gen_zsl/zsl-0.8b and zsl-swapped-0.8b")
    ar.add_argument("--small_runs", required=True, help="directory with the 0.8B A/B-logit test predictions (ctl-q35-0.8b-s0__step-200__eval-test-preds.jsonl)")
    a = ar.parse_args()
    gen, cl = Path(a.gen), Path(a.cluster)

    # ---------- load the full-set generation conditions, align on pair id ----------
    post = Path(a.post)
    rows = {c: jl(gen / f"gen_full__{c}__preds.jsonl") for c in FULL}
    rows.update({c: jl(post / "runs" / "gen_zsl" / c / "preds.jsonl") for c in ZSL})
    e08 = Path(a.e08b)
    z08 = Path(a.e08b_zsl)
    aih = Path(a.small_runs)
    rows.update({c: jl(z08 / "runs" / "gen_zsl" / c / "preds.jsonl") for c in ZSL08})
    # 0.8B fresh A/B-logit control test predictions (second machine evaluator format) as a generation-style row set
    rows["ctl-0.8b-fresh"] = [{"gold": ap.as_binary(r["gold"]), "pred": ap.as_binary(r["pred"]), "valid": True, "ms": 0.0,
                               "n_new_tokens": 0, "p_true": r["probs"]["true"], "meta": r["meta"]}
                              for r in jl(aih / "ctl-q35-0.8b-s0__step-200__eval-test-preds.jsonl")]
    ALL = FULL + ZSL + ZSL08 + ["ctl-0.8b-fresh"]
    def REPORT(c):
        if c in FULL: return gen / f"gen_full__{c}__report.json"
        if c in ZSL: return post / "runs" / "gen_zsl" / c / "report.json"
        if c in ZSL08: return z08 / "runs" / "gen_zsl" / c / "report.json"
        return None
    ids = sorted(r["meta"]["id"] for r in rows[FULL[0]])
    n = len(ids)
    pos = {i: k for k, i in enumerate(ids)}
    pat = None
    D = {}
    for c, rl in rows.items():
        assert len(rl) == n and {r["meta"]["id"] for r in rl} == set(ids), c
        gold = np.zeros(n, np.int8); pred = np.full(n, -1, np.int8); valid = np.zeros(n, bool); p_true = np.full(n, np.nan)
        pa = [None] * n
        for r in rl:
            k = pos[r["meta"]["id"]]
            gold[k] = int(r["gold"]); valid[k] = bool(r["valid"]); pa[k] = str(r["meta"]["patient_id"])
            if r["valid"]:
                pred[k] = int(r["pred"])
            if r.get("p_true") is not None:
                p_true[k] = r["p_true"]
        if pat is None:
            pat = pa
        assert pa == pat, c
        D[c] = {"gold": gold, "pred": pred, "valid": valid, "p_true": p_true}
    gold = D[FULL[0]]["gold"]
    for c in ALL:
        assert (D[c]["gold"] == gold).all(), c
    patients = sorted(set(pat))
    pidx = np.array([patients.index(p) for p in pat])
    npat = len(patients)
    print(f"{n} pairs, {npat} patients, gold eligible fraction {gold.mean():.4f}, majority floor {max(gold.mean(), 1 - gold.mean()):.4f}")

    # ---------- strict metrics + per-patient sufficient statistics ----------
    def eff_pred(c):   # invalid answer counts as the opposite of the gold class
        d = D[c]
        return np.where(d["valid"], d["pred"], 1 - gold).astype(np.int8)

    def per_patient_stats(c):
        e = eff_pred(c)
        out = np.zeros((npat, 7))
        for k in range(npat):
            m = pidx == k
            g, p = gold[m], e[m]
            out[k] = [m.sum(), (g == p).sum(), ((p == 1) & (g == 1)).sum(), ((p == 1) & (g == 0)).sum(),
                      ((p == 0) & (g == 1)).sum(), ((p == 0) & (g == 0)).sum(), (~D[c]["valid"][m]).sum()]
        return out

    def metrics_from(s, w):
        t = (w[:, None] * s).sum(0)
        n_, ok, tp1, fp1, fn1, tp0, inv = t[0], t[1], t[2], t[3], t[4], t[5], t[6]
        fp0, fn0 = fn1, fp1   # predicted excluded while eligible is a false negative for eligible and a false positive for excluded
        f1_1 = 2 * tp1 / (2 * tp1 + fp1 + fn1) if (2 * tp1 + fp1 + fn1) else 0.0
        f1_0 = 2 * tp0 / (2 * tp0 + fp0 + fn0) if (2 * tp0 + fp0 + fn0) else 0.0
        return ok / n_, (f1_0 + f1_1) / 2, inv / n_

    S = {c: per_patient_stats(c) for c in ALL}
    rng = np.random.default_rng(SEED)
    W = np.zeros((a.boot, npat))
    for b in range(a.boot):
        u, cnt = np.unique(rng.choice(npat, npat, replace=True), return_counts=True)
        W[b, u] = cnt
    ones = np.ones(npat)
    res = {}
    print("\n== full test set, strict scoring (invalid = wrong) ==")
    for c in ALL:
        acc, f1, inv = metrics_from(S[c], ones)
        boot = np.array([metrics_from(S[c], w)[0] for w in W])
        if REPORT(c) is not None:
            rep = json.load(open(REPORT(c)))
            assert abs(acc - rep["accuracy_strict_invalid_counts_wrong"]) < 1e-9, (c, acc, rep["accuracy_strict_invalid_counts_wrong"])
            assert abs(f1 - rep["macro_f1_strict_invalid_counts_wrong"]) < 1e-9, (c, f1, rep["macro_f1_strict_invalid_counts_wrong"])
            assert abs(inv - rep["invalid_rate"]) < 1e-9, c
        lo, hi = np.quantile(boot, [0.025, 0.975])
        ms = [r["ms"] for r in rows[c]]
        res[c] = {"accuracy": acc, "ci95": [float(lo), float(hi)], "macro_f1_strict": f1, "invalid_rate": inv,
                  "n_invalid": int((~D[c]["valid"]).sum()), "predicted_eligible_fraction": float(eff_pred(c).mean()),
                  "median_ms_full_run": float(st.median(ms)), "new_tokens_mean": float(np.mean([r["n_new_tokens"] for r in rows[c]]))}
        print(f"{c:28s} acc {pct(acc)} [{pct(lo)},{pct(hi)}]  F1 {pct(f1)}  invalid {res[c]['n_invalid']} ({100*inv:.2f}%)  "
              f"median {res[c]['median_ms_full_run']:.1f} ms  tokens {res[c]['new_tokens_mean']:.2f}   (matches report.json)")

    # ---------- AUROC of the scored (A/B-logit) conditions: zero-shot vs fine-tuned, same prompt ----------
    print("\n== AUROC of A/B-logit scored conditions (point and patient-bootstrap 95% interval; paired) ==")
    auc = {}
    scored_sys = ["zsl-0.8b", "ctl-0.8b-fresh", "zsl-9b", "lora-logits-9b", "zsl-27b", "lora-logits-27b"]
    nb = min(a.boot, 2000)
    for c in scored_sys:
        pt = D[c]["p_true"]
        assert not np.isnan(pt).any(), c
        auc[c] = {"point": float(ap.weighted_auroc(gold, pt, np.ones(n)))}
    boots = {c: [] for c in scored_sys}
    for b in range(nb):
        w = W[b][pidx]
        for c in scored_sys:
            boots[c].append(ap.weighted_auroc(gold, D[c]["p_true"], w))
    for c in scored_sys:
        lo, hi = np.quantile(boots[c], [0.025, 0.975])
        auc[c].update({"ci95": [float(lo), float(hi)], "bootstrap_resamples": nb})
        print(f"{c:16s} AUROC {auc[c]['point']:.4f} [{lo:.4f}, {hi:.4f}]")
    auc_diff = {}
    for fine, zero in (("lora-logits-27b", "zsl-27b"), ("lora-logits-9b", "zsl-9b")):
        d = np.array(boots[fine]) - np.array(boots[zero]); lo, hi = np.quantile(d, [0.025, 0.975])
        auc_diff[f"{fine} - {zero}"] = {"diff": auc[fine]["point"] - auc[zero]["point"], "ci95": [float(lo), float(hi)]}
        print(f"{fine} - {zero}: {auc[fine]['point'] - auc[zero]['point']:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # ---------- paired differences ----------
    pair = {}
    print("\n== paired differences in accuracy (patient bootstrap) ==")
    for label, A, B in PAIRED:
        pa_ = np.array([metrics_from(S[A], w)[0] - metrics_from(S[B], w)[0] for w in W])
        d = res[A]["accuracy"] - res[B]["accuracy"]
        lo, hi = np.quantile(pa_, [0.025, 0.975])
        p = float(min(1.0, 2 * min((pa_ <= 0).mean(), (pa_ >= 0).mean())))
        agree = float((eff_pred(A) == eff_pred(B)).mean())
        pair[label] = {"A": A, "B": B, "diff": d, "ci95": [float(lo), float(hi)], "p_two_sided": p, "prediction_agreement": agree}
        print(f"{label:100s} {100*d:+.2f} [{100*lo:+.2f},{100*hi:+.2f}] p={p:.4f}  agree {100*agree:.1f}%")

    # ---------- written answer vs scored letters: identical? ----------
    same = {}
    print("\n== LoRA: written letter vs scored letters, pair by pair ==")
    for sz in ("9b", "27b"):
        g_, l_ = D[f"lora-generate-{sz}"], D[f"lora-logits-{sz}"]
        dis = int((g_["pred"] != l_["pred"]).sum())
        same[sz] = {"disagreements": dis, "invalid_written": int((~g_["valid"]).sum())}
        print(f"{sz}: {dis} of {n} predictions differ; invalid written answers {same[sz]['invalid_written']}")

    # ---------- lora-logits full-set == earlier saved evaluation? ----------
    print("\n== re-scored vs earlier saved evaluation (same checkpoint) ==")
    earlier = {}
    for c, stem in EARLIER_PLAIN.items():
        old = {r["meta"]["id"]: r for r in jl(cl / f"{stem}__eval-test-preds.jsonl")}
        assert set(old) == set(ids)
        dis = sum(ap.as_binary(old[i]["pred"]) != int(D[c]["pred"][pos[i]]) for i in ids)
        dp = max(abs(old[i]["probs"]["true"] - D[c]["p_true"][pos[i]]) for i in ids)
        earlier[c] = {"pred_disagreements": dis, "max_abs_prob_difference": float(dp)}
        print(f"{c}: {dis} prediction differences vs {stem}; max |p| difference {dp:.4f}")

    # ---------- pointer: old path A vs one-pass path B ----------
    print("\n== pointer head: path A (prefix pass + cache branches) vs path B (one pass), full test set ==")
    ptr = {}
    for c in POINTER:
        rl = jl(gen / f"gen_full__{c}__preds.jsonl")
        rp = json.load(open(gen / f"gen_full__{c}__report.json"))
        assert len(rl) == n
        g_ = np.array([int(r["gold"]) for r in rl]); pA = np.array([r["p_A"] for r in rl]); pB = np.array([r["p_B"] for r in rl])
        order = np.array([pos[r["meta"]["id"]] for r in rl])
        assert len(set(r["meta"]["id"] for r in rl)) == n and set(r["meta"]["id"] for r in rl) == set(ids), c
        assert (gold[order] == g_).all()
        assert (np.array(pat)[order] == np.array([str(r["meta"]["patient_id"]) for r in rl])).all(), c
        a_, b_ = (pA > 0.5).astype(int), (pB > 0.5).astype(int)
        old = {r["meta"]["id"]: r for r in jl(cl / f"{EARLIER_PTR[c]}__eval-test-preds.jsonl")}
        o_pred = np.array([ap.as_binary(old[r["meta"]["id"]]["pred"]) for r in rl]); o_p = np.array([old[r["meta"]["id"]]["probs"]["true"] for r in rl])
        msA = np.array([r["ms_A"] for r in rl]); msB = np.array([r["ms_B"] for r in rl])
        ptr[c] = {"acc_A": float((a_ == g_).mean()), "acc_B": float((b_ == g_).mean()), "A_vs_B_disagreements": int((a_ != b_).sum()),
                  "A_vs_earlier_saved_disagreements": int((a_ != o_pred).sum()), "B_vs_earlier_saved_disagreements": int((b_ != o_pred).sum()),
                  "auroc_A": float(ap.weighted_auroc(g_, pA, np.ones(n))), "auroc_B": float(ap.weighted_auroc(g_, pB, np.ones(n))),
                  "max_abs_p_diff_A_vs_B": float(np.abs(pA - pB).max()),"max_abs_p_diff_A_vs_earlier": float(np.abs(pA - o_p).max()),
                  "median_ms_A": float(np.median(msA)), "median_ms_B": float(np.median(msB)),
                  "median_pairwise_ratio_A_over_B": float(np.median(msA / msB))}
        # paired bootstrap on A-B accuracy difference
        cA, cB = (a_ == g_).astype(float), (b_ == g_).astype(float)
        ptr[c]["acc_diff_B_minus_A"] = float(cB.mean() - cA.mean())
        print(f"{c:22s} acc A {pct(ptr[c]['acc_A'])}  B {pct(ptr[c]['acc_B'])}  A!=B on {ptr[c]['A_vs_B_disagreements']} pairs  "
              f"A!=earlier {ptr[c]['A_vs_earlier_saved_disagreements']}  B!=earlier {ptr[c]['B_vs_earlier_saved_disagreements']}  "
              f"max|dp| A-B {ptr[c]['max_abs_p_diff_A_vs_B']:.3f}  A {ptr[c]['median_ms_A']:.1f} ms  B {ptr[c]['median_ms_B']:.1f} ms  ratio {ptr[c]['median_pairwise_ratio_A_over_B']:.2f}")

    # ---------- latency: 992-pair sample, one chain per size on one GPU ----------
    print("\n== latency on the 992-pair timing sample (batch 1, same GPU within a size, run sequentially) ==")
    lat = {}
    for sz in ("9b", "27b"):
        L = {}
        for c in [f"zs-chat-{sz}", f"zs-chat-swapped-{sz}", f"lora-logits-{sz}", f"lora-generate-{sz}", f"lora-generate-swapped-{sz}"] + (["zs-plain-9b"] if sz == "9b" else []):
            rl = jl(gen / f"gen__{c}__preds.jsonl")
            L[c] = {r["meta"]["id"]: r["ms"] for r in rl}
        pr = {r["meta"]["id"]: (r["ms_A"], r["ms_B"]) for r in jl(gen / f"gen__pointer-timing-{sz}__preds.jsonl")}
        common = sorted(set.intersection(*[set(v) for v in L.values()]) & set(pr))
        assert len(common) == 992, (sz, len(common))
        med = {c: float(np.median([v[i] for i in common])) for c, v in L.items()}
        med["pointer-A"] = float(np.median([pr[i][0] for i in common])); med["pointer-B"] = float(np.median([pr[i][1] for i in common]))
        base = med[f"lora-logits-{sz}"]
        ratios = {c: float(np.median([ (pr[i][0] if c == "pointer-A" else pr[i][1] if c == "pointer-B" else L[c][i]) / L[f"lora-logits-{sz}"][i] for i in common])) for c in med}
        lat[sz] = {"median_ms": med, "median_pairwise_ratio_to_lora_logits": ratios}
        print(f"-- {sz}")
        for c in med:
            print(f"   {c:28s} median {med[c]:7.1f} ms   x{med[c]/base:.2f} of LoRA-scored letters (pairwise median ratio {ratios[c]:.2f})")

    # ---------- 0.8B timing (992 pairs, second machine, RTX 6000 Ada, fp32) ----------
    print("\n== 0.8B timing on the 992-pair sample (second machine, RTX 6000 Ada, fp32, one run) ==")
    t08 = {}
    base08 = e08 / "runs" / "gen"
    L8 = {r["meta"]["id"]: r for r in jl(base08 / "lora-logits-0.8b" / "preds.jsonl")}
    W8 = {r["meta"]["id"]: r for r in jl(base08 / "lora-generate-0.8b" / "preds.jsonl")}
    P8 = {r["meta"]["id"]: r for r in jl(base08 / "pointer-timing-0.8b" / "preds.jsonl")}
    common8 = sorted(set(L8) & set(W8) & set(P8))
    assert len(common8) == 992
    m_l = float(np.median([L8[i]["ms"] for i in common8])); m_w = float(np.median([W8[i]["ms"] for i in common8]))
    m_a = float(np.median([P8[i]["ms_A"] for i in common8])); m_b = float(np.median([P8[i]["ms_B"] for i in common8]))
    a_pred = np.array([P8[i]["p_A"] > 0.5 for i in common8]); b_pred = np.array([P8[i]["p_B"] > 0.5 for i in common8])
    n_empty = sum(1 for i in common8 if not W8[i]["valid"]); agree_valid = [W8[i]["pred"] == L8[i]["pred"] for i in common8 if W8[i]["valid"]]
    t08 = {"median_ms": {"lora-logits": m_l, "lora-generate": m_w, "pointer-A": m_a, "pointer-B": m_b},
           "ratio_to_lora_logits": {"lora-generate": m_w / m_l, "pointer-A": m_a / m_l, "pointer-B": m_b / m_l},
           "pointer_A_vs_B_prediction_differences": int((a_pred != b_pred).sum()),
           "lora_generate_invalid": n_empty, "lora_generate_valid_agree_with_logits": float(np.mean(agree_valid)), "n_valid": len(agree_valid),
           "lora_logits_accuracy_992": float(np.mean([L8[i]["correct"] for i in common8])),
           "lora_generate_accuracy_992_strict": float(np.mean([W8[i]["correct"] for i in common8]))}
    for k, v in t08["median_ms"].items():
        print(f"   {k:14s} {v:6.1f} ms  x{v / m_l:.2f}")
    print(f"   pointer A vs B differ on {t08['pointer_A_vs_B_prediction_differences']} of 992 pairs; LoRA writing: {n_empty} invalid of 992, "
          f"valid answers agree with logits on {t08['lora_generate_valid_agree_with_logits']:.3f} (n={len(agree_valid)}); strict accuracy {100*t08['lora_generate_accuracy_992_strict']:.2f}% vs scored {100*t08['lora_logits_accuracy_992']:.2f}% on the 992 sample")

    # ---------- per-GPU calibration ----------
    print("\n== quiet-node calibration (same 200 pairs, 9B LoRA letters) ==")
    cal = {}
    ov = Path(a.overnight)
    for g in range(4):
        r = jl(ov / f"gpucal_gpu{g}__preds.jsonl")
        cal[g] = float(st.median(x["ms"] for x in r)); print(f"GPU {g}: {cal[g]:.1f} ms (n={len(r)})")

    out = {"n_pairs": n, "n_patients": npat, "bootstrap": {"resamples": a.boot, "seed": SEED, "unit": "patient"}, "full_set": res,
           "paired": pair, "written_vs_scored": same, "rescored_vs_earlier": earlier, "pointer_paths": ptr, "latency_992": lat,
           "gpu_calibration_ms": cal, "timing_0.8b_992": t08, "auroc_scored": auc, "auroc_fine_minus_zero": auc_diff}
    (OUT / "gen_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote gen_stats.json")


if __name__ == "__main__":
    main()
