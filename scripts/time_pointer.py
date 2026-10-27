#!/usr/bin/env python3
"""Time the pointer-head model two ways on the SAME pairs, interleaved, on the same GPU.

  path A  `probs_and_prefix`  the scoring path used for the reported latencies: encode the patient state, then
          run the trial branch from a copy of the cached state (two passes + a cache copy for one question).
  path B  `forward_rows_batch` the training-style one-pass path: patient state + trial branch in a single row,
          one forward pass, no cache.

Both compute the same function (the script reports the largest probability difference between them).

    python scripts/time_pointer.py --run runs/run1-q35-9b-base/step-400 --data data/tm-v1-b4096 \
        --split test --allow-test --ids_file runs/gen/zs-chat-27b/sample_ids.json --dtype bf16 --out runs/gen/ptr-9b
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402

import generate_baselines as gb  # noqa: E402
from trialmatch.checkpoint import load, resolve_run  # noqa: E402
from trialmatch.model import MAX_BRANCH, MAX_STATE  # noqa: E402
from trialmatch.records import load_records, materialize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="pointer-head checkpoint dir")
    ap.add_argument("--data", required=True, help="dataset directory")
    ap.add_argument("--split", default="test")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--ids_file", default="")
    ap.add_argument("--per_patient", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        raise SystemExit("test split is locked: pass --allow-test")

    out_dir = Path(a.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already exists and is not empty; refusing to overwrite evidence")
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sync = (lambda: torch.cuda.synchronize()) if device.startswith("cuda") else (lambda: None)
    tok, model = load(a.run, device, dtype=torch.bfloat16 if a.dtype == "bf16" else torch.float32,
                      merge=a.dtype == "fp32")
    meta = torch.load(Path(resolve_run(a.run)) / "head.pt", map_location="cpu", weights_only=False)
    max_state, max_branch = meta.get("max_state", MAX_STATE), meta.get("max_branch", MAX_BRANCH)

    reqs = load_records(str(Path(a.data) / f"{a.split}.jsonl"), source="medjev")
    reqs = gb.pick_sample(reqs, a.per_patient, a.ids_file)
    n_requested = len(set(json.loads(Path(a.ids_file).read_text(encoding="utf-8")))) if a.ids_file else None
    if n_requested is not None and len(reqs) != n_requested:
        raise SystemExit(f"{n_requested - len(reqs)} of the {n_requested} requested pair ids were not found")
    n_sample = len(reqs)
    if a.limit:
        reqs = reqs[: a.limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    def time_path(path, enc):
        model.lm.config.use_cache = path == "A"
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            if path == "A":
                p = float(model.probs_and_prefix(enc)[0][0][1])
            else:
                p = float(torch.softmax(model.forward_rows_batch([enc])[0][0], -1)[1])
        sync()
        return 1000 * (time.perf_counter() - t0), p

    for req in reqs[: a.warmup]:
        enc = model.encode(tok, materialize(req), max_state=max_state, max_branch=max_branch)
        time_path("A", enc)
        time_path("B", enc)

    rows = []
    for i, req in enumerate(reqs):
        enc = model.encode(tok, materialize(req), max_state=max_state, max_branch=max_branch)
        first, second = ("A", "B") if i % 2 == 0 else ("B", "A")      # alternate who goes first
        res = {first: time_path(first, enc), second: time_path(second, enc)}
        gold = bool(req["questions"]["eligible"]["label"])
        rows.append({"gold": gold, "ms_A": res["A"][0], "ms_B": res["B"][0], "p_A": res["A"][1], "p_B": res["B"][1],
                     "meta": req["_meta"]})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(reqs)}  A {statistics.median(r['ms_A'] for r in rows):.0f} ms  "
                  f"B {statistics.median(r['ms_B'] for r in rows):.0f} ms", flush=True)

    n = len(rows)

    def stats(key):
        v = sorted(r[key] for r in rows)
        return {"median": statistics.median(v), "p95": v[int(0.95 * (n - 1))], "mean": statistics.mean(v)}

    import transformers
    report = {"run": a.run, "checkpoint_metadata": {k: meta.get(k) for k in ("base", "base_revision", "weights_dtype", "lora",
                                                                         "head_dim", "max_state", "max_branch", "step")},
              "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "python": sys.version.split()[0],
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "dtype": a.dtype, "device": device, "pairs": n, "pairs_in_full_sample": n_sample,
              "pair_ids_requested": n_requested, "batch_size": 1,
              "path_A_probs_and_prefix_ms": stats("ms_A"), "path_B_one_pass_ms": stats("ms_B"),
              "median_ratio_A_over_B": statistics.median(r["ms_A"] for r in rows) / statistics.median(r["ms_B"] for r in rows),
              "max_abs_prob_difference_A_vs_B": max(abs(r["p_A"] - r["p_B"]) for r in rows),
              "accuracy_A": sum((r["p_A"] > 0.5) == r["gold"] for r in rows) / n,
              "accuracy_B": sum((r["p_B"] > 0.5) == r["gold"] for r in rows) / n}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out_dir / "preds.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
