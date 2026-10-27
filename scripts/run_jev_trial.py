#!/usr/bin/env python3
"""Send TREC patient-trial pairs to the hosted Jev (TypeSafe System One) API and save the raw answers.

The processed records are already in the System One request shape (state + one typed question), so each request is
the record minus its label. The API key comes from the environment variable TYPESAFE_API_KEY or from --key_file (a
text file) and is never printed or written. The hosted model is an external baseline "without adaptation by us": we
cannot say what it was trained on.

Safeguards
  * a row counts as successful only if answers[<question id>]["noul"] is a finite number in [0, 1];
  * the returned model string must stay the same for the whole run and across resumes (it may be an alias, so the
    string and the time are saved with every row; the run stops if it changes or differs from --expect_model);
  * a resumed file is checked: rows must belong to the requested ids and carry the same gold label as the data;
    failed or invalid rows are retried, never counted as done;
  * latency is saved twice: the successful attempt (latency_ms) and the whole time including retries (elapsed_ms);
  * the run ends with a summary (ok, failed, missing, retries, latency percentiles) and a non-zero exit code if any
    requested pair has no valid answer.

    python scripts/run_jev_trial.py --data DIR --split test --out runs/jev_api/test.jsonl --dry_run
    python scripts/run_jev_trial.py --data DIR --split test --limit 20 --workers 1 --key_file KEY.txt --out runs/jev_api/pilot.jsonl
    python scripts/run_jev_trial.py --data DIR --split test --ids_file sample_ids.json --workers 1 --key_file KEY.txt --out runs/jev_api/timing992.jsonl
    python scripts/run_jev_trial.py --data DIR --split test --workers 6 --key_file KEY.txt --out runs/jev_api/test.jsonl

Latency here is client-side wall time of one request: network plus the hosted service. It is not comparable with the
local GPU timings and must be reported separately as hosted-API latency. Only synthetic TREC patient notes and public
registry text are sent; check TypeSafe's terms before a full run.
"""
import argparse
import datetime
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.typesafe.ai/v1/systemone"
FIELDS = ("type", "instructions", "criteria")
PRICE_PER_M_INPUT_TOKENS = 0.042   # USD as published by the vendor at the time of writing; output is free


def load_key(key_file=None, env_file=".env"):
    """Order: environment variable, then --key_file (a text file holding the key, alone or as TYPESAFE_API_KEY=...),
    then .env in the current folder. The key is never printed."""
    key = os.environ.get("TYPESAFE_API_KEY")
    for path in (key_file, env_file):
        if key or not path or not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8-sig"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("TYPESAFE_API_KEY="):
                line = line.split("=", 1)[1]
            key = line.strip().strip("'\"")
            break
    if not key:
        raise SystemExit("no API key: set TYPESAFE_API_KEY, or pass --key_file PATH (a text file with the key)")
    return key


def rid(rec):
    meta = rec["_meta"]
    if isinstance(meta, str):
        meta = json.loads(meta.replace("'", '"'))
    return meta["id"]


def qid_of(rec):
    return next(iter(rec["questions"]))


def gold_of(rec):
    return str(rec["questions"][qid_of(rec)]["label"]).lower() in ("true", "1")


def payload(rec, model):
    return {"state": rec["state"], "model": model,
            "questions": {qid: {k: v for k, v in q.items() if k in FIELDS} for qid, q in rec["questions"].items()}}


def valid_probability(resp, qid):
    """-> (p_true, answer dict) if the response holds a finite probability in [0, 1] for this question, else (None, why)."""
    ans = (resp.get("answers") or {}).get(qid)
    if not isinstance(ans, dict):
        return None, f"no answer for question '{qid}'"
    p = ans.get("noul")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
        return None, f"noul is not a probability: {p!r}"
    return float(p), ans


def call(body_obj, key, timeout, url, retries=5):
    """-> (response, ms of the successful attempt, ms including retries, retries used)."""
    body = json.dumps(body_obj).encode()
    last = None
    t_all = time.perf_counter()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
            now = time.perf_counter()
            return resp, (now - t0) * 1000, (now - t_all) * 1000, attempt
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
            if e.code < 500 and e.code != 429:
                raise RuntimeError(last)       # a real client error (bad key, bad request) must surface
        except Exception as e:                 # timeouts, resets, bad JSON
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(30, 2 ** attempt) + random.random())
    raise RuntimeError(f"giving up after {retries} attempts: {last}")


def read_existing(out, recs_by_id, expect_model):
    """Valid finished ids from a previous run; stops on rows that do not belong to this data."""
    done, models = {}, set()
    if not out.exists():
        return done, models
    for n, line in enumerate(open(out, encoding="utf-8"), 1):
        try:
            j = json.loads(line)
        except Exception:
            raise SystemExit(f"{out}: line {n} is not valid JSON; move the file aside or repair it")
        if "error" in j or j.get("p_true") is None:
            continue                            # failed rows are retried, never counted as done
        i = j.get("id")
        if i not in recs_by_id:
            raise SystemExit(f"{out}: id {i!r} (line {n}) is not among the requested pairs; wrong file or wrong --ids_file")
        if bool(j.get("gold")) != gold_of(recs_by_id[i]):
            raise SystemExit(f"{out}: gold label of {i!r} differs from the data; the file belongs to different data")
        if not isinstance(j["p_true"], (int, float)) or not math.isfinite(j["p_true"]) or not 0 <= j["p_true"] <= 1:
            continue
        done[i] = j
        if j.get("model"):
            models.add(j["model"])
    if len(models) > 1:
        raise SystemExit(f"{out}: rows carry several model strings {sorted(models)}; do not mix versions")
    if expect_model and models and models != {expect_model}:
        raise SystemExit(f"{out}: existing rows used {sorted(models)}, expected {expect_model}")
    return done, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with development.jsonl / test.jsonl (tm-v1-b4096)")
    ap.add_argument("--split", default="test", choices=["development", "test"])
    ap.add_argument("--ids_file", default=None, help="JSON list of pair ids to run (e.g. the 992-pair timing sample)")
    ap.add_argument("--model", default="jev-latest", help="model name sent in the request")
    ap.add_argument("--expect_model", default=None, help="stop if the API returns any other model string (e.g. jev-1.13.0)")
    ap.add_argument("--limit", type=int, default=None, help="first N unfinished pairs (pilot)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key_file", default=None, help="text file holding the API key (keep it outside the git repo)")
    ap.add_argument("--api_url", default=API_URL, help=argparse.SUPPRESS)
    ap.add_argument("--dry_run", action="store_true", help="count pairs, estimate tokens and cost; no network, no key")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(Path(a.data) / f"{a.split}.jsonl", encoding="utf-8")]
    if a.ids_file:
        want = set(json.load(open(a.ids_file, encoding="utf-8")))
        recs = [r for r in recs if rid(r) in want]
        if len(recs) != len(want):
            raise SystemExit(f"only {len(recs)} of the {len(want)} ids in {a.ids_file} were found in the data")
        print(f"{len(recs)} pairs match the {len(want)} ids in {a.ids_file}")
    by_id = {rid(r): r for r in recs}
    if len(by_id) != len(recs):
        raise SystemExit("duplicate pair ids in the data")
    out = Path(a.out)
    done, models = read_existing(out, by_id, a.expect_model)
    todo = [r for r in recs if rid(r) not in done]
    if a.limit:
        todo = todo[: a.limit]
    est_tok = sum(len(json.dumps(payload(r, a.model))) for r in todo) / 4
    print(f"{len(recs)} pairs, {len(done)} valid answers already saved, {len(todo)} to run; about {est_tok / 1e6:.2f}M input "
          f"tokens (chars/4 estimate), about ${est_tok / 1e6 * PRICE_PER_M_INPUT_TOKENS:.2f} at ${PRICE_PER_M_INPUT_TOKENS}/M")
    if a.dry_run:
        return
    key = load_key(a.key_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    expected = {"model": a.expect_model or (next(iter(models)) if models else None)}
    lock = threading.Lock()
    fh = open(out, "a", encoding="utf-8")
    st = {"ok": 0, "failed": 0, "retries": 0, "lat": [], "el": [], "tok": 0}
    stop = threading.Event()
    t_start = time.perf_counter()
    queue = list(reversed(todo))
    qlock = threading.Lock()

    def save(row):
        row["time"] = datetime.datetime.now().isoformat(timespec="seconds")
        fh.write(json.dumps(row) + "\n"); fh.flush()

    def work(rec):
        i, qid = rid(rec), qid_of(rec)
        try:
            resp, lat, el, retried = call(payload(rec, a.model), key, a.timeout, a.api_url)
        except Exception as e:
            with lock:
                save({"id": i, "error": str(e)}); st["failed"] += 1
                if str(e).startswith(("HTTP 401", "HTTP 403")):
                    print("authentication failed; stopping", file=sys.stderr); stop.set()
            return
        p, ans = valid_probability(resp, qid)
        model = resp.get("model")
        with lock:
            st["retries"] += retried
            if p is None:
                save({"id": i, "error": f"invalid answer: {ans}", "model": model, "retries": retried}); st["failed"] += 1
                return
            if expected["model"] is None:
                expected["model"] = model
                print(f"model string returned by the API: {model!r}", flush=True)
            if model != expected["model"]:
                save({"id": i, "error": f"model changed: {model!r} != {expected['model']!r}", "model": model}); st["failed"] += 1
                print(f"model string changed to {model!r} (expected {expected['model']!r}); stopping", file=sys.stderr)
                stop.set()
                return
            usage = resp.get("usage") or {}
            save({"id": i, "gold": gold_of(rec), "p_true": p, "answer": ans, "model": model, "latency_ms": round(lat, 1),
                  "elapsed_ms": round(el, 1), "retries": retried, "usage": usage})
            st["ok"] += 1; st["lat"].append(lat); st["el"].append(el); st["tok"] += usage.get("input_tokens") or 0
            n = st["ok"] + st["failed"]
            if n % 25 == 0 or n == len(todo):
                t = time.perf_counter() - t_start
                print(f"  {n}/{len(todo)} ok={st['ok']} failed={st['failed']} {t:.0f}s {n / t:.2f}/s "
                      f"median {statistics.median(st['lat']):.0f} ms", flush=True)

    def loop():
        while not stop.is_set():
            with qlock:
                if not queue:
                    return
                rec = queue.pop()
            work(rec)

    ts = [threading.Thread(target=loop, daemon=True) for _ in range(a.workers)]
    [t.start() for t in ts]; [t.join() for t in ts]
    fh.close()

    final, _ = read_existing(out, by_id, expected["model"])
    missing = [i for i in by_id if i not in final]
    if a.limit:
        missing = [i for i in missing if i in {rid(r) for r in todo}]
    q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))] if v else float("nan")
    print(f"summary: requested {len(by_id) if not a.limit else len(todo)}, valid {len(final)}, failed this run {st['failed']}, "
          f"retries {st['retries']}, missing {len(missing)}, model {expected['model']!r}, input tokens billed about {st['tok']}")
    if st["lat"]:
        print(f"latency ms (successful attempt) p50 {q(st['lat'], .5):.0f} p90 {q(st['lat'], .9):.0f} p99 {q(st['lat'], .99):.0f}; "
              f"incl. retries p50 {q(st['el'], .5):.0f} p99 {q(st['el'], .99):.0f}")
    if missing or stop.is_set():
        print(f"INCOMPLETE: {len(missing)} pairs have no valid answer; run the same command again to retry them", file=sys.stderr)
        sys.exit(1)
    print("COMPLETE")


if __name__ == "__main__":
    main()
