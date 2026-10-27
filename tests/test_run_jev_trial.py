"""The hosted-API runner: answer validation, model-string consistency, resume checks, failure accounting.
Everything runs against a local mock server; no key, no network, no cost."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_jev_trial as rj  # noqa: E402


def make_data(tmp_path, n=6):
    d = tmp_path / "data"; d.mkdir()
    with open(d / "test.jsonl", "w", encoding="utf-8") as f:
        for k in range(n):
            f.write(json.dumps({"state": f"note {k}", "_meta": {"id": f"trec2022/{k}/NCT{k}"},
                                "questions": {"eligible": {"type": "noul", "instructions": "Q?", "criteria": {"true": "E", "false": "X"},
                                                           "label": "true" if k % 2 == 0 else "false", "src": "s"}}}) + "\n")
    return d


class Server:
    def __init__(self, behave):
        self.behave, self.calls, self.bodies = behave, 0, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body)
                outer.calls += 1
                status, resp = outer.behave(outer.calls, body)
                data = json.dumps(resp).encode()
                self.send_response(status); self.send_header("Content-Length", str(len(data))); self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1/systemone"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def ok(p=0.7, model="jev-1.13.0"):
    return 200, {"answers": {"eligible": {"noul": p}}, "model": model, "usage": {"input_tokens": 10}}


def run(tmp_path, srv, extra=(), out="out.jsonl", n=6):
    d = make_data(tmp_path, n) if not (tmp_path / "data").exists() else tmp_path / "data"
    key = tmp_path / "key.txt"; key.write_text("secret-key\n")
    argv = ["run", "--data", str(d), "--out", str(tmp_path / out), "--key_file", str(key), "--api_url", srv.url, *extra]
    old = sys.argv; sys.argv = argv
    try:
        rj.main(); return 0
    except SystemExit as e:
        return e.code
    finally:
        sys.argv = old


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def test_label_and_key_never_leave_the_machine_in_the_request(tmp_path):
    srv = Server(lambda c, b: ok()); code = run(tmp_path, srv, ["--limit", "2"]); srv.close()
    assert code in (0, None)
    for b in srv.bodies:
        q = b["questions"]["eligible"]
        assert set(q) == {"type", "instructions", "criteria"} and "label" not in q
        assert set(b) == {"state", "model", "questions"}
    assert "secret-key" not in (tmp_path / "out.jsonl").read_text()


def test_success_rows_carry_both_latencies_and_complete(tmp_path):
    srv = Server(lambda c, b: ok(0.25)); code = run(tmp_path, srv); srv.close()
    assert code in (0, None)
    r = rows(tmp_path / "out.jsonl")
    assert len(r) == 6 and all(x["p_true"] == 0.25 and x["elapsed_ms"] >= x["latency_ms"] for x in r)
    assert {x["gold"] for x in r} == {True, False}


@pytest.mark.parametrize("bad", [{"answers": {}}, {"answers": {"eligible": {}}}, {"answers": {"eligible": {"noul": None}}},
                                 {"answers": {"eligible": {"noul": 1.5}}}, {"answers": {"eligible": {"noul": "0.4"}}},
                                 {"answers": {"eligible": {"noul": True}}}, {"answers": {"other": {"noul": 0.5}}}])
def test_invalid_probability_is_a_failure_not_a_success(tmp_path, bad):
    srv = Server(lambda c, b: (200, {**bad, "model": "jev-1.13.0"})); code = run(tmp_path, srv, ["--limit", "2"]); srv.close()
    assert code == 1
    r = rows(tmp_path / "out.jsonl")
    assert all("error" in x and x.get("p_true") is None for x in r)


def test_model_change_stops_the_run(tmp_path):
    srv = Server(lambda c, b: ok(model="jev-1.13.0" if c <= 2 else "jev-1.14.0")); code = run(tmp_path, srv); srv.close()
    assert code == 1
    r = rows(tmp_path / "out.jsonl")
    assert any("model changed" in x.get("error", "") for x in r)
    assert sum("p_true" in x for x in r) == 2


def test_expect_model_mismatch_is_caught(tmp_path):
    srv = Server(lambda c, b: ok(model="jev-9")); code = run(tmp_path, srv, ["--expect_model", "jev-1.13.0", "--limit", "2"]); srv.close()
    assert code == 1


def test_resume_skips_valid_rows_and_retries_failed_ones(tmp_path):
    calls = {"n": 0}

    def flaky(c, b):
        calls["n"] += 1
        return (200, {"answers": {}, "model": "jev-1.13.0"}) if c in (2, 4) else ok()
    srv = Server(flaky); code1 = run(tmp_path, srv); first = srv.calls
    code2 = run(tmp_path, srv); srv.close()
    assert code1 == 1 and code2 in (0, None)
    assert srv.calls - first == 2                       # only the two failed pairs were sent again
    final = {x["id"] for x in rows(tmp_path / "out.jsonl") if x.get("p_true") is not None}
    assert len(final) == 6


def test_resume_rejects_foreign_ids_and_wrong_gold(tmp_path):
    srv = Server(lambda c, b: ok()); run(tmp_path, srv, ["--limit", "2"])
    p = tmp_path / "out.jsonl"; r = rows(p)
    r[0]["gold"] = not r[0]["gold"]
    p.write_text("\n".join(json.dumps(x) for x in r) + "\n")
    assert run(tmp_path, srv) not in (0, None)
    r[0]["gold"] = not r[0]["gold"]; r[1]["id"] = "trec2022/zzz/NCT0"
    p.write_text("\n".join(json.dumps(x) for x in r) + "\n")
    assert run(tmp_path, srv) not in (0, None)
    srv.close()


def test_auth_error_stops_immediately(tmp_path):
    srv = Server(lambda c, b: (401, {"error": "bad key"})); code = run(tmp_path, srv); srv.close()
    assert code == 1 and srv.calls <= 2


def test_ids_file_with_unknown_ids_is_refused(tmp_path):
    d = make_data(tmp_path)
    ids = tmp_path / "ids.json"; ids.write_text(json.dumps(["trec2022/0/NCT0", "nope"]))
    srv = Server(lambda c, b: ok()); code = run(tmp_path, srv, ["--ids_file", str(ids)]); srv.close()
    assert code not in (0, None) and srv.calls == 0


def test_dry_run_needs_no_key_and_no_network(tmp_path):
    d = make_data(tmp_path)
    old = sys.argv; sys.argv = ["run", "--data", str(d), "--out", str(tmp_path / "o.jsonl"), "--dry_run"]
    try:
        rj.main()
    finally:
        sys.argv = old
    assert not (tmp_path / "o.jsonl").exists()
