"""M7-B2: the experimental worker is exact, isolated between requests and fails without dying."""
import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from accelsched_kai import p2h_producer as base, p2h_worker_experiment as w, p2hpc_producer as pc, schema
from test_p2hpc_producer import FORE, cohort, request as pc_request, scarce_projection
from test_p2h_producer import projection, request as p2h_request

w.boot()
NAMES = ("envelope", "artifact", "snapshot", "execution", "controls")


def line(kind, request):
    return json.dumps(dict(op="produce", kind=kind, request=request))


def decode(resp):
    return [base64.b64decode(resp[n]) for n in NAMES]


def same(cold, warm):
    return [cold[i] for i in (0, 1, 2, 4)] == [warm[i] for i in (0, 1, 2, 4)]  # execution holds volatile timing


def test_worker_output_is_byte_identical_to_the_cold_producers():
    r = p2h_request(projection(4))
    resp = w.handle(line("p2h", r))
    assert resp["ok"] and same(base.produce(r), decode(resp))
    r2 = pc_request()
    resp2 = w.handle(line("p2hpc", r2))
    assert resp2["ok"] and same(pc.produce(r2), decode(resp2))


def test_interleaved_requests_do_not_leak_state():
    a, b, c = p2h_request(projection(2), alpha=0.), p2h_request(projection(4), alpha=.5), pc_request(alpha=.5)
    expected = [base.produce(a), base.produce(b), pc.produce(c)]
    order = [0, 1, 2, 0, 2, 1, 1, 0]
    reqs = [("p2h", a), ("p2h", b), ("p2hpc", c)]
    for i in order:
        resp = w.handle(line(*reqs[i]))
        assert resp["ok"] and same(expected[i], decode(resp))


def test_failed_request_does_not_poison_later_requests_and_bad_input_is_reported():
    good = p2h_request(projection(3))
    bad = deepcopy_request(good)
    bad["phase"] = 99
    assert w.handle(line("p2h", bad))["ok"] is False
    assert w.handle("{not json")["ok"] is False
    assert w.handle(json.dumps(dict(op="produce", kind="nope", request={})))["ok"] is False
    assert w.handle(json.dumps(dict(op="selftest_crash")))["ok"] is False  # disabled without the explicit self-test switch
    assert same(base.produce(good), decode(w.handle(line("p2h", good))))


def deepcopy_request(r):
    import copy
    return copy.deepcopy(r)


def test_stage_timings_are_non_overlapping_and_sum_to_the_total():
    resp = w.handle(line("p2hpc", pc_request()))
    t = resp["timings"]
    assert t["snapshot"] + t["build"] + t["load"] + t["other"] == t["total"] and min(t["snapshot"], t["build"], t["load"], t["other"]) >= 0
    assert t["build"] > 0 and t["load"] > 0 and 0 <= t["lp_solve_s"] * 1e9 <= t["build"]


def test_no_projection_artifact_or_price_is_cached_between_requests():
    src = Path(w.__file__).read_text()
    assert "lru_cache" not in src and "_cache" not in src.replace("no cache", "")
    r = p2h_request(projection(4))
    first = decode(w.handle(line("p2h", r)))
    other = p2h_request(projection(4, request=(6, 4, 2)))
    assert not same(first, decode(w.handle(line("p2h", other))))
    assert same(first, decode(w.handle(line("p2h", r))))


def test_process_level_worker_serves_many_requests_and_selftests_are_gated(tmp_path):
    src = str(Path(__file__).resolve().parents[1] / "src")
    code = "import runpy,sys;sys.path.insert(0,sys.argv[1]);runpy.run_module('accelsched_kai.p2h_worker_experiment',run_name='__main__')"
    p = subprocess.Popen([sys.executable, "-I", "-B", "-c", code, src], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    ready = json.loads(p.stdout.readline())
    assert ready["op"] == "ready" and ready["boot"]["import_ns"] > 0
    r = p2h_request(projection(4))
    for _ in range(3):
        p.stdin.write(line("p2h", r) + "\n"); p.stdin.flush()
        resp = json.loads(p.stdout.readline())
        assert resp["ok"] and same(base.produce(r), decode(resp))
    p.stdin.write(json.dumps(dict(op="selftest_crash")) + "\n"); p.stdin.flush()
    assert json.loads(p.stdout.readline())["ok"] is False and p.poll() is None  # still alive: self-test disabled
    p.stdin.close(); p.wait(timeout=20)
    assert p.returncode == 0
