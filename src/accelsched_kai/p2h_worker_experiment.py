"""M7-B2 EXPERIMENT ONLY: a long-lived controlled Python worker for the unchanged P2H producers.

This is a latency experiment, NOT a production path and NOT wired to any plugin or grant. A request
carries a complete fresh native projection (and cohort) exactly as the cold one-process-per-task
producer receives it; every request is recomputed from scratch (full M3-A build, full primal-dual
re-certification on load) and the worker holds NO cache of projections, artifacts, prices or Advice
between requests. Outputs are byte-identical to ``p2h_producer.produce`` / ``p2hpc_producer.produce``.
Time is attributed to non-overlapping stages: parse, snapshot, build (includes the LP solve and the
original certificate), load (full recertification) and other (validation + serialization).
"""
import base64
import json
import os
import sys
import time

_BOOT = {}


def boot():
    """Import the heavy modules once and record spawn-to-first-line and import cost."""
    t_first = time.time_ns()
    epoch = int(os.environ.get("M7_SPAWN_EPOCH_NS", "0") or 0)
    t0 = time.perf_counter_ns()
    global schema, base, pc, ba
    from . import bounded_advice as ba, p2h_producer as base, p2hpc_producer as pc, schema  # noqa: E402  (heavy: numpy, scipy, frozen AccelSched)
    t1 = time.perf_counter_ns()
    _BOOT.update(spawn_to_first_line_ns=(t_first - epoch) if epoch else None, import_ns=t1 - t0, pid=os.getpid())
    return dict(_BOOT)


class _Timers:
    def __init__(self):
        self.ns = dict(snapshot=0, build=0, load=0)

    def wrap(self, owner, name, key):
        original = getattr(owner, name)

        def timed(*a, **k):
            t = time.perf_counter_ns()
            try:
                return original(*a, **k)
            finally:
                self.ns[key] += time.perf_counter_ns() - t
        setattr(owner, name, timed)
        return original


def _produce(kind, request):
    timers = _Timers()
    saved = [(base, "snapshot_from_projection", timers.wrap(base, "snapshot_from_projection", "snapshot")),
             (ba, "build_artifact", timers.wrap(ba, "build_artifact", "build")),
             (ba, "load_artifact", timers.wrap(ba, "load_artifact", "load"))]
    t0 = time.perf_counter_ns()
    try:
        out = (base if kind == "p2h" else pc).produce(request)
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)
    total = time.perf_counter_ns() - t0
    lp = sum(a.get("elapsed_seconds", 0.0) for a in schema.loads(out[3]).get("attempts", []))
    stages = dict(timers.ns, total=total)
    stages["other"] = total - timers.ns["snapshot"] - timers.ns["build"] - timers.ns["load"]
    stages["lp_solve_s"] = lp
    return out, stages


def handle(line, selftest=False):
    """One request line -> one response dict. Never raises; failures are reported and the worker keeps serving."""
    try:
        req = schema.loads(line)
        op = req.get("op")
        if op == "ping":
            return dict(ok=True, op="ping", boot=_BOOT)
        if op in ("selftest_crash", "selftest_hang") and selftest:
            if op == "selftest_crash":
                os._exit(17)
            time.sleep(3600)
        if op != "produce" or req.get("kind") not in ("p2h", "p2hpc") or not isinstance(req.get("request"), dict):
            return dict(ok=False, error="unsupported op/kind")
        t = time.perf_counter_ns()
        request = req["request"]
        out, stages = _produce(req["kind"], request)
        stages["parse_ns"] = 0
        names = ("envelope", "artifact", "snapshot", "execution", "controls")
        resp = {n: base64.b64encode(b).decode() for n, b in zip(names, out)}
        stages["serialize_ns"] = time.perf_counter_ns() - t - stages["total"]
        return dict(ok=True, op="produce", timings=stages, **resp)
    except Exception as exc:  # noqa: BLE001 - bounded, reported, non-fatal
        return dict(ok=False, error=type(exc).__name__ + ": " + str(exc)[:300])


def serve(stdin=None, stdout=None):
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    selftest = os.environ.get("M7_WORKER_SELFTEST") == "1"
    ready = boot()
    stdout.write(json.dumps(dict(ok=True, op="ready", boot=ready), sort_keys=True) + "\n")
    stdout.flush()
    for line in stdin:
        if len(line) > 1_048_576:
            resp = dict(ok=False, error="request too large")
        else:
            resp = handle(line, selftest)
        stdout.write(json.dumps(resp, sort_keys=True) + "\n")
        stdout.flush()


def oneshot(request_path, out_dir):
    """Cold path: the same instrumented producer, one process per task; writes the five outputs and timings."""
    ready = boot()
    t = time.perf_counter_ns()
    req = schema.loads(open(request_path, "rb").read())
    out, stages = _produce(req["kind"], req["request"])
    os.makedirs(out_dir, exist_ok=False)
    for name, data in zip(("envelope.json", "artifact.json", "snapshot.json", "execution.json", "controls.json"), out):
        open(os.path.join(out_dir, name), "wb").write(data)
    stages.update(ready, wall_inside_python_ns=time.perf_counter_ns() - t)
    open(os.path.join(out_dir, "timings.json"), "w").write(json.dumps(stages, sort_keys=True))
    sys.stdout.buffer.write(out[0])


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--oneshot":
        oneshot(sys.argv[2], sys.argv[3])
    else:
        serve()
