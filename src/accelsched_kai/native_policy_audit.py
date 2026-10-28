"""M4-B independent audit of Stage A native records (Fake Session CPU; PROBE_ONLY).

Reads the normalized JSON written by the Go native differential and re-derives, with the
independent offline Oracle and the unchanged M4-A simulator, what the model-equivalent
state must be. Everything labelled OFFLINE_* below is a secondary, model-equivalent
computation and is NEVER presented as native KAI behaviour. The native Allocate results
(Bind/Cache/Task state) come only from the Go record.
"""
import argparse
import hashlib
from pathlib import Path
import sys

from . import oracle, policy_benefit as pb, schema
from .schema import InputError, digest, encoded

VERSION = "M4B-AUDIT-1"
NODES = ("node0", "node1")
GIB = 1024 ** 3
FAMILY = "CPU_ONLY_TEST"


def _need(cond, message):
    if not cond:
        raise InputError(message)


def residual_from_native(resources):
    """Native IDLE (GPU, CPU milli, bytes) -> model units (whole GPUs, cores, GiB), exact integers."""
    out = []
    for name in NODES:
        gpu, milli, mem = resources[name]["idle"]
        _need(gpu == 0 and milli % 1000 == 0 and mem % GIB == 0, "non-integral or GPU native unit")
        out.append([0, milli // 1000, mem // GIB])
    return out


def model_trace(tr):
    """The common trace expressed in the M4-A trace schema (one individual arrival per stage)."""
    ids = list(NODES)
    nodes = [dict(id=i, gpu_family=FAMILY, shape="S", capacity=[0, int(c), 8]) for i, c in zip(ids, tr["node_cores"])]
    task = lambda i, c: dict(id="native-job-%d" % i, demand=[0, c, 1], value=c, gpu_family=None, eligible=ids[:])
    arrivals = [[task(i, c)] for i, c in enumerate(tr["arrival_cores"])]
    forecast = [dict(id=f["id"], demand=list(f["demand"]), weight=f["weight"], gpu_family=None, eligible=list(f["eligible"]))
                for f in tr["forecast"]]
    base = len(tr["arrival_cores"])
    realized = [dict(task(base + k, c), id="R%d" % k) for k, c in enumerate(tr["held_out_future_cores"])]
    return dict(id=tr["name"], regime="native_stage_a", nodes=nodes, arrivals=arrivals, forecast=forecast, realized=realized)


def audit_arm(record):
    tr = record["trace"]
    trace = model_trace(tr)
    errors, residual = [], [n["capacity"][:] for n in trace["nodes"]]
    placement = []
    for step, batch in zip(record["steps"], trace["arrivals"]):
        task = batch[0]
        if residual_from_native(step["before_resources"]) != residual:
            errors.append("native before-state differs from model replay at " + task["id"])
        node = step["chosen_node"]
        _need(node in NODES and step["task"] == task["id"], "unexpected native decision identity")
        i = NODES.index(node)
        if not (oracle.allowed(trace["nodes"][i], task) and oracle.fits(residual[i], task["demand"])):
            errors.append("model-infeasible native acceptance at " + task["id"])
        residual[i] = [r - d for r, d in zip(residual[i], task["demand"])]
        placement.append(i)
        if residual_from_native(step["after_resources"]) != residual:
            errors.append("native after-state differs from model replay at " + task["id"])
    snapshot = pb._full(trace)
    physical = oracle.audit(snapshot, placement, residual)
    if physical["status"] != "PHYSICAL_VALID":
        errors += physical["errors"]
    # Secondary, model-equivalent (NOT native) evaluation of the same frozen post-current residual.
    evaluation = oracle.evaluate(pb._full(trace, realized=True), residual)
    # Native future decisions replayed against the model for feasibility agreement only.
    future_residual, agreement = [r[:] for r in residual], True
    for job, row in zip(trace["realized"], record["future"]):
        fits = [i for i, n in enumerate(trace["nodes"]) if oracle.allowed(n, job) and oracle.fits(future_residual[i], job["demand"])]
        if row["outcome"] == "ACCEPTED":
            i = NODES.index(row["node"])
            agreement &= i in fits
            if i in fits:
                future_residual[i] = [r - d for r, d in zip(future_residual[i], job["demand"])]
        else:
            agreement &= not fits and row["reason"] == "NO_FEASIBLE_NODE"
    if residual_from_native(record["final_resources"]) != future_residual:
        errors.append("native final state differs from model future replay")
    forecast_snapshot = pb._stage_snapshot(trace, residual, trace["arrivals"][0])
    return dict(status="AUDIT_PASS" if not errors and agreement else "AUDIT_FAIL", errors=errors,
                native_future_feasibility_agrees_with_model=agreement, placement=placement,
                post_current_residual=residual, post_future_residual=future_residual,
                predicted_surrogate_F_explanatory_only=oracle.future_score(forecast_snapshot, residual),
                fragmentation=pb._metrics(trace, residual),
                offline_oracle_secondary=dict(label="OFFLINE_MODEL_EQUIVALENT_NOT_NATIVE_KAI",
                    sequential_first_fit=evaluation["sequential"], offline_fixed_current=evaluation["offline_fixed_current"]))


def offline_references(tr):
    trace = model_trace(tr)
    out = {}
    for name in (pb.PRIMARY_REFERENCE, pb.REFERENCE + ":first_fit", pb.DYNAMIC):
        arm = pb.run_arm(trace, name)
        out[name] = dict(label="OFFLINE_NOT_NATIVE_KAI", status=arm["status"], placement=arm["placement"],
                         sequential_admitted=arm["outcome"]["sequential"]["count"],
                         offline_fixed_current_admitted=arm["outcome"]["offline_fixed_current"]["count"])
    return out


def audit_record(normalized):
    _need(normalized.get("version") == "M4B-A1", "unknown native record version")
    out = {}
    for name, arms in normalized["traces"].items():
        audits = {arm: audit_arm(rec) for arm, rec in arms.items()}
        off = arms["native_off"]
        row = dict(audits=audits, offline_references=offline_references(off["trace"]), arms={})
        for arm, rec in arms.items():
            placement = [s["chosen_node"] for s in rec["steps"]]
            flips = [i for i, (a, b) in enumerate(zip(placement, [s["chosen_node"] for s in off["steps"]])) if a != b]
            row["arms"][arm] = dict(
                placement=placement, flip_indices_vs_native_off=flips,
                native_future_admitted=rec["future_admitted"], native_future_total=rec["future_total"],
                native_future_admitted_cores=rec["future_admitted_cores"],
                delta_admitted_vs_native_off=rec["future_admitted"] - off["future_admitted"],
                delta_admitted_cores_vs_native_off=rec["future_admitted_cores"] - off["future_admitted_cores"],
                ranking_flip_observed=bool(flips) or any(s["native_order_S"] != s["native_order_S0"] for s in rec["steps"]),
                decision_status="BASELINE" if arm == "native_off" else ("FLIP" if flips else "NO_FLIP"))
        out[name] = row
    return dict(version=VERSION, scope="Stage A native Fake Session CPU; offline-oracle fields are secondary and not native",
                traces=out, simulated_gpu="SIMULATED_GPU_NOT_RUN")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("normalized", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    raws = [p.read_bytes() for p in args.normalized]
    identical = all(r == raws[0] for r in raws)
    report = audit_record(schema.loads(raws[0]))
    report["runs"] = len(raws)
    report["normalized_sha256"] = [hashlib.sha256(r).hexdigest() for r in raws]
    report["byte_identical_across_runs"] = identical
    report["audit_sha256"] = digest(report)
    args.out.write_bytes(encoded(report))
    ok = identical and all(a["status"] == "AUDIT_PASS" for t in report["traces"].values() for a in t["audits"].values())
    sys.stdout.write(report["audit_sha256"] + ("\n" if ok else " FAIL\n"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
