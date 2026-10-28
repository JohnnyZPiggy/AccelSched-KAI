"""M5 independent audit of the Stage A/B native whole-GPU records (Fake Session; PROBE_ONLY).

Re-derives the model-equivalent state with the unchanged Oracle and the unchanged M4-A
simulator. Fields labelled OFFLINE_* are secondary computations and are NEVER native KAI
behaviour; native Allocate/Bind/Cache/Task facts come only from the Go records. Simulated
Fake-Session GPUs are not physical GPUs.
"""
import argparse
import hashlib
from pathlib import Path
import sys

from . import m5_producer, oracle, policy_benefit as pb, schema
from .schema import InputError, digest, encoded

VERSION = "M5-AUDIT-1"
NODES = ("node0", "node1")
GIB = 1024 ** 3
MEM_GIB = 64


def _need(cond, message):
    if not cond:
        raise InputError(message)


def residual_from_native(resources):
    """Native IDLE [GPU, CPU milli, RAM bytes] -> exact [whole GPU, cores, GiB]; remainders are refused."""
    out = []
    for name in NODES:
        gpu, milli, mem = resources[name]["idle"]
        _need(type(gpu) is int and gpu >= 0 and milli % 1000 == 0 and mem % GIB == 0, "non-integral native unit")
        out.append([gpu, milli // 1000, mem // GIB])
    return out


def model_trace(tr):
    ids = list(NODES)
    nodes = [dict(id=i, gpu_family=m5_producer.FAMILY, shape="G%dC%dM%d" % (n["gpus"], int(n["cores"]), MEM_GIB),
                  capacity=[n["gpus"], int(n["cores"]), MEM_GIB]) for i, n in zip(ids, tr["nodes"])]
    task = lambda i, j: dict(id="native-job-%d" % i, demand=[j["gpus"], j["cores"], 1], value=j["gpus"], gpu_family=None, eligible=ids[:])
    arrivals = [[task(i, j)] for i, j in enumerate(tr["arrivals"])]
    forecast = [dict(id=f["id"], demand=list(f["demand"]), weight=f["weight"], gpu_family=None, eligible=list(f["eligible"])) for f in tr["forecast"]]
    base = len(tr["arrivals"])
    realized = [dict(task(base + k, j), id="R%d" % k) for k, j in enumerate(tr["held_out_future"])]
    return dict(id=tr["name"], regime="native_whole_gpu", nodes=nodes, arrivals=arrivals, forecast=forecast, realized=realized)


def audit_arm(record):
    tr = record["trace"]
    trace = model_trace(tr)
    errors, residual, placement = [], [n["capacity"][:] for n in trace["nodes"]], []
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
    physical = oracle.audit(pb._full(trace), placement, residual)
    if physical["status"] != "PHYSICAL_VALID":
        errors += physical["errors"]
    evaluation = oracle.evaluate(pb._full(trace, realized=True), residual)
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


def audit_stage_a(a):
    """Structural re-check of the Stage A record: every unsupported case refused, equivalence tables agree."""
    _need(a.get("version") == m5_producer.P2, "unknown Stage A version")
    names = a["mapping_proof"]["index_names"]
    _need(names == ["cpu", "memory", "gpu", "pods"], "native index map differs from P2")
    refused = {f["case"]: f["gate"] for f in a["fail_closed"]}
    _need(refused and all(g in ("go_projection", "python_producer", "construction_panic") for g in refused.values()), "an unsupported case was not refused")
    for row in a["eligibility_table"]:
        for node in NODES:
            _need(row[node]["native"] == row[node]["model"], "native/model candidate disagreement")
    trust = a["trust"]
    _need(all(trust[k] is True for k in ("spent_grant_rejected", "stale_projection_rejected", "mutated_hooks_rejected", "forged_uid_rejected",
                                       "foreign_session_rejected", "resource_mutation_rejected", "incomplete_duplicate_mapping_rejected",
                                       "fresh_grant_after_faults_activates", "neutral_scores_equal_native")), "trust gate missing")
    return dict(status="STAGE_A_PASS", cases_refused=len(refused), gates=sorted(set(refused.values())), eligibility_rows=len(a["eligibility_table"]))


def audit_record(normalized, stage_a):
    _need(normalized.get("version") == "M5-B1", "unknown native record version")
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
                native_future_admitted_gpus=rec["future_admitted_gpus"],
                delta_admitted_vs_native_off=rec["future_admitted"] - off["future_admitted"],
                delta_admitted_gpus_vs_native_off=rec["future_admitted_gpus"] - off["future_admitted_gpus"],
                decision_status="BASELINE" if arm == "native_off" else ("FLIP" if flips else "NO_FLIP"))
        out[name] = row
    return dict(version=VERSION, scope="whole-GPU native Fake Session; offline-oracle fields are secondary and not native; GPUs are simulated",
                stage_a=audit_stage_a(stage_a), traces=out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    norms = [(d / "normalized.json").read_bytes() for d in args.run_dirs]
    stages = [(d / "stage_a.json").read_bytes() for d in args.run_dirs]
    identical = all(n == norms[0] for n in norms) and all(s == stages[0] for s in stages)
    report = audit_record(schema.loads(norms[0]), schema.loads(stages[0]))
    report.update(runs=len(norms), normalized_sha256=[hashlib.sha256(n).hexdigest() for n in norms],
                  stage_a_sha256=[hashlib.sha256(s).hexdigest() for s in stages], byte_identical_across_runs=identical)
    report["audit_sha256"] = digest(report)
    args.out.write_bytes(encoded(report))
    ok = identical and all(a["status"] == "AUDIT_PASS" for t in report["traces"].values() for a in t["audits"].values())
    sys.stdout.write(report["audit_sha256"] + ("\n" if ok else " FAIL\n"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
