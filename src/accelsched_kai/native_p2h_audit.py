"""M6 independent audit and Stage C analysis of the P2H native whole-GPU records (Fake Session; PROBE_ONLY).

Re-derives each arm's model-equivalent state with the unchanged Oracle and unchanged M4-A simulator.
Fields labelled OFFLINE_* are secondary and are NEVER native KAI behaviour; native Allocate/Bind/Cache
facts come only from the Go records. Distinct workload seeds are independent scene observations; fresh
process repetitions only check determinism. Simulated Fake-Session GPUs are not physical GPUs.
"""
import argparse
import hashlib
from pathlib import Path
import statistics
import sys

from . import bounded_advice as ba, oracle, p2h_producer, policy_benefit as pb, schema
from .schema import InputError, digest, encoded

VERSION = "M6-AUDIT-1"
GIB = 1024 ** 3
SIZE_CLASSES = (1, 2, 4, 6, 8)  # whole-GPU sizes probed for the fragmentation size-fit counts (2 cores, 1 GiB)


def _need(cond, message):
    if not cond:
        raise InputError(message)


def residual_from_native(resources, names):
    out = []
    for name in names:
        gpu, milli, mem = resources[name]["idle"]
        _need(type(gpu) is int and gpu >= 0 and milli % 1000 == 0 and mem % GIB == 0, "non-integral native unit")
        out.append([gpu, milli // 1000, mem // GIB])
    return out


def model_trace(scene):
    names = ["node%d" % i for i in range(len(scene["nodes"]))]
    nodes = [dict(id=n, gpu_family=p2h_producer.FAMILY, shape="G%dC%dM%d" % (v["gpus"], int(v["cores"]), int(v["mem_gib"])),
                  capacity=[v["gpus"], int(v["cores"]), int(v["mem_gib"])]) for n, v in zip(names, scene["nodes"])]
    task = lambda i, j: dict(id="native-job-%d" % i, demand=[j["gpus"], j["cores"], j["gib"]], value=j["gpus"], gpu_family=None, eligible=names[:])
    arrivals = [[task(i, j)] for i, j in enumerate(scene["arrivals"])]
    forecast = [dict(id=f["id"], demand=list(f["demand"]), weight=f["weight"], gpu_family=None, eligible=list(f["eligible"])) for f in scene["forecast"]]
    base = len(scene["arrivals"])
    realized = [dict(task(base + k, j), id="R%d" % k) for k, j in enumerate(scene["held_out_future"])]
    return dict(id=scene["id"], regime=scene["regime"], nodes=nodes, arrivals=arrivals, forecast=forecast, realized=realized), names


def fragmentation(residual):
    fits = {str(g): sum(r[0] >= g and r[1] >= 2 and r[2] >= 1 for r in residual) for g in SIZE_CLASSES}
    stranded = sum(r[0] for r in residual if r[1] < 2 or r[2] < 1)  # GPUs on nodes that cannot host even a 1-GPU/2-core job
    return dict(per_node_idle=[r[:] for r in residual], total_free_gpus=sum(r[0] for r in residual), largest_free_gpu_block=max(r[0] for r in residual),
                size_fit_counts=fits, stranded_free_gpus=stranded, units="per_node_idle: whole GPUs, integer cores, GiB; size fit: nodes that can host a job of that GPU size with 2 cores and 1 GiB")


def audit_arm(record):
    scene = record["scene"]
    trace, names = model_trace(scene)
    errors, residual, placement = [], [n["capacity"][:] for n in trace["nodes"]], []
    for step, batch in zip(record["steps"], trace["arrivals"]):
        task = batch[0]
        if residual_from_native(step["before_resources"], names) != residual:
            errors.append("native before-state differs from model replay at " + task["id"])
        node = step["chosen_node"]
        _need(node in names and step["task"] == task["id"], "unexpected native decision identity")
        i = names.index(node)
        model_candidates = [n for n, nd in zip(names, trace["nodes"]) if oracle.allowed(nd, task) and oracle.fits(residual[names.index(n)], task["demand"])]
        if model_candidates != step["candidates"]:
            errors.append("native candidate set differs from the model candidate set at " + task["id"])
        if node not in model_candidates:
            errors.append("model-infeasible native acceptance at " + task["id"])
        residual[i] = [r - d for r, d in zip(residual[i], task["demand"])]
        placement.append(i)
        if residual_from_native(step["after_resources"], names) != residual:
            errors.append("native after-state differs from model replay at " + task["id"])
    physical = oracle.audit(pb._full(trace), placement, residual)
    if physical["status"] != "PHYSICAL_VALID":
        errors += physical["errors"]
    evaluation = oracle.evaluate(pb._full(trace, realized=True), residual)
    future_residual, agreement = [r[:] for r in residual], True
    for job, row in zip(trace["realized"], record["future"]):
        fits = [n for n, nd in zip(names, trace["nodes"]) if oracle.allowed(nd, job) and oracle.fits(future_residual[names.index(n)], job["demand"])]
        agreement &= fits == row["candidates"]
        if row["outcome"] == "ACCEPTED":
            i = names.index(row["node"])
            agreement &= row["node"] in fits
            if row["node"] in fits:
                future_residual[i] = [r - d for r, d in zip(future_residual[i], job["demand"])]
        else:
            agreement &= not fits and row["reason"] == "NO_FEASIBLE_NODE"
    # A withdrawn rejected job holds no resources, so the final state equals the model replay.
    if residual_from_native(record["final_resources"], names) != future_residual:
        errors.append("native final state differs from model future replay")
    return dict(status="AUDIT_PASS" if not errors and agreement else "AUDIT_FAIL", errors=errors, native_future_feasibility_agrees_with_model=agreement,
                placement=placement, post_current_residual=residual, post_future_residual=future_residual,
                predicted_surrogate_F_explanatory_only=oracle.future_score(pb._stage_snapshot(trace, residual, trace["arrivals"][0]), residual),
                fragmentation=fragmentation(residual),
                offline_oracle_secondary=dict(label="OFFLINE_MODEL_EQUIVALENT_NOT_NATIVE_KAI",
                    sequential_first_fit=evaluation["sequential"], offline_fixed_current=evaluation["offline_fixed_current"]))


def offline_dynamic(scene):
    trace, _ = model_trace(scene)
    out = {}
    for name in (pb.PRIMARY_REFERENCE, pb.DYNAMIC):
        arm = pb.run_arm(trace, name)
        out[name] = dict(label="OFFLINE_NOT_NATIVE_KAI", status=arm["status"], placement=arm["placement"], sequential_admitted=arm["outcome"]["sequential"]["count"])
    return out


def lp_prices(scenes):
    """Inspect every certified per-arrival Shape-LP price; report whether any is nonzero and whether it changed a ranking."""
    total = nonzero = changed_ranking = changed_choice = price_term_pairs = 0
    examples = []
    for sc in scenes:
        off = sc["arms"]["native_off"]
        for arm in ("native_dynamic_pgfp_lp",):
            lp, pg = sc["arms"][arm], sc["arms"]["native_dynamic_pgfp"]
            for sl, sp in zip(lp["steps"], pg["steps"]):
                prices = sl["artifact_pairs"]["prices"]
                total += 1
                if any(any(v != 0 for v in row) for row in prices):
                    nonzero += 1
                    if len(examples) < 5:
                        examples.append(dict(scene=sc["id"], prices=prices))
                price_term_pairs += sum(1 for pair in sl["artifact_pairs"].get("pairs", []) if pair["p"] != 0 or pair["AP"] != 0)
                if sl["native_order_S"] != sp["native_order_S"]:
                    changed_ranking += 1
                if sl["chosen_node"] != sp["chosen_node"]:
                    changed_choice += 1
    return dict(certified_artifacts=total, nonzero_price_artifacts=nonzero, nonzero_price_term_pairs=price_term_pairs, ranking_differs_from_pgfp=changed_ranking, chosen_node_differs_from_pgfp=changed_choice, nonzero_examples=examples)


def lp_ranking_effect(scenes):
    """Does the (1-alpha) blend scaling (alone) change the native ranking/choice/admission versus alpha=0?"""
    differs = same = 0
    for sc in scenes:
        a, b = sc["arms"]["native_dynamic_pgfp"], sc["arms"]["native_dynamic_pgfp_lp"]
        if [s["chosen_node"] for s in a["steps"]] != [s["chosen_node"] for s in b["steps"]] or a["future_admitted"] != b["future_admitted"]:
            differs += 1
        else:
            same += 1
    return dict(scenes_where_lp_arm_differs_from_pgfp=differs, scenes_identical=same)


def offline_multi_pending_lp_probe():
    """OFFLINE (not native): certified Shape-LP prices CAN be nonzero when several tasks compete for a scarce Shape.

    The per-arrival bridge scores ONE Pending target, whose single-task LP is never capacity-scarce while a node fits it;
    this probe shows the unchanged M3-A machinery on a 3-task snapshot, only to explain why LP prices are zero natively.
    """
    ids = ["node0", "node1"]
    nodes = [dict(id="node0", gpu_family=p2h_producer.FAMILY, shape="SCARCE", capacity=[2, 8, 16]), dict(id="node1", gpu_family=p2h_producer.FAMILY, shape="PLENTY", capacity=[8, 32, 64])]
    task = lambda i, g, v: dict(id="T%d" % i, demand=[g, 2, 2], value=v, eligible=ids[:], gpu_family=p2h_producer.FAMILY)
    snap = dict(schema_version="0.1", units=list(schema.UNITS), nodes=nodes, tasks=[task(0, 2, 9), task(1, 2, 9), task(2, 8, 12)],
                future=[dict(id="F0", demand=[2, 2, 2], weight=1, eligible=ids[:], gpu_family=p2h_producer.FAMILY)], realized_future=[], constraints={})
    out = {}
    for alpha in (0., .5):
        result = ba.build_artifact(snap, BF=4., BP=8., alpha=alpha)
        _need(result["status"] == ba.STATUS, "offline LP probe must certify")
        a = result["artifact"]
        out["alpha_%s" % alpha] = dict(prices=a["lp"]["prices"], p=[v["p"] for v in a["pairs"]], scores=[v["score"] for v in a["pairs"]],
                                       certificate_certified=a["lp"]["certificate"]["certified"])
    return dict(label="OFFLINE_MULTI_PENDING_LP_PROBE_NOT_NATIVE", **out,
                nonzero_price=any(v != 0 for row in out["alpha_0.5"]["prices"] for v in row))


def paired(scenes, arm):
    rows = []
    for sc in scenes:
        a, o = sc["arms"][arm], sc["arms"]["native_off"]
        rows.append(dict(scene=sc["id"], arm_admitted=a["future_admitted"], off_admitted=o["future_admitted"], denominator=a["future_total"],
                         arm_gpus=a["future_admitted_gpus"], off_gpus=o["future_admitted_gpus"], arm_cores=a["future_admitted_cores"], off_cores=o["future_admitted_cores"],
                         delta=a["future_admitted"] - o["future_admitted"], delta_gpus=a["future_admitted_gpus"] - o["future_admitted_gpus"]))
    return rows


def tally(rows):
    num = sum(r["delta"] for r in rows)
    den = sum(r["denominator"] for r in rows)
    deltas = [r["delta"] for r in rows]
    return dict(scenes=len(rows), improved=sum(d > 0 for d in deltas), unchanged=sum(d == 0 for d in deltas), harmed=sum(d < 0 for d in deltas),
                numerator_arm=sum(r["arm_admitted"] for r in rows), numerator_off=sum(r["off_admitted"] for r in rows), denominator=den,
                delta_admission_rate=num / den if den else None, delta_gpus=sum(r["delta_gpus"] for r in rows),
                per_scene_delta_mean=statistics.mean(deltas) if deltas else None, per_scene_delta_stdev=statistics.pstdev(deltas) if deltas else None)


def stage_c(normalized):
    scenes = normalized["scenes"]
    by_regime = {}
    for sc in scenes:
        by_regime.setdefault(sc["id"].split(":")[0], []).append(sc)
    out = dict(regimes={}, adverse_examples=[], beneficial_examples=[])
    for regime, group in by_regime.items():
        out["regimes"][regime] = {arm: tally(paired(group, arm)) for arm in ("native_dynamic_pgfp", "native_dynamic_pgfp_lp")}
    for arm in ("native_dynamic_pgfp",):
        for r in paired(scenes, arm):
            if r["delta"] < 0:
                out["adverse_examples"].append(dict(r, arm=arm))
            if r["delta"] > 0:
                out["beneficial_examples"].append(dict(r, arm=arm))
    out["all_scenes"] = {arm: tally(paired(scenes, arm)) for arm in ("native_dynamic_pgfp", "native_dynamic_pgfp_lp")}
    out["search_scope"] = dict(regimes=sorted(by_regime), seeds_per_regime=normalized["seeds_per_regime"], scenes=len(scenes),
                               statement="FOUND/NOT_FOUND hold only inside this enumerated regime x seed matrix; no population claim")
    out["lp"] = lp_prices(scenes)
    out["lp_ranking_effect"] = lp_ranking_effect(scenes)
    price_free = out["lp"]["nonzero_price_artifacts"] == 0 and out["lp"]["nonzero_price_term_pairs"] == 0
    out["lp_verdict"] = "LP_NO_INCREMENTAL_BENEFIT" if price_free else "LP_PRICE_TERM_OBSERVED"
    out["lp_note"] = ("Every certified Shape-LP price is zero, so the Shape-LP term A_P is identically zero; any difference between the LP arm and "
                      "PGFP comes only from the (1-alpha) scaling of the marginal term, not from a Shape-LP price contribution") if price_free else "nonzero Shape-LP price terms occurred; inspect nonzero_examples"
    out["offline_multi_pending_lp_probe"] = offline_multi_pending_lp_probe()
    return out


def audit_stage_a(a):
    _need(a.get("version") == p2h_producer.P2H, "unknown Stage A version")
    refused = {f["case"]: f["gate"] for f in a["fail_closed"]}
    _need(refused and all(g in ("go_projection", "python_producer", "construction_panic") for g in refused.values()), "an unsupported case was not refused")
    for row in a["eligibility_table"]:
        for node, v in row.items():
            if node.startswith("node"):
                _need(v["native"] == v["model"], "native/model candidate disagreement")
    sizes = sorted({p["nodes"] for p in a["positive"]})
    _need(sizes == [2, 3, 4], "positive coverage must include 2, 3 and 4 nodes")
    t = a["trust"]
    _need(all(t[k] is True for k in ("spent_grant_rejected", "stale_projection_rejected", "mutated_hooks_rejected", "forged_uid_rejected", "foreign_session_rejected",
                                    "resource_mutation_rejected", "node_identity_change_rejected", "pod_slot_pressure_rejected", "fresh_grant_after_faults_activates",
                                    "neutral_scores_equal_native", "frozen_p1_p2_receivers_and_p2h_reject_each_other")), "trust gate missing")
    _need(len(t["receiver_mutations_rejected"]) >= 20, "receiver mutation coverage")
    _need(a["state_evolution"]["max_phase_used"] >= 5, "state evolution must reach phase 5")
    return dict(status="STAGE_A_PASS", cases_refused=len(refused), gates=sorted(set(refused.values())), eligibility_rows=len(a["eligibility_table"]),
                receiver_mutations=len(t["receiver_mutations_rejected"]), nodes=sizes)


def audit_record(normalized, stage_a_record):
    _need(normalized.get("version") == "M6-B1", "unknown native record version")
    scenes = normalized["scenes"]
    audits = {}
    offline = {}
    for sc in scenes:
        audits[sc["id"]] = {arm: audit_arm(rec) for arm, rec in sc["arms"].items()}
        offline[sc["id"]] = offline_dynamic(sc["arms"]["native_off"]["scene"])
    return dict(version=VERSION, scope="whole-GPU multi-node native Fake Session; offline-oracle fields are secondary and not native; GPUs are simulated",
                stage_a=audit_stage_a(stage_a_record), audits=audits, offline_references=offline, stage_c=stage_c(normalized), same_batch=normalized["same_batch"])


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
    ok = identical and all(a["status"] == "AUDIT_PASS" for arms in report["audits"].values() for a in arms.values())
    sys.stdout.write(report["audit_sha256"] + ("\n" if ok else " FAIL\n"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
