"""M6 independent audit and Stage C analysis on synthetic native-shaped records (no Go, no Kubernetes)."""
from copy import deepcopy
import json

import pytest

from accelsched_kai import native_p2h_audit as npa, policy_benefit as pb, schema

GIB = 1024 ** 3
NODES = [dict(gpus=4, cores=16, mem_gib=64), dict(gpus=8, cores=32, mem_gib=128), dict(gpus=2, cores=8, mem_gib=32), dict(gpus=8, cores=16, mem_gib=64)]
NAMES = ["node0", "node1", "node2", "node3"]
SCENE = dict(id="matched:0", regime="matched", seed=0, nodes=NODES, arrivals=[dict(gpus=2, cores=4, gib=2), dict(gpus=1, cores=2, gib=1), dict(gpus=2, cores=4, gib=2), dict(gpus=1, cores=8, gib=8)],
             forecast=[dict(id="F0", demand=[4, 8, 4], weight=2, eligible=NAMES)],
             held_out_future=[dict(gpus=6, cores=8, gib=4), dict(gpus=4, cores=8, gib=4), dict(gpus=1, cores=2, gib=1)])


def resources(residual):
    out = {}
    for name, left, n in zip(NAMES, residual, NODES):
        alloc = [n["gpus"], int(n["cores"]) * 1000, int(n["mem_gib"]) * GIB]
        idle = [left[0], left[1] * 1000, left[2] * GIB]
        out[name] = dict(allocatable=alloc, idle=idle, used=[a - i for a, i in zip(alloc, idle)], releasing=[0, 0, 0], members=[])
    return out


def arm_record(arm, choose, scene=SCENE, order=None):
    trace, names = npa.model_trace(scene)
    residual = [n["capacity"][:] for n in trace["nodes"]]
    steps = []
    for batch, pick in zip(trace["arrivals"], choose):
        task, before = batch[0], [r[:] for r in residual]
        cands = [n for n, r in zip(names, residual) if all(a >= b for a, b in zip(r, task["demand"]))]
        node = names.index(pick)
        residual[node] = [r - d for r, d in zip(residual[node], task["demand"])]
        steps.append(dict(task=task["id"], chosen_node=pick, candidates=cands, before_resources=resources(before), after_resources=resources(residual),
                          native_order_S=order or cands, artifact_pairs=dict(prices=[[0, 0, 0]])))
    future = []
    for job in trace["realized"]:
        cands = [n for n, r in zip(names, residual) if all(a >= b for a, b in zip(r, job["demand"]))]
        if cands:
            node = names.index(cands[0])
            residual[node] = [r - d for r, d in zip(residual[node], job["demand"])]
            future.append(dict(candidates=cands, outcome="ACCEPTED", node=cands[0]))
        else:
            future.append(dict(candidates=[], outcome="REJECTED", reason="NO_FEASIBLE_NODE"))
    acc = [j for j, f in zip(trace["realized"], future) if f["outcome"] == "ACCEPTED"]
    return dict(arm=arm, scene=scene, steps=steps, future=future, future_admitted=len(acc), future_total=len(future), future_admitted_gpus=sum(j["demand"][0] for j in acc),
                future_admitted_cores=sum(j["demand"][1] for j in acc), final_resources=resources(residual))


def scene_record(off_choice=("node0", "node0", "node1", "node1"), pg_choice=("node2", "node1", "node3", "node1")):
    return {"id": SCENE["id"], "arms": {"native_off": arm_record("native_off", off_choice), "native_dynamic_pgfp": arm_record("native_dynamic_pgfp", pg_choice),
                                         "native_dynamic_pgfp_lp": arm_record("native_dynamic_pgfp_lp", pg_choice)}}


def test_faithful_record_passes_with_candidate_sets_and_secondary_labels():
    audit = npa.audit_arm(scene_record()["arms"]["native_off"])
    assert audit["status"] == "AUDIT_PASS" and audit["errors"] == []
    assert audit["offline_oracle_secondary"]["label"] == "OFFLINE_MODEL_EQUIVALENT_NOT_NATIVE_KAI"
    f = audit["fragmentation"]
    assert f["total_free_gpus"] == sum(r[0] for r in audit["post_current_residual"]) and set(f["size_fit_counts"]) == {"1", "2", "4", "6", "8"}


def test_units_exact_and_remainders_rejected():
    assert npa.residual_from_native(resources([[3, 4, 5], [6, 16, 64], [1, 1, 1], [0, 0, 0]]), NAMES)[0] == [3, 4, 5]
    for mutate in (lambda r: r["node0"]["idle"].__setitem__(1, 1500), lambda r: r["node1"]["idle"].__setitem__(2, GIB + 1), lambda r: r["node2"]["idle"].__setitem__(0, 1.5)):
        bad = resources([[3, 4, 5], [6, 16, 64], [1, 1, 1], [0, 0, 0]])
        mutate(bad)
        with pytest.raises(schema.InputError):
            npa.residual_from_native(bad, NAMES)


@pytest.mark.parametrize("tamper", [
    lambda r: r["steps"][1]["after_resources"]["node1"]["idle"].__setitem__(0, 0),
    lambda r: r["steps"][0].update(chosen_node="node1"),
    lambda r: r["steps"][2]["before_resources"]["node3"]["idle"].__setitem__(0, 1),
    lambda r: r["steps"][0]["candidates"].pop(),
    lambda r: r["steps"][0]["candidates"].reverse(),
    lambda r: r["final_resources"]["node1"]["idle"].__setitem__(0, 0),
    lambda r: r["future"][0].update(outcome="ACCEPTED", node="node1", candidates=["node1"]),
    lambda r: r["future"][2].update(node="node3" if r["future"][2]["node"] != "node3" else "node2"),
])
def test_tampered_native_records_fail_the_independent_audit(tamper):
    bad = scene_record()["arms"]["native_off"]
    tamper(bad)
    try:
        assert npa.audit_arm(bad)["status"] == "AUDIT_FAIL"
    except schema.InputError:
        pass


def test_model_infeasible_native_acceptance_is_detected():
    rec = scene_record()["arms"]["native_off"]
    rec["scene"] = deepcopy(SCENE)
    rec["scene"]["arrivals"][0] = dict(gpus=9, cores=4, gib=2)
    assert npa.audit_arm(rec)["status"] == "AUDIT_FAIL"


def test_paired_tally_exact_numerators_and_harm_help_neutral():
    sc = scene_record()
    sc["arms"]["native_dynamic_pgfp"]["future_admitted"] += 1
    sc["arms"]["native_dynamic_pgfp"]["future_admitted_gpus"] += 4
    harmed = deepcopy(sc)
    harmed["id"] = "matched:1"
    harmed["arms"]["native_dynamic_pgfp"]["future_admitted"] = harmed["arms"]["native_off"]["future_admitted"] - 1
    rows = npa.paired([sc, harmed], "native_dynamic_pgfp")
    t = npa.tally(rows)
    assert (t["improved"], t["harmed"], t["unchanged"]) == (1, 1, 0) and t["denominator"] == 6 and t["delta_admission_rate"] == 0
    assert t["numerator_arm"] - t["numerator_off"] == sum(r["delta"] for r in rows)


def test_lp_prices_zero_means_no_incremental_benefit_and_nonzero_is_detected():
    normalized = dict(version="M6-B1", seeds_per_regime=1, scenes=[scene_record()])
    c = npa.stage_c(normalized)
    assert c["lp_verdict"] == "LP_NO_INCREMENTAL_BENEFIT" and c["lp"]["nonzero_price_artifacts"] == 0 and c["lp"]["certified_artifacts"] == 4
    assert "identically zero" in c["lp_note"]
    nz = deepcopy(normalized)
    nz["scenes"][0]["arms"]["native_dynamic_pgfp_lp"]["steps"][0]["artifact_pairs"]["prices"] = [[0.5, 0, 0]]
    c2 = npa.stage_c(nz)
    assert c2["lp"]["nonzero_price_artifacts"] == 1 and c2["lp_verdict"] == "LP_PRICE_TERM_OBSERVED"
    nz2 = deepcopy(normalized)
    nz2["scenes"][0]["arms"]["native_dynamic_pgfp_lp"]["steps"][0]["artifact_pairs"]["pairs"] = [dict(p=-3.0, AP=-0.375)]
    assert npa.stage_c(nz2)["lp"]["nonzero_price_term_pairs"] == 1 and npa.stage_c(nz2)["lp_verdict"] == "LP_PRICE_TERM_OBSERVED"
    assert c["search_scope"]["scenes"] == 1 and "no population claim" in c["search_scope"]["statement"]


def test_offline_multi_pending_lp_probe_has_certified_nonzero_prices_and_is_labelled_not_native():
    probe = npa.offline_multi_pending_lp_probe()
    assert probe["label"] == "OFFLINE_MULTI_PENDING_LP_PROBE_NOT_NATIVE" and probe["nonzero_price"] is True
    assert probe["alpha_0.0"]["certificate_certified"] and probe["alpha_0.5"]["certificate_certified"]
    assert probe["alpha_0.0"]["scores"] != probe["alpha_0.5"]["scores"]


def test_offline_dynamic_reuses_m4a_and_is_not_native():
    out = npa.offline_dynamic(SCENE)
    assert set(out) == {pb.PRIMARY_REFERENCE, pb.DYNAMIC} and all(v["label"] == "OFFLINE_NOT_NATIVE_KAI" and v["status"] == "PHYSICAL_VALID" for v in out.values())


def stage_a():
    ok = {k: True for k in ("spent_grant_rejected", "stale_projection_rejected", "mutated_hooks_rejected", "forged_uid_rejected", "foreign_session_rejected",
                            "resource_mutation_rejected", "node_identity_change_rejected", "pod_slot_pressure_rejected", "fresh_grant_after_faults_activates",
                            "neutral_scores_equal_native", "frozen_p1_p2_receivers_and_p2h_reject_each_other")}
    ok["receiver_mutations_rejected"] = ["m%d" % i for i in range(24)]
    return dict(version="P2H-1", positive=[dict(nodes=n) for n in (2, 3, 4)], state_evolution=dict(max_phase_used=5),
                fail_closed=[dict(case="x", gate="go_projection", reason="y")],
                eligibility_table=[dict(request=dict(gpus=1), candidates=["node0"], node0=dict(native=True, model=True), node1=dict(native=False, model=False))], trust=ok)


def test_stage_a_structural_audit_and_failures():
    assert npa.audit_stage_a(stage_a())["status"] == "STAGE_A_PASS"
    for mutate in (lambda a: a["fail_closed"][0].update(gate="ACCEPTED"), lambda a: a["eligibility_table"][0]["node1"].update(model=True),
                   lambda a: a["trust"].update(spent_grant_rejected=False), lambda a: a.update(version="P2-WholeGPU-1"), lambda a: a.update(fail_closed=[]),
                   lambda a: a["positive"].pop(), lambda a: a["state_evolution"].update(max_phase_used=3), lambda a: a["trust"].update(receiver_mutations_rejected=["a"])):
        bad = stage_a()
        mutate(bad)
        with pytest.raises((schema.InputError, KeyError)):
            npa.audit_stage_a(bad)


def test_cli_flags_non_identical_repetitions(tmp_path):
    good = dict(version="M6-B1", seeds_per_regime=1, scenes=[scene_record()], same_batch=dict(ok=True))
    dirs = []
    for i, extra in enumerate((0, 0, 1)):
        d = tmp_path / ("run%d" % i)
        d.mkdir()
        (d / "normalized.json").write_bytes(schema.encoded(dict(good, extra=extra)))
        (d / "stage_a.json").write_bytes(schema.encoded(stage_a()))
        dirs.append(str(d))
    assert npa.main(dirs[:2] + ["--out", str(tmp_path / "ok.json")]) == 0
    assert json.loads((tmp_path / "ok.json").read_text())["byte_identical_across_runs"]
    assert npa.main(dirs + ["--out", str(tmp_path / "bad.json")]) == 1
    with pytest.raises(schema.InputError):
        npa.audit_record(dict(version="other", scenes=[]), stage_a())
