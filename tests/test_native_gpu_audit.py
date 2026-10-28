"""M5 independent audit logic on synthetic native-shaped whole-GPU records (no Go, no Kubernetes)."""
from copy import deepcopy
import json

import pytest

from accelsched_kai import native_gpu_audit as nga, policy_benefit as pb, schema

GIB = 1024 ** 3
TRACE = dict(name="split_vs_concentrated", nodes=[dict(gpus=6, cores=16), dict(gpus=6, cores=16)],
             arrivals=[dict(gpus=2, cores=2), dict(gpus=2, cores=2)],
             forecast=[dict(id="F0", demand=[4, 2, 1], weight=2, eligible=["node0", "node1"])],
             held_out_future=[dict(gpus=4, cores=2), dict(gpus=4, cores=2)])


def resources(residual, nodes=TRACE["nodes"]):
    out = {}
    for name, left, n in zip(nga.NODES, residual, nodes):
        alloc = [n["gpus"], int(n["cores"]) * 1000, nga.MEM_GIB * GIB]
        idle = [left[0], left[1] * 1000, left[2] * GIB]
        out[name] = dict(allocatable=alloc, idle=idle, used=[a - i for a, i in zip(alloc, idle)], releasing=[0, 0, 0], members=[])
    return out


def native_record(placement, outcomes, tr=TRACE):
    trace = nga.model_trace(tr)
    residual = [n["capacity"][:] for n in trace["nodes"]]
    steps = []
    for batch, node in zip(trace["arrivals"], placement):
        task, before = batch[0], [r[:] for r in residual]
        residual[node] = [r - d for r, d in zip(residual[node], task["demand"])]
        steps.append(dict(task=task["id"], chosen_node=nga.NODES[node], before_resources=resources(before, tr["nodes"]), after_resources=resources(residual, tr["nodes"])))
    future = []
    for job, outcome in zip(trace["realized"], outcomes):
        if outcome is None:
            assert not [i for i in range(2) if all(r >= d for r, d in zip(residual[i], job["demand"]))]
            future.append(dict(outcome="REJECTED", reason="NO_FEASIBLE_NODE"))
        else:
            residual[outcome] = [r - d for r, d in zip(residual[outcome], job["demand"])]
            future.append(dict(outcome="ACCEPTED", node=nga.NODES[outcome]))
    accepted = [j for j, f in zip(trace["realized"], future) if f["outcome"] == "ACCEPTED"]
    return dict(arm="native_off", trace=tr, steps=steps, future=future, future_admitted=len(accepted), future_total=len(future),
                future_admitted_gpus=sum(j["demand"][0] for j in accepted), final_resources=resources(residual, tr["nodes"]))


def faithful():
    return native_record([0, 0], [1, None])


def stage_a():
    return dict(version="P2-WholeGPU-1", mapping_proof=dict(index_names=["cpu", "memory", "gpu", "pods"]),
                fail_closed=[dict(case="fractional_gpu", gate="go_projection", reason="x"), dict(case="one_node_ineligible", gate="python_producer", reason="y")],
                eligibility_table=[dict(request=dict(gpus=1, cores=2), node0=dict(native=True, model=True), node1=dict(native=False, model=False))],
                trust={k: True for k in ("spent_grant_rejected", "stale_projection_rejected", "mutated_hooks_rejected", "forged_uid_rejected", "foreign_session_rejected",
                                         "resource_mutation_rejected", "incomplete_duplicate_mapping_rejected", "fresh_grant_after_faults_activates", "neutral_scores_equal_native")})


def test_faithful_record_passes_with_whole_gpu_units_and_secondary_labels():
    audit = nga.audit_arm(faithful())
    assert audit["status"] == "AUDIT_PASS" and audit["errors"] == [] and audit["post_current_residual"] == [[2, 12, 62], [6, 16, 64]]
    assert audit["offline_oracle_secondary"]["label"] == "OFFLINE_MODEL_EQUIVALENT_NOT_NATIVE_KAI"
    assert audit["predicted_surrogate_F_explanatory_only"] == 2


def test_units_are_exact_and_gpu_remainders_rejected():
    assert nga.residual_from_native(resources([[3, 4, 5], [6, 16, 64]])) == [[3, 4, 5], [6, 16, 64]]
    for mutate in (lambda r: r["node0"]["idle"].__setitem__(1, 1500), lambda r: r["node0"]["idle"].__setitem__(2, GIB + 1),
                   lambda r: r["node0"]["idle"].__setitem__(0, 1.5)):
        bad = resources([[3, 4, 5], [6, 16, 64]])
        mutate(bad)
        with pytest.raises(schema.InputError):
            nga.residual_from_native(bad)


@pytest.mark.parametrize("tamper", [
    lambda r: r["steps"][1]["after_resources"]["node0"]["idle"].__setitem__(0, 0),
    lambda r: r["steps"][0].update(chosen_node="node1"),
    lambda r: r["steps"][1]["before_resources"]["node1"]["idle"].__setitem__(0, 1),
    lambda r: r["final_resources"]["node1"]["idle"].__setitem__(0, 0),
    lambda r: r["future"][1].update(outcome="ACCEPTED", node="node1"),
    lambda r: r["future"][0].update(node="node0"),
])
def test_tampered_native_records_fail_the_independent_audit(tamper):
    bad = faithful()
    tamper(bad)
    try:
        assert nga.audit_arm(bad)["status"] == "AUDIT_FAIL"
    except schema.InputError:
        pass


def test_model_infeasible_native_acceptance_is_detected():
    rec = faithful()
    rec["trace"] = deepcopy(TRACE)
    rec["trace"]["arrivals"] = [dict(gpus=7, cores=2), dict(gpus=7, cores=2)]
    assert nga.audit_arm(rec)["status"] == "AUDIT_FAIL"


def test_offline_references_reuse_m4a_and_are_not_native():
    refs = nga.offline_references(TRACE)
    assert all(r["label"] == "OFFLINE_NOT_NATIVE_KAI" and r["status"] == "PHYSICAL_VALID" for r in refs.values())
    assert refs[pb.DYNAMIC]["placement"] == [0, 1] and refs[pb.PRIMARY_REFERENCE]["placement"] == [0, 0]
    assert refs[pb.DYNAMIC]["sequential_admitted"] == 2 and refs[pb.PRIMARY_REFERENCE]["sequential_admitted"] == 1


def test_stage_a_structural_audit_and_failures():
    assert nga.audit_stage_a(stage_a())["status"] == "STAGE_A_PASS"
    for mutate in (lambda a: a["fail_closed"][0].update(gate="ACCEPTED"), lambda a: a["eligibility_table"][0]["node1"].update(model=True),
                   lambda a: a["trust"].update(spent_grant_rejected=False), lambda a: a["mapping_proof"].update(index_names=["gpu"]),
                   lambda a: a.update(version="P1"), lambda a: a.update(fail_closed=[])):
        bad = stage_a()
        mutate(bad)
        with pytest.raises((schema.InputError, KeyError)):
            nga.audit_stage_a(bad)


def test_cli_flags_non_identical_repetitions(tmp_path):
    good = dict(version="M5-B1", traces={"split_vs_concentrated": {"native_off": faithful()}})
    dirs = []
    for i, extra in enumerate((0, 0, 1)):
        d = tmp_path / ("run%d" % i)
        d.mkdir()
        (d / "normalized.json").write_bytes(schema.encoded(dict(good, extra=extra)))
        (d / "stage_a.json").write_bytes(schema.encoded(stage_a()))
        dirs.append(str(d))
    assert nga.main(dirs[:2] + ["--out", str(tmp_path / "ok.json")]) == 0
    assert json.loads((tmp_path / "ok.json").read_text())["byte_identical_across_runs"]
    assert nga.main(dirs + ["--out", str(tmp_path / "bad.json")]) == 1
    with pytest.raises(schema.InputError):
        nga.audit_record(dict(version="other", traces={}), stage_a())
