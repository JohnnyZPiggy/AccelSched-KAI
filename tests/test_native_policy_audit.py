"""M4-B independent audit logic on synthetic native-shaped records (no Go, no Kubernetes)."""
from copy import deepcopy
import json

import pytest

from accelsched_kai import native_policy_audit as npa, policy_benefit as pb, schema

GIB = 1024 ** 3
FRAG = dict(name="fragmentation", node_cores=[6, 6], arrival_cores=[2, 2],
            forecast=[dict(id="F0", demand=[0, 4, 1], weight=2, eligible=["node0", "node1"])], held_out_future_cores=[4, 4])


def resources(residual, cores=(6, 6)):
    out = {}
    for name, left, total in zip(npa.NODES, residual, cores):
        idle = [0, left[1] * 1000, left[2] * GIB]
        alloc = [0, total * 1000, 8 * GIB]
        out[name] = dict(allocatable=alloc, idle=idle, used=[a - i for a, i in zip(alloc, idle)], releasing=[0, 0, 0], members=[])
    return out


def native_record(tr, placement, outcomes):
    """Build what a faithful native run would report for the given decisions (model replay)."""
    trace = npa.model_trace(tr)
    residual = [n["capacity"][:] for n in trace["nodes"]]
    steps = []
    for batch, node in zip(trace["arrivals"], placement):
        task, before = batch[0], [r[:] for r in residual]
        residual[node] = [r - d for r, d in zip(residual[node], task["demand"])]
        steps.append(dict(task=task["id"], chosen_node=npa.NODES[node], before_resources=resources(before, tr["node_cores"]),
                          after_resources=resources(residual, tr["node_cores"]), native_order_S=["node0", "node1"], native_order_S0=["node0", "node1"]))
    future = []
    for job, outcome in zip(trace["realized"], outcomes):
        if outcome is None:
            fits = [i for i in range(2) if all(r >= d for r, d in zip(residual[i], job["demand"]))]
            assert not fits
            future.append(dict(outcome="REJECTED", reason="NO_FEASIBLE_NODE"))
        else:
            residual[outcome] = [r - d for r, d in zip(residual[outcome], job["demand"])]
            future.append(dict(outcome="ACCEPTED", node=npa.NODES[outcome]))
    admitted = sum(f["outcome"] == "ACCEPTED" for f in future)
    return dict(arm="native_off", trace=tr, steps=steps, future=future, future_admitted=admitted, future_total=len(future),
                future_admitted_cores=sum(j["demand"][1] for j, f in zip(trace["realized"], future) if f["outcome"] == "ACCEPTED"),
                final_resources=resources(residual, tr["node_cores"]))


def record(placement=(0, 0), outcomes=(1, None)):
    return native_record(FRAG, list(placement), list(outcomes))


def test_faithful_record_passes_and_secondary_fields_are_labelled_not_native():
    audit = npa.audit_arm(record())
    assert audit["status"] == "AUDIT_PASS" and audit["errors"] == [] and audit["native_future_feasibility_agrees_with_model"]
    assert audit["offline_oracle_secondary"]["label"] == "OFFLINE_MODEL_EQUIVALENT_NOT_NATIVE_KAI"
    assert audit["post_current_residual"] == [[0, 2, 6], [0, 6, 8]] and audit["predicted_surrogate_F_explanatory_only"] == 2


def test_units_are_exact_and_unsupported_resources_are_rejected():
    assert npa.residual_from_native(resources([[0, 3, 5], [0, 6, 8]])) == [[0, 3, 5], [0, 6, 8]]
    bad = resources([[0, 3, 5], [0, 6, 8]])
    bad["node0"]["idle"][1] += 1  # not whole cores
    with pytest.raises(schema.InputError):
        npa.residual_from_native(bad)
    gpu = resources([[0, 3, 5], [0, 6, 8]])
    gpu["node1"]["idle"][0] = 1
    with pytest.raises(schema.InputError):
        npa.residual_from_native(gpu)


@pytest.mark.parametrize("tamper", [
    lambda r: r["steps"][1]["after_resources"]["node0"]["idle"].__setitem__(1, 0),
    lambda r: r["steps"][0].update(chosen_node="node1"),
    lambda r: r["steps"][1]["before_resources"]["node1"]["idle"].__setitem__(2, 0),
    lambda r: r["final_resources"]["node1"]["idle"].__setitem__(1, 0),
    lambda r: r["future"][1].update(outcome="ACCEPTED", node="node1"),
    lambda r: r["future"][0].update(node="node0"),
])
def test_tampered_native_records_fail_the_independent_audit(tamper):
    bad = record()
    tamper(bad)
    try:
        result = npa.audit_arm(bad)
    except schema.InputError:
        return
    assert result["status"] == "AUDIT_FAIL"


def test_model_infeasible_native_acceptance_is_detected():
    # Same native claims, but the trace says the arrivals were larger than any node can host.
    over = deepcopy(FRAG)
    over["arrival_cores"] = [7, 7]
    rec = record()
    rec["trace"] = over
    result = npa.audit_arm(rec)
    assert result["status"] == "AUDIT_FAIL" and result["errors"]


def test_offline_references_reuse_m4a_and_are_labelled_not_native():
    refs = npa.offline_references(FRAG)
    assert set(refs) == {pb.PRIMARY_REFERENCE, pb.REFERENCE + ":first_fit", pb.DYNAMIC}
    assert all(r["label"] == "OFFLINE_NOT_NATIVE_KAI" and r["status"] == "PHYSICAL_VALID" for r in refs.values())
    assert refs[pb.DYNAMIC]["placement"] == [0, 1] and refs[pb.PRIMARY_REFERENCE]["placement"] == [0, 0]


def test_cli_flags_non_identical_repetitions_and_unknown_versions(tmp_path):
    good = dict(version="M4B-A1", traces={"fragmentation": {"native_off": record()}})
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_bytes(schema.encoded(good))
    b.write_bytes(schema.encoded(dict(good, extra=1)))
    out = tmp_path / "audit.json"
    assert npa.main([str(a), str(a), str(a), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["byte_identical_across_runs"] and report["simulated_gpu"] == "SIMULATED_GPU_NOT_RUN"
    assert npa.main([str(a), str(b), "--out", str(tmp_path / "x.json")]) == 1
    with pytest.raises(schema.InputError):
        npa.audit_record(dict(version="other", traces={}))
