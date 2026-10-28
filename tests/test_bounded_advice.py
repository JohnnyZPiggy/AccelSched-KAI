"""M3-A numerical, artifact, failure and provenance acceptance (offline CPU)."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from accelsched_kai import adapter, advisor, bounded_advice as ba, oracle, schema
from test_poc import load, duals, multi_snapshot


def build(s=None, **params):
    return ba.build_artifact(load() if s is None else s, **dict(BF=2., BP=2., alpha=.5) | params)


def accepted(s=None, **params):
    result = build(s, **params)
    assert result["status"] == ba.STATUS, result
    return result["artifact"]


def rehash(artifact):
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = schema.digest(artifact)
    return schema.encoded(artifact)


def denied(result):
    assert result["status"] != ba.STATUS, result
    assert "artifact" not in result and "advice" not in result and "pairs" not in result
    schema.encoded(result)


@pytest.mark.parametrize("index", range(4))
def test_G1_G3_G5_source_certified_roundtrip(index):
    s = load(index)
    artifact = accepted(s)
    assert artifact["upstream_sha"] == adapter.BASELINE
    assert artifact["frozen_source_sha256"] == adapter.SOURCE_HASHES
    assert artifact["mapping"] == adapter.instance(s)[1]
    assert artifact["lp"]["status"] == "OPTIMAL"
    cert = artifact["lp"]["certificate"]
    assert cert["certified"] and cert["original_hash_match"]
    assert cert["objective_gap"] == 0 and cert["stationarity_residual"] == 0
    assert cert["native_objective_fields_match"]
    assert artifact["placement"]["independent_audit"]["status"] == "PHYSICAL_VALID"
    with patch.object(adapter.frozen()["lp_backends"], "solve_lp", side_effect=AssertionError("load must not re-solve/select another dual")):
        assert ba.load_artifact(schema.encoded(artifact), s) == dict(status=ba.STATUS, artifact=artifact)
    assert len(artifact["pairs"]) + len(artifact["excluded_pairs"]) == len(s["nodes"]) * len(s["tasks"])


@pytest.mark.parametrize("index,expected", [(0, [-1., 0.]), (1, [-3., -1.]), (3, [-1.])])
def test_G2_independent_fixed_state_marginals(index, expected):
    a = accepted(load(index))
    assert [p["g"] for p in a["pairs"]] == expected
    for p in a["pairs"]:
        assert p["g"] == p["g_oracle"] and p["g"] <= 0
        assert -1 <= p["score"] <= 1
    if index == 0:
        # Same Shape prices, different physical future eligibility -> different g.
        assert a["pairs"][0]["p"] == a["pairs"][1]["p"]
        assert a["pairs"][0]["g"] != a["pairs"][1]["g"]


def test_G2_eligibility_zero_saturation_and_fixed_not_sequential_state():
    s = multi_snapshot()
    s["nodes"].reverse()  # Preserve first-occurrence shape mapping, sort pairs by IDs.
    a = accepted(s)
    keys = [(p["task_id"], p["node_id"]) for p in a["pairs"]]
    assert keys == sorted(keys) and a["mapping"]["nodes"] == ["N1", "N0"]
    assert any(p["reason"] == "INELIGIBLE" and p["task_id"] == "T3" for p in a["excluded_pairs"])
    # Each of the identical-demand feasible tasks gets the same original-state g,
    # even though the sequential replay cannot place all three.
    for node in ("N0", "N1"):
        assert len({p["g"] for p in a["pairs"] if p["node_id"] == node}) == 1
    assert -1 in a["placement"]["replay"]["placement"]
    s = load()
    s["future"][0]["weight"] = 0
    assert all(p["g"] == 0 for p in accepted(s)["pairs"])
    s["future"][0]["weight"] = 1
    s["future"][0]["demand"] = [5, 4, 4]  # Already does not fit: zero loss.
    assert all(p["g"] == 0 for p in accepted(s)["pairs"])
    s["tasks"][0]["gpu_family"] = "B"
    a = accepted(s)
    assert [p["node_id"] for p in a["pairs"]] == ["N1"]
    s["tasks"][0]["demand"] = [5, 2, 2]
    a = accepted(s)
    assert not a["pairs"] and a["placement"]["independent_audit"]["accepted"] == 0
    assert {p["reason"] for p in a["excluded_pairs"]} == {"INELIGIBLE", "RESOURCE_INFEASIBLE"}


def test_G2_independent_marginal_fault_is_detected():
    s = load()
    modules = adapter.frozen()
    original = accepted(s)["placement"]["replay"]
    # A frozen replay witness isolates the new fixed-state marginal check; this
    # is explicitly injected evidence, not an alternative algorithm source.
    with patch.object(modules["certified_pgfp"], "_replay", return_value=original), \
         patch.object(modules["m0_oracle"], "marginal", return_value=123.):
        result = build(s)
    denied(result)
    assert result["status"] == "MARGINAL_MISMATCH"


def test_G3_dual_nonuniqueness_retained_not_manufactured_unique_price():
    s = load(3)
    native = build(s)
    assert native["execution"]["attempts"][0]["raw"]["backend"] == "highs"
    # Independently certified analytical witnesses prove only nonuniqueness;
    # they are not passed off as native solver executions or built artifacts.
    witnesses = duals()
    assert all(w["certificate"]["certified"] for w in witnesses)
    assert witnesses[0]["raw"]["pi"] != witnesses[1]["raw"]["pi"]
    assert native["artifact"]["lp"]["price_sha256"] == schema.digest(native["artifact"]["lp"]["prices"])


@pytest.mark.parametrize("BF,expected,clipped", [(4., -.75, False), (3., -1., False), (2., -1., True)])
def test_G4_clip_boundaries(BF, expected, clipped):
    p = accepted(load(1), BF=BF, alpha=0)["pairs"][0]
    assert p["g"] == -3 and p["AF"] == expected and p["score"] == expected
    assert p["clipped_F"] is clipped


@pytest.mark.parametrize("BP,AP,clipped", [(2., -.5, False), (1., -1., False), (.5, -1., True)])
def test_G4_price_sign_clip_and_alpha_endpoints(BP, AP, clipped):
    # Scarce 1GPU, value2, demand2 -> x=.5; native GPU price=1.
    s = load(3)
    s["tasks"][0].update(demand=[2, 0, 0], value=2)
    # Task bound x<=1 and resource x<=.5 makes the capacity price unambiguous.
    s["nodes"][0]["capacity"] = [1, 0, 0]
    # No feasible pair here, so use a second smaller current task (value0), whose
    # valid candidate sees the same native certified scarcity price.
    s["tasks"].append(dict(s["tasks"][0], id="small", demand=[1, 0, 0], value=0))
    a = accepted(s, BP=BP, alpha=1)
    p = a["pairs"][0]
    assert p["task_id"] == "small" and p["p"] == -1.
    assert p["AP"] == AP and p["score"] == AP and p["clipped_P"] is clipped


@pytest.mark.parametrize("params", [dict(BF=0), dict(BF=-1), dict(BP=0), dict(BP=float("nan")),
    dict(BF=float("inf")), dict(alpha=-.01), dict(alpha=1.01), dict(alpha=True),
    dict(BF=True), dict(BF=10**1000), dict(time_limit=0), dict(time_limit=11),
    dict(BF=1e-320), dict(BP=1e-320)])
def test_G4_invalid_and_nonfinite_normalization_fail_closed(params):
    # A positive nonzero scarcity price is required to exercise BP overflow.
    s = load(3)
    s["tasks"][0].update(demand=[2, 0, 0], value=2)
    s["tasks"].append(dict(s["tasks"][0], id="small", demand=[1, 0, 0], value=0))
    denied(build(s, **params))


ARTIFACT_MUTATIONS = [
    lambda a: a.update(version="M3A-2"),
    lambda a: a.update(upstream_sha="0" * 40),
    lambda a: a["frozen_source_sha256"].update(m0_oracle="0" * 64),
    lambda a: a["implementation_source_sha256"].update(bounded_advice="0" * 64),
    lambda a: a.update(prediction_sha256="0" * 64),
    lambda a: a.update(resources_sha256="0" * 64),
    lambda a: a.update(future_sha256="0" * 64),
    lambda a: a["mapping"]["nodes"].reverse(),
    lambda a: a["mapping"]["shapes"].append("foreign"),
    lambda a: a["pairs"][0].update(node_id="foreign"),
    lambda a: a["pairs"][0].update(task_id="foreign"),
    lambda a: a["pairs"].append(deepcopy(a["pairs"][0])),
    lambda a: a["pairs"].pop(),
    lambda a: a["pairs"].reverse(),
    lambda a: a["pairs"][0].update(score=1.),
    lambda a: a["pairs"][0].update(g=-99.),
    lambda a: a["pairs"][0].update(AF=True),
    lambda a: a["normalization"].update(alpha=True),
    lambda a: a["normalization"].update(extra=0),
    lambda a: a["lp"]["prices"][0].__setitem__(0, 99.),
    lambda a: a["lp"].update(price_sha256="0" * 64),
    lambda a: a["lp"]["raw"]["pi"].__setitem__(0, 1.),
    lambda a: a["lp"]["raw"]["x"].__setitem__(0, 99.),
    lambda a: a["lp"]["raw"]["rc"].__setitem__(0, 1.),
    lambda a: a["lp"]["raw"].update(objective=-99.),
    lambda a: a["lp"]["raw"]["native_status"].update(code=1),
    lambda a: a["lp"]["certificate"].update(certified=False),
    lambda a: a["lp"]["certificate"].update(primal_residual=123.),
    lambda a: a["lp"]["attempts"].clear(),
    lambda a: a["lp"]["original_model"]["c"].__setitem__(0, -99.),
    lambda a: a["lp"].update(source_hash="0" * 64),
    lambda a: a["tolerances"].update(lp_absolute=1.),
    lambda a: a["placement"]["replay"].update(placement=[0]),
    lambda a: a["placement"]["independent_audit"].update(accepted=99),
    lambda a: a.update(extra="unknown"),
]


@pytest.mark.parametrize("mutation", ARTIFACT_MUTATIONS)
def test_G4_G6_even_rehashed_tampering_is_recomputed(mutation):
    a = accepted()
    mutation(a)
    denied(ba.load_artifact(rehash(a), load()))


@pytest.mark.parametrize("change", [
    lambda s: s["nodes"][0]["capacity"].__setitem__(0, 3),
    lambda s: s["tasks"][0]["demand"].__setitem__(0, 1),
    lambda s: s["tasks"][0].update(id="foreign"),
    lambda s: s["future"][0].update(weight=2),
    lambda s: s["future"][0].update(eligible=[]),
    lambda s: s["nodes"].reverse(),
])
def test_G1_G6_stale_snapshot_rejected(change):
    raw = schema.encoded(accepted())
    s = load(); change(s)
    denied(ba.load_artifact(raw, s))


@pytest.mark.parametrize("kind", ["timeout", "infeasible", "error", "uncertified", "hash", "primal", "dual", "rc", "objective", "nonfinite", "missing", "price-sign"])
def test_G3_G6_dishonest_solver_flags_never_bypass_certificate(kind):
    modules = adapter.frozen()
    a, _ = adapter.instance(load())
    model, _ = modules["certified_pgfp"].canonical_shape(a)
    result = modules["lp_backends"].solve_lp(model, backend="highs", fallback="none", time_limit=5)
    raw = result.raw
    if kind in ("timeout", "infeasible", "error"):
        raw.status = {"timeout": "TIME_LIMIT", "infeasible": "INFEASIBLE", "error": "ERROR"}[kind]
    elif kind == "uncertified": result.attempts[0].certificate["certified"] = False
    elif kind == "hash": result.source_hash = "0" * 64
    elif kind == "primal": raw.x[0] = 99.
    elif kind == "dual": raw.pi[0] = 1.
    elif kind == "rc": raw.rc[0] = 1.
    elif kind == "objective": raw.objective = -99.
    elif kind == "nonfinite": raw.x[0] = np.nan
    elif kind == "missing": raw.pi = None
    elif kind == "price-sign": raw.pi[1] = 1e-9  # Cert tolerance does not grant a negative scarcity price.
    with patch.object(modules["lp_backends"], "solve_lp", return_value=result):
        out = build(alpha=0)
    denied(out)
    if kind == "timeout": assert out["status"] == "SOLVER_TIMEOUT"


def test_G5_relaxation_is_not_discrete_eligibility_or_placement_certificate():
    s = load(3)
    s["tasks"][0]["eligible"] = []
    a = accepted(s)
    # Shape-LP ignores model eligibility: x=1 and objective=-1 remain optimal.
    assert a["lp"]["raw"]["x"] == [1.]
    assert a["lp"]["certificate"]["primal_objective"] == -1.
    assert a["lp"]["certificate"]["certified"]
    assert a["pairs"] == [] and len(a["excluded_pairs"]) == 1
    assert oracle.audit(s, [0])["status"] == "PHYSICAL_INVALID"
    assert a["placement"]["replay"]["placement"] == [-1]
    assert a["placement"]["independent_audit"]["accepted"] == 0
    assert a["placement"]["independent_audit"]["status"] == "PHYSICAL_VALID"


def test_G5_corrupted_replay_and_physical_audit_fail_closed():
    modules = adapter.frozen()
    replay = accepted()["placement"]["replay"]
    replay["placement"] = [99]
    with patch.object(modules["certified_pgfp"], "_replay", return_value=replay):
        denied(build())
    with patch.object(oracle, "audit", return_value=dict(status="PHYSICAL_INVALID", errors=["injected independent physical failure"])):
        denied(build())


@pytest.mark.parametrize("case", load(4)["cases"], ids=lambda x: x["case"])
def test_G5_unsupported_no_artifact(case):
    out = build(case["snapshot"])
    denied(out)
    assert out["status"] == "UNSUPPORTED_CONSTRAINT"


def test_G1_G6_bad_source_and_input_fail_closed(monkeypatch):
    s = load(); s["tasks"].append(deepcopy(s["tasks"][0]))
    denied(build(s))
    s = load(); s["tasks"][0]["eligible"] = ["unknown"]
    denied(build(s))
    s = load(); s.update(schema_version="foreign")
    denied(build(s))
    monkeypatch.setitem(adapter.SOURCE_HASHES, "m0_oracle", "0" * 64)
    denied(build())


def test_G4_noncanonical_decode_corruption_and_size():
    a = accepted(); raw = schema.encoded(a)
    for invalid in (raw + b"\n", b" " + raw, raw[:-1], b'{}', b'{"x":1,"x":2}',
                    b'{"x":NaN}', b'\xff', b' ' * (ba.MAX_BYTES + 1), raw.decode(),
                    raw.replace(b'"alpha":0.5', b'"alpha":5e-1')):
        denied(ba.load_artifact(invalid, load()))
    a["artifact_sha256"] = "0" * 64
    denied(ba.load_artifact(schema.encoded(a), load()))


def test_G6_no_hindsight_input_mutation_or_optional_device_import():
    s = load(); original = deepcopy(s)
    first = accepted(s)
    assert s == original
    changed = load(2)
    assert changed["realized_future"] != s["realized_future"]
    assert accepted(changed) == first
    assert ba.load_artifact(schema.encoded(first), changed)["status"] == ba.STATUS
    assert not set(sys.modules) & {"torch", "cupy", "cupdlpx", "hpdlp", "kubernetes", "google.colab"}


def record_evidence(directory):
    """Fresh-process native receipts, separate from injected test witnesses."""
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    scenarios = [("F01", load()), ("F02", load(1)), ("F03", load(2)), ("F04", load(3)),
                 ("multi-shape", multi_snapshot())]
    relaxed = load(3); relaxed["tasks"][0]["eligible"] = []
    scenarios.append(("relaxed-ineligible", relaxed))
    for name, snapshot in scenarios:
        result = build(snapshot)
        assert result["status"] == ba.STATUS, result
        artifact = result["artifact"]
        raw = schema.encoded(artifact)
        assert ba.load_artifact(raw, snapshot)["status"] == ba.STATUS
        (directory / (name + ".artifact.json")).write_bytes(raw)
        (directory / (name + ".execution.json")).write_bytes(schema.encoded(result["execution"]))
        (directory / (name + ".snapshot.json")).write_bytes(schema.encoded(snapshot))
        records.append(dict(name=name, bytes=len(raw), file_sha256=hashlib.sha256(raw).hexdigest(),
                            artifact_sha256=artifact["artifact_sha256"], prediction_sha256=artifact["prediction_sha256"],
                            model_source_hash=artifact["lp"]["source_hash"], price_sha256=artifact["lp"]["price_sha256"],
                            pairs=artifact["pairs"], certificate=artifact["lp"]["certificate"]))
    # Analytical evidence is explicitly distinct from native artifacts above.
    (directory / "analytical-dual-witnesses.json").write_bytes(schema.encoded(duals()))
    (directory / "summary.json").write_bytes(schema.encoded(records))
    print(json.dumps(dict(status="PASS", native_runs=len(records), artifact_hashes={r["name"]: r["file_sha256"] for r in records})))


if __name__ == "__main__":
    record_evidence(Path(sys.argv[1]))
