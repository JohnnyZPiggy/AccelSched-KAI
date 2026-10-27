"""Bounded CPU acceptance tests and reproducible evidence recorder.

Run directly after pytest to retain every fixture, failure and numerical witness.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest
from accelsched_kai import adapter, advisor, oracle, schema

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = sorted((ROOT / "fixtures").glob("*.json"))


def load(index=0):
    return schema.loads(FIXTURES[index].read_bytes())


def duals():
    a, _ = adapter.instance(load(3)); modules = adapter.frozen()
    model, _ = modules["certified_pgfp"].canonical_shape(a)
    evidence = []
    for mu, lam in ((1., 0.), (0., 1.)):
        raw = modules["lp_backends"].RawResult("analytical_test_only", None, "OPTIMAL",
            native_status="analytical exact certificate; not a native solver run",
            x=np.array([1.]), pi=np.array([-mu, -lam, 0., 0.]), rc=np.array([0.]),
            objective=-1., dual_objective=-1., gap=0.)
        _, _, check = advisor.certificate(a, raw, model.source_hash)
        evidence.append(dict(mu=mu, lambda_gpu=lam, original_hash=model.source_hash,
                             raw=advisor.plain(asdict(raw)), certificate=check))
    return evidence


def multi_snapshot():
    s = load(); s["nodes"][1]["shape"] = "other"
    s["tasks"] = [deepcopy(s["tasks"][0]) for _ in range(4)]
    for j, t in enumerate(s["tasks"]):
        t.update(id="T" + str(j), value=[1, 2, 2, 0][j], demand=[3, 2, 2])
    s["tasks"][3]["eligible"] = []
    return s


def raw_failure(kind):
    s = load(); a, _ = adapter.instance(s); modules = adapter.frozen()
    model, _ = modules["certified_pgfp"].canonical_shape(a)
    real = modules["lp_backends"].solve_lp(model, backend="highs", fallback="none", time_limit=5)
    raw = deepcopy(real.raw)
    if kind == "timeout":
        raw.status = "TIME_LIMIT"; raw.native_status = {"code": 1, "message": "injected CPU timeout"}
    elif kind == "bad_primal":
        raw.x[0] = 2.
    elif kind == "bad_dual":
        raw.pi[0] = 2.
    elif kind == "bad_objective":
        raw.objective = -99.
    elif kind == "nonfinite":
        raw.x[0] = np.nan
    elif kind == "missing":
        raw.pi = None
    elif kind == "infeasible":
        raw.status = "INFEASIBLE"
    elif kind == "solver_error":
        raw.status = "ERROR"; raw.error = "injected CPU adapter error"
    check = modules["lp_certify"].certify(model, raw)
    # Intentionally dishonest outer success flags exercise independent rechecking.
    attempt = modules["lp_backends"].Attempt(raw, "OPTIMAL", {"certified": True}, 0.)
    result = modules["lp_backends"].CertifiedResult("TIME_LIMIT" if kind == "timeout" else "OPTIMAL", (attempt,), 0.,
        "0" * 64 if kind == "wrong_hash" else model.source_hash)
    with patch.object(modules["lp_backends"], "solve_lp", return_value=result):
        out = advisor.advise(s)
    return dict(injected=True, scenario=kind, independent_raw_check=check, result=out)


@pytest.mark.parametrize("index", range(4))
def test_certified_fixtures(index):
    result = advisor.advise(load(index))
    assert result["status"] == "PHYSICAL_VALID", result
    assert result["lp"]["certificate"]["certified"]
    assert result["lp"]["certificate"]["original_hash_match"]
    assert result["lp"]["certificate"]["objective_gap"] == 0
    assert result["replay_verification"]["status"] == "EXACT_FIXED_PRICE_REPLAY"
    assert all(v["oracle"]["status"] == "PHYSICAL_VALID" for v in result["policy_audits"].values())
    expected_hash = result.pop("output_sha256")
    assert schema.digest(result) == expected_hash


def test_f01_and_f02_math():
    for index, expected_initial, expected_after, scores in ((0, 1, (0, 1), [-1, 0]), (1, 4, (1, 3), [-3, -1])):
        s = load(index); out = advisor.advise(s)
        assert out["advice"] == {"first_fit": [0], "lp_best_fit": [0], "true_pgfp": [1]}
        assert oracle.future_score(s, [n["capacity"] for n in s["nodes"]]) == expected_initial
        assert out["replay"]["ledger_steps"][0]["attempts"] == [{"shape": 0, "candidates": [0, 1], "scores": scores}]
        audits = out["policy_audits"]
        assert (audits["lp_best_fit"]["future"]["predicted_surrogate_F"], audits["true_pgfp"]["future"]["predicted_surrogate_F"]) == expected_after


def test_f03_shift_no_hindsight():
    predicted = advisor.advise(load()); shifted = advisor.advise(load(2))
    assert predicted["replay"] == shifted["replay"]
    assert predicted["lp"]["source_hash"] == shifted["lp"]["source_hash"]
    policies = shifted["policy_audits"]
    for key, expected in (("true_pgfp", 0), ("lp_best_fit", 1)):
        future = policies[key]["future"]
        assert future["sequential"]["value"] == expected
        assert future["offline_fixed_current"]["value"] == expected
    assert policies["true_pgfp"]["future"]["predicted_surrogate_F"] == 1


def test_f04_two_valid_duals_not_a_placement_claim():
    first, second = duals()
    assert first["raw"]["pi"] != second["raw"]["pi"]
    for witness in (first, second):
        assert witness["certificate"]["certified"]
        assert witness["certificate"]["primal_objective"] == -1
        assert witness["certificate"]["dual_objective"] == -1
        assert witness["certificate"]["row_complementarity"] == 0
        assert witness["certificate"]["stationarity_residual"] == 0


@pytest.mark.parametrize("case", load(4)["cases"], ids=lambda x: x["case"])
def test_f05_unsupported(case):
    out = advisor.advise(case["snapshot"])
    assert out["status"] == "UNSUPPORTED_CONSTRAINT"
    assert out["input_sha256"] == schema.digest(case["snapshot"])
    assert "advice" not in out and "policy_audits" not in out and "lp" not in out


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(extra_mandatory=True),
    lambda s: s.update(schema_version="0.2"),
    lambda s: s.update(units=["GPU", "millicores", "bytes"]),
    lambda s: s.update(constraints={"unknown": True}),
    lambda s: s.update(tasks=[]),
    lambda s: s["nodes"][1].update(id="N0"),
    lambda s: s["tasks"][0].update(eligible=["missing"]),
    lambda s: s["tasks"][0].update(eligible=["N0", "N0"]),
    lambda s: s["tasks"][0].update(value=True),
    lambda s: s["tasks"][0].update(value=float("inf")),
    lambda s: s["tasks"][0].update(demand=[1, -1, 2]),
    lambda s: s["tasks"][0].update(demand=[True, 2, 2]),
    lambda s: s["tasks"][0].update(demand=[1, 2.5, 2]),
    lambda s: s["tasks"][0].update(demand=[0, 0, 0]),
    lambda s: s["tasks"][0].update(required_rack="rack0"),
    lambda s: s.update(nodes="invalid"),
    lambda s: s.update(realized_future=s["realized_future"] * 9),
])
def test_invalid_schema(mutation):
    s = load(); mutation(s); out = advisor.advise(s)
    assert out["status"] == "INVALID_INPUT"
    assert "advice" not in out


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{', '"unterminated', b'\xff'])
def test_strict_json(text):
    with pytest.raises(schema.InputError):
        schema.loads(text)


@pytest.mark.parametrize("kind", ["timeout", "bad_primal", "bad_dual", "bad_objective", "nonfinite", "missing", "wrong_hash", "infeasible", "solver_error"])
def test_raw_failure_integrity(kind):
    out = raw_failure(kind)["result"]
    assert out["status"] == ("SOLVER_TIMEOUT" if kind == "timeout" else "LP_UNCERTIFIED"), out
    assert "advice" not in out
    assert len(out["lp"]["attempts"]) == 1
    schema.encoded(out)


@pytest.mark.parametrize("field", ["task_order", "shape_order", "candidates", "scores", "chosen", "before", "after", "placement", "residual", "greedy_residual"])
def test_complete_replay_fault_detection(field):
    s = load(); out = advisor.advise(s); replay = deepcopy(out["replay"])
    if field == "task_order": replay[field] = []
    elif field == "shape_order": replay[field] = [[]]
    elif field in ("candidates", "scores"): replay["ledger_steps"][0]["attempts"][0][field] = []
    elif field == "chosen": replay["ledger_steps"][0][field] = 0
    elif field in ("before", "after"): replay["ledger_steps"][0][field][0][0] = 99
    elif field == "placement": replay[field] = [0]
    else: replay[field][0][0] = 99
    lp = {"prices": np.array(out["lp"]["prices"])}
    assert advisor.verify_replay(s, lp, replay)["status"] == "REPLAY_MISMATCH"
    modules = adapter.frozen()
    with patch.object(modules["certified_pgfp"], "_replay", return_value=replay):
        rejected = advisor.advise(s)
    assert rejected["status"] == "REPLAY_MISMATCH" and "advice" not in rejected


def test_multiple_shapes_order_ties_rejections_and_manual_price_only():
    s = multi_snapshot(); a, _ = adapter.instance(s); modules = adapter.frozen()
    for prices in (np.zeros((2, 3)), np.array([[1., 0., 0.], [0., 0., 0.]])):
        lp = {"shapes": [0, 1], "prices": prices}
        replay = modules["certified_pgfp"]._replay(a, lp)
        assert advisor.verify_replay(s, lp, replay)["status"] == "EXACT_FIXED_PRICE_REPLAY"
        assert replay["task_order"] == [1, 2, 0, 3]
        assert replay["placement"][0] == -1 and replay["placement"][3] == -1
        assert replay["shape_order"][0] == ([0, 1] if prices.sum() == 0 else [1, 0])
    # These manual test prices have NO certificate and never enter advise().
    out = advisor.advise(s)
    assert out["status"] == "PHYSICAL_VALID"


def test_oracle_independence_faults_and_lawful_rejection():
    s = load(); a, _ = adapter.instance(s); modules = adapter.frozen()
    with patch.object(modules["m0_oracle"], "validate", side_effect=AssertionError("must not call scheduler")):
        assert oracle.audit(s, [1])["status"] == "PHYSICAL_VALID"
        assert oracle.audit(s, [-1])["status"] == "PHYSICAL_VALID"
        assert oracle.audit(s, [1], [[0, 0, 0], [0, 0, 0]])["status"] == "PHYSICAL_INVALID"
        for bad in ([2], [True], [0, 1], [[0, 1]], [0.0]):
            assert oracle.audit(s, bad)["status"] == "PHYSICAL_INVALID"
    s["tasks"][0]["eligible"] = ["N0"]
    assert oracle.audit(s, [1])["status"] == "PHYSICAL_INVALID"
    s["tasks"][0]["demand"] = [5, 2, 2]
    assert oracle.audit(s, [0])["status"] == "PHYSICAL_INVALID"
    assert advisor.advise(s)["status"] == "PHYSICAL_VALID"  # rejection is lawful


def test_mapping_family_is_not_shape():
    s = load(); a, mapping = adapter.instance(s)
    assert a.shapes.tolist() == [0, 0] and mapping["nodes"] == ["N0", "N1"]
    assert a.eligible.tolist() == [[True], [True]]
    assert a.future_eligible.tolist() == [[True], [False]]
    s["tasks"][0]["gpu_family"] = "B"; a, _ = adapter.instance(s)
    assert a.eligible.tolist() == [[False], [True]]
    s["nodes"].reverse(); a, mapping = adapter.instance(s)
    assert mapping["nodes"] == ["N1", "N0"] and a.eligible.tolist() == [[True], [False]]


def test_future_sequential_vs_exact_offline():
    s = load(3); s["nodes"][0]["capacity"] = [2, 0, 0]
    template = s["tasks"][0]
    s["realized_future"] = [dict(template, id="low", demand=[2, 0, 0], value=1),
                            dict(template, id="high", demand=[2, 0, 0], value=5)]
    result = oracle.evaluate(s, [[2, 0, 0]])
    assert result["sequential"]["value"] == 1
    assert result["offline_fixed_current"]["value"] == 5
    assert result["offline_fixed_current"]["placement"] == [-1, 0]


def test_source_pin_and_internal_failure(monkeypatch):
    monkeypatch.setitem(adapter.SOURCE_HASHES, "m0_oracle", "0" * 64)
    out = advisor.advise(load())
    assert out["status"] == "INTERNAL_ERROR" and "source mismatch" in out["error"]
    assert "advice" not in out


def test_physical_invalid_does_not_emit_advice(monkeypatch):
    original = advisor.baseline
    def corrupt(s, lp, policy):
        out = original(s, lp, policy); out["placement"] = [99]; return out
    monkeypatch.setattr(advisor, "baseline", corrupt)
    out = advisor.advise(load())
    assert out["status"] == "PHYSICAL_INVALID" and "advice" not in out


@pytest.mark.parametrize("limit", [0, -1, True, float("nan"), 11])
def test_invalid_time_limit(limit):
    assert advisor.advise(load(), time_limit=limit)["status"] == "INVALID_INPUT"


def test_cpu_import_boundary():
    adapter.frozen()
    assert not set(sys.modules) & {"torch", "cupy", "cupdlpx", "hpdlp", "kubernetes", "google.colab"}


def record_evidence():
    outdir = ROOT / "evidence"; outdir.mkdir(exist_ok=True)
    runs = []
    for index in range(4):
        runs.append(dict(fixture=FIXTURES[index].name, result=advisor.advise(load(index))))
    for case in load(4)["cases"]:
        runs.append(dict(fixture=FIXTURES[4].name, case=case["case"], result=advisor.advise(case["snapshot"])))
    failures = [raw_failure(kind) for kind in ["timeout", "bad_primal", "bad_dual", "bad_objective", "nonfinite", "missing", "wrong_hash", "infeasible", "solver_error"]]
    s = load(); base = advisor.advise(s); changed = deepcopy(base["replay"]); changed["ledger_steps"][0]["attempts"][0]["scores"][0] = 999
    with patch.object(adapter.frozen()["certified_pgfp"], "_replay", return_value=changed):
        failures.append(dict(injected=True, scenario="replay_tamper", result=advisor.advise(s)))
    original = advisor.baseline
    def bad_baseline(s, lp, policy):
        result = original(s, lp, policy); result["placement"] = [99]; return result
    with patch.object(advisor, "baseline", side_effect=bad_baseline):
        failures.append(dict(injected=True, scenario="physical_tamper", result=advisor.advise(s)))
    with patch.dict(adapter.SOURCE_HASHES, {"m0_oracle": "0" * 64}):
        failures.append(dict(injected=True, scenario="source_mismatch", result=advisor.advise(s)))
    invalid = deepcopy(s); invalid["tasks"][0]["mandatory_unknown"] = True
    failures.append(dict(injected=True, scenario="invalid_input", result=advisor.advise(invalid)))
    oracle_faults = [oracle.audit(s, bad) for bad in ([99], [True], [0, 1], [[0, 1]])]
    oracle_faults.append(oracle.audit(s, [1], [[0, 0, 0], [0, 0, 0]]))
    manual = []
    ms = multi_snapshot(); a, _ = adapter.instance(ms)
    for prices in (np.zeros((2, 3)), np.array([[1., 0., 0.], [0., 0., 0.]])):
        lp = dict(shapes=[0, 1], prices=prices); replay = adapter.frozen()["certified_pgfp"]._replay(a, lp)
        manual.append(dict(label="MANUAL_FIXED_PRICE_TEST_ONLY_NOT_CERTIFIED", input=ms, prices=prices.tolist(), replay=replay, verification=advisor.verify_replay(ms, lp, replay)))
    files = {"fixture-runs.json": runs, "dual-witnesses.json": duals(), "failures.json": failures,
             "oracle-faults.json": oracle_faults, "manual-replay.json": manual}
    for name, obj in files.items():
        (outdir / name).write_bytes(schema.encoded(advisor.plain(obj)) + b"\n")
    manifest = dict(baseline=adapter.BASELINE, source_hashes=adapter.SOURCE_HASHES,
        engineer_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        python=sys.version, platform=platform.platform(),
        dependencies={n: importlib.metadata.version(n) for n in ("numpy", "scipy", "pytest", "setuptools", "accelsched")},
        mode="PROBE_ONLY_CPU_OFFLINE", fixture_cases=len(runs), native_lp_fixture_runs=4, injected_failure_runs=len(failures),
        denominator_statuses={status: sum(run["result"]["status"] == status for run in runs + failures) for status in sorted(schema.STATUSES)},
        files={str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [*FIXTURES, *sorted((ROOT / "src").rglob("*.py")), Path(__file__),
                         ROOT / "pyproject.toml", *[outdir / name for name in files]]})
    (outdir / "manifest.json").write_bytes(schema.encoded(manifest) + b"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    record_evidence()
