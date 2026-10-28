"""M4-A acceptance G1-G6: offline multi-task policy-benefit falsification (CPU, PROBE_ONLY)."""
from copy import deepcopy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from accelsched_kai import adapter, bounded_advice, oracle, policy_benefit as pb, schema

SRC = str(Path(__file__).resolve().parents[1] / "src")


def scene(regime="matched", k=0):
    return pb.make_scene(regime, k)


@pytest.fixture(scope="module")
def small():
    return pb.experiment(scenes_per_regime=1)


def test_G1_identity_schema_and_common_initial_state():
    assert adapter.BASELINE == "9dc0c263d2aefeb151d565166b6b950e2ac9f6a5"
    adapter.frozen()  # raises on any frozen-blob mismatch
    for regime in pb.REGIMES:
        trace = scene(regime, 3)
        schema.validate(pb._full(trace)); schema.validate(pb._full(trace, realized=True))
        initial = pb.digest([n["capacity"] for n in trace["nodes"]])
        record = pb.run_scene(trace)
        assert list(record["arms"]) == list(pb.ARMS)
        assert all(a["steps"][0]["before_sha256"] == initial for a in record["arms"].values())


def test_G1_scene_generation_is_pure_and_outcome_independent():
    assert scene("shift_small_to_large", 5) == scene("shift_small_to_large", 5)
    base = scene("matched", 2)
    other = pb.make_scene("matched", 2)
    other["realized"] = []
    assert base["forecast"] == other["forecast"] and base["arrivals"] == other["arrivals"]


def test_G2_physical_validity_and_stepwise_conservation(small):
    assert small["all_arms_physically_valid"]
    for s in small["scenes"]:
        for arm in s["arms"].values():
            assert arm["status"] == "PHYSICAL_VALID" and arm["errors"] == []
            for before, step in zip([None] + arm["steps"], arm["steps"]):
                assert all(x >= 0 for row in step["after"] for x in row)


def test_G2_receipt_checker_rejects_violations():
    trace = scene()
    nodes, task = trace["nodes"], trace["arrivals"][0][0]
    before = [n["capacity"][:] for n in nodes]
    good = [r[:] for r in before]; good[0] = [a - b for a, b in zip(good[0], task["demand"])]
    assert pb._receipt_check(nodes, task, before, 0, good) == []
    leaked = [r[:] for r in good]; leaked[1][0] -= 1
    assert pb._receipt_check(nodes, task, before, 0, leaked)
    assert pb._receipt_check(nodes, task, before, -1, good)  # rejection must leave residual unchanged
    narrow = deepcopy(task); narrow["eligible"] = [nodes[1]["id"]]
    assert "ineligible acceptance" in pb._receipt_check(nodes, narrow, before, 0, good)


def test_G3_no_hindsight_realized_never_changes_decisions():
    trace = scene("shift_small_to_large", 4)
    mutated = deepcopy(trace)
    mutated["realized"] = [dict(r, demand=[1, 1, 1], id=r["id"]) for r in trace["realized"]]
    for arm in pb.ARMS:
        a, b = pb.run_arm(trace, arm), pb.run_arm(mutated, arm)
        assert a["placement"] == b["placement"] and a["steps"] == b["steps"]
    for batch in trace["arrivals"]:
        assert pb._stage_snapshot(trace, [n["capacity"] for n in trace["nodes"]], batch)["realized_future"] == []


def test_G3_static_is_initial_only_dynamic_refreshes_and_provenance_is_immutable():
    trace = scene("scarce_matched", 1)
    static, safe, dynamic = (pb.run_arm(trace, a) for a in (pb.STATIC, pb.STATIC_SAFE, pb.DYNAMIC_LP))
    assert len({s["artifact_sha256"] for s in static["stages"]}) == 1
    assert static["stages"][0]["state_is_initial"]
    assert all(s["advice"] == "NEUTRAL_AFTER_MUTATION" for s in safe["stages"] if not s["state_is_initial"])
    residual = [n["capacity"] for n in trace["nodes"]]
    expected, index = [], 0
    for s, batch in enumerate(trace["arrivals"]):
        expected.append(pb.digest(residual if s == 0 else dynamic["steps"][index - 1]["after"]))
        index += len(batch)
    assert [s["residual_sha256"] for s in dynamic["stages"]] == expected
    advice = pb._static_advice(trace, pb.CONTROLS)
    assert advice and hashlib.sha256(advice["raw"]).hexdigest() == advice["raw_sha256"]


def test_G3_uncertified_output_is_rejected_and_stage_goes_neutral():
    trace = scene("matched", 0)
    with patch.object(bounded_advice, "build_artifact", return_value=dict(status="LP_UNCERTIFIED", error="forced")):
        record = pb.run_arm(trace, pb.DYNAMIC_LP)
    assert record["status"] == "PHYSICAL_VALID"
    assert all(s["advice"] == "NEUTRAL_UNCERTIFIED" and s["lp"] == dict(status="LP_UNCERTIFIED") for s in record["stages"])
    with patch.object(bounded_advice, "build_artifact", return_value=dict(status="LP_UNCERTIFIED")):
        assert pb.run_arm(trace, pb.STATIC)["status"] == "STATIC_ADVICE_UNCERTIFIED"


def test_G4_upstream_marginal_matches_independent_formula_and_tie_rule():
    trace = scene("matched", 6)
    stage = pb._stage_snapshot(trace, [n["capacity"] for n in trace["nodes"]], trace["arrivals"][0])
    scores = pb._marginal_scores(stage, pb.CONTROLS)
    assert scores and all(-1 <= v <= 0 for v in scores.values())
    nodes = [dict(id="A", gpu_family="x", capacity=[4, 4, 4]), dict(id="B", gpu_family="x", capacity=[4, 4, 4])]
    task = dict(id="T", demand=[1, 1, 1], eligible=["A", "B"], gpu_family=None)
    tie = {("T", "A"): -.5, ("T", "B"): -.5}
    assert pb._pick(pb.DYNAMIC, task, nodes, [[4, 4, 4], [4, 4, 4]], tie) == 0
    assert pb._pick(pb.DYNAMIC, task, nodes, [[4, 4, 4], [4, 4, 4]], {("T", "A"): -.5, ("T", "B"): -.1}) == 1
    assert pb._pick(pb.DYNAMIC, task, nodes, [[0, 0, 0], [0, 0, 0]], tie) == -1


def test_G4_constant_price_within_shape_is_identified(small):
    for case in small["same_shape"].values():
        assert case["status"] == bounded_advice.STATUS and case["constant_within_every_shape"]
    assert small["same_shape"]["single_shape"]["price_by_task_and_shape"]["T0"].keys() == {"S4"}
    lp_stages = [st for s in small["scenes"] for st in s["arms"][pb.DYNAMIC_LP]["stages"] if "lp" in st]
    assert lp_stages and all(st["lp"]["constant_price_groups"] <= st["lp"]["price_groups"] for st in lp_stages if st["lp"]["status"] == bounded_advice.STATUS)


def test_G5_paired_outcomes_exact_denominators_and_single_post_freeze_evaluation():
    trace = scene("shift_small_to_large", 2)
    calls = []
    real = oracle.evaluate
    with patch.object(oracle, "evaluate", side_effect=lambda snap, rest: calls.append(snap["realized_future"]) or real(snap, rest)):
        record = pb.run_scene(trace)
    assert len(calls) == len(pb.ARMS) and all(len(c) == len(trace["realized"]) for c in calls)
    for name, p in record["paired"].items():
        arm, ref = record["arms"][name], record["arms"][pb.PRIMARY_REFERENCE]
        d = p["sequential"]
        assert d["denominator"] == len(trace["realized"]) == arm["outcome"]["number_future"]
        assert d["arm_numerator"] - d["reference_numerator"] == d["delta_admission_rate_num"]
        assert d["arm_numerator"] == arm["outcome"]["sequential"]["count"] and d["reference_numerator"] == ref["outcome"]["sequential"]["count"]
        assert p["F_difference"] == arm["metrics"]["F"] - ref["metrics"]["F"]


def test_G5_counterexample_search_is_bounded_deterministic_and_reports_status(small):
    c = small["counterexample_search"]
    assert c["instances"] == 9 * 64 * 3 * 4 and c["status"] in ("FOUND", "NOT_FOUND")
    assert c["admissions_dynamic_greater"] + c["admissions_equal"] + c["admissions_dynamic_fewer"] == c["instances"]
    assert (c["status"] == "FOUND") == (c["matches"] > 0) and len(c["first_witnesses"]) == min(3, c["matches"])
    for w in c["first_witnesses"]:
        assert w["F_dynamic"] > w["F_reference"] and w["admitted_dynamic"] <= w["admitted_reference"]
    assert c["matches_with_strictly_fewer_admissions"] <= c["matches"]


def test_G5_shift_scenes_and_original_policy_vs_proxy_labels_stay_distinct(small):
    assert set(small["aggregate"]) == set(pb.REGIMES)
    assert {pb.DYNAMIC, pb.DYNAMIC_LP, pb.STATIC} <= set(small["aggregate"]["matched"])
    for regime in small["witnesses"].values():
        for w in regime.values():
            assert set(w) == {"harmed_count", "harmed_first", "unchanged_count", "unchanged_first"}
    assert "Native KAI" not in "".join(pb.ARMS) and small["claim_scope"].startswith("offline simulation only")


def test_G6_in_process_repeat_is_byte_identical(small):
    again = pb.experiment(scenes_per_regime=1)
    assert schema.encoded(again) == schema.encoded(small)
    assert "timing" not in json.dumps(small).lower()


def test_G6_fresh_processes_are_byte_identical(tmp_path):
    digests = []
    env = dict(os.environ, PYTHONPATH=SRC)
    for i in range(2):
        out = tmp_path / ("run" + str(i))
        done = subprocess.run([sys.executable, "-m", "accelsched_kai.policy_benefit", "--out", str(out),
                               "--scenes-per-regime", "1"], capture_output=True, text=True, env=env, cwd=tmp_path)
        assert done.returncode == 0, done.stderr
        digests.append(hashlib.sha256((out / "normalized.json").read_bytes()).hexdigest())
        assert done.stdout.strip() == digests[-1]
        assert (out / "timings.json").exists()
    assert digests[0] == digests[1]


def test_frozen_sources_unchanged_after_experiment(small):
    assert adapter.frozen() and small["identity"]["frozen_source_sha256"] == adapter.SOURCE_HASHES
    assert small["record_sha256"] == schema.digest({k: v for k, v in small.items() if k != "record_sha256"})
