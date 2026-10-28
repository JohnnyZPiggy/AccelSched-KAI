"""M7 analysis logic on synthetic native-shaped records and on genuine M3-A probes (no Go, no Kubernetes)."""
from copy import deepcopy

import pytest

from accelsched_kai import m7_analysis as m7, p2h_producer as base, schema


def step(i, chosen, s0, advice, order=None, domain="d0", prices=None):
    names = sorted(s0)
    s = {n: s0[n] + 40 * advice.get(n, 0) for n in names}
    rank = lambda sc: sorted(names, key=lambda n: (-sc[n], names.index(n)))
    return dict(index=i, task="native-job-%d" % i, candidates=names, chosen_node=chosen, native_scores_S0=s0, native_scores_S=s, native_order_S0=rank(s0), native_order_S=order or rank(s),
                before_domain_sha256=domain, artifact_pairs=dict(pairs=[dict(node_id=n, score=advice.get(n, 0), p=0, AF=advice.get(n, 0), AP=0) for n in names],
                                                                 prices=prices or [[0, 0, 0]]), controls=dict(shapes=["S"]))


def arm(chosen, admitted, steps=None, total=3):
    return dict(steps=steps or [step(0, chosen, {"node0": 1000, "node1": 1009}, {"node0": 0, "node1": -0.5})], future=[dict(outcome="ACCEPTED", node="node0")] * admitted,
                future_admitted=admitted, future_total=total, future_admitted_gpus=admitted, future_admitted_cores=admitted)


def test_tie_rule_and_same_state_counterfactual_flip_thresholds():
    s0, a = {"node0": 1000, "node1": 1009}, {"node0": 0, "node1": -0.5}
    chosen = m7._ranked(s0, a, 40, ["node0", "node1"])[0]
    assert chosen == "node0" and m7._ranked(s0, a, 0, ["node0", "node1"])[0] == "node1"
    # at lambda = 18 the totals tie (1000 vs 1000) and the lowest node index wins; just below it the native pick survives
    assert m7._ranked(s0, a, 17.9999, ["node0", "node1"])[0] == "node1" and m7._ranked(s0, a, 18, ["node0", "node1"])[0] == "node0"
    scene = {"id": "matched:0", "arms": {"pgfp_l40": arm("node0", 1)}}
    assert m7.tie_rule_validated([scene])["validated"]
    cf = m7.same_state_counterfactual([scene])
    r = cf["detail"][0]
    assert r["native_margin"] == 9 and r["advice_spread"] == 0.5 and r["min_lambda_to_flip"] == 18.0
    assert cf["rank_flips_vs_native_pick"]["0"] == 0 and cf["rank_flips_vs_native_pick"]["20"] == 1 and cf["rank_flips_vs_native_pick"]["5"] == 0


def test_single_candidate_decisions_have_no_margin_and_cannot_flip():
    only = step(0, "node0", {"node0": 1000}, {"node0": -0.5})
    scene = {"id": "x:0", "arms": {"pgfp_l40": arm("node0", 1, [only])}}
    cf = m7.same_state_counterfactual([scene])
    assert cf["detail"][0]["native_margin"] is None and cf["single_candidate_decisions"] == 1 and all(v == 0 for v in cf["rank_flips_vs_native_pick"].values())


def scene_with(name, off, pg):
    return {"id": name, "arms": {"native_off": off, "pgfp_l40": pg}}


def test_baseline_reproduction_and_unverified_when_hashes_or_decisions_differ():
    sc = {"id": "matched:0", "arms": {"native_off": arm("node1", 2), "pgfp_l40": arm("node0", 1), "lp_mix_single": arm("node0", 1)}}
    m6 = {"id": "matched:0", "arms": {"native_off": arm("node1", 2), "native_dynamic_pgfp": arm("node0", 1), "native_dynamic_pgfp_lp": arm("node0", 1)}}
    ok = m7.baseline([sc], [m6], m7.M6_PUBLISHED_NORMALIZED_SHA256, m7.M6_PUBLISHED_NORMALIZED_SHA256)
    assert ok["status"] == "M6_BASELINE_REPRODUCED" and ok["pgfp_vs_off"]["harmed"] == 1
    assert m7.baseline([sc], [m6], "0" * 64, m7.M6_PUBLISHED_NORMALIZED_SHA256)["status"] == "M6_BASELINE_UNVERIFIED"
    bad = deepcopy(m6)
    bad["arms"]["native_dynamic_pgfp"] = arm("node1", 1)
    assert m7.baseline([sc], [bad], m7.M6_PUBLISHED_NORMALIZED_SHA256, m7.M6_PUBLISHED_NORMALIZED_SHA256)["decision_mismatches_m7_vs_m6_rerun"] == [("matched:0", "pgfp_l40")]


def b1_scenes(price=None, flip=False, gain=False):
    prices = price or [[0, 0, 0]]
    base_arm = lambda c, adm: arm(c, adm, [step(0, c, {"node0": 1000, "node1": 1009}, {"node0": 0, "node1": -0.5}, prices=prices if c == "x" else None)])
    sc = {"id": "matched:0", "arms": {k: arm("node0", 1) for k in ("native_off", "pgfp_l40", "lp_mix_single", "pgfp_l20", "pc_alpha0_l20_cohort", "lp_mix_cohort", "lp_mix_cohort_oracle")}}
    sc["arms"]["lp_mix_cohort"]["steps"][0]["artifact_pairs"]["prices"] = prices
    if flip:
        sc["arms"]["lp_mix_cohort"]["steps"][0]["chosen_node"] = "node1"
        sc["arms"]["lp_mix_cohort"]["steps"][0]["native_order_S"] = ["node1", "node0"]
    if gain:
        sc["arms"]["lp_mix_cohort"]["future_admitted"] = 2
    return [sc]


def test_b1_verdicts_zero_price_nonzero_without_gain_and_effect():
    assert m7.b1_chain(b1_scenes())["verdict"] == "LP_NO_INCREMENTAL_BENEFIT"
    nz = m7.b1_chain(b1_scenes(price=[[4.25, 0, 0]]))
    assert nz["verdict"] == "LP_NONZERO_NO_DECISION_GAIN" and nz["decisions_with_nonzero_cohort_price"] == 1
    assert nz["equal_coefficient"]["pgfp_coefficient_control"] == nz["equal_coefficient"]["pgfp_coefficient_treatment"] == 20
    # a ranking change below the top that leaves every choice and admission unchanged is its own, weaker, verdict
    order_only = b1_scenes(price=[[4.25, 0, 0]])
    order_only[0]["arms"]["lp_mix_cohort"]["steps"][0]["native_order_S"] = ["node1", "node0"]
    order_only[0]["arms"]["pgfp_l20"]["steps"][0]["native_order_S"] = ["node0", "node1"]
    assert m7.b1_chain(order_only)["verdict"] == "LP_NONZERO_RANKING_CHANGE_NO_CHOICE_OR_ADMISSION_GAIN"
    assert m7.b1_chain(b1_scenes(price=[[4.25, 0, 0]], flip=True))["verdict"] == "LP_DECISION_EFFECT_OBSERVED"
    assert m7.b1_chain(b1_scenes(price=[[4.25, 0, 0]], gain=True))["verdict"] == "LP_DECISION_EFFECT_OBSERVED"
    assert m7.b1_chain([{"id": "x:0", "arms": {"native_off": arm("node0", 1)}}])["status"] == "NATIVE_NONZERO_LP_NOT_RUN"


def test_f_vs_admission_finds_and_reports_the_scope():
    from test_native_p2h_audit import SCENE, scene_record
    sc = scene_record()
    out = m7.f_vs_admission([dict(id=sc["id"], arms={"native_off": sc["arms"]["native_off"], "pgfp_l40": sc["arms"]["native_dynamic_pgfp"]})])
    assert out["status"] in ("FOUND", "NOT_FOUND") and out["scenes_checked"] == 1 and "not an outcome" in out["scope"]
    for c in out["counterexamples"]:
        assert c["F_pgfp"] > c["F_off"] and c["admitted_pgfp"] < c["admitted_off"]


def test_latency_distribution_decomposes_without_double_counting():
    stages = lambda total: dict(spawn_to_first_line_ns=1e8, import_ns=5e8, snapshot=1e6, build=3e7, load=2e7, other=1e6, total=total, lp_solve_s=0.01)
    one = [dict(request=i, rep=r, wall_seconds=0.8, stages=stages(5.2e7)) for i in range(2) for r in range(3)]
    worker = [dict(request=i, rep=r, order=i, wall_seconds=0.06, stages=stages(5.2e7)) for i in range(2) for r in range(2)]
    native = [dict(arm="pgfp_l40", scene="a", seconds=dict(Producer=3.0, LPSolve=0.05, Fixture=.1, Allocate=.001)), dict(arm="pgfp_l40_worker", scene="a", seconds=dict(Producer=.3, LPSolve=0.05, Fixture=.1, Allocate=.001))]
    out = m7.latency(dict(cold_oneshot=one, worker=worker, worker_boot={}), native)
    assert abs(out["per_task_producer_wall_speedup_median"] - 0.8 / 0.06) < 1e-9
    e2e = out["end_to_end_matched_arms"]
    assert abs(e2e["cold_per_task_process"]["producer_exclusive_of_lp_per_scene"]["median"] - 2.95) < 1e-9
    assert out["cold_one_process_per_task"]["interpreter_exit_pipe_and_output_residual"]["median"] == pytest.approx(0.8 - (1e8 + 5e8 + 5.2e7) * 1e-9)
    init = out["first_call_initialisation_inside_build"]
    assert init["difference"] == pytest.approx(init["cold_process_build_median"] - init["warm_worker_build_median"]) and "initialisation" in init["note"]


def test_genuine_lp_probe_reports_certified_nonzero_cohort_price_and_single_target_zero():
    nodes = [("G4C16M64", [4, 16, 64]), ("G8C32M128", [8, 32, 128])]
    fc = [dict(id="F0", demand=[4, 4, 1], weight=2, eligible=["node0", "node1"])]
    single = m7._snapshot(nodes, [("T", [1, 2, 1])], fc)
    scarce = m7._snapshot(nodes, [("T", [1, 2, 1]), ("c0", [8, 8, 4]), ("c1", [4, 4, 2])], fc)
    a0, p0, c0 = m7._prices(single)
    a1, p1, c1 = m7._prices(scarce)
    assert c0 and c1 and all(v == 0 for row in p0 for v in row) and any(v != 0 for row in p1 for v in row)
    assert [x[0] for x in m7._target_pairs(a1, "T")] == ["node0", "node1"]


def test_independent_dual_resolves_are_certified_by_the_frozen_certificate():
    out = m7.dual_nonuniqueness(cases=1)
    assert out["cases"] and out["all_alternatives_certified"] and "no solver witness is invented" in out["note"]


def test_price_term_is_read_from_the_targets_own_pairs_not_the_cohort_members():
    sc = b1_scenes(price=[[4.25, 0, 0]])
    s = sc[0]["arms"]["lp_mix_cohort"]["steps"][0]
    s["artifact_pairs"]["pairs"] = [dict(task_id="native-job-0", node_id="node0", score=0, p=-1.0, AF=0, AP=-0.2), dict(task_id="native-job-0", node_id="node1", score=0, p=-2.0, AF=0, AP=-0.6),
                                    dict(task_id="plan:0", node_id="node0", score=0, p=-9.0, AF=0, AP=-1.0), dict(task_id="plan:0", node_id="node1", score=0, p=-9.0, AF=0, AP=-1.0)]
    out = m7.b1_chain(sc)
    assert out["decisions_with_shape_differential_target_price_term"] == 1 and abs(out["price_term_spread_points_median"] - 20 * 0.4) < 1e-9
