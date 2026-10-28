"""M7 analysis: baseline reproduction, PGFP failure attribution, Shape-LP price information and producer latency.

Pure analysis of the Go native records plus offline, GENUINE M3-A/original-LP probes. Nothing here edits a
price, injects a score or reads a held-out job as a forecast except where a field is explicitly labelled
OFFLINE_ORACLE_INFORMATION_NOT_NATIVE. The deterministic ``analysis`` document is byte-compared across fresh
processes; wall-clock numbers live only in the separate ``timing`` document.
"""
import argparse
import hashlib
import itertools
from pathlib import Path
import statistics
import sys

import numpy as np

from . import adapter, advisor, bounded_advice as ba, native_p2h_audit as npa, oracle, p2h_producer as base, policy_benefit as pb, schema
from .schema import InputError, digest, encoded

VERSION = "M7-ANALYSIS-1"
GRID = (0, 1, 2, 5, 10, 20, 40, 60, 80, 120)  # predeclared lambda grid for the same-state counterfactual
NATIVE_GRID_ARMS = {0: "pgfp_l0", 5: "pgfp_l5", 20: "pgfp_l20", 40: "pgfp_l40", 80: "pgfp_l80"}
M6_PUBLISHED_NORMALIZED_SHA256 = "8550b4ebc040b60ac5623a9f2807c2ad9cfb2014e1754bc38c1074172e7d84ca"
BF, BP = 4., 8.


def _need(cond, message):
    if not cond:
        raise InputError(message)


def decisions(rec):
    return [(s["chosen_node"], s["native_scores_S"]) for s in rec["steps"]], [(f["outcome"], f.get("node")) for f in rec["future"]], rec["future_admitted"]


def m6_arm(m6_scene, name):
    return m6_scene["arms"][name]


# ---- 1. baseline reproduction --------------------------------------------------------------------------------

def baseline(scenes, m6_scenes, m6_raw_sha, m6_expected):
    by6 = {s["id"]: s for s in m6_scenes}
    mapping = (("native_off", "native_off"), ("pgfp_l40", "native_dynamic_pgfp"), ("lp_mix_single", "native_dynamic_pgfp_lp"))
    mismatches = []
    for sc in scenes:
        for mine, theirs in mapping:
            a, b = sc["arms"][mine], by6[sc["id"]]["arms"][theirs]
            if decisions(a) != decisions(b):
                mismatches.append((sc["id"], mine))
    totals = {}
    for mine, theirs in mapping:
        admitted = sum(sc["arms"][mine]["future_admitted"] for sc in scenes)
        totals[mine] = dict(admitted=admitted, denominator=sum(sc["arms"][mine]["future_total"] for sc in scenes))
    pg, off = totals["pgfp_l40"]["admitted"], totals["native_off"]["admitted"]
    per = [sc["arms"]["pgfp_l40"]["future_admitted"] - sc["arms"]["native_off"]["future_admitted"] for sc in scenes]
    status = "M6_BASELINE_REPRODUCED" if m6_raw_sha == m6_expected and not mismatches else "M6_BASELINE_UNVERIFIED"
    return dict(status=status, m6_rerun_normalized_sha256=m6_raw_sha, m6_published_normalized_sha256=m6_expected, m6_rerun_byte_identical_to_published=m6_raw_sha == m6_expected,
                decision_mismatches_m7_vs_m6_rerun=mismatches, scenes=len(scenes), totals=totals, pgfp_vs_off=dict(improved=sum(d > 0 for d in per), unchanged=sum(d == 0 for d in per), harmed=sum(d < 0 for d in per), admitted_pgfp=pg, admitted_off=off),
                reproduced_m6_counts=(pg, off, totals["pgfp_l40"]["denominator"], sum(d > 0 for d in per), sum(d == 0 for d in per), sum(d < 0 for d in per)))


# ---- 2. attribution ------------------------------------------------------------------------------------------------

def _ranked(s0, advice, lam, names):
    return sorted(names, key=lambda n: (-(s0[n] + lam * advice[n]), names.index(n)))


def tie_rule_validated(scenes):
    """The recorded native order equals sort by (-score, ascending node index) at lambda=40 and at S0; else the rule is not used."""
    ok = total = 0
    for sc in scenes:
        for s in sc["arms"]["pgfp_l40"]["steps"]:
            names = list(s["candidates"])
            by_s = sorted(names, key=lambda n: (-s["native_scores_S"][n], names.index(n)))
            by_s0 = sorted(names, key=lambda n: (-s["native_scores_S0"][n], names.index(n)))
            total += 1
            ok += (by_s == s["native_order_S"]) and (by_s0 == s["native_order_S0"]) and s["chosen_node"] == s["native_order_S"][0]
    return dict(decisions=total, consistent=ok, validated=ok == total)


def same_state_counterfactual(scenes):
    """For every PGFP decision, re-rank the SAME native state with S0 + lambda*A for the predeclared grid (A = recorded certified advice)."""
    rows, flips = [], {str(l): 0 for l in GRID}
    margins = []
    for sc in scenes:
        for s in sc["arms"]["pgfp_l40"]["steps"]:
            names = list(s["candidates"])
            a = {p["node_id"]: p["score"] for p in s["artifact_pairs"]["pairs"]}
            s0 = s["native_scores_S0"]
            native_pick = _ranked(s0, a, 0, names)[0]
            ordered = sorted((s0[n] for n in names), reverse=True)
            margin = (ordered[0] - ordered[1]) if len(names) > 1 else None
            spread = max(a[n] for n in names) - min(a[n] for n in names)
            needed = None if (margin is None or spread == 0) else margin / spread
            picks = {str(l): _ranked(s0, a, l, names)[0] for l in GRID}
            for l in GRID:
                flips[str(l)] += picks[str(l)] != native_pick
            if margin is not None:
                margins.append(margin)
            rows.append(dict(scene=sc["id"], index=s["index"], candidates=len(names), native_pick=native_pick, native_margin=margin, advice_spread=spread,
                             min_lambda_to_flip=needed, recorded_pick_lambda40=s["chosen_node"], picks=picks))
    per_scene = {}
    for r in rows:
        per_scene.setdefault(r["scene"], []).append(r)
    return dict(decisions=len(rows), single_candidate_decisions=sum(r["candidates"] == 1 for r in rows), rank_flips_vs_native_pick=flips,
                native_margin_median=statistics.median(margins) if margins else None,
                decisions_with_zero_advice_spread=sum(r["advice_spread"] == 0 for r in rows),
                decisions_flippable_at_some_grid_lambda=sum(any(r["picks"][str(l)] != r["native_pick"] for l in GRID) for r in rows),
                min_lambda_to_flip_values=sorted({r["min_lambda_to_flip"] for r in rows if r["min_lambda_to_flip"] is not None})[:12],
                detail=rows)


def immediate_F(scene, residual, task):
    trace, names = npa.model_trace(scene)
    snap = pb._stage_snapshot(trace, residual, trace["arrivals"][0])
    return oracle.future_score(snap, residual)


def full_path_lambda(scenes):
    """State-diverging lambda sweep (native): paired future admission vs NATIVE_OFF per lambda."""
    out = {}
    for lam, arm in NATIVE_GRID_ARMS.items():
        per = [sc["arms"][arm]["future_admitted"] - sc["arms"]["native_off"]["future_admitted"] for sc in scenes]
        diverged = sum([s["chosen_node"] for s in sc["arms"][arm]["steps"]] != [s["chosen_node"] for s in sc["arms"]["native_off"]["steps"]] for sc in scenes)
        out[str(lam)] = dict(arm=arm, improved=sum(d > 0 for d in per), unchanged=sum(d == 0 for d in per), harmed=sum(d < 0 for d in per),
                             admitted=sum(sc["arms"][arm]["future_admitted"] for sc in scenes), admitted_off=sum(sc["arms"]["native_off"]["future_admitted"] for sc in scenes),
                             denominator=sum(sc["arms"][arm]["future_total"] for sc in scenes), scenes_with_diverged_placements=diverged,
                             admitted_gpus=sum(sc["arms"][arm]["future_admitted_gpus"] for sc in scenes), admitted_gpus_off=sum(sc["arms"]["native_off"]["future_admitted_gpus"] for sc in scenes))
    return out


def forecast_variants(scenes):
    out = {}
    for arm in ("pgfp_l40", "pgfp_l40_accurate_forecast", "pgfp_l40_clairvoyant_forecast"):
        by = {}
        for sc in scenes:
            regime = sc["id"].split(":")[0]
            d = by.setdefault(regime, dict(delta=0, scenes=0, harmed=0, improved=0))
            x = sc["arms"][arm]["future_admitted"] - sc["arms"]["native_off"]["future_admitted"]
            d["delta"] += x
            d["scenes"] += 1
            d["harmed"] += x < 0
            d["improved"] += x > 0
        out[arm] = dict(label="OFFLINE_ORACLE_INFORMATION_NOT_NATIVE" if "clairvoyant" in arm else "native_declared_or_distributionally_accurate_forecast", by_regime=by,
                        total_delta=sum(v["delta"] for v in by.values()), harmed=sum(v["harmed"] for v in by.values()), improved=sum(v["improved"] for v in by.values()))
    return out


def f_vs_admission(scenes):
    """Counterexamples where immediate surrogate F improves yet native future admission is worse."""
    found, checked, both_improve = [], 0, 0
    for sc in scenes:
        off = npa.audit_arm(sc["arms"]["native_off"])
        pg = npa.audit_arm(sc["arms"]["pgfp_l40"])
        checked += 1
        f_off, f_pg = off["predicted_surrogate_F_explanatory_only"], pg["predicted_surrogate_F_explanatory_only"]
        a_off, a_pg = sc["arms"]["native_off"]["future_admitted"], sc["arms"]["pgfp_l40"]["future_admitted"]
        if f_pg > f_off and a_pg < a_off:
            found.append(dict(scene=sc["id"], F_off=f_off, F_pgfp=f_pg, admitted_off=a_off, admitted_pgfp=a_pg,
                              placements_off=[s["chosen_node"] for s in sc["arms"]["native_off"]["steps"]], placements_pgfp=[s["chosen_node"] for s in sc["arms"]["pgfp_l40"]["steps"]],
                              held_out=[(j["gpus"], j["cores"], j["gib"]) for j in sc["arms"]["native_off"]["scene"]["held_out_future"]],
                              fragmentation_off=off["fragmentation"], fragmentation_pgfp=pg["fragmentation"]))
        both_improve += f_pg > f_off and a_pg > a_off
    return dict(status="FOUND" if found else "NOT_FOUND", scenes_checked=checked, counterexamples=found, f_and_admission_both_improve=both_improve,
                scope="the 31 declared scenes x the pgfp_l40 arm; F is the weighted surrogate fit count, not an outcome")


# ---- 3. genuine Shape-LP structural probes (offline) ---------------------------------------------------------------

def _cohort_demands(forecast):
    out = []
    for f in forecast:
        for _ in range(f["weight"]):
            if len(out) < 8:
                out.append(list(f["demand"]))
    return out


def _snapshot(nodes, tasks, forecast):
    names = ["node%d" % i for i in range(len(nodes))]
    ns = [dict(id=n, gpu_family=base.FAMILY, shape=s, capacity=cap) for n, (s, cap) in zip(names, nodes)]
    ts = [dict(id=i, demand=d, value=1 + 4 * d[0] + d[1] // 8, eligible=names[:], gpu_family=base.FAMILY) for i, d in tasks]
    fu = [dict(id=f["id"], demand=list(f["demand"]), weight=f["weight"], gpu_family=base.FAMILY, eligible=list(f["eligible"])) for f in forecast]
    return dict(schema_version="0.1", units=list(schema.UNITS), nodes=ns, tasks=ts, future=fu, realized_future=[], constraints={})


def _prices(snapshot, alpha=.5):
    r = ba.build_artifact(snapshot, BF=BF, BP=BP, alpha=alpha)
    _need(r["status"] == ba.STATUS, "genuine M3-A artifact must certify: " + r["status"])
    a = r["artifact"]
    return a, a["lp"]["prices"], a["lp"]["certificate"]["certified"]


def _target_pairs(a, tid):
    return [(p["node_id"], p["p"], p["AF"], p["AP"], p["score"]) for p in a["pairs"] if p["task_id"] == tid]


def lp_probe_on_m6_states(scenes):
    """Single Pending target vs legitimately declared cohorts, on every M6 decision state (offline, genuine M3-A)."""
    out = dict(decisions=0, single_nonzero=0, forecast_cohort_nonzero=0, queue_lookahead_nonzero=0, oracle_cohort_nonzero=0,
               forecast_cohort_target_price_differs_across_candidates=0, certified_all=True, examples=[])
    for sc in scenes:
        rec = sc["arms"]["pgfp_l40"]
        scene = rec["scene"]
        names = ["node%d" % i for i in range(len(scene["nodes"]))]
        shapes = ["G%dC%dM%d" % (n["gpus"], int(n["cores"]), int(n["mem_gib"])) for n in scene["nodes"]]
        for s in rec["steps"]:
            residual = npa.residual_from_native(s["before_resources"], names)
            req = [s["request"]["gpus"], s["request"]["cores"], s["request"]["gib"]]
            nodes = list(zip(shapes, residual))
            target = ("T", req)
            rest = [[j["gpus"], j["cores"], j["gib"]] for j in scene["arrivals"][s["index"] + 1:]]
            variants = {
                "single": [target],
                "forecast_cohort": [target] + [("plan:%d" % k, d) for k, d in enumerate(_cohort_demands(scene["forecast"]))],
                "queue_lookahead": [target] + [("queue:%d" % k, d) for k, d in enumerate(rest)],
                "oracle_cohort": [target] + [("oracle:%d" % k, [j["gpus"], j["cores"], j["gib"]]) for k, j in enumerate(scene["held_out_future"])],
            }
            out["decisions"] += 1
            for name, tasks in variants.items():
                a, pr, cert = _prices(_snapshot(nodes, tasks, scene["forecast"]))
                out["certified_all"] &= cert
                nz = any(v != 0 for row in pr for v in row)
                key = {"single": "single_nonzero", "forecast_cohort": "forecast_cohort_nonzero", "queue_lookahead": "queue_lookahead_nonzero", "oracle_cohort": "oracle_cohort_nonzero"}[name]
                out[key] += nz
                if name == "forecast_cohort":
                    ps = {round(p[1], 9) for p in _target_pairs(a, "T") if p[0] in s["candidates"]}
                    out["forecast_cohort_target_price_differs_across_candidates"] += len(ps) > 1
                    if nz and len(out["examples"]) < 5:
                        out["examples"].append(dict(scene=sc["id"], index=s["index"], prices=pr, shapes=sorted(set(shapes)), target_pairs=_target_pairs(a, "T")))
    return out


TYPES = {"small": [1, 2, 1], "mid": [2, 4, 2], "large": [4, 8, 4], "wide": [6, 8, 4], "cpuheavy": [1, 8, 8]}
POOL = [("G4C16M64", [4, 16, 64]), ("G8C32M128", [8, 32, 128]), ("G2C8M32", [2, 8, 32]), ("G8C16M64", [8, 16, 64])]


def lp_design_search():
    """Bounded exhaustive scarcity search over the M6 shape pool: node subsets x target x cohort multisets (<=4 jobs)."""
    forecast = [dict(id="F0", demand=[4, 4, 1], weight=1, eligible=[])]
    stats = dict(builds=0, nonzero=0, shape_differential=0, target_price_differs=0, certified_all=True, nonzero_not_differential=0)
    examples = []
    for k in (2, 3):
        for idx in itertools.combinations(range(4), k):
            nodes = [POOL[i] for i in idx]
            names = ["node%d" % i for i in range(len(nodes))]
            fc = [dict(forecast[0], eligible=names)]
            for tgt in ("small", "mid"):
                for size in range(1, 5):
                    for combo in itertools.combinations_with_replacement(sorted(TYPES), size):
                        tasks = [("T", TYPES[tgt])] + [("c%d" % i, TYPES[n]) for i, n in enumerate(combo)]
                        a, pr, cert = _prices(_snapshot([(s, cap) for s, cap in nodes], tasks, fc))
                        stats["builds"] += 1
                        stats["certified_all"] &= cert
                        if any(v != 0 for row in pr for v in row):
                            stats["nonzero"] += 1
                            differential = len({tuple(row) for row in pr}) > 1
                            stats["shape_differential"] += differential
                            stats["nonzero_not_differential"] += not differential
                            tp = {round(p[1], 9) for p in _target_pairs(a, "T")}
                            if len(tp) > 1:
                                stats["target_price_differs"] += 1
                                if len(examples) < 4:
                                    examples.append(dict(nodes=[s for s, _ in nodes], target=tgt, cohort=list(combo), prices=pr, target_pairs=_target_pairs(a, "T")))
    stats["status"] = "FOUND" if stats["target_price_differs"] else "NOT_FOUND"
    stats["examples"] = examples
    stats["scope"] = "node subsets of the 4-shape M6 pool (size 2-3) x target in {small,mid} x cohort multisets of 1-4 jobs from 5 types; one forecast item"
    return stats


def dual_nonuniqueness(cases=6):
    """For certified nonzero-price cohorts, re-solve the ORIGINAL canonical LP with two simplex/IPM backends and
    certify each dual independently with the frozen certificate; no price is edited."""
    from scipy.optimize import linprog
    modules = adapter.frozen()
    forecast = [dict(id="F0", demand=[4, 4, 1], weight=1, eligible=["node0", "node1", "node2"])]
    nodes = [POOL[0], POOL[1], POOL[2]]
    found = []
    for tgt in ("small", "mid"):
        for size in (2, 3, 4):
            for combo in itertools.combinations_with_replacement(sorted(TYPES), size):
                tasks = [("T", TYPES[tgt])] + [("c%d" % i, TYPES[n]) for i, n in enumerate(combo)]
                snap = _snapshot(nodes, tasks, forecast)
                a, pr, cert = _prices(snap)
                if not any(v != 0 for row in pr for v in row):
                    continue
                inst, _ = adapter.instance(snap)
                model, labels = modules["certified_pgfp"].canonical_shape(inst)
                row = dict(combo=list(combo), target=tgt, solver_prices=pr)
                for method in ("highs-ds", "highs-ipm"):
                    res = linprog(model.c, A_ub=model.A, b_ub=model.row_hi, bounds=(0, None), method=method)
                    raw = modules["lp_backends"].RawResult("independent_scipy_" + method, None, "OPTIMAL", native_status="independent re-solve", x=np.asarray(res.x),
                                                          pi=np.asarray(res.ineqlin.marginals), rc=np.asarray(res.lower.marginals), objective=float(res.fun),
                                                          dual_objective=float(res.fun), gap=0.)
                    _, _, check = advisor.certificate(inst, raw, model.source_hash)
                    prices = (-raw.pi[len(inst.value):]).reshape(len(labels), 3)
                    row[method] = dict(certified=bool(check["certified"]), prices=[[round(float(v), 9) for v in r] for r in prices])
                row["dual_nonunique"] = row["highs-ds"]["prices"] != row["highs-ipm"]["prices"]
                found.append(row)
                if len(found) == cases:
                    return dict(cases=found, nonunique=sum(r["dual_nonunique"] for r in found), all_alternatives_certified=all(r[m]["certified"] for r in found for m in ("highs-ds", "highs-ipm")),
                                note="no solver witness is invented: both duals come from independent scipy HiGHS solves of the ORIGINAL canonical model and are certified by the frozen certificate")
    return dict(cases=found, nonunique=sum(r["dual_nonunique"] for r in found), all_alternatives_certified=all(r[m]["certified"] for r in found for m in ("highs-ds", "highs-ipm")), note="fewer cases than requested")


# ---- 4. B1: legitimate nonzero LP versus the equal-PGFP-coefficient control ---------------------------------------------

def _target_ap(step):
    """Price-term values of the scoring target's own pairs only (a cohort artifact also scores the cohort members)."""
    return {p["node_id"]: p for p in step["artifact_pairs"]["pairs"] if p.get("task_id", step["task"]) == step["task"]}


def b1_chain(scenes):
    arms = ("lp_mix_cohort", "pc_alpha0_l20_cohort", "pgfp_l20", "lp_mix_single", "pgfp_l40", "lp_mix_cohort_oracle")
    if not scenes or any(a not in scenes[0]["arms"] for a in arms):
        return dict(status="NATIVE_NONZERO_LP_NOT_RUN")
    lam, alpha = 40., .5
    nonzero_decisions, chain, same_state, flips, differ_ranking, price_spread, top_penalised = 0, [], 0, 0, 0, 0, 0
    spreads, margins, nonzero_by_scene = [], [], {}
    ranking_examples = []
    for sc in scenes:
        lp, ctl = sc["arms"]["lp_mix_cohort"], sc["arms"]["pgfp_l20"]
        for sl, sc_ in zip(lp["steps"], ctl["steps"]):
            prices = sl["artifact_pairs"]["prices"]
            nz = any(v != 0 for row in prices for v in row)
            same = sl["before_domain_sha256"] == sc_["before_domain_sha256"]  # identical authoritative native state => a clean paired comparison
            if same:
                same_state += 1
                flips += sl["chosen_node"] != sc_["chosen_node"]
                differ_ranking += sl["native_order_S"] != sc_["native_order_S"]
            if not nz:
                continue
            nonzero_decisions += 1
            nonzero_by_scene[sc["id"]] = nonzero_by_scene.get(sc["id"], 0) + 1
            pairs = _target_ap(sl)
            names = list(sl["candidates"])
            ap = {n: pairs[n]["AP"] for n in names}
            spread = lam * alpha * (max(ap.values()) - min(ap.values()))
            ordered = sorted((sl["native_scores_S0"][n] for n in names), reverse=True)
            margin = (ordered[0] - ordered[1]) if len(names) > 1 else None
            if spread > 0:
                price_spread += 1
                spreads.append(spread)
                top = sl["native_order_S0"][0]
                top_penalised += ap[top] == min(ap.values()) and spread > 0
            if margin is not None:
                margins.append(margin)
            if same and sl["native_order_S"] != sc_["native_order_S"] and len(ranking_examples) < 4:
                ranking_examples.append(dict(scene=sc["id"], index=sl["index"], lp_order=sl["native_order_S"], control_order=sc_["native_order_S"], chosen=sl["chosen_node"],
                                             control_chosen=sc_["chosen_node"], price_term_spread_points=spread, native_margin=margin, shape_price_vectors=prices, shapes=sl["controls"]["shapes"],
                                             target_ap={n: pairs[n]["AP"] for n in names}))
            if len(chain) < 6 and spread > 0:
                chain.append(dict(scene=sc["id"], index=sl["index"], shape_price_vectors=prices, shapes=sl["controls"]["shapes"],
                                  target_pairs={n: dict(p=pairs[n]["p"], AF=pairs[n]["AF"], AP=pairs[n]["AP"], score=pairs[n]["score"]) for n in names}, S0=sl["native_scores_S0"], S=sl["native_scores_S"],
                                  order=sl["native_order_S"], control_order=sc_["native_order_S"], chosen=sl["chosen_node"], control_chosen=sc_["chosen_node"], price_term_spread_points=spread, native_margin=margin,
                                  chain="lambda_shape -> p=-lambda_shape.d -> AP=clip(p/BP) -> S0+lambda*((1-alpha)AF+alpha AP) -> OrderedNodesByTask -> Allocate/Bind"))

    def tally(a, b):
        d = [sc["arms"][a]["future_admitted"] - sc["arms"][b]["future_admitted"] for sc in scenes]
        return dict(improved=sum(x > 0 for x in d), unchanged=sum(x == 0 for x in d), harmed=sum(x < 0 for x in d), admitted_a=sum(sc["arms"][a]["future_admitted"] for sc in scenes),
                    admitted_b=sum(sc["arms"][b]["future_admitted"] for sc in scenes), denominator=sum(sc["arms"][a]["future_total"] for sc in scenes))
    placements_equal = sum([s["chosen_node"] for s in sc["arms"]["lp_mix_cohort"]["steps"]] == [s["chosen_node"] for s in sc["arms"]["pgfp_l20"]["steps"]] for sc in scenes)
    adm = tally("lp_mix_cohort", "pgfp_l20")
    if nonzero_decisions == 0:
        verdict = "LP_NO_INCREMENTAL_BENEFIT"
    elif flips == 0 and placements_equal == len(scenes) and adm["improved"] == adm["harmed"] == 0:
        verdict = "LP_NONZERO_NO_DECISION_GAIN" if differ_ranking == 0 else "LP_NONZERO_RANKING_CHANGE_NO_CHOICE_OR_ADMISSION_GAIN"
    else:
        verdict = "LP_DECISION_EFFECT_OBSERVED"
    return dict(status="B1_EXECUTED", verdict=verdict, scenes=len(scenes), decisions_with_nonzero_cohort_price=nonzero_decisions, scenes_with_nonzero_price=len(nonzero_by_scene),
                decisions_with_shape_differential_target_price_term=price_spread, decisions_where_the_native_top_candidate_carries_the_lowest_price_term=top_penalised,
                price_term_spread_points_median=statistics.median(spreads) if spreads else None, native_margin_median_at_nonzero_decisions=statistics.median(margins) if margins else None,
                same_state_decisions=same_state, native_choice_flips_vs_equal_coefficient_control=flips, native_ranking_differences_vs_control=differ_ranking,
                scenes_with_identical_placements=placements_equal, ranking_difference_examples=ranking_examples, chain_examples=chain,
                equal_coefficient=dict(control="pgfp_l20 (alpha=0, lambda=20)", treatment="lp_mix_cohort (alpha=.5, lambda=40)", pgfp_coefficient_control=20, pgfp_coefficient_treatment=20, admission=adm),
                alpha0_pipeline_control=tally("pc_alpha0_l20_cohort", "pgfp_l20"),
                legacy_blend=dict(single_target=tally("lp_mix_single", "pgfp_l40"), cohort_vs_single_blend=tally("lp_mix_cohort", "lp_mix_single")),
                cohort_vs_native_off=tally("lp_mix_cohort", "native_off"),
                oracle_cohort_diagnostic=dict(label="OFFLINE_ORACLE_INFORMATION_NOT_NATIVE", vs_equal_coefficient_control=tally("lp_mix_cohort_oracle", "pgfp_l20")))


# ---- 5. B2 latency ------------------------------------------------------------------------------------------------------

def _q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]


def _dist(v):
    return dict(n=len(v), median=statistics.median(v), p10=_q(v, .1), p90=_q(v, .9), min=min(v), max=max(v))


def latency(timings, native_timings):
    one, worker = timings["cold_oneshot"], timings["worker"]
    ns = 1e-9
    def stage_dist(rows, key):
        return _dist([r["stages"][key] * ns for r in rows if key in r["stages"]])
    out = dict(label="EXPERIMENT_ONLY_NOT_WIRED_TO_PLUGIN", note="few fixtures: no throughput/p99 claim; OS page cache cannot be purged without privileges so 'cold' means a fresh process, not a cold disk")
    out["cold_one_process_per_task"] = dict(wall=_dist([r["wall_seconds"] for r in one]), first_run_in_study_wall=one[0]["wall_seconds"],
        warm_repeat_wall=_dist([r["wall_seconds"] for r in one if r["rep"] > 0]),
        spawn_to_first_python_line=stage_dist(one, "spawn_to_first_line_ns"), imports=stage_dist(one, "import_ns"),
        snapshot_projection_check=stage_dist(one, "snapshot"), build_incl_original_lp_solve_and_certificate=stage_dist(one, "build"),
        lp_solve_nested_in_build=_dist([r["stages"]["lp_solve_s"] for r in one]), load_full_recertification=stage_dist(one, "load"), other_validation_serialization=stage_dist(one, "other"),
        python_compute_total=stage_dist(one, "total"))
    residual = [r["wall_seconds"] - (r["stages"]["spawn_to_first_line_ns"] + r["stages"]["import_ns"] + r["stages"]["total"]) * ns for r in one]
    out["cold_one_process_per_task"]["interpreter_exit_pipe_and_output_residual"] = _dist(residual)
    out["long_lived_worker"] = dict(wall=_dist([r["wall_seconds"] for r in worker]), first_request_after_start_wall=worker[0]["wall_seconds"],
        warm_wall=_dist([r["wall_seconds"] for r in worker if r["order"] > 0 or r["rep"] > 0]),
        boot=timings.get("worker_boot"), snapshot_projection_check=stage_dist(worker, "snapshot"), build_incl_original_lp_solve_and_certificate=stage_dist(worker, "build"),
        lp_solve_nested_in_build=_dist([r["stages"]["lp_solve_s"] for r in worker]), load_full_recertification=stage_dist(worker, "load"),
        other_validation_serialization=stage_dist(worker, "other"), python_compute_total=stage_dist(worker, "total"),
        transit_and_json_overhead=_dist([r["wall_seconds"] - r["stages"]["total"] * ns for r in worker]))
    cm, wm = out["cold_one_process_per_task"]["wall"]["median"], out["long_lived_worker"]["warm_wall"]["median"]
    out["per_task_producer_wall_speedup_median"] = cm / wm if wm else None
    cold_build, warm_build = out["cold_one_process_per_task"]["build_incl_original_lp_solve_and_certificate"]["median"], out["long_lived_worker"]["build_incl_original_lp_solve_and_certificate"]["median"]
    first = worker[0]["stages"]["build"] * ns
    out["first_call_initialisation_inside_build"] = dict(
        cold_process_build_median=cold_build, warm_worker_build_median=warm_build, difference=cold_build - warm_build, first_worker_request_build=first,
        note="lazy first-call imports/initialisation (scipy.optimize, HiGHS) land inside build in a fresh process; the nested true LP solve is milliseconds, so most of the cold 'build' is initialisation, not optimisation")
    # end-to-end per scene from the matched native arms: cold transport vs worker transport (identical decisions)
    pairs = {}
    for row in native_timings:
        if row["arm"] in ("pgfp_l40", "pgfp_l40_worker"):
            pairs.setdefault(row["scene"], {})[row["arm"]] = row
    e2e = {}
    for arm, label in (("pgfp_l40", "cold_per_task_process"), ("pgfp_l40_worker", "worker")):
        rows = [p[arm]["seconds"] for p in pairs.values() if arm in p]
        tot = [sum(r.values()) for r in rows]
        e2e[label] = dict(scenes=len(rows), end_to_end_per_scene=_dist(tot),
                          stage_medians={k: statistics.median(r[k] for r in rows) for k in rows[0]},
                          producer_exclusive_of_lp_per_scene=_dist([r["Producer"] - r["LPSolve"] for r in rows]), lp_solve_per_scene=_dist([r["LPSolve"] for r in rows]))
    e2e["note"] = "Producer is inclusive of the nested LP solve; 'producer_exclusive_of_lp' subtracts it so no time is counted twice. Fixture is the Fake Session build."
    out["end_to_end_matched_arms"] = e2e
    return out


# ---- driver ---------------------------------------------------------------------------------------------------------------

def analyse(m7, m6, m6_sha, latency_norm):
    scenes = m7["scenes"]
    out = dict(version=VERSION, scope="whole-GPU multi-node native Fake Session (simulated GPUs); offline fields are secondary, probes are genuine M3-A/original LP, oracle fields are not deployable")
    out["baseline"] = baseline(scenes, m6["scenes"], m6_sha, M6_PUBLISHED_NORMALIZED_SHA256)
    out["attribution"] = dict(tie_rule=tie_rule_validated(scenes), same_state=same_state_counterfactual(scenes), full_path_lambda=full_path_lambda(scenes),
                              forecast_variants=forecast_variants(scenes), f_vs_admission=f_vs_admission(scenes))
    out["lp_structure"] = dict(m6_states=lp_probe_on_m6_states(scenes), design_search=lp_design_search(), dual_nonuniqueness=dual_nonuniqueness())
    out["b1"] = b1_chain(scenes) if m7.get("cohort_arms_run") else dict(status="NATIVE_NONZERO_LP_NOT_RUN", reason=m7.get("b1_status"))
    out["trust_boundary"] = m7.get("trust_boundary")
    out["b2_normalized"] = latency_norm
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--m6", type=Path, required=True, help="normalized.json of a fresh TestM6P2H rerun")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing-out", type=Path, required=True)
    args = parser.parse_args(argv)
    norms = [(d / "normalized.json").read_bytes() for d in args.run_dirs]
    lats = [(d / "latency_normalized.json").read_bytes() for d in args.run_dirs]
    identical = all(n == norms[0] for n in norms) and all(l == lats[0] for l in lats)
    m6_raw = args.m6.read_bytes()
    report = analyse(schema.loads(norms[0]), schema.loads(m6_raw), hashlib.sha256(m6_raw).hexdigest(), schema.loads(lats[0]))
    audits = {}
    m7 = schema.loads(norms[0])
    for sc in m7["scenes"]:
        audits[sc["id"]] = {arm: npa.audit_arm(rec)["status"] for arm, rec in sc["arms"].items()}
    report["audit_all_arms"] = dict(arm_records=sum(len(v) for v in audits.values()), failures=sorted((s, a) for s, v in audits.items() for a, st in v.items() if st != "AUDIT_PASS"))
    report.update(runs=len(norms), normalized_sha256=[hashlib.sha256(n).hexdigest() for n in norms], latency_normalized_sha256=[hashlib.sha256(l).hexdigest() for l in lats],
                  byte_identical_across_runs=identical)
    report["analysis_sha256"] = digest(report)
    args.out.write_bytes(encoded(report))
    tdoc = {}
    for i, d in enumerate(args.run_dirs):
        import json
        lt = json.loads((d / "latency_timings.json").read_text())
        nt = json.loads((d / "timings.json").read_text())
        tdoc["run%d" % (i + 1)] = latency(lt, nt)
    import json
    args.timing_out.write_text(json.dumps(tdoc, indent=1, sort_keys=True))
    ok = identical and not report["audit_all_arms"]["failures"] and report["baseline"]["status"] == "M6_BASELINE_REPRODUCED"
    sys.stdout.write(report["analysis_sha256"] + ("\n" if ok else " FAIL\n"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
