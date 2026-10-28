"""M4-A: offline multi-task policy-benefit falsification (PROBE_ONLY, CPU).

This is NOT a native KAI benchmark. Every arm below is an offline simulation over one
finite arrival trace. Forecasts (the `future` catalogue) never contain realized
outcomes; realized jobs are read only by `oracle.evaluate` after placements are frozen.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import statistics
import time

from . import adapter, advisor, bounded_advice, oracle, schema
from .schema import digest, encoded

VERSION = "M4A-1"
MODE = "PROBE_ONLY_CPU_OFFLINE"
REFERENCE, STATIC, STATIC_SAFE = ("OFFLINE_REFERENCE_ONLY", "STATIC_ADVICE_COUNTERFACTUAL",
                                  "STATIC_SAFE_FALLBACK")
DYNAMIC, DYNAMIC_LP = "DYNAMIC_PGFP", "DYNAMIC_PGFP_PLUS_LP"
ARMS = (REFERENCE + ":first_fit", REFERENCE + ":best_fit", STATIC, STATIC_SAFE, DYNAMIC, DYNAMIC_LP)
PRIMARY_REFERENCE = REFERENCE + ":best_fit"
# Identical controls, fixed a priori (never tuned on outcomes). BF/BP only bound the scores.
CONTROLS = dict(BF=4., BP=8., alpha_lp=.5, alpha_marginal=0., time_limit=5.)
TIE_RULE = "highest bounded score; ties to the lowest node index in original node order"
UNITS = schema.UNITS
SHAPE_TYPES = {"S2": [2, 16, 64], "S4": [4, 32, 128], "S8": [8, 64, 256]}
TASK_TYPES = {"small": [1, 4, 16], "mid": [2, 8, 32], "large": [4, 16, 64], "cpu": [0, 8, 32]}
ARRIVAL_MIX = (40, 30, 20, 10)  # percent, order of TASK_TYPES
REGIMES = {  # (forecast mix, realized mix, arrival mix) in percent, order of TASK_TYPES
    "matched": ((10, 30, 50, 10), (10, 30, 50, 10), ARRIVAL_MIX),
    "shift_large_to_small": ((10, 30, 50, 10), (50, 30, 10, 10), ARRIVAL_MIX),
    "shift_small_to_large": ((50, 30, 10, 10), (10, 30, 50, 10), ARRIVAL_MIX),
    "scarce_matched": ((10, 30, 50, 10), (10, 30, 50, 10), (10, 30, 50, 10)),
}
SCENES_PER_REGIME = 24
TASKS_PER_TRACE, FORECAST_ITEMS, REALIZED_ITEMS = 8, 4, 6


def _h(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def _draw(mix, *parts):
    point, total = _h(*parts) % 100, 0
    for name, share in zip(TASK_TYPES, mix):
        total += share
        if point < total:
            return name
    raise AssertionError("mix must sum to 100")


def _value(demand):
    return 1 + 4 * demand[0] + demand[1] // 8


def make_scene(regime, index):
    """Deterministic pure function of (regime, index); no outcome-dependent selection."""
    forecast_mix, realized_mix, arrival_mix = REGIMES[regime]
    tag = regime + ":" + str(index)
    count = 3 + _h(tag, "nodes") % 2
    shapes = list(SHAPE_TYPES)
    nodes = []
    for n in range(count):
        shape = shapes[_h(tag, "shape", n) % 3]
        nodes.append(dict(id="N" + str(n), gpu_family="A", shape=shape, capacity=SHAPE_TYPES[shape][:]))
    ids = [n["id"] for n in nodes]
    arrivals, made = [], 0
    while made < TASKS_PER_TRACE:
        size = min(1 + _h(tag, "batch", len(arrivals)) % 2, TASKS_PER_TRACE - made)
        batch = []
        for k in range(size):
            kind = _draw(arrival_mix, tag, "arrival", made)
            demand = TASK_TYPES[kind][:]
            skip = _h(tag, "elig", made)
            eligible = ids[:] if skip % 4 else [i for j, i in enumerate(ids) if j != (skip // 4) % count]
            batch.append(dict(id="T" + str(made), demand=demand, value=_value(demand), gpu_family=None,
                              eligible=eligible, kind=kind))
            made += 1
        arrivals.append(batch)
    forecast = [dict(id="F" + str(i), demand=TASK_TYPES[_draw(forecast_mix, tag, "forecast", i)][:],
                     weight=1 + _h(tag, "weight", i) % 3, gpu_family=None, eligible=ids[:])
                for i in range(FORECAST_ITEMS)]
    realized = []
    for i in range(REALIZED_ITEMS):
        demand = TASK_TYPES[_draw(realized_mix, tag, "realized", i)][:]
        realized.append(dict(id="R" + str(i), demand=demand, value=_value(demand), gpu_family=None, eligible=ids[:]))
    return dict(id=tag, regime=regime, nodes=nodes, arrivals=arrivals, forecast=forecast, realized=realized)


def _clean(job):
    return {k: v for k, v in job.items() if k != "kind"}


def snapshot(nodes, tasks, forecast, realized=()):
    return dict(schema_version="0.1", units=UNITS[:], nodes=[dict(n, capacity=list(n["capacity"])) for n in nodes],
                tasks=[_clean(t) for t in tasks], future=[dict(f) for f in forecast],
                realized_future=[_clean(r) for r in realized], constraints={})


def _stage_snapshot(trace, residual, batch):
    nodes = [dict(n, capacity=[int(x) for x in residual[i]]) for i, n in enumerate(trace["nodes"])]
    return snapshot(nodes, batch, trace["forecast"])  # realized_future is empty by construction


def _full(trace, realized=False):
    tasks = [t for batch in trace["arrivals"] for t in batch]
    return snapshot(trace["nodes"], tasks, trace["forecast"], trace["realized"] if realized else ())


def _clip(x):
    return max(-1., min(1., x))


def _marginal_scores(stage, controls):
    """Frozen m0_oracle.marginal on the stage's residual state, independently cross-checked."""
    a, _ = adapter.instance(stage)
    m0 = adapter.frozen()["m0_oracle"]
    rest = a.capacity.astype(float).copy()
    out = {}
    for t, task in enumerate(stage["tasks"]):
        for n, node in enumerate(stage["nodes"]):
            if not oracle.allowed(node, task) or not oracle.fits(node["capacity"], task["demand"]):
                continue
            g = float(m0.marginal(a, rest, n, t))
            after = [r - d for r, d in zip(node["capacity"], task["demand"])]
            independent = sum(f["weight"] * (int(oracle.fits(after, f["demand"])) - int(oracle.fits(node["capacity"], f["demand"])))
                              for f in stage["future"] if oracle.allowed(node, f))
            if abs(g - independent) > 1e-9 or g > 0:
                raise ValueError("marginal disagreement or positive sign")
            out[(task["id"], node["id"])] = _clip(g / controls["BF"]) * (1 - controls["alpha_marginal"])
    return out


def _lp_scores(stage, controls):
    """Fresh certified M3-A artifact on the stage's residual state; uncertified output is rejected."""
    result = bounded_advice.build_artifact(stage, BF=controls["BF"], BP=controls["BP"],
                                           alpha=controls["alpha_lp"], time_limit=controls["time_limit"])
    if result["status"] != bounded_advice.STATUS:
        return None, dict(status=result["status"]), None
    artifact = result["artifact"]
    raw = encoded(artifact)
    loaded = bounded_advice.load_artifact(raw, stage)
    if loaded["status"] != bounded_advice.STATUS or encoded(loaded["artifact"]) != raw:
        return None, dict(status="ARTIFACT_RELOAD_REJECTED"), None
    scores = {(p["task_id"], p["node_id"]): p["score"] for p in artifact["pairs"]}
    by_shape = {}
    for p in artifact["pairs"]:
        by_shape.setdefault((p["task_id"], p["shape"]), set()).add(p["p"])
    summary = dict(status=bounded_advice.STATUS, artifact_sha256=artifact["artifact_sha256"],
                   constant_price_groups=sum(len(v) == 1 for v in by_shape.values()),
                   price_groups=len(by_shape), nonzero_price_pairs=sum(p["p"] != 0 for p in artifact["pairs"]),
                   pairs=len(artifact["pairs"]), clipped=sum(p["clipped_F"] or p["clipped_P"] for p in artifact["pairs"]))
    return scores, summary, artifact


def _static_advice(trace, controls):
    initial = [n["capacity"] for n in trace["nodes"]]
    stage = _stage_snapshot(trace, initial, [t for batch in trace["arrivals"] for t in batch])
    result = bounded_advice.build_artifact(stage, BF=controls["BF"], BP=controls["BP"],
                                           alpha=controls["alpha_marginal"], time_limit=controls["time_limit"])
    if result["status"] != bounded_advice.STATUS:
        return None
    artifact = result["artifact"]
    raw = encoded(artifact)
    if bounded_advice.load_artifact(raw, stage)["status"] != bounded_advice.STATUS:
        return None
    return dict(raw=raw, raw_sha256=hashlib.sha256(raw).hexdigest(), sha256=artifact["artifact_sha256"], initial=initial,
                scores={(p["task_id"], p["node_id"]): p["score"] for p in artifact["pairs"]},
                excluded={(p["task_id"], p["node_id"]) for p in artifact["excluded_pairs"]})


def _pick(arm, task, nodes, residual, scores):
    candidates = [i for i, n in enumerate(nodes) if oracle.allowed(n, task) and oracle.fits(residual[i], task["demand"])]
    if not candidates:
        return -1
    if arm == REFERENCE + ":first_fit":
        return candidates[0]
    if arm == PRIMARY_REFERENCE:
        return min(candidates, key=lambda i: (sum(r - d for r, d in zip(residual[i], task["demand"])), i))
    return max(candidates, key=lambda i: (scores.get((task["id"], nodes[i]["id"]), 0.), -i))


def _receipt_check(nodes, task, before, chosen, after):
    errors = []
    expected = [r - d for r, d in zip(before[chosen], task["demand"])] if chosen >= 0 else None
    for i, (b, a) in enumerate(zip(before, after)):
        if i == chosen and a != expected or i != chosen and a != b:
            errors.append("conservation violated at node " + str(i))
    if chosen >= 0 and not oracle.allowed(nodes[chosen], task):
        errors.append("ineligible acceptance")
    if any(x < 0 for row in after for x in row):
        errors.append("negative residual")
    return errors


def _metrics(trace, residual):
    forecast_snapshot = _stage_snapshot(trace, residual, trace["arrivals"][0])
    nodes = forecast_snapshot["nodes"]
    fits = [sum(oracle.allowed(n, f) and oracle.fits(n["capacity"], f["demand"]) for n in nodes) for f in trace["forecast"]]
    stranded = [0, 0, 0]
    for i, n in enumerate(nodes):
        if not any(oracle.allowed(n, f) and oracle.fits(residual[i], f["demand"]) for f in trace["forecast"]):
            stranded = [s + r for s, r in zip(stranded, residual[i])]
    return dict(F=oracle.future_score(forecast_snapshot, residual), per_node_slack=[list(r) for r in residual],
                size_fit_counts=fits, stranded_capacity=stranded,
                units="F: weighted forecast-node fits; slack/stranded: " + ", ".join(UNITS))


def run_arm(trace, arm, controls=CONTROLS):
    """One arm over the trace; returns a normalized (timing-free) record."""
    nodes = trace["nodes"]
    residual = [list(n["capacity"]) for n in nodes]
    flat = [t for batch in trace["arrivals"] for t in batch]
    placement, steps, stages, errors = [], [], [], []
    static = _static_advice(trace, controls) if arm in (STATIC, STATIC_SAFE) else None
    if arm in (STATIC, STATIC_SAFE) and static is None:
        return dict(arm=arm, status="STATIC_ADVICE_UNCERTIFIED")
    for s, batch in enumerate(trace["arrivals"]):
        scores, info = {}, dict(stage=s, advice="NONE")
        if arm == DYNAMIC:
            scores = _marginal_scores(_stage_snapshot(trace, residual, batch), controls)
            info.update(advice="FRESH_MARGINAL_ALPHA0", residual_sha256=digest(residual))
        elif arm == DYNAMIC_LP:
            scores, lp_info, _ = _lp_scores(_stage_snapshot(trace, residual, batch), controls)
            info.update(advice="FRESH_CERTIFIED_LP" if scores is not None else "NEUTRAL_UNCERTIFIED",
                        residual_sha256=digest(residual), lp=lp_info)
            scores = scores or {}
        elif arm in (STATIC, STATIC_SAFE):
            if hashlib.sha256(static["raw"]).hexdigest() != static["raw_sha256"]:
                raise AssertionError("static advice artifact mutated after certification")
            fresh = residual == static["initial"]
            info.update(advice="INITIAL_STATE_ADVICE" if arm == STATIC or fresh else "NEUTRAL_AFTER_MUTATION",
                        artifact_sha256=static["sha256"], state_is_initial=fresh)
            scores = static["scores"] if arm == STATIC or fresh else {}
        stages.append(info)
        for task in batch:
            before = [r[:] for r in residual]
            chosen = _pick(arm, task, nodes, residual, scores)
            if static is not None and chosen >= 0 and (task["id"], nodes[chosen]["id"]) in static["excluded"]:
                errors.append("static exclusion contradicted by live state")
            if chosen >= 0:
                residual[chosen] = [r - d for r, d in zip(residual[chosen], task["demand"])]
            errors += _receipt_check(nodes, task, before, chosen, residual)
            placement.append(chosen)
            steps.append(dict(stage=s, task=task["id"], node=nodes[chosen]["id"] if chosen >= 0 else None,
                              before_sha256=digest(before), after_sha256=digest(residual), after=[r[:] for r in residual]))
    audit = oracle.audit(_full(trace), placement, residual)
    status = "PHYSICAL_VALID" if audit["status"] == "PHYSICAL_VALID" and not errors else "PHYSICAL_INVALID"
    record = dict(arm=arm, status=status, errors=errors + audit["errors"], placement=placement, steps=steps,
                  stages=stages, accepted_current=audit["accepted"], placed_value=audit["placed_value"],
                  final_residual=residual, metrics=_metrics(trace, residual))
    if status == "PHYSICAL_VALID":
        out = oracle.evaluate(_full(trace, realized=True), residual)  # realized jobs enter only here
        record["outcome"] = dict(sequential=out["sequential"], offline_fixed_current=out["offline_fixed_current"],
                                 predicted_surrogate_F=out["predicted_surrogate_F"], number_future=len(trace["realized"]))
    return record


def _delta(arm, ref, key):
    d = dict(arm_numerator=arm["outcome"][key]["count"], reference_numerator=ref["outcome"][key]["count"],
             denominator=arm["outcome"]["number_future"], value_difference=arm["outcome"][key]["value"] - ref["outcome"][key]["value"])
    d["delta_admission_rate_num"] = d["arm_numerator"] - d["reference_numerator"]
    return d


def paired(arms):
    ref = arms[PRIMARY_REFERENCE]
    out = {}
    for name, arm in arms.items():
        if name == PRIMARY_REFERENCE or arm["status"] != "PHYSICAL_VALID" or ref["status"] != "PHYSICAL_VALID":
            continue
        out[name] = dict(sequential=_delta(arm, ref, "sequential"), offline_fixed_current=_delta(arm, ref, "offline_fixed_current"),
                         F_difference=arm["metrics"]["F"] - ref["metrics"]["F"])
    return out


def run_scene(trace, controls=CONTROLS):
    arms = {name: run_arm(trace, name, controls) for name in ARMS}
    return dict(id=trace["id"], regime=trace["regime"], trace_sha256=digest(dict(
        nodes=trace["nodes"], arrivals=trace["arrivals"], forecast=trace["forecast"], realized=trace["realized"])),
        arms=arms, paired=paired(arms))


@contextmanager
def verified_frozen_once():
    """Verify the five frozen blobs at entry and exit; reuse the verified modules in between."""
    modules = adapter.frozen()
    original = adapter.frozen
    adapter.frozen = lambda: modules
    try:
        yield modules
    finally:
        adapter.frozen = original
        adapter.frozen()


def _tally(scenes, arm, key="sequential"):
    rows = [s["paired"][arm][key] for s in scenes if arm in s["paired"]]
    num = sum(r["delta_admission_rate_num"] for r in rows)
    den = sum(r["denominator"] for r in rows)
    return dict(scenes=len(rows), improved=sum(r["delta_admission_rate_num"] > 0 for r in rows),
                unchanged=sum(r["delta_admission_rate_num"] == 0 for r in rows),
                harmed=sum(r["delta_admission_rate_num"] < 0 for r in rows),
                numerator_difference=num, denominator=den, delta_admission_rate=num / den if den else None,
                value_difference=sum(r["value_difference"] for r in rows))


def aggregate(scenes):
    out = {}
    for regime in REGIMES:
        subset = [s for s in scenes if s["regime"] == regime]
        out[regime] = {arm: dict(sequential=_tally(subset, arm), offline_fixed_current=_tally(subset, arm, "offline_fixed_current"))
                       for arm in ARMS if arm != PRIMARY_REFERENCE}
    return out


def witnesses(scenes, arm, regime):
    harmed = [s["id"] for s in scenes if s["regime"] == regime and arm in s["paired"]
              and s["paired"][arm]["sequential"]["delta_admission_rate_num"] < 0]
    unchanged = [s["id"] for s in scenes if s["regime"] == regime and arm in s["paired"]
                 and s["paired"][arm]["sequential"]["delta_admission_rate_num"] == 0]
    return dict(harmed_count=len(harmed), harmed_first=harmed[:3], unchanged_count=len(unchanged), unchanged_first=unchanged[:3])


# ---- bounded exhaustive counterexample search ------------------------------------------
SEARCH_CAPS = ("S2", "S4", "S8")
SEARCH_FORECASTS = {"F_large": [("large", 2)], "F_mixed": [("large", 2), ("small", 1)], "F_mid": [("mid", 2), ("small", 1)]}
SEARCH_REALIZED = {"R_large3": ["large"] * 3, "R_small3": ["small"] * 3, "R_mid3": ["mid"] * 3, "R_mix": ["large", "mid", "small"]}


def _search_trace(caps, kinds, forecast, realized):
    ids = ["N0", "N1"]
    nodes = [dict(id=ids[i], gpu_family="A", shape="S", capacity=SHAPE_TYPES[c][:]) for i, c in enumerate(caps)]
    arrivals = [[dict(id="T" + str(i), demand=TASK_TYPES[k][:], value=_value(TASK_TYPES[k]), gpu_family=None,
                      eligible=ids[:])] for i, k in enumerate(kinds)]
    fc = [dict(id="F" + str(i), demand=TASK_TYPES[k][:], weight=w, gpu_family=None, eligible=ids[:]) for i, (k, w) in enumerate(forecast)]
    rz = [dict(id="R" + str(i), demand=TASK_TYPES[k][:], value=_value(TASK_TYPES[k]), gpu_family=None, eligible=ids[:])
          for i, k in enumerate(realized)]
    return dict(id="search", regime="search", nodes=nodes, arrivals=arrivals, forecast=fc, realized=rz)


def _run_simple(trace, arm):
    """Marginal-only dynamic arm or best-fit reference; no LP (the search is the cheap path)."""
    nodes, residual = trace["nodes"], [list(n["capacity"]) for n in trace["nodes"]]
    for s, batch in enumerate(trace["arrivals"]):
        scores = _marginal_scores(_stage_snapshot(trace, residual, batch), CONTROLS) if arm == DYNAMIC else {}
        for task in batch:
            chosen = _pick(arm, task, nodes, residual, scores)
            if chosen >= 0:
                residual[chosen] = [r - d for r, d in zip(residual[chosen], task["demand"])]
    out = oracle.evaluate(_full(trace, realized=True), residual)
    return oracle.future_score(_stage_snapshot(trace, residual, trace["arrivals"][0]), residual), out["sequential"]["count"], residual


def counterexample_search():
    from itertools import product
    kinds_space = list(TASK_TYPES)
    total = found = strict = dyn_better = equal = dyn_worse = 0
    first = []
    with verified_frozen_once():
        for caps in product(SEARCH_CAPS, repeat=2):
            for kinds in product(kinds_space, repeat=3):
                for fname, forecast in SEARCH_FORECASTS.items():
                    for rname, realized in SEARCH_REALIZED.items():
                        trace = _search_trace(caps, kinds, forecast, realized)
                        fd, ad, _ = _run_simple(trace, DYNAMIC)
                        fr, ar, _ = _run_simple(trace, PRIMARY_REFERENCE)
                        total += 1
                        dyn_better += ad > ar
                        equal += ad == ar
                        dyn_worse += ad < ar
                        if fd > fr and ad <= ar:
                            found += 1
                            strict += ad < ar
                            if len(first) < 3:
                                first.append(dict(caps=list(caps), arrivals=list(kinds), forecast=fname, realized=rname,
                                                  F_dynamic=fd, F_reference=fr, admitted_dynamic=ad, admitted_reference=ar))
    return dict(status="FOUND" if found else "NOT_FOUND", criterion="F_dynamic > F_reference and sequential admissions_dynamic <= admissions_reference",
                reference=PRIMARY_REFERENCE, space=dict(nodes=2, node_shapes=list(SEARCH_CAPS), arrivals=3, arrival_kinds=kinds_space,
                forecasts=list(SEARCH_FORECASTS), realized=list(SEARCH_REALIZED)),
                instances=total, matches=found, matches_with_strictly_fewer_admissions=strict, admissions_dynamic_greater=dyn_better, admissions_equal=equal,
                admissions_dynamic_fewer=dyn_worse, first_witnesses=first,
                scope="NOT_FOUND/FOUND is only within this enumerated space")


# ---- same-Shape constant-price demonstration ------------------------------------------
def same_shape_case():
    def build(shapes):
        ids = ["N0", "N1", "N2"]
        nodes = [dict(id=i, gpu_family="A", shape=s, capacity=SHAPE_TYPES["S4"][:]) for i, s in zip(ids, shapes)]
        batch = [dict(id="T0", demand=TASK_TYPES["mid"][:], value=_value(TASK_TYPES["mid"]), gpu_family=None, eligible=ids[:]),
                 dict(id="T1", demand=TASK_TYPES["large"][:], value=_value(TASK_TYPES["large"]), gpu_family=None, eligible=ids[:])]
        forecast = [dict(id="F0", demand=TASK_TYPES["large"][:], weight=2, gpu_family=None, eligible=ids[:])]
        return snapshot(nodes, batch, forecast)

    out = {}
    for name, shapes in (("single_shape", ["S4", "S4", "S4"]), ("mixed_shape", ["S4", "S4", "OTHER"])):
        result = bounded_advice.build_artifact(build(shapes), BF=CONTROLS["BF"], BP=CONTROLS["BP"], alpha=CONTROLS["alpha_lp"])
        if result["status"] != bounded_advice.STATUS:
            out[name] = dict(status=result["status"])
            continue
        prices = {}
        for p in result["artifact"]["pairs"]:
            prices.setdefault(p["task_id"], {}).setdefault(p["shape"], []).append(p["p"])
        out[name] = dict(status=bounded_advice.STATUS, price_by_task_and_shape=prices,
                         constant_within_every_shape=all(len(set(v)) == 1 for t in prices.values() for v in t.values()),
                         artifact_sha256=result["artifact"]["artifact_sha256"])
    return out


def source_identity():
    root = Path(__file__).resolve().parent
    return dict(upstream_sha=adapter.BASELINE, frozen_source_sha256=dict(adapter.SOURCE_HASHES),
                implementation_source_sha256={n: hashlib.sha256(root.joinpath(n + ".py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                              for n in ("policy_benefit", "bounded_advice", "adapter", "advisor", "oracle", "schema")})


def experiment(scenes_per_regime=SCENES_PER_REGIME):
    adapter.frozen()  # fail closed on any frozen-source mismatch before any arm runs
    scenes = [run_scene(make_scene(r, k)) for r in REGIMES for k in range(scenes_per_regime)]
    valid = all(a["status"] == "PHYSICAL_VALID" for s in scenes for a in s["arms"].values())
    record = dict(version=VERSION, mode=MODE, claim_scope="offline simulation only; not Native KAI, GPU, production or general performance",
                  identity=source_identity(), controls=CONTROLS, tie_rule=TIE_RULE, regimes={k: list(v) for k, v in REGIMES.items()},
                  scenes_per_regime=scenes_per_regime, all_arms_physically_valid=valid, scenes=scenes, aggregate=aggregate(scenes),
                  witnesses={r: {arm: witnesses(scenes, arm, r) for arm in (DYNAMIC, DYNAMIC_LP, STATIC)} for r in REGIMES},
                  counterexample_search=counterexample_search(), same_shape=same_shape_case())
    adapter.frozen()  # frozen blobs unchanged at the end
    record["record_sha256"] = digest(record)
    return record


def timing_report(repeats=5, scenes=3):
    """Wall-clock only; deliberately excluded from the normalized, byte-compared record."""
    out = {}
    traces = [make_scene("matched", k) for k in range(scenes)]
    for arm in (PRIMARY_REFERENCE, DYNAMIC, DYNAMIC_LP):
        cold = []
        for trace in traces[:1]:
            t0 = time.perf_counter(); run_arm(trace, arm); cold.append(time.perf_counter() - t0)
        warm = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for trace in traces:
                run_arm(trace, arm)
            warm.append((time.perf_counter() - t0) / len(traces))
        out[arm] = dict(first_call_seconds=cold[0], warm_per_trace_median=statistics.median(warm),
                        warm_min=min(warm), warm_max=max(warm), repeats=repeats, traces=len(traces))
    return out


def main(argv=None):
    import argparse, json, sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes-per-regime", type=int, default=SCENES_PER_REGIME)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    record = experiment(args.scenes_per_regime)
    raw = encoded(record)
    (args.out / "normalized.json").write_bytes(raw)
    timing = timing_report()
    (args.out / "timings.json").write_text(json.dumps(timing, indent=2, sort_keys=True))
    sys.stdout.write(hashlib.sha256(raw).hexdigest() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
