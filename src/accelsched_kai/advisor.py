"""Certified fixed-price reference replay, baselines, evidence and a CPU-only CLI."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import numpy as np
from .adapter import BASELINE, SOURCE_HASHES, frozen, instance
from . import oracle
from .schema import InputError, digest, encoded, loads, validate


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def certificate(a, raw, source_hash):
    """Rebuild and certify the original model; prices alone never qualify."""
    modules = frozen()
    model, labels = modules["certified_pgfp"].canonical_shape(a)
    model.assert_intact()
    check = modules["lp_certify"].certify(model, raw)
    check["original_hash_match"] = source_hash == model.source_hash
    check["certified"] = bool(check["certified"] and check["original_hash_match"])
    return model, labels, check


def verify_replay(snapshot, lp, replay):
    """Check every reference ledger decision against original normalized inputs.

    This verifier does not call the scheduler's validate/marginal functions.
    """
    nodes, tasks = snapshot["nodes"], snapshot["tasks"]
    labels = list(dict.fromkeys(n["shape"] for n in nodes))
    prices = np.asarray(lp["prices"])
    utility = np.array([j["value"] for j in tasks])[None, :] - prices @ np.array([j["demand"] for j in tasks]).T
    order = sorted(range(len(tasks)), key=lambda j: (-max(utility[:, j]), j))
    shape_orders = [sorted(range(len(labels)), key=lambda h: (-utility[h, j], h)) for j in range(len(tasks))]
    errors = []
    if replay["task_order"] != order or replay["shape_order"] != shape_orders:
        errors.append("complete task/Shape order mismatch")
    rest = [n["capacity"][:] for n in nodes]; placement = [-1] * len(tasks)
    if len(replay["ledger_steps"]) != len(tasks):
        errors.append("ledger length mismatch")
    else:
        for position, j in enumerate(order):
            step = replay["ledger_steps"][position]; job = tasks[j]
            expected = {"task": j, "before": [r[:] for r in rest], "attempts": [], "chosen": -1}
            for h in shape_orders[j]:
                ids = [i for i, n in enumerate(nodes) if n["shape"] == labels[h]
                       and oracle.allowed(n, job) and oracle.fits(rest[i], job["demand"])]
                scores = []
                for i in ids:
                    after = [r - d for r, d in zip(rest[i], job["demand"])]
                    # Unchanged nodes cancel exactly; avoid subtracting two large F sums.
                    scores.append(sum(f["weight"] * (int(oracle.fits(after, f["demand"]))
                        - int(oracle.fits(rest[i], f["demand"]))) for f in snapshot["future"]
                        if oracle.allowed(nodes[i], f)))
                expected["attempts"].append({"shape": h, "candidates": ids, "scores": scores})
                if ids:
                    chosen = max(ids, key=lambda i: (scores[ids.index(i)], -i))
                    placement[j] = chosen; expected["chosen"] = chosen
                    rest[chosen] = [r - d for r, d in zip(rest[chosen], job["demand"])]
                    break
            expected["after"] = [r[:] for r in rest]
            # Compare all fields; allow only summation roundoff in marginal scores.
            observed = json.loads(json.dumps(step))
            for actual, wanted in zip(observed.get("attempts", []), expected["attempts"]):
                scores = actual.get("scores", [])
                if len(scores) == len(wanted["scores"]) and all(math.isclose(x, y, abs_tol=1e-9, rel_tol=1e-12) for x, y in zip(scores, wanted["scores"])):
                    actual["scores"] = wanted["scores"]
            if observed != expected:
                errors.append("ledger mismatch at position " + str(position))
    physical = oracle.audit(snapshot, replay["placement"], replay["residual"])
    if (replay["placement"] != placement or replay["residual"] != rest
            or replay["greedy_residual"] != rest or replay["value"] != physical.get("placed_value")
            or not math.isclose(replay["future_score"], oracle.future_score(snapshot, rest), abs_tol=1e-9, rel_tol=1e-12)):
        errors.append("final placement/residual/objective mismatch")
    return {"status": "REPLAY_MISMATCH" if errors else "EXACT_FIXED_PRICE_REPLAY", "errors": errors,
            "utility": utility.tolist(), "physical": physical}


def baseline(snapshot, lp, policy):
    tasks, nodes = snapshot["tasks"], snapshot["nodes"]
    labels = list(dict.fromkeys(n["shape"] for n in nodes))
    utility = np.array([t["value"] for t in tasks])[None, :] - lp["prices"] @ np.array([t["demand"] for t in tasks]).T
    order = list(range(len(tasks))) if policy == "first_fit" else sorted(range(len(tasks)), key=lambda j: (-max(utility[:, j]), j))
    rest = [n["capacity"][:] for n in nodes]; placement = [-1] * len(tasks); steps = []
    for j in order:
        step = {"task": j, "before": [r[:] for r in rest], "attempts": [], "chosen": -1}
        shapes = [None] if policy == "first_fit" else sorted(range(len(labels)), key=lambda h: (-utility[h, j], h))
        for h in shapes:
            ids = [i for i, n in enumerate(nodes) if (h is None or n["shape"] == labels[h])
                   and oracle.allowed(n, tasks[j]) and oracle.fits(rest[i], tasks[j]["demand"])]
            step["attempts"].append({"shape": h, "candidates": ids})
            if ids:
                i = ids[0] if policy == "first_fit" else min(ids, key=lambda i: (sum(r - d for r, d in zip(rest[i], tasks[j]["demand"])), i))
                placement[j] = i; step["chosen"] = i
                rest[i] = [r - d for r, d in zip(rest[i], tasks[j]["demand"])]
                break
        step["after"] = [r[:] for r in rest]; steps.append(step)
    return {"placement": placement, "residual": rest, "task_order": order, "ledger_steps": steps}


def advise(snapshot, *, time_limit=5.0):
    report = {"baseline_sha": BASELINE, "mode": "PROBE_ONLY_CPU_OFFLINE", "status": "INTERNAL_ERROR"}
    try:
        validate(snapshot)
        report["input_sha256"] = digest(snapshot)
        if type(time_limit) not in (int, float) or not math.isfinite(time_limit) or not 0 < time_limit <= 10:
            raise InputError("time_limit must be finite, positive and <= 10 seconds")
        a, mapping = instance(snapshot); modules = frozen()
        report.update(mapping=mapping, frozen_source_sha256=SOURCE_HASHES)
        model, labels = modules["certified_pgfp"].canonical_shape(a)
        result = modules["lp_backends"].solve_lp(model, backend="highs", fallback="none", time_limit=time_limit)
        rebuilt, labels, check = certificate(a, result.raw, result.source_hash)
        report["lp"] = {"status": result.status, "source_hash": result.source_hash,
            "certificate": check, "attempts": [plain(asdict(attempt)) for attempt in result.attempts],
            "original_model": {"A": rebuilt.A.toarray().tolist(), "c": rebuilt.c.tolist(),
                "row_upper": rebuilt.row_hi.tolist(), "variable_lower": rebuilt.var_lo.tolist(),
                "row_lower": "all -infinity", "variable_upper": "all +infinity"},
            "eligibility_relaxed": True, "price_scope": "Shape aggregate; not discrete family feasibility"}
        if result.raw.status == "TIME_LIMIT" or result.status == "TIME_LIMIT":
            report["status"] = "SOLVER_TIMEOUT"
        elif result.status != "OPTIMAL" or not result.certified or not check["certified"]:
            report["status"] = "LP_UNCERTIFIED"
        else:
            prices = -result.raw.pi[len(a.value):].reshape(len(labels), 3)
            lp = {"prices": prices, "shapes": labels}
            report["lp"].update(prices=prices.tolist(), price_sha256=digest(prices.tolist()))
            replay = modules["certified_pgfp"]._replay(a, lp)
            verification = verify_replay(snapshot, lp, replay)
            report.update(replay=replay, replay_verification=verification)
            if verification["status"] != "EXACT_FIXED_PRICE_REPLAY":
                report["status"] = "REPLAY_MISMATCH"
            else:
                policies = {"first_fit": baseline(snapshot, lp, "first_fit"),
                            "lp_best_fit": baseline(snapshot, lp, "lp_best_fit"), "true_pgfp": replay}
                outcomes = {}
                for name, trace in policies.items():
                    physical = oracle.audit(snapshot, trace["placement"], trace["residual"])
                    outcomes[name] = {"trace": trace, "oracle": physical}
                    if physical["status"] == "PHYSICAL_VALID":
                        outcomes[name]["future"] = oracle.evaluate(snapshot, physical["residual"])
                report["policy_audits"] = outcomes
                report["status"] = "PHYSICAL_VALID" if all(o["oracle"]["status"] == "PHYSICAL_VALID" for o in outcomes.values()) else "PHYSICAL_INVALID"
                # No advice is exposed until every independent audit has passed.
                if report["status"] == "PHYSICAL_VALID":
                    report["advice"] = {name: o["trace"]["placement"] for name, o in outcomes.items()}
    except InputError as exc:
        report.update(status=exc.status, error=str(exc))
    except Exception as exc:
        report.update(status="INTERNAL_ERROR", error=type(exc).__name__ + ": " + str(exc))
    report = plain(report)
    # Solver evidence may contain nonfinite native fields; retain them explicitly as
    # strings rather than emitting invalid JSON or discarding the failing attempt.
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        return value
    report = finite(report)
    report["output_sha256"] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    try:
        raw = args.snapshot.read_bytes()
        snapshot = loads(raw)
        out = advise(snapshot)
    except (InputError, OSError) as exc:
        out = {"status": getattr(exc, "status", "INVALID_INPUT"), "error": str(exc)}
    sys.stdout.buffer.write(encoded(out) + b"\n")
    return 0 if out["status"] == "PHYSICAL_VALID" else 1


if __name__ == "__main__":
    sys.exit(main())
