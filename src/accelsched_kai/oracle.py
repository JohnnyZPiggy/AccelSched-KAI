"""Independent physical audit and bounded exact future evaluation; no scheduler imports."""
from functools import lru_cache
from .schema import validate


def allowed(node, job):
    return node["id"] in job["eligible"] and (job["gpu_family"] is None or job["gpu_family"] == node["gpu_family"])


def fits(rest, demand):
    return all(r >= d for r, d in zip(rest, demand))


def future_score(snapshot, rest):
    return sum(j["weight"] for i, n in enumerate(snapshot["nodes"]) for j in snapshot["future"]
               if allowed(n, j) and fits(rest[i], j["demand"]))


def audit(snapshot, placement, claimed_residual=None):
    validate(snapshot)
    rest = [n["capacity"][:] for n in snapshot["nodes"]]
    errors = []
    if not isinstance(placement, list) or len(placement) != len(snapshot["tasks"]):
        return {"status": "PHYSICAL_INVALID", "errors": ["exactly one node index or -1 per task required"]}
    value = 0
    for j, i in enumerate(placement):
        task = snapshot["tasks"][j]
        if type(i) is not int or not -1 <= i < len(rest):
            errors.append("invalid assignment for task " + str(j)); continue
        if i == -1:
            continue
        if not allowed(snapshot["nodes"][i], task):
            errors.append("ineligible assignment for task " + str(j))
        rest[i] = [r - d for r, d in zip(rest[i], task["demand"])]
        value += task["value"]
    if any(r < 0 for row in rest for r in row):
        errors.append("capacity exceeded")
    if claimed_residual is not None and claimed_residual != rest:
        errors.append("claimed residual mismatch")
    return {"status": "PHYSICAL_INVALID" if errors else "PHYSICAL_VALID", "errors": errors,
            "placement": placement, "residual": rest, "placed_value": value,
            "accepted": sum(type(i) is int and i >= 0 for i in placement)}


def evaluate(snapshot, rest):
    """Actual catalogue is evaluated only AFTER current placement is frozen.

    Sequential first-fit uses input order; offline optimum may reject/reorder future
    admissions but cannot move current jobs. Neither is the weighted surrogate F.
    """
    jobs = snapshot["realized_future"]
    current = [row[:] for row in rest]; sequential = []; value = 0
    for job in jobs:
        candidates = [i for i, n in enumerate(snapshot["nodes"]) if allowed(n, job) and fits(current[i], job["demand"])]
        i = candidates[0] if candidates else -1
        sequential.append(i)
        if i >= 0:
            current[i] = [r - d for r, d in zip(current[i], job["demand"])]
            value += job["value"]

    @lru_cache(None)
    def optimum(j, state):
        if j == len(jobs):
            return 0, ()
        job = jobs[j]; best_value, tail = optimum(j + 1, state); best = (best_value, (-1,) + tail)
        for i, node in enumerate(snapshot["nodes"]):
            if allowed(node, job) and fits(state[i], job["demand"]):
                changed = list(state); changed[i] = tuple(r - d for r, d in zip(state[i], job["demand"]))
                v, p = optimum(j + 1, tuple(changed)); v += job["value"]
                if v > best[0]:
                    best = v, (i,) + p
        return best

    offline_value, offline = optimum(0, tuple(tuple(row) for row in rest))
    return {"predicted_surrogate_F": future_score(snapshot, rest),
            "sequential": {"placement": sequential, "count": sum(i >= 0 for i in sequential), "value": value, "residual": current},
            "offline_fixed_current": {"placement": list(offline), "count": sum(i >= 0 for i in offline), "value": offline_value}}
