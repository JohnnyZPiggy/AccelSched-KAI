"""Strict v0.1 JSON boundary and explicit failure vocabulary."""
import hashlib
import json
import math

STATUSES = frozenset({"INVALID_INPUT", "UNSUPPORTED_CONSTRAINT", "LP_UNCERTIFIED",
    "SOLVER_TIMEOUT", "REPLAY_MISMATCH", "PHYSICAL_INVALID", "PHYSICAL_VALID", "INTERNAL_ERROR"})
UNITS = ["whole GPUs", "integer CPU cores", "integer RAM GiB"]
UNSUPPORTED = {"gang", "required_topology", "fractional_gpu", "mig", "dra", "preemption"}


class InputError(ValueError):
    def __init__(self, message, status="INVALID_INPUT"):
        super().__init__(message)
        self.status = status


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def loads(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise InputError("duplicate JSON key: " + key)
            out[key] = value
        return out
    try:
        return json.loads(text, object_pairs_hook=pairs,
                          parse_constant=lambda s: (_ for _ in ()).throw(InputError("nonfinite JSON: " + s)))
    except (ValueError, TypeError) as exc:
        raise InputError(str(exc)) from exc


def fields(obj, required, optional=()):
    if not isinstance(obj, dict) or set(obj) - set(required) - set(optional) or set(required) - set(obj):
        raise InputError("missing/unknown fields; required=" + repr(required))


def text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise InputError("nonempty string (maximum 128 characters) required")


def scalar(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1_000_000:
        raise InputError("finite nonnegative value/weight <= 1000000 required")


def vector(value):
    if not isinstance(value, list) or len(value) != 3:
        raise InputError("three resource coordinates required")
    if type(value[0]) is float and math.isfinite(value[0]) and value[0] != int(value[0]):
        raise InputError("fractional GPU is unsupported", "UNSUPPORTED_CONSTRAINT")
    if any(type(x) is not int or not 0 <= x <= 1_000_000 for x in value):
        raise InputError("integer nonnegative resource coordinates <= 1000000 required")


def validate(snapshot):
    fields(snapshot, ("schema_version", "units", "nodes", "tasks", "future", "realized_future", "constraints"))
    if snapshot["schema_version"] != "0.1" or snapshot["units"] != UNITS:
        raise InputError("unsupported schema version or units")
    constraints = snapshot["constraints"]
    if not isinstance(constraints, dict) or set(constraints) - UNSUPPORTED:
        raise InputError("unknown mandatory constraint")
    if constraints:
        raise InputError("unsupported semantics: " + ",".join(sorted(constraints)), "UNSUPPORTED_CONSTRAINT")
    for name, low, high in (("nodes", 1, 8), ("tasks", 1, 16), ("future", 1, 8), ("realized_future", 0, 8)):
        if not isinstance(snapshot[name], list) or not low <= len(snapshot[name]) <= high:
            raise InputError(name + " exceeds bounded PoC-0 size or is empty")
    if (len(snapshot["nodes"]) + 1) ** len(snapshot["realized_future"]) > 100_000:
        raise InputError("offline future search exceeds 100000 assignment bound")
    node_ids = []
    for node in snapshot["nodes"]:
        fields(node, ("id", "gpu_family", "shape", "capacity"))
        for key in ("id", "gpu_family", "shape"):
            text(node[key])
        vector(node["capacity"])
        node_ids.append(node["id"])
    if len(set(node_ids)) != len(node_ids):
        raise InputError("duplicate node ID")
    for name in ("tasks", "future", "realized_future"):
        ids = []
        for job in snapshot[name]:
            objective = "weight" if name == "future" else "value"
            fields(job, ("id", "demand", objective, "eligible", "gpu_family"))
            text(job["id"]); vector(job["demand"]); scalar(job[objective])
            if not any(job["demand"]):
                raise InputError("zero resource demand unsupported in PoC-0")
            if job["gpu_family"] is not None:
                text(job["gpu_family"])
            eligible = job["eligible"]
            if (not isinstance(eligible, list) or any(not isinstance(i, str) for i in eligible)
                    or len(set(eligible)) != len(eligible) or set(eligible) - set(node_ids)):
                raise InputError("eligible must be unique known node IDs (empty means reject)")
            ids.append(job["id"])
        if len(set(ids)) != len(ids):
            raise InputError("duplicate " + name + " ID")
    return snapshot
