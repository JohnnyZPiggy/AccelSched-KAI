"""M3-A: certified offline fixed-state scores, never a native KAI policy.

Artifact integrity is not a signature or proof of who ran a solver. Reading
rebuilds the original LP and recertifies its supplied witness without re-solving.
Only the prediction snapshot is bound; realized outcomes never enter this policy.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import numpy as np

from . import adapter, advisor, oracle, schema
from .schema import InputError, digest, encoded, fields, loads, validate

VERSION = "M3A-1"
STATUS = "CERTIFIED_OFFLINE_ADVICE"
MARGINAL_ABS = 1e-9
MARGINAL_REL = 1e-12
MAX_BYTES = 1024 * 1024


def _finite(value):
    if type(value) not in (int, float):
        raise InputError("finite real scalar required (not bool)")
    try:
        value = float(value)
    except OverflowError as exc:
        raise InputError("scalar overflow") from exc
    if not math.isfinite(value):
        raise InputError("nonfinite arithmetic or parameter")
    return value


def _parameters(BF, BP, alpha, time_limit):
    BF, BP, alpha, time_limit = map(_finite, (BF, BP, alpha, time_limit))
    if BF <= 0 or BP <= 0 or not 0 <= alpha <= 1 or not 0 < time_limit <= 10:
        raise InputError("BF/BP > 0, alpha in [0,1], time_limit in (0,10] required")
    return dict(BF=BF, BP=BP, alpha=alpha, time_limit=time_limit)


def _prediction(snapshot):
    validate(snapshot)
    prediction = deepcopy(snapshot)
    prediction["realized_future"] = []
    return prediction


def _raw_payload(raw):
    out = advisor.plain(asdict(raw))
    out.pop("timing")  # Volatile timing remains in the separate execution receipt.
    return out


def _decode_raw(payload, modules, params):
    required = ("backend", "version", "status", "native_status", "x", "pi", "rc",
                "objective", "dual_objective", "gap", "witness", "error", "settings")
    fields(payload, required)
    if (payload["backend"] != "highs" or payload["status"] != "OPTIMAL"
            or not isinstance(payload["version"], str) or not payload["version"].startswith("SciPy ")
            or payload["error"] is not None or payload["witness"] is not None):
        raise InputError("only an OPTIMAL HiGHS witness qualifies", "LP_UNCERTIFIED")
    fields(payload["native_status"], ("code", "message"))
    if type(payload["native_status"]["code"]) is not int or payload["native_status"]["code"] != 0:
        raise InputError("native solver did not report optimal", "LP_UNCERTIFIED")
    if not isinstance(payload["native_status"]["message"], str):
        raise InputError("invalid native status message")
    settings = dict(time_limit=params["time_limit"], primal_feasibility_tolerance=1e-9,
                    dual_feasibility_tolerance=1e-9, ipm_optimality_tolerance=1e-10)
    if encoded(payload["settings"]) != encoded(settings):
        raise InputError("solver settings mismatch")
    values = deepcopy(payload)
    for key in ("x", "pi", "rc"):
        if not isinstance(values[key], list):
            raise InputError("missing solver vector", "LP_UNCERTIFIED")
        values[key] = np.array([_finite(x) for x in values[key]], dtype=np.float64)
    for key in ("objective", "dual_objective", "gap"):
        if values[key] is not None:
            values[key] = _finite(values[key])
    return modules["lp_backends"].RawResult(**values)


def _assemble(prediction, params, raw_payload, source_hash):
    a, mapping = adapter.instance(prediction)
    modules = adapter.frozen()
    raw = _decode_raw(raw_payload, modules, params)
    model, labels, check = advisor.certificate(a, raw, source_hash)
    if not check["certified"] or raw.status != "OPTIMAL":
        raise InputError("independent original LP certificate rejected", "LP_UNCERTIFIED")
    prices = -raw.pi[len(a.value):].reshape(len(labels), 3)
    if not np.isfinite(prices).all() or (prices < 0).any():
        raise InputError("invalid scarcity price sign/finiteness", "LP_UNCERTIFIED")
    lp = dict(prices=prices, shapes=labels)
    replay = modules["certified_pgfp"]._replay(a, lp)
    verification = advisor.verify_replay(prediction, lp, replay)
    if verification["status"] != "EXACT_FIXED_PRICE_REPLAY":
        raise InputError("original fixed-price replay mismatch", "REPLAY_MISMATCH")
    physical = oracle.audit(prediction, replay["placement"], replay["residual"])
    if physical["status"] != "PHYSICAL_VALID":
        raise InputError("independent discrete placement audit failed", "PHYSICAL_INVALID")

    pairs, excluded = [], []
    rest = a.capacity.astype(float).copy()  # All marginals use ONE original state.
    for t, task in sorted(enumerate(prediction["tasks"]), key=lambda item: item[1]["id"]):
        for n, node in sorted(enumerate(prediction["nodes"]), key=lambda item: item[1]["id"]):
            identity = dict(task_id=task["id"], node_id=node["id"], shape=node["shape"])
            if not oracle.allowed(node, task) or not oracle.fits(node["capacity"], task["demand"]):
                reason = "INELIGIBLE" if not oracle.allowed(node, task) else "RESOURCE_INFEASIBLE"
                excluded.append(dict(identity, reason=reason))
                continue
            g = _finite(modules["m0_oracle"].marginal(a, rest, n, t))
            after = [r - d for r, d in zip(node["capacity"], task["demand"])]
            independent = _finite(math.fsum(f["weight"] *
                (int(oracle.fits(after, f["demand"])) - int(oracle.fits(node["capacity"], f["demand"])))
                for f in prediction["future"] if oracle.allowed(node, f)))
            if g > 0 or independent > 0 or not math.isclose(g, independent, abs_tol=MARGINAL_ABS, rel_tol=MARGINAL_REL):
                raise InputError("independent marginal disagreement/sign", "MARGINAL_MISMATCH")
            p = _finite(-float(prices[int(a.shapes[n])] @ a.demand[t]))
            scaled_F, scaled_P = _finite(g / params["BF"]), _finite(p / params["BP"])
            AF, AP = max(-1., min(1., scaled_F)), max(-1., min(1., scaled_P))
            score = _finite((1 - params["alpha"]) * AF + params["alpha"] * AP)
            if not -1 <= score <= 1:
                raise InputError("bounded score violated")
            pairs.append(dict(identity, g=g, g_oracle=independent, p=p, AF=AF, AP=AP, score=score,
                              clipped_F=abs(scaled_F) > 1, clipped_P=abs(scaled_P) > 1))

    body = dict(version=VERSION, status=STATUS, mode="PROBE_ONLY_CPU_OFFLINE",
        policy="SCOPED_HYPOTHESIS: bounded blend, not original PGFP replay ordering or KAI optimality",
        snapshot_scope="validated prediction snapshot with realized_future=[]; no outcome evaluation",
        upstream_sha=adapter.BASELINE, frozen_source_sha256=dict(adapter.SOURCE_HASHES),
        implementation_source_sha256={name: hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name, path in dict(bounded_advice=__file__, adapter=adapter.__file__, advisor=advisor.__file__,
                                   oracle=oracle.__file__, schema=schema.__file__).items()},
        prediction_sha256=digest(prediction),
        resources_sha256=digest(dict(nodes=prediction["nodes"], tasks=prediction["tasks"])),
        future_sha256=digest(prediction["future"]), mapping=mapping, normalization=params,
        tolerances=dict(marginal_absolute=MARGINAL_ABS, marginal_relative=MARGINAL_REL,
                        lp_absolute=modules["lp_certify"].TOL),
        lp=dict(status="OPTIMAL", source_hash=model.source_hash, certificate=check,
                raw=_raw_payload(raw), prices=prices.tolist(), price_sha256=digest(prices.tolist()),
                attempts=[dict(status="OPTIMAL", raw=_raw_payload(raw), certificate=check)],
                original_model=dict(A=model.A.toarray().tolist(), c=model.c.tolist(),
                    row_upper=model.row_hi.tolist(), variable_lower=model.var_lo.tolist(),
                    row_lower="all -infinity", variable_upper="all +infinity"),
                scope="original eligibility-relaxed Shape aggregate LP, not physical eligibility"),
        pairs=pairs, excluded_pairs=excluded,
        placement=dict(replay=advisor.plain(replay), replay_verification=advisor.plain(verification),
                       independent_audit=physical, scope="this discrete replay only; candidate pairs are not placement certificates"))
    body = advisor.plain(body)
    body["artifact_sha256"] = digest(body)
    if len(encoded(body)) > MAX_BYTES:
        raise InputError("bounded artifact exceeds 1 MiB")
    return body


def _safe_diagnostic(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _safe_diagnostic(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_diagnostic(v) for v in value]
    return value


def build_artifact(snapshot, *, BF, BP, alpha, time_limit=5.):
    """Solve the pinned original LP once; publish a complete artifact or none."""
    report = dict(status="INTERNAL_ERROR", mode="PROBE_ONLY_CPU_OFFLINE")
    try:
        params = _parameters(BF, BP, alpha, time_limit)
        prediction = _prediction(snapshot)
        a, _ = adapter.instance(prediction)
        modules = adapter.frozen()
        model, _ = modules["certified_pgfp"].canonical_shape(a)
        result = modules["lp_backends"].solve_lp(model, backend="highs", fallback="none", time_limit=params["time_limit"])
        report["execution"] = _safe_diagnostic(advisor.plain(asdict(result)))
        if result.status == "TIME_LIMIT" or result.raw.status == "TIME_LIMIT":
            raise InputError("native solver time limit", "SOLVER_TIMEOUT")
        if result.status != "OPTIMAL" or not result.certified or result.raw.status != "OPTIMAL" or len(result.attempts) != 1:
            raise InputError("original LP not certified OPTIMAL", "LP_UNCERTIFIED")
        artifact = _assemble(prediction, params, _raw_payload(result.raw), result.source_hash)
        report.update(status=STATUS, artifact=artifact)
    except InputError as exc:
        report.update(status=exc.status, error=str(exc))
    except Exception as exc:
        report.update(status="INTERNAL_ERROR", error=type(exc).__name__ + ": " + str(exc))
    return report


def load_artifact(raw, snapshot):
    """Fail closed on stale input, noncanonical JSON or altered numerical claims.

    Recompute from the supplied certified dual, not a newly selected dual. Thus
    legitimate dual nonuniqueness is preserved and no solver is invoked on read.
    """
    try:
        if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
            raise InputError("canonical bytes up to 1 MiB required")
        artifact = loads(raw)
        if not isinstance(artifact, dict) or encoded(artifact) != raw:
            raise InputError("noncanonical artifact encoding")
        if artifact.get("version") != VERSION or artifact.get("status") != STATUS:
            raise InputError("unknown artifact version/status")
        payload = dict(artifact)
        claimed_hash = payload.pop("artifact_sha256", None)
        if claimed_hash != digest(payload):
            raise InputError("artifact integrity mismatch")
        prediction = _prediction(snapshot)
        params = artifact["normalization"]
        fields(params, ("BF", "BP", "alpha", "time_limit"))
        params = _parameters(**params)
        expected = _assemble(prediction, params, artifact["lp"]["raw"], artifact["lp"]["source_hash"])
        if encoded(expected) != raw:
            raise InputError("artifact identity, certificate, coverage or numerical evidence mismatch")
        return dict(status=STATUS, artifact=expected)
    except InputError as exc:
        return dict(status=exc.status, error=str(exc))
    except Exception as exc:
        return dict(status="INVALID_ARTIFACT", error=type(exc).__name__ + ": " + str(exc))
