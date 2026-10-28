"""M7 P2H-PC-1: separately versioned planning-cohort producer (controlled, test-only).

The frozen P2H-1 envelope/receiver admit ONE scoring target and its single-task LP, whose certified
Shape prices are zero whenever a node fits it. P2H-PC-1 keeps the SAME native projection, the SAME
single genuine Pending target and the unchanged M3-A ``build_artifact``/``load_artifact`` and original
``canonical_shape`` LP, but lets the LP also see a declared *planning cohort*: extra tasks that the
decision maker legitimately knows at decision time (a published queue lookahead or the immutable
declared forecast). Cohort provenance (source, as-of arrival/phase, horizon, exact members) is hashed
into the envelope. Only the target's score pairs are exported. Nothing here edits a price or score; an
OFFLINE_ORACLE_INFORMATION_NOT_NATIVE source exists solely for clearly-labelled diagnostics.
"""
import base64
from pathlib import Path
import struct
import sys

from . import bounded_advice as ba, p2h_producer as base, schema

PC_VERSION = "P2H-PC-1"
SOURCES = ("PUBLISHED_QUEUE_LOOKAHEAD", "DECLARED_FORECAST", "OFFLINE_ORACLE_INFORMATION_NOT_NATIVE")
NATIVE_SOURCES = SOURCES[:2]
MAX_MEMBERS = 8


def cohort_hash(cohort):
    return base.sha(schema.encoded(cohort))


def validate_cohort(c, phase, target_id, names):
    schema.fields(c, ("source", "as_of_arrival", "as_of_phase", "horizon", "members"))
    if c["source"] not in SOURCES:
        raise schema.InputError("cohort source must be declared and known")
    if type(c["as_of_arrival"]) is not int or c["as_of_arrival"] < 0 or c["as_of_phase"] != phase or type(c["horizon"]) is not int or not 0 <= c["horizon"] <= MAX_MEMBERS:
        raise schema.InputError("cohort time provenance (as-of arrival, phase, horizon) required and must match the grant phase")
    if not isinstance(c["members"], list) or len(c["members"]) > min(c["horizon"], MAX_MEMBERS):
        raise schema.InputError("cohort larger than its declared horizon")
    seen, out = {target_id}, []
    for m in c["members"]:
        schema.fields(m, ("id", "demand"))
        schema.text(m["id"])
        d = m["demand"]
        if m["id"] in seen or not isinstance(d, list) or len(d) != 3 or not all(type(x) is int and x >= 0 for x in d) or not any(d):
            raise schema.InputError("cohort member identity/demand invalid or duplicated")
        seen.add(m["id"])
        out.append(dict(id=m["id"], demand=d[:], value=1 + 4 * d[0] + d[1] // 8, eligible=names[:], gpu_family=base.FAMILY))
    return out


def produce(request):
    schema.fields(request, ("projection_b64", "phase", "alpha", "BF", "BP", "forecast", "cohort"))
    if type(request["phase"]) is not int or not 1 <= request["phase"] <= base.MAX_PHASE:
        raise schema.InputError("bounded phase required")
    projection_raw = base64.b64decode(request["projection_b64"], validate=True)
    if not 0 < len(projection_raw) <= 262144:
        raise schema.InputError("projection size")
    p = schema.loads(projection_raw)
    snapshot = base.snapshot_from_projection(p)  # frozen P2H-1 projection checks, unchanged
    names, target = p["node_names"], snapshot["tasks"][0]
    members = validate_cohort(request["cohort"], request["phase"], target["id"], names)
    snapshot["tasks"] = [target] + members
    snapshot["future"] = base.forecast_items(request["forecast"], names)
    schema.validate(snapshot)
    built = ba.build_artifact(snapshot, BF=request["BF"], BP=request["BP"], alpha=request["alpha"])
    if built["status"] != ba.STATUS:
        raise schema.InputError("M3-A build failed: " + built["status"])
    artifact_raw = schema.encoded(built["artifact"])
    loaded = ba.load_artifact(artifact_raw, snapshot)
    if loaded["status"] != ba.STATUS:
        raise schema.InputError("M3-A load failed: " + loaded["status"])
    a = loaded["artifact"]
    mine = [v for v in a["pairs"] if v["task_id"] == target["id"]]
    excluded = sorted(v["node_id"] for v in a["excluded_pairs"] if v["task_id"] == target["id"])
    if [v["node_id"] for v in mine] != p["candidates"] or excluded != sorted(set(names) - set(p["candidates"])):
        raise schema.InputError("target score pairs differ from the exact native candidate set")
    cohort = request["cohort"]
    body = dict(version=PC_VERSION, producer=base.PRODUCER,
        producer_sha256=base.sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        artifact_sha256=base.sha(artifact_raw), artifact_b64=base64.b64encode(artifact_raw).decode(),
        prediction_sha256=a["prediction_sha256"], projection_sha256=base.sha(projection_raw),
        session=p["session"], phase=request["phase"], task_uid=target["id"], candidates=list(p["candidates"]),
        nodes=[dict(name=n["name"], uid=n["uid"]) for n in p["nodes"]],
        pairs=[dict(task_uid=v["task_id"], node=v["node_id"], score_bits=struct.pack(">d", v["score"]).hex()) for v in mine],
        cohort_sha256=cohort_hash(cohort), cohort_source=cohort["source"], cohort_ids=[m["id"] for m in members])
    raw = schema.encoded(body)
    if len(raw) > 131072:
        raise schema.InputError("bridge envelope size")
    prices = a["lp"]["prices"]
    controls = schema.encoded(dict(alpha=request["alpha"], BF=request["BF"], BP=request["BP"], projection=base.P2H, envelope=PC_VERSION,
        forecast=snapshot["future"], cohort=cohort, cohort_sha256=body["cohort_sha256"], prices=prices, price_sha256=a["lp"]["price_sha256"],
        shapes=a["mapping"]["shapes"], nonzero_price=any(v != 0 for row in prices for v in row),
        target_pairs=[dict(node=v["node_id"], g=v["g"], p=v["p"], AF=v["AF"], AP=v["AP"], score=v["score"]) for v in mine]))
    return raw, artifact_raw, schema.encoded(snapshot), schema.encoded(built["execution"]), controls


if __name__ == "__main__":
    request = schema.loads(Path(sys.argv[1]).read_bytes())
    raw, artifact, snapshot, execution, controls = produce(request)
    out = Path(sys.argv[2]); out.mkdir(exist_ok=False)
    for name, data in (("envelope.json", raw), ("artifact.json", artifact), ("snapshot.json", snapshot),
                       ("execution.json", execution), ("controls.json", controls)):
        (out / name).write_bytes(data)
    sys.stdout.buffer.write(raw)
