"""M4-B controlled offline test producer with explicit (alpha, BF, BP, forecast) provenance.

Narrow, parameterized sibling of the frozen M3-B ``bridge_envelope``: the same audited
P1 projection, one-Pending-target mapping, unchanged M3-A ``build_artifact`` then
``load_artifact``, and the same ``M3B-1`` envelope bytes. It only lifts the two M3-B
constants (fixed forecast, alpha .5/BF=BP=2) into validated request fields so the
Dynamic PGFP (alpha=0, genuine upstream m0 marginal) and Dynamic PGFP+LP arms can be
produced from the live per-arrival native state. Scores are never edited or invented.
Bytes alone confer no trust on Go; this is not a signature or production producer.
"""
import base64
from pathlib import Path
import struct
import sys

from . import bounded_advice as ba, bridge_envelope as be, schema

VERSION = be.VERSION
PRODUCER = be.PRODUCER
NODES = ("node0", "node1")
FAMILY = "CPU_ONLY_TEST"
MAX_FORECAST = 4


def forecast_items(items):
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_FORECAST:
        raise schema.InputError("bounded forecast required")
    out = []
    for item in items:
        schema.fields(item, ("id", "demand", "weight", "eligible"))
        schema.text(item["id"])
        d = item["demand"]
        if (not isinstance(d, list) or len(d) != 3 or any(type(x) is not int for x in d)
                or d[0] != 0 or d[1] < 0 or d[2] < 0):
            raise schema.InputError("P1 CPU-only forecast demand [0, cores, GiB] required")
        if not isinstance(item["eligible"], list) or not item["eligible"] or set(item["eligible"]) - set(NODES):
            raise schema.InputError("forecast eligibility must name P1 nodes")
        out.append(dict(id=item["id"], demand=d[:], weight=item["weight"], gpu_family=FAMILY,
                        eligible=list(item["eligible"])))
    return out


def produce(request):
    schema.fields(request, ("projection_b64", "phase", "alpha", "BF", "BP", "forecast"))
    if type(request["phase"]) is not int or not 1 <= request["phase"] <= 3:
        raise schema.InputError("bounded phase required")
    projection_raw = base64.b64decode(request["projection_b64"], validate=True)
    if not 0 < len(projection_raw) <= 65536:
        raise schema.InputError("projection size")
    p = schema.loads(projection_raw)
    snapshot = be.snapshot_from_projection(p)
    snapshot["future"] = forecast_items(request["forecast"])  # forecast comes from the trace, never from realized jobs
    schema.validate(snapshot)
    built = ba.build_artifact(snapshot, BF=request["BF"], BP=request["BP"], alpha=request["alpha"])
    if built["status"] != ba.STATUS:
        raise schema.InputError("M3-A build failed: " + built["status"])
    artifact_raw = schema.encoded(built["artifact"])
    loaded = ba.load_artifact(artifact_raw, snapshot)
    if loaded["status"] != ba.STATUS:
        raise schema.InputError("M3-A load failed: " + loaded["status"])
    a = loaded["artifact"]
    if len(a["pairs"]) != 2 or a["excluded_pairs"]:
        raise schema.InputError("incomplete supported candidate coverage")
    body = dict(version=VERSION, producer=PRODUCER,
        producer_sha256=be.sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        artifact_sha256=be.sha(artifact_raw), artifact_b64=base64.b64encode(artifact_raw).decode(),
        prediction_sha256=a["prediction_sha256"], projection_sha256=be.sha(projection_raw),
        session=p["session"], phase=request["phase"], task_uid=snapshot["tasks"][0]["id"],
        nodes=[dict(name=n["name"], uid=n["uid"]) for n in p["nodes"]],
        pairs=[dict(task_uid=v["task_id"], node=v["node_id"], score_bits=struct.pack(">d", v["score"]).hex()) for v in a["pairs"]])
    raw = schema.encoded(body)
    if len(raw) > 65536:
        raise schema.InputError("bridge envelope size")
    controls = schema.encoded(dict(alpha=request["alpha"], BF=request["BF"], BP=request["BP"],
                                   forecast=snapshot["future"], prices=a["lp"]["prices"], price_sha256=a["lp"]["price_sha256"]))
    return raw, artifact_raw, schema.encoded(snapshot), schema.encoded(built["execution"]), controls


if __name__ == "__main__":
    request = schema.loads(Path(sys.argv[1]).read_bytes())
    raw, artifact, snapshot, execution, controls = produce(request)
    out = Path(sys.argv[2]); out.mkdir(exist_ok=False)
    for name, data in (("envelope.json", raw), ("artifact.json", artifact), ("snapshot.json", snapshot),
                       ("execution.json", execution), ("controls.json", controls)):
        (out / name).write_bytes(data)
    sys.stdout.buffer.write(raw)
