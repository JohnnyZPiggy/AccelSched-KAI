"""M3-B controlled offline test producer; bytes alone confer no trust on Go.

P1 supports the audited CPU-only B4 fake model, one Pending candidate and at most
two Binding context tasks. Snapshot capacities are native IDLE, never total.
Native context and callback/object identity are additionally guarded by the Go
test harness. This is not a general native KAI predicate projection.
"""
import base64
import hashlib
from pathlib import Path
import struct
import sys

from . import bounded_advice as ba, schema

VERSION = "M3B-1"
PRODUCER = "SAE_CONTROLLED_OFFLINE_TEST_HARNESS"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def vector(v):
    if not isinstance(v, list) or len(v) != 3 or any(type(x) is not int or x < 0 for x in v):
        raise schema.InputError("native integer vector required")
    if v[0] != 0 or v[1] % 1000 or v[2] % (1024**3):
        raise schema.InputError("P1 CPU-only, integral cores/GiB required")
    return [0, v[1] // 1000, v[2] // (1024**3)]


def snapshot_from_projection(p):
    schema.fields(p, ("version", "session", "nodes", "tasks", "domain", "config", "hooks", "semantics"))
    if p["version"] != "P1" or p["semantics"] != "AUDITED_B4_CPU_ONLY_SINGLE_PENDING":
        raise schema.InputError("unsupported native projection")
    schema.text(p["session"])
    if not isinstance(p["nodes"], list) or len(p["nodes"]) != 2 or not isinstance(p["tasks"], list) or not 1 <= len(p["tasks"]) <= 3:
        raise schema.InputError("bounded two-node P1 fixture required")
    nodes = []
    ids = []
    for n in p["nodes"]:
        schema.fields(n, ("name", "uid", "shape", "family", "allocatable", "idle", "used", "releasing", "members", "native"))
        schema.text(n["name"])
        if type(n["uid"]) is not str or n["shape"] != "S" or n["family"] != "CPU_ONLY_TEST":
            raise schema.InputError("unknown node identity/Shape/family")
        alloc, idle, used, releasing = [vector(n[k]) for k in ("allocatable", "idle", "used", "releasing")]
        if any(releasing) or alloc != [x + y for x, y in zip(idle, used)]:
            raise schema.InputError("idle/used/total state mismatch or releasing unsupported")
        nodes.append(dict(id=n["name"], gpu_family=n["family"], shape=n["shape"], capacity=idle))
        ids.append(n["name"])
    if ids != ["node0", "node1"]:
        raise schema.InputError("P1 exact ordered node identities required")
    pending, known = [], {}
    for t in p["tasks"]:
        schema.fields(t, ("uid", "pod_uid", "status", "node", "request", "accepted", "native"))
        schema.text(t["uid"])
        if t["uid"] != t["pod_uid"] or t["uid"] in known:
            raise schema.InputError("task UID/Pod UID mismatch or duplicate")
        request = vector(t["request"])
        vector(t["accepted"])
        if t["status"] == "Pending" and t["node"] == "":
            pending.append(t)
        elif t["status"] != "Binding" or t["node"] not in ids:
            raise schema.InputError("unrepresented native task state")
        known[t["uid"]] = (t, request)
    if len(pending) != 1:
        raise schema.InputError("exactly one Pending scoring target required")
    for n in p["nodes"]:
        expected = sorted(uid for uid, (t, _) in known.items() if t["status"] == "Binding" and t["node"] == n["name"])
        if n["members"] != expected:
            raise schema.InputError("node/task membership mismatch")
        used = [sum(known[uid][1][k] for uid in expected) for k in range(3)]
        if vector(n["used"]) != used:
            raise schema.InputError("native used resources not represented by context")
    target = pending[0]
    snapshot = dict(schema_version="0.1", units=list(schema.UNITS), nodes=nodes,
        tasks=[dict(id=target["uid"], demand=vector(target["request"]), value=1,
                    eligible=ids, gpu_family="CPU_ONLY_TEST")],
        future=[dict(id="fixed-test-forecast", demand=[0, 4, 1], weight=1,
                     eligible=["node0"], gpu_family="CPU_ONLY_TEST")], realized_future=[], constraints={})
    return schema.validate(snapshot)


def produce(request):
    schema.fields(request, ("projection_b64", "phase"))
    if type(request["phase"]) is not int or not 1 <= request["phase"] <= 3:
        raise schema.InputError("bounded phase required")
    projection_raw = base64.b64decode(request["projection_b64"], validate=True)
    if not 0 < len(projection_raw) <= 65536:
        raise schema.InputError("projection size")
    p = schema.loads(projection_raw)
    snapshot = snapshot_from_projection(p)
    built = ba.build_artifact(snapshot, BF=2., BP=2., alpha=.5)
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
        producer_sha256=sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        artifact_sha256=sha(artifact_raw), artifact_b64=base64.b64encode(artifact_raw).decode(),
        prediction_sha256=a["prediction_sha256"], projection_sha256=sha(projection_raw),
        session=p["session"], phase=request["phase"], task_uid=snapshot["tasks"][0]["id"],
        nodes=[dict(name=n["name"], uid=n["uid"]) for n in p["nodes"]],
        pairs=[dict(task_uid=v["task_id"], node=v["node_id"], score_bits=struct.pack(">d", v["score"]).hex()) for v in a["pairs"]])
    raw = schema.encoded(body)
    if len(raw) > 65536:
        raise schema.InputError("bridge envelope size")
    return raw, artifact_raw, schema.encoded(snapshot), schema.encoded(built["execution"])


if __name__ == "__main__":
    request = schema.loads(Path(sys.argv[1]).read_bytes())
    raw, artifact, snapshot, execution = produce(request)
    out = Path(sys.argv[2]); out.mkdir(exist_ok=False)
    for name, data in (("envelope.json", raw), ("artifact.json", artifact), ("snapshot.json", snapshot), ("execution.json", execution)):
        (out / name).write_bytes(data)
    sys.stdout.buffer.write(raw)
