"""M6 P2H-1: strict 2-to-4-node heterogeneous whole-GPU projection and controlled producer.

A NEW versioned domain: ``m5_producer`` (P2-WholeGPU-1, exactly two nodes) and ``bridge_envelope``
(P1 CPU_ONLY) are untouched. Python re-derives every model coordinate from the RAW native vectors
and requires the Go-reported native candidate set to equal the independently computed model
candidate set, node by node. The envelope (version ``P2H-1``) carries the ordered candidate list
and exactly one score pair per candidate; M3-A ``build_artifact`` then ``load_artifact`` are called
unchanged, and any mismatch fails closed. Bytes alone confer no trust on Go; this is a controlled
test producer, not a signature or a production activation path.
"""
import base64
import hashlib
from pathlib import Path
import struct
import sys

from . import bounded_advice as ba, schema

P2H = "P2H-1"
SEMANTICS = "AUDITED_WHOLE_GPU_MULTI_NODE_SINGLE_PENDING"
ENVELOPE_VERSION = "P2H-1"
PRODUCER = "SAE_CONTROLLED_OFFLINE_TEST_HARNESS"
FAMILY = "WHOLE_GPU_TEST"
GIB = 1024 ** 3
INDEX_MAP = {"cpu": 0, "gpu": 2, "memory": 1, "pods": 3}
MIN_NODES, MAX_NODES = 2, 4
MIN_FREE_POD_SLOTS = 4
MAX_TASKS = 8
MAX_PHASE = 8
MAX_FORECAST = 4
BOUNDS = {"max_nodes": MAX_NODES, "max_phase": MAX_PHASE, "max_tasks": MAX_TASKS}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _int(x):
    return type(x) is int and 0 <= x <= 10 ** 15


def native_vector(v):
    if not isinstance(v, list) or len(v) != 4 or not all(_int(x) for x in v):
        raise schema.InputError("four exact nonnegative native integers required (floats/NaN/negative rejected)")
    return v


def model_from_native(v):
    cpu, mem, gpu, _pods = native_vector(v)
    if cpu % 1000 or mem % GIB:
        raise schema.InputError("CPU milli/1000 and RAM bytes/2^30 must be exact integers (no rounding)")
    return [gpu, cpu // 1000, mem // GIB]


def shape_of(model):
    return "G%dC%dM%d" % tuple(model)


def node_names(n):
    return ["node%d" % i for i in range(n)]


def snapshot_from_projection(p):
    schema.fields(p, ("version", "session", "semantics", "units", "index_map", "bounds", "node_names", "nodes", "tasks", "eligibility", "candidates", "domain", "config", "hooks"))
    if p["version"] != P2H or p["semantics"] != SEMANTICS or p["units"] != schema.UNITS or p["index_map"] != INDEX_MAP or p["bounds"] != BOUNDS:
        raise schema.InputError("unsupported native projection version/units/index map/bounds")
    schema.text(p["session"])
    names = p["node_names"]
    if not isinstance(names, list) or not MIN_NODES <= len(names) <= MAX_NODES or names != node_names(len(names)):
        raise schema.InputError("P2H exact ordered node0..nodeN-1 identities with 2<=N<=4 required")
    if not isinstance(p["nodes"], list) or len(p["nodes"]) != len(names) or not isinstance(p["tasks"], list) or not 1 <= len(p["tasks"]) <= MAX_TASKS:
        raise schema.InputError("bounded P2H fixture required (node/task counts)")
    nodes, used_pods = [], {}
    for want, n in zip(names, p["nodes"]):
        schema.fields(n, ("name", "uid", "shape", "family", "allocatable", "idle", "used", "releasing", "native_raw", "members", "native"))
        if n["name"] != want or type(n["uid"]) is not str or n["family"] != FAMILY:
            raise schema.InputError("node identity/order/GPU family mismatch")
        schema.fields(n["native_raw"], ("allocatable", "idle", "used", "releasing"))
        raw = {k: native_vector(n["native_raw"][k]) for k in ("allocatable", "idle", "used", "releasing")}
        if any(raw["releasing"]):
            raise schema.InputError("releasing/preemption resources unsupported")
        if any(a != i + u for a, i, u in zip(raw["allocatable"], raw["idle"], raw["used"])):
            raise schema.InputError("native idle/used/total mismatch (extra coordinate or accounting error)")
        model = {k: model_from_native(raw[k]) for k in ("allocatable", "idle", "used")}
        for k in model:
            if n[k] != model[k]:
                raise schema.InputError("Go/Python unit conversion disagreement on " + k)
        if n["releasing"] != [0, 0, 0] or n["shape"] != shape_of(model["allocatable"]):
            raise schema.InputError("unsupported releasing/Shape identity")
        if raw["idle"][3] < MIN_FREE_POD_SLOTS or raw["allocatable"][3] < MIN_FREE_POD_SLOTS:
            raise schema.InputError("pod-slot bottleneck is not representable in the model")
        used_pods[want] = raw["used"][3]
        nodes.append(dict(id=want, gpu_family=FAMILY, shape=n["shape"], capacity=model["idle"]))
    pending, known = [], {}
    for t in p["tasks"]:
        schema.fields(t, ("uid", "pod_uid", "status", "node", "request", "accepted", "native_request", "native"))
        schema.text(t["uid"])
        if t["uid"] != t["pod_uid"] or t["uid"] in known:
            raise schema.InputError("task UID/Pod UID mismatch or duplicate")
        native = native_vector(t["native_request"])
        if native[3] != 1 or t["request"] != model_from_native(native) or not any(t["request"]):
            raise schema.InputError("request must be one pod slot, non-zero, and match the independent conversion")
        accepted = t["accepted"]
        if not isinstance(accepted, list) or len(accepted) != 3 or not all(_int(x) for x in accepted):
            raise schema.InputError("accepted vector must be exact model integers")
        if t["status"] == "Pending" and t["node"] == "":
            pending.append(t)
        elif t["status"] != "Binding" or t["node"] not in names:
            raise schema.InputError("unrepresented native task state")
        known[t["uid"]] = t
    if len(pending) != 1:
        raise schema.InputError("exactly one Pending scoring target required")
    for n in p["nodes"]:
        members = sorted(uid for uid, t in known.items() if t["status"] == "Binding" and t["node"] == n["name"])
        if n["members"] != members:
            raise schema.InputError("node/task membership mismatch")
        used = [sum(known[uid]["request"][k] for uid in members) for k in range(3)]
        if n["used"] != used or used_pods[n["name"]] != len(members):
            raise schema.InputError("native used resources/pod slots not represented by the task context")
    target = pending[0]
    schema.fields(p["eligibility"], names)
    model_candidates = []
    for n in p["nodes"]:
        e = p["eligibility"][n["name"]]
        schema.fields(e, ("native", "model", "fitting_gpus"))
        model_fit = all(i >= d for i, d in zip(n["idle"], target["request"]))
        if type(e["native"]) is not bool or e["native"] is not model_fit or e["model"] is not model_fit:
            raise schema.InputError("native and independent model candidate sets disagree at " + n["name"])
        if model_fit:
            if not isinstance(e["fitting_gpus"], int) or e["fitting_gpus"] < target["request"][0]:
                raise schema.InputError("FittingGPUs does not cover a feasible whole-GPU request")
            model_candidates.append(n["name"])
    if p["candidates"] != model_candidates or not model_candidates:
        raise schema.InputError("exact ordered native candidate list required (non-empty, no extra/duplicate/misordered nodes)")
    value = 1 + 4 * target["request"][0] + target["request"][1] // 8
    snapshot = dict(schema_version="0.1", units=list(schema.UNITS), nodes=nodes,
        tasks=[dict(id=target["uid"], demand=target["request"][:], value=value, eligible=names[:], gpu_family=FAMILY)],
        future=[dict(id="placeholder", demand=[1, 1, 1], weight=1, eligible=names[:], gpu_family=FAMILY)],
        realized_future=[], constraints={})
    return schema.validate(snapshot)


def forecast_items(items, names):
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_FORECAST:
        raise schema.InputError("bounded forecast required")
    out = []
    for item in items:
        schema.fields(item, ("id", "demand", "weight", "eligible"))
        schema.text(item["id"])
        d = item["demand"]
        if not isinstance(d, list) or len(d) != 3 or not all(type(x) is int and x >= 0 for x in d) or not any(d):
            raise schema.InputError("whole-GPU forecast demand [GPU, cores, GiB] required")
        if not isinstance(item["eligible"], list) or not item["eligible"] or set(item["eligible"]) - set(names):
            raise schema.InputError("forecast eligibility must name P2H nodes")
        out.append(dict(id=item["id"], demand=d[:], weight=item["weight"], gpu_family=FAMILY, eligible=list(item["eligible"])))
    return out


def produce(request):
    schema.fields(request, ("projection_b64", "phase", "alpha", "BF", "BP", "forecast"))
    if type(request["phase"]) is not int or not 1 <= request["phase"] <= MAX_PHASE:
        raise schema.InputError("bounded phase required")
    projection_raw = base64.b64decode(request["projection_b64"], validate=True)
    if not 0 < len(projection_raw) <= 262144:
        raise schema.InputError("projection size")
    p = schema.loads(projection_raw)
    snapshot = snapshot_from_projection(p)
    names = p["node_names"]
    snapshot["future"] = forecast_items(request["forecast"], names)  # trace forecast only, never realized jobs
    schema.validate(snapshot)
    built = ba.build_artifact(snapshot, BF=request["BF"], BP=request["BP"], alpha=request["alpha"])
    if built["status"] != ba.STATUS:
        raise schema.InputError("M3-A build failed: " + built["status"])
    artifact_raw = schema.encoded(built["artifact"])
    loaded = ba.load_artifact(artifact_raw, snapshot)
    if loaded["status"] != ba.STATUS:
        raise schema.InputError("M3-A load failed: " + loaded["status"])
    a = loaded["artifact"]
    scored = [v["node_id"] for v in a["pairs"]]
    excluded = sorted(v["node_id"] for v in a["excluded_pairs"])
    if scored != p["candidates"] or excluded != sorted(set(names) - set(p["candidates"])) or len(set(scored)) != len(scored):
        raise schema.InputError("artifact coverage differs from the exact native candidate set")
    body = dict(version=ENVELOPE_VERSION, producer=PRODUCER,
        producer_sha256=sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        artifact_sha256=sha(artifact_raw), artifact_b64=base64.b64encode(artifact_raw).decode(),
        prediction_sha256=a["prediction_sha256"], projection_sha256=sha(projection_raw),
        session=p["session"], phase=request["phase"], task_uid=snapshot["tasks"][0]["id"],
        candidates=list(p["candidates"]),
        nodes=[dict(name=n["name"], uid=n["uid"]) for n in p["nodes"]],
        pairs=[dict(task_uid=v["task_id"], node=v["node_id"], score_bits=struct.pack(">d", v["score"]).hex()) for v in a["pairs"]])
    raw = schema.encoded(body)
    if len(raw) > 131072:
        raise schema.InputError("bridge envelope size")
    controls = schema.encoded(dict(alpha=request["alpha"], BF=request["BF"], BP=request["BP"], projection=P2H,
        forecast=snapshot["future"], prices=a["lp"]["prices"], price_sha256=a["lp"]["price_sha256"], shapes=a["mapping"]["shapes"],
        candidates=p["candidates"], excluded=excluded))
    return raw, artifact_raw, schema.encoded(snapshot), schema.encoded(built["execution"]), controls


if __name__ == "__main__":
    request = schema.loads(Path(sys.argv[1]).read_bytes())
    raw, artifact, snapshot, execution, controls = produce(request)
    out = Path(sys.argv[2]); out.mkdir(exist_ok=False)
    for name, data in (("envelope.json", raw), ("artifact.json", artifact), ("snapshot.json", snapshot),
                       ("execution.json", execution), ("controls.json", controls)):
        (out / name).write_bytes(data)
    sys.stdout.buffer.write(raw)
