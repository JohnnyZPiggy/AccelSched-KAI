"""M6 P2H-1: 2-to-4-node heterogeneous projection, exact candidate set, fail-closed domain."""
import base64
from copy import deepcopy
import struct

import pytest

from accelsched_kai import m5_producer, p2h_producer as mp, schema

GIB = 1024 ** 3
SHAPES = [(4, 16, 64), (8, 32, 128), (2, 8, 32), (8, 16, 64)]


def raw(gpu, cores, gib, pods):
    return [cores * 1000, gib * GIB, gpu, pods]


def node(i, shape, used=(0, 0, 0), members=(), pods_used=0):
    alloc = raw(*shape, 110)
    use = [used[1] * 1000, used[2] * GIB, used[0], pods_used]
    idle = [a - u for a, u in zip(alloc, use)]
    m = mp.model_from_native
    return dict(name="node%d" % i, uid="", shape=mp.shape_of(m(alloc)), family=mp.FAMILY, allocatable=m(alloc), idle=m(idle), used=m(use),
                releasing=[0, 0, 0], native_raw=dict(allocatable=alloc, idle=idle, used=use, releasing=[0, 0, 0, 0]), members=list(members), native={})


def projection(n=4, request=(2, 2, 1), shapes=None):
    nodes = [node(i, s) for i, s in enumerate((shapes or SHAPES)[:n])]
    task = dict(uid="native-job-0", pod_uid="native-job-0", status="Pending", node="", request=list(request), accepted=[0, 0, 0],
                native_request=raw(*request, 1), native={})
    elig, cands = {}, []
    for nd in nodes:
        fit = all(i >= d for i, d in zip(nd["idle"], request))
        elig[nd["name"]] = dict(native=fit, model=fit, fitting_gpus=nd["idle"][0] if fit else 0)
        if fit:
            cands.append(nd["name"])
    return dict(version=mp.P2H, session="p2h-session", semantics=mp.SEMANTICS, units=list(schema.UNITS), index_map=dict(mp.INDEX_MAP), bounds=dict(mp.BOUNDS),
                node_names=mp.node_names(n), nodes=nodes, tasks=[task], eligibility=elig, candidates=cands, domain={}, config={}, hooks={})


def request(p=None, alpha=0., BF=4., BP=8., forecast=None):
    return dict(phase=1, projection_b64=base64.b64encode(schema.encoded(projection() if p is None else p)).decode(), alpha=alpha, BF=BF, BP=BP,
                forecast=forecast if forecast is not None else [dict(id="F0", demand=[4, 4, 1], weight=2, eligible=["node0", "node1"])])


def bits(envelope):
    return [struct.unpack(">d", bytes.fromhex(p["score_bits"]))[0] for p in schema.loads(envelope)["pairs"]]


@pytest.mark.parametrize("n", [2, 3, 4])
def test_variable_node_counts_yield_exactly_one_pair_per_native_candidate(n):
    out = mp.produce(request(projection(n)))
    env = schema.loads(out[0])
    assert env["version"] == "P2H-1" and env["candidates"] == mp.node_names(n) and [p["node"] for p in env["pairs"]] == mp.node_names(n)
    assert [v["name"] for v in env["nodes"]] == mp.node_names(n) and len(bits(out[0])) == n
    assert schema.loads(out[1])["excluded_pairs"] == []


def test_native_infeasible_nodes_are_excluded_and_candidate_list_is_exact():
    p = projection(4, request=(6, 2, 1))  # only the two 8-GPU nodes fit
    assert p["candidates"] == ["node1", "node3"]
    out = mp.produce(request(p))
    env = schema.loads(out[0])
    assert env["candidates"] == ["node1", "node3"] and [x["node"] for x in env["pairs"]] == ["node1", "node3"]
    assert sorted(x["node_id"] for x in schema.loads(out[1])["excluded_pairs"]) == ["node0", "node2"]
    assert schema.loads(out[4])["excluded"] == ["node0", "node2"]


def test_heterogeneous_shapes_and_alpha_zero_marginal_scores():
    out = mp.produce(request(projection(4), alpha=0.))
    a = schema.loads(out[1])
    assert a["mapping"]["shapes"] == ["G4C16M64", "G8C32M128", "G2C8M32", "G8C16M64"]
    for pair in a["pairs"]:
        assert pair["g"] == pair["g_oracle"] and pair["score"] == pair["AF"] == max(-1., min(1., pair["g"] / 4.))
    # the 4-GPU/4-core forecast fits node0..node3 minus node2; only node0 (4 GPUs) loses its fit when a 2-GPU task lands there
    assert bits(out[0]) == [-.5, 0., 0., 0.]
    again = mp.produce(request(projection(4), alpha=0.))
    assert [again[i] for i in (0, 1, 2, 4)] == [out[i] for i in (0, 1, 2, 4)]


def test_used_state_membership_and_idle_capacity():
    p = projection(3)
    p["nodes"][0] = node(0, SHAPES[0], used=(2, 2, 1), members=["native-job-1"], pods_used=1)
    p["tasks"].append(dict(uid="native-job-1", pod_uid="native-job-1", status="Binding", node="node0", request=[2, 2, 1], accepted=[2, 2, 1],
                           native_request=raw(2, 2, 1, 1), native={}))
    s = mp.snapshot_from_projection(p)
    assert s["nodes"][0]["capacity"] == [2, 14, 63] and s["nodes"][1]["capacity"] == [8, 32, 128]


def test_p1_and_p2_modules_are_untouched_and_distinct():
    assert m5_producer.P2 == "P2-WholeGPU-1" and mp.P2H == "P2H-1" and m5_producer.NODES == ("node0", "node1")
    with pytest.raises(schema.InputError):
        m5_producer.snapshot_from_projection(projection(3))  # a P2H projection is not accepted by the frozen P2 producer


def mutate_candidates(fn):
    def apply(p):
        fn(p["candidates"])
    return apply


@pytest.mark.parametrize("change", [
    lambda p: p.update(version="P2-WholeGPU-1"), lambda p: p.update(semantics="x"), lambda p: p.update(extra=1), lambda p: p["bounds"].update(max_nodes=8),
    lambda p: p["index_map"].update(gpu=0), lambda p: p.update(units=["a", "b", "c"]),
    # candidate set integrity
    mutate_candidates(lambda c: c.reverse()), mutate_candidates(lambda c: c.pop()), mutate_candidates(lambda c: c.append("node0")),
    mutate_candidates(lambda c: c.append("node9")), lambda p: p.update(candidates=[]), lambda p: p.update(candidates=["node0", "node0", "node1", "node2", "node3"]),
    lambda p: p["eligibility"]["node1"].update(native=False), lambda p: p["eligibility"]["node1"].update(model=False),
    lambda p: p["eligibility"]["node0"].update(fitting_gpus=0), lambda p: p["eligibility"].pop("node3"),
    # node identity/order/count
    lambda p: p["node_names"].reverse(), lambda p: p["node_names"].append("node4"), lambda p: p["nodes"].pop(),
    lambda p: p["nodes"][1].update(name="node0"), lambda p: p["nodes"][0].update(uid=None), lambda p: p["nodes"][0].update(family="OTHER"),
    lambda p: p["nodes"][0].update(shape="G9C9M9"), lambda p: p["nodes"][0].update(members=["ghost"]),
    lambda p: p.update(node_names=["node0"], nodes=p["nodes"][:1]),
    # raw native vectors / units
    lambda p: p["nodes"][0]["native_raw"]["releasing"].__setitem__(2, 1), lambda p: p["nodes"][0]["native_raw"]["idle"].__setitem__(3, 100),
    lambda p: p["nodes"][0]["native_raw"]["idle"].__setitem__(0, 3), lambda p: p["nodes"][0]["native_raw"]["allocatable"].__setitem__(0, 4.5),
    lambda p: p["nodes"][0]["native_raw"]["allocatable"].__setitem__(0, -4), lambda p: p["nodes"][0]["native_raw"]["allocatable"].__setitem__(0, True),
    lambda p: p["nodes"][0]["native_raw"]["allocatable"].append(5), lambda p: p["nodes"][0]["native_raw"]["idle"].__setitem__(1, 64 * GIB - 1),
    lambda p: p["nodes"][0]["idle"].__setitem__(0, 3),
    # tasks
    lambda p: p["tasks"][0].update(request=[1, 2, 1]), lambda p: p["tasks"][0]["native_request"].__setitem__(3, 2),
    lambda p: p["tasks"][0].update(request=[0, 0, 0], native_request=raw(0, 0, 0, 1)), lambda p: p["tasks"][0].update(status="Releasing", node="node0"),
    lambda p: p["tasks"][0].update(node="node0"), lambda p: p["tasks"][0].update(accepted=[0.5, 0, 0]),
    lambda p: p["tasks"].append(deepcopy(p["tasks"][0])), lambda p: p["tasks"].clear(), lambda p: p["tasks"][0].update(pod_uid="other"),
])
def test_unsupported_or_inconsistent_p2h_projections_fail_closed(change):
    p = projection(4)
    change(p)
    with pytest.raises((schema.InputError, KeyError, TypeError, ValueError)):
        mp.produce(request(p))


def test_phase_and_forecast_bounds_and_envelope_caps():
    for phase in (0, mp.MAX_PHASE + 1, True):
        r = request(); r["phase"] = phase
        with pytest.raises(schema.InputError):
            mp.produce(r)
    r = request(); r["phase"] = mp.MAX_PHASE
    assert mp.produce(r)
    for change in (lambda r: r.update(realized=[]), lambda r: r.pop("forecast"), lambda r: r.update(forecast=[]),
                   lambda r: r["forecast"][0].update(demand=[0, 0, 0]), lambda r: r["forecast"][0].update(demand=[1.5, 1, 1]),
                   lambda r: r["forecast"][0].update(eligible=["node9"]), lambda r: r["forecast"][0].update(weight=-1),
                   lambda r: r.update(forecast=r["forecast"] * 5), lambda r: r.update(alpha=2.)):
        bad = request(); change(bad)
        with pytest.raises((schema.InputError, KeyError, TypeError)):
            mp.produce(bad)


def test_uncertified_build_never_produces_an_envelope():
    from unittest.mock import patch
    from accelsched_kai import bounded_advice as ba
    with patch.object(ba, "build_artifact", return_value=dict(status="LP_UNCERTIFIED")):
        with pytest.raises(schema.InputError):
            mp.produce(request())
    with patch.object(ba, "load_artifact", return_value=dict(status="INVALID_ARTIFACT")):
        with pytest.raises(schema.InputError):
            mp.produce(request())
