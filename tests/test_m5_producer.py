"""M5 P2-WholeGPU-1: independent unit derivation, fail-closed domain, parameter provenance."""
import base64
from copy import deepcopy
import struct

import pytest

from accelsched_kai import m5_producer as mp, schema

GIB = 1024 ** 3


def raw(gpu, cores, gib, pods):
    return [cores * 1000, gib * GIB, gpu, pods]


def node(i, gpus, cores, gib, used=(0, 0, 0), members=(), pods_used=0):
    alloc = raw(gpus, cores, gib, 110)
    use = [used[1] * 1000, used[2] * GIB, used[0], pods_used]
    idle = [a - u for a, u in zip(alloc, use)]
    model = lambda v: mp.model_from_native(v)
    return dict(name="node%d" % i, uid="", shape=mp.shape_of(model(alloc)), family=mp.FAMILY,
                allocatable=model(alloc), idle=model(idle), used=model(use), releasing=[0, 0, 0],
                native_raw=dict(allocatable=alloc, idle=idle, used=use, releasing=[0, 0, 0, 0]), members=list(members), native={})


def projection(request=(2, 2, 1), shapes=((4, 16, 64), (8, 16, 64))):
    nodes = [node(i, *s) for i, s in enumerate(shapes)]
    task = dict(uid="native-job-0", pod_uid="native-job-0", status="Pending", node="", request=list(request), accepted=[0, 0, 0],
                native_request=raw(request[0], request[1], request[2], 1), native={})
    fits = {n["name"]: dict(native=True, model=True, fitting_gpus=n["idle"][0]) for n in nodes}
    return dict(version=mp.P2, session="m5-session", semantics=mp.SEMANTICS, units=list(schema.UNITS), index_map=dict(mp.INDEX_MAP),
                nodes=nodes, tasks=[task], eligibility=fits, domain={}, config={}, hooks={})


def request(p=None, alpha=0., BF=4., BP=8., forecast=None, **extra):
    return dict(phase=1, projection_b64=base64.b64encode(schema.encoded(projection() if p is None else p)).decode(),
                alpha=alpha, BF=BF, BP=BP,
                forecast=forecast if forecast is not None else [dict(id="F0", demand=[4, 4, 1], weight=2, eligible=["node0", "node1"])], **extra)


def bits(envelope):
    return [struct.unpack(">d", bytes.fromhex(p["score_bits"]))[0] for p in schema.loads(envelope)["pairs"]]


def test_exact_independent_unit_conversion():
    assert mp.model_from_native([16000, 64 * GIB, 8, 110]) == [8, 16, 64]
    for bad in ([16500, 64 * GIB, 8, 110], [16000, 64 * GIB + 1, 8, 110], [16000, 64 * GIB, 8.0, 110], [16000, 64 * GIB, -1, 110],
                [16000, 64 * GIB, True, 110], [16000, 64 * GIB, 8], [float("nan"), 0, 0, 0], [10 ** 16, 0, 0, 0]):
        with pytest.raises(schema.InputError):
            mp.model_from_native(bad)


def test_snapshot_uses_idle_not_total_and_carries_whole_gpus():
    p = projection()
    p["nodes"][0] = node(0, 4, 16, 64, used=(2, 2, 1), members=["native-job-1"], pods_used=1)
    p["tasks"].append(dict(uid="native-job-1", pod_uid="native-job-1", status="Binding", node="node0", request=[2, 2, 1], accepted=[2, 2, 1],
                           native_request=raw(2, 2, 1, 1), native={}))
    p["eligibility"]["node0"]["fitting_gpus"] = 2
    s = mp.snapshot_from_projection(p)
    assert s["nodes"][0]["capacity"] == [2, 14, 63] and s["nodes"][1]["capacity"] == [8, 16, 64]
    assert s["tasks"][0]["demand"] == [2, 2, 1] and s["nodes"][0]["shape"] == "G4C16M64" and s["nodes"][1]["shape"] == "G8C16M64"


def test_alpha_zero_scores_are_the_clipped_frozen_marginal_and_alpha_blends():
    zero = mp.produce(request(alpha=0.))
    a = schema.loads(zero[1])
    for pair in a["pairs"]:
        assert pair["g"] == pair["g_oracle"] and pair["score"] == pair["AF"] == max(-1., min(1., pair["g"] / 4.))
    assert bits(zero[0]) == [-.5, 0.]  # placing the 2-GPU task on node0 destroys its only 4-GPU forecast fit; node1 keeps one
    assert bits(mp.produce(request(alpha=.5))[0]) != bits(zero[0])
    controls = schema.loads(zero[4])
    assert controls["projection"] == mp.P2 and controls["alpha"] == 0 and controls["shapes"] == ["G4C16M64", "G8C16M64"]
    again = mp.produce(request(alpha=0.))
    assert [again[i] for i in (0, 1, 2, 4)] == [zero[i] for i in (0, 1, 2, 4)]


def test_same_shape_nodes_have_constant_lp_price_and_equal_blend_offset():
    p = projection(shapes=((8, 16, 64), (8, 16, 64)))
    a = schema.loads(mp.produce(request(p, alpha=.5))[1])
    assert a["mapping"]["shapes"] == ["G8C16M64"] and len({pair["p"] for pair in a["pairs"]}) == 1


@pytest.mark.parametrize("change", [
    lambda p: p.update(version="P1"), lambda p: p.update(semantics="x"), lambda p: p.update(extra=1),
    lambda p: p["index_map"].update(gpu=0), lambda p: p.update(units=["a", "b", "c"]),
    lambda p: p["nodes"][0]["native_raw"]["releasing"].__setitem__(2, 1),                       # releasing
    lambda p: p["nodes"][0]["native_raw"]["idle"].__setitem__(3, 100),                            # pod slot accounting
    lambda p: p["nodes"][0]["native_raw"]["idle"].__setitem__(0, 3),                              # GPU idle+used != total
    lambda p: p["nodes"][0]["native_raw"]["allocatable"].__setitem__(0, 4.5),                     # fractional GPU
    lambda p: p["nodes"][0]["native_raw"]["allocatable"].__setitem__(0, -4),
    lambda p: p["nodes"][0]["native_raw"]["allocatable"].__setitem__(0, True),
    lambda p: p["nodes"][0]["native_raw"]["allocatable"].append(5),                               # extra coordinate
    lambda p: p["nodes"][0]["native_raw"]["idle"].__setitem__(1, 64 * GIB - 1),                   # RAM unit mismatch
    lambda p: p["nodes"][0]["idle"].__setitem__(0, 3),                                            # Go/Python disagreement
    lambda p: p["nodes"][0].update(shape="G9C9M9"), lambda p: p["nodes"][0].update(family="OTHER_GPU"),
    lambda p: p["nodes"][0].update(uid=None), lambda p: p["nodes"][0].update(name="node9"),
    lambda p: p["nodes"][1].update(name="node0"), lambda p: p["nodes"].pop(),
    lambda p: p["tasks"][0].update(request=[1, 2, 1]),                                            # request != native
    lambda p: p["tasks"][0]["native_request"].__setitem__(3, 2),                                  # more than one pod slot
    lambda p: p["tasks"][0].update(request=[0, 0, 0], native_request=raw(0, 0, 0, 1)),            # zero demand
    lambda p: p["tasks"][0].update(status="Releasing", node="node0"),
    lambda p: p["tasks"][0].update(node="node0"),                                                 # Pending with a node
    lambda p: p["tasks"][0].update(accepted=[0.5, 0, 0]),
    lambda p: p["tasks"].append(deepcopy(p["tasks"][0])),                                         # duplicate / two Pending
    lambda p: p["tasks"].clear(),
    lambda p: p["eligibility"]["node1"].update(native=False),
    lambda p: p["eligibility"]["node1"].update(model=False),
    lambda p: p["eligibility"]["node0"].update(fitting_gpus=0),
    lambda p: p["eligibility"].pop("node1"),
    lambda p: p["nodes"][0].update(members=["ghost"]),
])
def test_unsupported_or_inconsistent_native_projections_fail_closed(change):
    p = projection()
    change(p)
    with pytest.raises((schema.InputError, KeyError, TypeError)):
        mp.produce(request(p))


def test_model_ineligible_node_is_refused_even_if_native_flags_lie():
    p = projection(request=(6, 2, 1))  # node0 has only 4 GPUs
    with pytest.raises(schema.InputError):
        mp.produce(request(p))


@pytest.mark.parametrize("change", [
    lambda r: r.update(realized=[]), lambda r: r.pop("forecast"), lambda r: r.update(forecast=[]),
    lambda r: r["forecast"][0].update(demand=[0, 0, 0]), lambda r: r["forecast"][0].update(demand=[1.5, 1, 1]),
    lambda r: r["forecast"][0].update(eligible=["node9"]), lambda r: r["forecast"][0].update(weight=-1),
    lambda r: r.update(forecast=r["forecast"] * 5), lambda r: r.update(alpha=2.), lambda r: r.update(phase=4),
])
def test_invalid_requests_fail_closed(change):
    r = request()
    change(r)
    with pytest.raises((schema.InputError, KeyError, TypeError)):
        mp.produce(r)


def test_uncertified_build_never_produces_an_envelope():
    from unittest.mock import patch
    from accelsched_kai import bounded_advice as ba
    with patch.object(ba, "build_artifact", return_value=dict(status="LP_UNCERTIFIED")):
        with pytest.raises(schema.InputError):
            mp.produce(request())
    with patch.object(ba, "load_artifact", return_value=dict(status="INVALID_ARTIFACT")):
        with pytest.raises(schema.InputError):
            mp.produce(request())
