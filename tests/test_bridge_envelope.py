"""Offline bridge format tests; these synthetic inputs do not mint Go grants."""
import base64
from copy import deepcopy
import hashlib
import struct
from unittest.mock import patch

import pytest
from accelsched_kai import bridge_envelope as bridge, bounded_advice as ba, schema


def projection():
    return dict(version="P1", session="bridge-session", semantics="AUDITED_B4_CPU_ONLY_SINGLE_PENDING",
        nodes=[dict(name="node"+str(i), uid="", shape="S", family="CPU_ONLY_TEST",
                    allocatable=[0, cpu*1000, 8*1024**3], idle=[0, cpu*1000, 8*1024**3],
                    used=[0, 0, 0], releasing=[0, 0, 0], members=[], native={}) for i, cpu in enumerate([4,8])],
        tasks=[dict(uid="native-job-0", pod_uid="native-job-0", status="Pending", node="",
                    request=[0, 2000, 1024**3], accepted=[0,0,0], native={})], domain={},config={},hooks={})


def request(p=None):
    return dict(phase=1, projection_b64=base64.b64encode(schema.encoded(projection() if p is None else p)).decode())


def test_real_m3a_roundtrip_envelope_and_float_bits():
    req=request(); raw, artifact, snapshot, execution=bridge.produce(req)
    env=schema.loads(raw); a=schema.loads(artifact)
    assert raw==schema.encoded(env)
    assert hashlib.sha256(artifact).hexdigest()==env["artifact_sha256"]
    assert ba.load_artifact(artifact,schema.loads(snapshot))["status"]==ba.STATUS
    assert [struct.unpack(">d",bytes.fromhex(p["score_bits"]))[0] for p in env["pairs"]]==[-.25,0.]
    assert a["pairs"][0]["g"]==-1 and a["pairs"][1]["g"]==0
    assert schema.loads(execution)["status"]=="OPTIMAL"


@pytest.mark.parametrize("change",[
    lambda p:p.update(version="wrong"),lambda p:p.update(extra="unsupported"),
    lambda p:p["nodes"][0]["idle"].__setitem__(1,3999),
    lambda p:p["nodes"][0]["idle"].__setitem__(2,8*1024**3-1),
    lambda p:p["nodes"][0]["idle"].__setitem__(0,1),
    lambda p:p["nodes"][0]["used"].__setitem__(1,1000),
    lambda p:p["nodes"][0]["releasing"].__setitem__(1,1000),
    lambda p:p["nodes"][0].update(shape="foreign"),lambda p:p["nodes"][0].update(family="GPU"),
    lambda p:p["nodes"][0].update(name="node1"),lambda p:p["nodes"].reverse(),
    lambda p:p["tasks"][0].update(pod_uid="reused"),lambda p:p["tasks"][0].update(status="Allocated"),
    lambda p:p["tasks"].append(deepcopy(p["tasks"][0])),lambda p:p["tasks"].clear(),
    lambda p:p["nodes"][0]["members"].append("foreign"),
])
def test_projection_faults_reject(change):
    p=projection();change(p)
    with pytest.raises((schema.InputError,ValueError,KeyError)):
        bridge.produce(request(p))


def test_validator_cannot_be_replaced_with_claimed_status():
    with patch.object(ba,"load_artifact",return_value={"status":"FAKE_CERTIFIED"}):
        with pytest.raises(schema.InputError):bridge.produce(request())
    req=request();req["certified"]=True
    with pytest.raises(schema.InputError):bridge.produce(req)


@pytest.mark.parametrize("phase",[0,True,1.0,4,"1"])
def test_phase_types(phase):
    req=request();req["phase"]=phase
    with pytest.raises(schema.InputError):bridge.produce(req)
