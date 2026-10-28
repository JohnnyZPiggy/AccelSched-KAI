"""M4-B controlled producer: parameter provenance, M3-B equivalence and fail-closed requests."""
import base64
from copy import deepcopy
import struct

import pytest

from accelsched_kai import bounded_advice as ba, bridge_envelope as bridge, m4b_producer as mp, schema
from test_bridge_envelope import projection

M3B_FORECAST = [dict(id="fixed-test-forecast", demand=[0, 4, 1], weight=1, eligible=["node0"])]


def request(alpha=.5, BF=2., BP=2., forecast=None, p=None, **extra):
    raw = schema.encoded(projection() if p is None else p)
    return dict(phase=1, projection_b64=base64.b64encode(raw).decode(), alpha=alpha, BF=BF, BP=BP,
                forecast=deepcopy(M3B_FORECAST if forecast is None else forecast), **extra)


def bits(envelope):
    return [struct.unpack(">d", bytes.fromhex(p["score_bits"]))[0] for p in schema.loads(envelope)["pairs"]]


def test_equivalent_to_frozen_m3b_producer_under_its_constants():
    old = bridge.produce(dict(phase=1, projection_b64=request()["projection_b64"]))
    new = mp.produce(request())
    assert old[1] == new[1] and old[2] == new[2]  # artifact and snapshot bytes (execution holds volatile timing)
    assert bits(old[0]) == bits(new[0]) == [-.25, 0.]
    assert schema.loads(old[0])["producer_sha256"] != schema.loads(new[0])["producer_sha256"]


def test_alpha_zero_is_genuine_upstream_marginal_and_alpha_changes_the_blend():
    zero = mp.produce(request(alpha=0.))
    artifact = schema.loads(zero[1])
    assert artifact["normalization"]["alpha"] == 0
    for pair in artifact["pairs"]:
        assert pair["g"] == pair["g_oracle"] and pair["score"] == pair["AF"] == max(-1., min(1., pair["g"] / 2.))
    assert bits(zero[0]) == [-.5, 0.]
    assert bits(mp.produce(request(alpha=.5))[0]) != bits(zero[0])
    controls = schema.loads(zero[4])
    assert controls["alpha"] == 0 and controls["BF"] == 2 and controls["forecast"][0]["id"] == "fixed-test-forecast"
    assert controls["price_sha256"] == artifact["lp"]["price_sha256"]


def test_forecast_is_bound_into_the_prediction_and_changes_advice():
    other = [dict(id="f", demand=[0, 2, 1], weight=3, eligible=["node0", "node1"])]
    a, b = mp.produce(request(alpha=0.)), mp.produce(request(alpha=0., forecast=other))
    assert schema.loads(a[0])["prediction_sha256"] != schema.loads(b[0])["prediction_sha256"]
    assert bits(a[0]) != bits(b[0])
    again = mp.produce(request(alpha=0.))
    assert [again[i] for i in (0, 1, 2, 4)] == [a[i] for i in (0, 1, 2, 4)]  # deterministic bytes (execution holds timing)


@pytest.mark.parametrize("change", [
    lambda r: r.update(realized=[]), lambda r: r.pop("forecast"), lambda r: r.update(forecast=[]),
    lambda r: r["forecast"][0].update(demand=[1, 4, 1]), lambda r: r["forecast"][0].update(eligible=["node9"]),
    lambda r: r["forecast"][0].update(eligible=[]), lambda r: r["forecast"][0].update(weight=-1),
    lambda r: r["forecast"][0].update(demand=[0, 4]), lambda r: r["forecast"][0].update(extra=1),
    lambda r: r.update(forecast=r["forecast"] * 5), lambda r: r.update(alpha=2.), lambda r: r.update(BF=0.),
    lambda r: r.update(phase=4), lambda r: r.update(alpha=True),
])
def test_invalid_requests_fail_closed(change):
    r = request()
    change(r)
    with pytest.raises((schema.InputError, KeyError, TypeError)):
        mp.produce(r)


def test_uncertified_build_never_produces_an_envelope():
    from unittest.mock import patch
    with patch.object(ba, "build_artifact", return_value=dict(status="LP_UNCERTIFIED")):
        with pytest.raises(schema.InputError):
            mp.produce(request())
    with patch.object(ba, "load_artifact", return_value=dict(status="INVALID_ARTIFACT")):
        with pytest.raises(schema.InputError):
            mp.produce(request())
