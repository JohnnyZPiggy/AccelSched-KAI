"""M7 P2H-PC-1: planning cohort provenance, certified LP price visibility, unchanged single-target behaviour."""
import base64
from copy import deepcopy
import struct

import pytest

from accelsched_kai import p2h_producer as base, p2hpc_producer as pc, schema
from test_p2h_producer import node, projection, raw

FORE = [dict(id="F0", demand=[4, 4, 1], weight=2, eligible=["node0", "node1"])]


def scarce_projection():
    """node0: 4-GPU Shape; node1: 8-GPU Shape. A 1-GPU target; a queued 8-GPU and a 4-GPU job compete for GPUs."""
    return projection(2, request=(1, 2, 1), shapes=[(4, 16, 64), (8, 32, 128)])


def cohort(members, phase=1, source="PUBLISHED_QUEUE_LOOKAHEAD", arrival=0, horizon=None):
    return dict(source=source, as_of_arrival=arrival, as_of_phase=phase, horizon=len(members) if horizon is None else horizon,
                members=[dict(id="queue:%d" % i, demand=list(d)) for i, d in enumerate(members)])


def request(p=None, alpha=.5, members=((8, 8, 4), (4, 4, 2)), **kw):
    return dict(phase=1, projection_b64=base64.b64encode(schema.encoded(scarce_projection() if p is None else p)).decode(), alpha=alpha, BF=4., BP=8.,
                forecast=deepcopy(FORE), cohort=kw.pop("cohort", cohort(members)), **kw)


def target_scores(envelope):
    return [struct.unpack(">d", bytes.fromhex(p["score_bits"]))[0] for p in schema.loads(envelope)["pairs"]]


def test_empty_cohort_equals_the_frozen_single_target_producer_scores():
    single = base.produce(dict(phase=1, projection_b64=request()["projection_b64"], alpha=.5, BF=4., BP=8., forecast=deepcopy(FORE)))
    plan = pc.produce(request(members=()))
    assert target_scores(single[0]) == target_scores(plan[0])
    assert schema.loads(plan[0])["version"] == "P2H-PC-1" and schema.loads(plan[0])["cohort_ids"] == []
    assert schema.loads(plan[4])["nonzero_price"] is False


def test_single_pending_price_is_zero_but_cohort_scarcity_gives_a_certified_nonzero_price():
    out = pc.produce(request())
    a, c = schema.loads(out[1]), schema.loads(out[4])
    assert a["lp"]["certificate"]["certified"] and c["nonzero_price"] is True
    prices = c["prices"]
    assert any(v != 0 for row in prices for v in row) and c["shapes"] == ["G4C16M64", "G8C32M128"]
    # priced Shape/resource: the GPU column of at least one Shape
    assert any(row[0] != 0 for row in prices)
    # chain: p = -price.demand, AP = clip(p/BP), score = (1-alpha)AF + alpha AP
    for t in c["target_pairs"]:
        assert t["AP"] == max(-1., min(1., t["p"] / 8.)) and abs(t["score"] - (.5 * t["AF"] + .5 * t["AP"])) < 1e-12


def test_cohort_price_changes_target_scores_but_not_the_pgfp_marginal():
    zero_alpha = pc.produce(request(alpha=0.))
    mixed = pc.produce(request(alpha=.5))
    assert [t["AF"] for t in schema.loads(zero_alpha[4])["target_pairs"]] == [t["AF"] for t in schema.loads(mixed[4])["target_pairs"]]
    assert target_scores(zero_alpha[0]) != target_scores(mixed[0])
    # no price is injected: the exported pairs are exactly the artifact's target pairs
    a = schema.loads(mixed[1])
    assert [p["score"] for p in a["pairs"] if p["task_id"] == "native-job-0"] == target_scores(mixed[0])


def test_cohort_provenance_is_hashed_into_the_envelope_and_changes_with_content():
    a, b = pc.produce(request()), pc.produce(request(members=((8, 8, 4),)))
    ea, eb = schema.loads(a[0]), schema.loads(b[0])
    assert ea["cohort_sha256"] == pc.cohort_hash(request()["cohort"]) and ea["cohort_sha256"] != eb["cohort_sha256"]
    assert ea["cohort_ids"] == ["queue:0", "queue:1"] and ea["cohort_source"] == "PUBLISHED_QUEUE_LOOKAHEAD"
    again = pc.produce(request())
    assert [again[i] for i in (0, 1, 2, 4)] == [a[i] for i in (0, 1, 2, 4)]


@pytest.mark.parametrize("change", [
    lambda r: r["cohort"].update(source="INVENTED"), lambda r: r["cohort"].pop("source"), lambda r: r["cohort"].update(as_of_phase=2),
    lambda r: r["cohort"].update(as_of_arrival=-1), lambda r: r["cohort"].update(horizon=1), lambda r: r["cohort"].update(horizon=99),
    lambda r: r["cohort"]["members"][0].update(id="native-job-0"), lambda r: r["cohort"]["members"][1].update(id="queue:0"),
    lambda r: r["cohort"]["members"][0].update(demand=[0, 0, 0]), lambda r: r["cohort"]["members"][0].update(demand=[1.5, 1, 1]),
    lambda r: r["cohort"]["members"][0].update(extra=1), lambda r: r["cohort"].update(extra=1),
    lambda r: r["cohort"].update(members=[dict(id="q%d" % i, demand=[1, 1, 1]) for i in range(9)], horizon=9),
    lambda r: r.pop("cohort"), lambda r: r.update(realized=[]),
])
def test_invalid_or_leaky_cohort_requests_fail_closed(change):
    r = request()
    change(r)
    with pytest.raises((schema.InputError, KeyError, TypeError)):
        pc.produce(r)


def test_oracle_information_source_is_explicitly_labelled_in_the_envelope():
    out = pc.produce(request(cohort=cohort([(8, 8, 4)], source="OFFLINE_ORACLE_INFORMATION_NOT_NATIVE")))
    assert schema.loads(out[0])["cohort_source"] == "OFFLINE_ORACLE_INFORMATION_NOT_NATIVE"
    assert pc.NATIVE_SOURCES == ("PUBLISHED_QUEUE_LOOKAHEAD", "DECLARED_FORECAST")


def test_frozen_p2h_projection_checks_still_apply():
    p = scarce_projection()
    p["candidates"].reverse()
    with pytest.raises(schema.InputError):
        pc.produce(request(p))


def test_uncertified_artifact_never_yields_an_envelope():
    from unittest.mock import patch
    from accelsched_kai import bounded_advice as ba
    with patch.object(ba, "build_artifact", return_value=dict(status="LP_UNCERTIFIED")):
        with pytest.raises(schema.InputError):
            pc.produce(request())
    with patch.object(ba, "load_artifact", return_value=dict(status="INVALID_ARTIFACT")):
        with pytest.raises(schema.InputError):
            pc.produce(request())
