from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from tools.argon4_forward import risk_view as rv


T = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)


def clock():
    return {"status": "CALIBRATED", "source": rv.SOURCE, "server_minus_local_seconds": 2.,
            "uncertainty_seconds": .3, "calibrated_monotonic": 95.,
            "calibrated_at_local": (T-timedelta(seconds=5)).isoformat(), "receipt_sha256": "a"*64}


def position(identity, end_seconds, *, coin="BTC", cost=10., fee=.1):
    return {"order_id": identity, "underlying": coin, "cost_usd": cost, "fee_usd": fee,
            "window_end": (T+timedelta(seconds=end_seconds)).isoformat()}


@pytest.fixture(autouse=True)
def fixed_monotonic(monkeypatch):
    monkeypatch.setattr(rv.time, "monotonic", lambda: 100.)


def test_partition_keeps_all_fees_and_risk_and_does_not_mutate_input():
    s={"locked_cost": 30.3, "positions": [position("active", 10), position("pending", 0),
       position("boundary", 2, coin="SOL")], "cash": 10000.}
    original=deepcopy(s); c=clock(); original_clock=deepcopy(c)
    r=rv.split_open_risk(s, c, T)
    assert s == original and c == original_clock
    assert r["active_window_cost_usd"] == r["awaiting_resolution_cost_usd"] == r["unknown_cost_usd"] == 10.1
    assert r["total_locked_cost_usd"] == 30.3 and r["locked_cost_atoms"] == 30_300_000
    assert r["reconciled"] is True and r["status"] == "PARTIAL_UNKNOWN"
    assert r["by_underlying"]["BTC"]["total_locked_cost_usd"] == 20.2
    assert r["by_underlying"]["SOL"]["unknown_cost_usd"] == 10.1
    assert r["risk_released_usd"] == r["reusable_cash_from_unsettled_usd"] == 0
    assert r["awaiting_resolution_releases_risk"] is False
    assert [p["risk_state"] for p in r["positions"]] == [rv.ACTIVE, rv.PENDING, rv.UNKNOWN]


@pytest.mark.parametrize("endpoint, expected", [(1.7,rv.PENDING),(2.3,rv.UNKNOWN),(2.300001,rv.ACTIVE)])
def test_clock_interval_endpoint_semantics(endpoint, expected):
    r=rv.split_open_risk({"locked_cost":10.1,"positions":[position("x",endpoint)]},clock(),T)
    assert r["positions"][0]["risk_state"] == expected


def test_aware_as_of_and_explicit_now_without_clock_do_not_certify_expiry():
    r=rv.split_open_risk({"locked_cost":10.1,"positions":[position("x",-100)],"as_of":T.isoformat()},now=T)
    assert r["unknown_cost_usd"] == 10.1 and r["time_bounds"] is None
    assert r["positions"][0]["reason"] == "CLOCK_UNAVAILABLE"


@pytest.mark.parametrize("change", [
    {"status":"UNCALIBRATED"}, {"source":"HOST_CLOCK"}, {"uncertainty_seconds":.51},
    {"uncertainty_seconds":-.1}, {"uncertainty_seconds":float("nan")},
    {"server_minus_local_seconds":float("inf")}, {"calibrated_monotonic":39.},
    {"calibrated_monotonic":101.}, {"receipt_sha256":"unverified"},
    {"calibrated_at_local":T.isoformat()}, {"calibrated_at_local":"2026-11-07T12:00:00"},
])
def test_invalid_stale_or_stepped_clock_never_classifies_past_window(change):
    c={**clock(),**change}
    r=rv.split_open_risk({"locked_cost":10.1,"positions":[position("x",-100)]},c,T)
    assert r["unknown_cost_usd"] == 10.1 and r["awaiting_resolution_cost_usd"] == 0
    assert r["clock_status"] == "UNAVAILABLE"


@pytest.mark.parametrize("end", [None,"bad","2026-11-07T10:00:00"])
def test_invalid_window_end_is_unknown_and_stays_locked(end):
    p=position("x",-100);p["window_end"]=end
    r=rv.split_open_risk({"locked_cost":10.1,"positions":[p]},clock(),T)
    assert r["unknown_cost_usd"] == 10.1
    assert r["positions"][0]["reason"] == "WINDOW_END_INVALID"


def test_timezone_offsets_are_compared_in_utc():
    p=position("x",0);p["window_end"]="2026-11-07T20:00:00+08:00"
    r=rv.split_open_risk({"locked_cost":10.1,"positions":[p]},clock(),T)
    assert r["awaiting_resolution_cost_usd"] == 10.1
    assert r["positions"][0]["window_end"] == T.isoformat()


def test_micro_precision_reconciliation_and_zero_cost_position_count():
    ps=[position("x",-2,cost=.000001,fee=.000002),position("zero",2,cost=0,fee=0)]
    r=rv.split_open_risk({"locked_cost":.000003,"positions":ps},clock(),T)
    assert r["locked_cost_atoms"] == 3 and r["counts"][rv.UNKNOWN] == 1
    assert r["status"] == "PARTIAL_UNKNOWN"


@pytest.mark.parametrize("bad", [None,True,-1,float("nan"),float("inf"),"not-money",.0000001])
def test_invalid_money_raises_instead_of_fabricating_unknown_amount(bad):
    with pytest.raises(rv.RiskViewError):
        rv.split_open_risk({"locked_cost":10.1,"positions":[position("x",0,cost=bad)]},clock(),T)


def test_mismatched_total_missing_fee_and_duplicate_identity_fail_closed():
    s={"locked_cost":10.2,"positions":[position("x",0)]}
    with pytest.raises(rv.RiskViewError,match="reconcile"):
        rv.split_open_risk(s,clock(),T)
    s["locked_cost"]=10.1;del s["positions"][0]["fee_usd"]
    with pytest.raises(rv.RiskViewError,match="fee required"):
        rv.split_open_risk(s,clock(),T)
    with pytest.raises(rv.RiskViewError,match="duplicated"):
        rv.split_open_risk({"locked_cost":20.2,"positions":[position("x",0),position("x",1)]},clock(),T)


def test_empty_account_is_reconciled_without_inventing_a_clock():
    r=rv.split_open_risk({"locked_cost":0,"positions":[]})
    assert r["reconciled"] is True and r["total_locked_cost_usd"] == 0
    assert r["clock_status"] == "UNAVAILABLE" and r["positions"] == []


def test_production_now_path_uses_one_real_local_instant(monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls,tz=None): return T
    monkeypatch.setattr(rv,"datetime",FixedDateTime)
    r=rv.split_open_risk({"locked_cost":10.1,"positions":[position("x",0)]},clock())
    assert r["awaiting_resolution_cost_usd"] == 10.1
