from datetime import datetime, timedelta, timezone
import pytest
from tools.argon4_forward.clock_sync import bounds_from_samples, project_local
from tools.argon4_forward.clock_sync import calibrate
import hashlib
from types import SimpleNamespace

T = datetime(2026, 11, 6, tzinfo=timezone.utc)


def sample(start, end, server_offset=5):
    a, b = T+timedelta(seconds=start), T+timedelta(seconds=end)
    return {"requested_at_local": a.isoformat(), "received_at_local": b.isoformat(),
            "server_second": int(a.timestamp()+server_offset), "monotonic_elapsed_seconds": end-start}


def test_seconds_resolution_is_not_claimed_as_millisecond_precision():
    c = bounds_from_samples([sample(.1,.3)]*3)
    assert c["uncertainty_seconds"] > .5


def test_interval_intersection_bounds_the_real_clock_offset():
    c = bounds_from_samples([sample(.1,.3),sample(.5,.7),sample(.9,1.1)])
    assert c["offset_lower_seconds"] <= 5 <= c["offset_upper_seconds"]
    assert c["uncertainty_seconds"] < .5
    assert project_local(T,c,upper=False) <= T+timedelta(seconds=5) <= project_local(T,c)


def test_inconsistent_server_or_local_clock_step_fails_closed():
    with pytest.raises(ValueError):
        bounds_from_samples([sample(.1,.3),sample(.5,.7),sample(.9,1.1,50)])
    bad = sample(.9,1.1)
    bad["monotonic_elapsed_seconds"] = 2
    with pytest.raises(ValueError):
        bounds_from_samples([sample(.1,.3),sample(.5,.7),bad])


def test_persistent_transport_clock_keeps_six_actual_samples():
    calls = []
    def get(kind):
        i = len(calls)
        calls.append(kind)
        start = T + timedelta(seconds=.1+.4*i)
        end = start + timedelta(seconds=.2)
        second = int(start.timestamp()) + 5
        raw = str(second)
        return second, {"requested_at": start.isoformat(), "received_at": end.isoformat(),
                        "body_complete": True, "http_status": 200, "raw_text": raw,
                        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                        "transport_timing": {"request_elapsed_seconds": .2}}, None
    result = calibrate(transport=SimpleNamespace(get=get))
    assert calls == ["time"]*6
    assert result["status"] == "CALIBRATED"
    assert len(result["receipts"]) == 6
    assert result["offset_lower_seconds"] <= 5 <= result["offset_upper_seconds"]
    assert result["receipts"][0]["requested_at_local"] != result["receipts"][0]["received_at_local"]


def test_failed_clock_transport_is_not_retried_or_made_fresh():
    calls = []
    failed = {"body_complete": False, "http_status": 429, "raw_text": None}
    def get(kind):
        calls.append(kind)
        return None, failed, "HTTP_429"
    result = calibrate(transport=SimpleNamespace(get=get))
    assert calls == ["time"]
    assert result["status"] == "UNCALIBRATED" and result["receipts"] == []
    assert "observed_at" not in result
    assert result["failed_receipt"] == failed
