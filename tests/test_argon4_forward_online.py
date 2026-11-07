"""Offline evidence tests: fixture candles are never production observations."""
import hashlib
import json
import sqlite3

import pandas as pd
import pytest

from tools.argon4_forward.online import (
    OnlineInputs, PublicInputsHTTP, SYMBOLS, STEPS, _merge, _parameters,
)


class FixtureHTTP:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.bad_clock = False
        self.fail_clock = False
        self.conflicting = False

    def get(self, kind, params=None):
        params = params or {}
        self.calls.append((kind, params.copy()))
        now = self.clock()
        if kind == "clock" and self.fail_clock:
            raise ValueError("fixture failure")
        if kind.endswith("clock"):
            server = now + (pd.Timedelta(seconds=20) if self.bad_clock else pd.Timedelta(0))
            payload = {"serverTime": int(server.timestamp() * 1000)}
        elif kind == "dvol":
            start = pd.to_datetime(params["start_timestamp"], unit="ms", utc=True).ceil("D")
            end = pd.to_datetime(params["end_timestamp"], unit="ms", utc=True).floor("D")
            payload = {"result": {"data": [[int(t.timestamp()*1000), 40, 42, 38, 41]
                                           for t in pd.date_range(start, end, freq="D")]}}
        elif kind == "funding":
            start = pd.to_datetime(params["startTime"], unit="ms", utc=True).ceil("8h")
            end = pd.to_datetime(params["endTime"], unit="ms", utc=True).floor("8h")
            payload = [{"symbol": params["symbol"], "fundingTime": int(t.timestamp()*1000),
                        "fundingRate": "0.0001"} for t in pd.date_range(start, end, freq="8h")][:params["limit"]]
        else:
            step = STEPS[params["interval"]]
            start = pd.to_datetime(params["startTime"], unit="ms", utc=True).ceil(step)
            end = pd.to_datetime(params["endTime"], unit="ms", utc=True).floor(step)
            dates = pd.date_range(start, end, freq=step)[:params["limit"]]
            # Fixed values ensure paging/overlap does not alter fixture history.
            payload = [[int(t.timestamp()*1000), "100", "102", "98",
                        "100.5" if self.conflicting else "101", "0" if kind == "mark" else "10",
                        int((t + step).timestamp()*1000)-1, "1000", 12, "4", "400", "0"] for t in dates]
        raw = json.dumps(payload).encode()
        return payload, {"url": "https://fixture.invalid/"+kind,
                         "requested_at": (now-pd.Timedelta(milliseconds=20)).isoformat(),
                         "received_at": now.isoformat(), "sha256": hashlib.sha256(raw).hexdigest(),
                         "http_status": 200}


@pytest.fixture
def online(tmp_path):
    moment = [pd.Timestamp("2026-11-06T15:12:00Z")]
    client = FixtureHTTP(lambda: moment[0])
    inputs = OnlineInputs(tmp_path, client=client, clock=lambda: moment[0])
    return inputs, client, moment


def test_complete_closed_inputs_have_schema_and_real_receipt_gate(online):
    inputs, client, moment = online
    status = inputs.refresh()
    assert status["status"] == "READY", status
    assert status["latest_closed_5m"] == "2026-11-06T15:10:00+00:00"
    assert status["feature_available_at"] == moment[0].isoformat()
    for symbol in SYMBOLS:
        frame = pd.read_parquet(inputs._path("futures", symbol, "5m"))
        assert len(frame) >= 4320
        assert {"open_time", "quote_volume", "count", "taker_buy_quote_volume", "first_received_at"} <= set(frame)
        assert (frame.open_time + STEPS["5m"] <= moment[0]).all()
        assert frame.first_received_at.eq(moment[0]).all()
    basis = pd.read_parquet(inputs._path("basis"))
    assert isinstance(basis.index, pd.DatetimeIndex)
    assert basis.index.name == "date"
    unavailable = inputs.snapshot(moment[0]-pd.Timedelta(seconds=1))
    assert unavailable["status"] == "BLOCKED"
    assert unavailable["latest_closed_5m"] is None
    assert all(s["rows"] == 0 for s in unavailable["sources"].values())
    evidence = inputs.mark_registry_evidence()
    assert len(evidence) == 4
    assert all(e["source_endpoint"].endswith("/markPriceKlines") and len(e["sha256"]) == 64 for e in evidence)


def test_incremental_refresh_preserves_first_receipts_and_uses_overlap(online):
    inputs, client, moment = online
    initial = inputs.refresh()
    assert initial["status"] == "READY", initial
    before = pd.read_parquet(inputs._path("futures", "BTCUSDT", "5m"))
    moment[0] += pd.Timedelta(minutes=5)
    client.calls.clear()
    assert inputs.refresh()["status"] == "READY"
    after = pd.read_parquet(inputs._path("futures", "BTCUSDT", "5m"))
    assert len(after) == len(before)+1
    pd.testing.assert_frame_equal(before, after.iloc[:-1].reset_index(drop=True))
    assert after.first_received_at.iloc[-1] == moment[0]
    call = next(p for k, p in client.calls if k == "futures" and p["symbol"] == "BTCUSDT" and p["interval"] == "5m")
    assert call["startTime"] == int((before.date.max()-STEPS["5m"]*2).timestamp()*1000)
    assert not any(k in ("mark", "spot", "dvol") or p.get("interval") in ("1h", "4h") for k, p in client.calls)
    state = inputs.snapshot()
    funding = state["sources"]["BTCUSDT:funding"]
    assert funding["latest_receipt_at"] == moment[0].isoformat()
    assert funding["latest_received_at"] == (moment[0]-pd.Timedelta(minutes=5)).isoformat()


def test_missing_bar_blocks_without_imputation(online):
    inputs, _, _ = online
    inputs.refresh()
    path = inputs._path("futures", "ETHUSDT", "5m")
    frame = pd.read_parquet(path)
    frame.drop(index=100).to_parquet(path, index=False)
    status = inputs.snapshot()
    assert status["status"] == "BLOCKED"
    assert "CANDLE_GAPS" in status["sources"]["ETHUSDT:futures:5m"]["issues"]
    assert len(pd.read_parquet(path)) == len(frame)-1


def test_changed_closed_observation_fails_and_preserves_file(online):
    inputs, client, _ = online
    inputs.refresh()
    path = inputs._path("futures", "BTCUSDT", "5m")
    before = path.read_bytes()
    client.conflicting = True
    status = inputs.refresh()
    assert status["status"] == "BLOCKED"
    assert any("CONFLICTING_CLOSED_OBSERVATION" in error["error"] for error in status["errors"])
    assert path.read_bytes() == before


def test_funding_is_event_series_with_empirical_interval_freshness(online):
    inputs, _, moment = online
    inputs.refresh()
    path = inputs._path("funding", "SOLUSDT")
    frame = pd.read_parquet(path)
    # Irregular observations remain irregular. No hourly/grid rows are inserted.
    frame = frame.drop(index=[5, 6]).reset_index(drop=True)
    frame.to_parquet(path, index=False)
    status = inputs.snapshot()
    funding = status["sources"]["SOLUSDT:funding"]
    assert funding["max_observed_interval_seconds"] == 24*3600
    assert funding["observed_interval_seconds"] == [8*3600, 24*3600]
    assert funding["resampled"] is False
    assert funding["rows"] == len(frame)
    assert len(pd.read_parquet(path)) == len(frame)
    # A long observation gap is conservative freshness evidence, not proof the
    # event history is complete, or permission to synthesize cashflows.


def test_clock_skew_blocks_and_official_spot_clock_can_fallback(online):
    inputs, client, _ = online
    client.fail_clock = True
    assert inputs.refresh()["clock"]["status"] == "VERIFIED"
    assert inputs.snapshot()["clock"]["source"].endswith("spot_clock")
    client.bad_clock = True
    result = inputs.refresh()
    assert result["status"] == "BLOCKED"
    assert result["clock"]["status"] == "CLOCK_SKEW_OR_LATENCY"


def test_absence_future_and_stale_snapshot_fail_closed(online):
    inputs, _, moment = online
    assert inputs.snapshot()["status"] == "BLOCKED"
    with pytest.raises(ValueError, match="Future"):
        inputs.snapshot(moment[0]+pd.Timedelta(minutes=1))
    inputs.refresh()
    moment[0] += pd.Timedelta(minutes=6)
    status = inputs.snapshot()
    assert status["status"] == "BLOCKED"
    assert status["clock"]["status"] == "STALE_OR_NOT_YET_RECEIVED"
    assert "LATEST_CLOSED_BAR_MISSING" in status["sources"]["BTCUSDT:futures:5m"]["issues"]


def test_parameter_allowlist_rejects_order_auth_and_arbitrary_symbols():
    for kind, params in (("order", {}), ("clock", {"apiKey": "not-a-secret"}),
                         ("futures", {"symbol": "OTHER", "interval": "5m", "startTime": 1, "endTime": 2, "limit": 2})):
        with pytest.raises(ValueError):
            _parameters(kind, params)


def test_http_uses_only_get_without_environment_or_redirects_and_saves_exact_bytes(tmp_path, monkeypatch):
    import tools.argon4_forward.online as module
    raw = b'{"serverTime": 1793977920000}'
    observed = {}

    class Response:
        status_code = 302
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): yield raw

    class Session:
        trust_env = True
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            observed.update(trust_env=self.trust_env, url=url, **kwargs)
            return Response()

    monkeypatch.setattr(module.requests, "Session", Session)
    path = tmp_path / "receipts.sqlite3"
    http = PublicInputsHTTP(path, min_spacing=0)
    with pytest.raises(ValueError, match="302"):
        http.get("clock")
    assert observed["trust_env"] is False
    assert observed["allow_redirects"] is False
    assert observed["url"] == "https://fapi.binance.com/fapi/v1/time"
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT sha256,raw,http_status FROM receipts").fetchone()
    assert row == (hashlib.sha256(raw).hexdigest(), raw, 302)


def test_duplicate_conflict_is_not_hidden_by_receipt_columns():
    row = {"date": pd.Timestamp("2026-01-01T00:00Z"), "close": 10,
           "first_received_at": pd.Timestamp("2026-01-02T00:00Z"), "receipt_sha256": "a"}
    old = pd.DataFrame([row])
    new = pd.DataFrame([{**row, "close": 11, "receipt_sha256": "b"}])
    with pytest.raises(ValueError, match="CONFLICTING"):
        _merge(old, new, ["date"])


def test_rate_limit_is_persisted_and_prevents_fallback_binance_requests(tmp_path, monkeypatch):
    import tools.argon4_forward.online as module
    now = pd.Timestamp("2026-11-06T00:00:00Z")
    until = now + pd.Timedelta(hours=1)
    raw = json.dumps({"code": -1003, "msg": f"IP banned until {int(until.timestamp()*1000)}."}).encode()
    calls = []

    class Response:
        status_code = 418
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): yield raw

    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            calls.append(url)
            return Response()

    monkeypatch.setattr(module.requests, "Session", Session)
    path = tmp_path / "receipts.sqlite3"
    http = PublicInputsHTTP(path, clock=lambda: now, min_spacing=0)
    with pytest.raises(ValueError, match="418"):
        http.get("clock")
    restarted = PublicInputsHTTP(path, clock=lambda: now, min_spacing=0)
    with pytest.raises(ValueError, match="PUBLIC_RATE_LIMIT_BACKOFF"):
        restarted.get("spot_clock")
    assert len(calls) == 1
    shared = json.loads(path.with_name("binance_backoff.json").read_text())
    assert pd.Timestamp(shared["blocked_until_utc"]) == until + pd.Timedelta(seconds=15)
    assert shared["source_receipt_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("http_status", [418, 429])
def test_concurrent_waiters_never_send_after_first_rate_limit_before_receipt_saved(tmp_path, monkeypatch, http_status):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import tools.argon4_forward.online as module

    now = pd.Timestamp("2026-11-06T00:00:00Z")
    until = now + pd.Timedelta(hours=1)
    raw = json.dumps({"code": -1003, "msg": f"IP banned until {int(until.timestamp()*1000)}."}).encode()
    calls = []
    calls_lock = threading.Lock()
    callers_ready = threading.Barrier(5)
    receipt_pending = threading.Event()
    release_receipt = threading.Event()
    unexpected_request = threading.Event()

    class Response:
        status_code = http_status
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): yield raw

    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            with calls_lock:
                calls.append(url)
                if len(calls) > 1:
                    unexpected_request.set()
            return Response()

    monkeypatch.setattr(module.requests, "Session", Session)
    path = tmp_path / "receipts.sqlite3"
    http = PublicInputsHTTP(path, clock=lambda: now, min_spacing=0)
    save = http._save

    def delayed_save(receipt, body):
        # Model the original race: the 418 has arrived, but durable evidence
        # and shared backoff have not yet been written.
        receipt_pending.set()
        assert release_receipt.wait(5), "test failed to release receipt writer"
        save(receipt, body)

    monkeypatch.setattr(http, "_save", delayed_save)

    def call(kind):
        callers_ready.wait(timeout=5)
        try:
            http.get(kind)
        except ValueError as exc:
            return str(exc)
        pytest.fail("rate-limited request unexpectedly succeeded")

    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(call, kind) for kind in ("clock", "spot_clock", "clock", "spot_clock")]
        callers_ready.wait(timeout=5)
        try:
            assert receipt_pending.wait(5), "no response reached the receipt writer"
            # All worker calls are concurrent; no response/error handling may
            # be overtaken by any of the already queued request workers.
            assert not unexpected_request.wait(.2)
            with calls_lock:
                assert len(calls) == 1
        finally:
            release_receipt.set()
        errors = [future.result(timeout=5) for future in pending]
    assert len(calls) == 1
    assert errors.count("PUBLIC_HTTP_STATUS_" + str(http_status)) == 1
    assert sum(error.startswith("PUBLIC_RATE_LIMIT_BACKOFF_UNTIL_") for error in errors) == 3
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 1
    shared = json.loads(path.with_name("binance_backoff.json").read_text())
    assert shared["source_receipt_sha256"] == hashlib.sha256(raw).hexdigest()


def test_calibrated_clock_keeps_system_receipt_and_evidence(tmp_path, monkeypatch):
    import tools.argon4_forward.online as module
    system = pd.Timestamp("2026-11-06T00:00:00Z")
    upper = system + pd.Timedelta(seconds=5.2)
    raw = b'{"serverTime": 1793923205000}'
    monkeypatch.setattr(module, "_now", lambda: system)

    class Response:
        status_code = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): yield raw

    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs): return Response()

    monkeypatch.setattr(module.requests, "Session", Session)
    inputs = OnlineInputs(tmp_path, clock=lambda: upper,
                          time_basis="CALIBRATED_UTC_UPPER_BOUND", clock_evidence_sha256="a"*64)
    _, receipt = inputs.client.get("clock")
    assert receipt["received_at_local"] == system.isoformat()
    assert receipt["received_at"] == upper.isoformat()
    assert receipt["clock_evidence_sha256"] == "a"*64
    assert inputs.snapshot()["time_basis"] == "CALIBRATED_UTC_UPPER_BOUND"
    with pytest.raises(ValueError, match="evidence"):
        OnlineInputs(tmp_path, time_basis="CALIBRATED_UTC_UPPER_BOUND")


def test_later_page_failure_preserves_actual_first_page_without_claiming_ready(online):
    inputs, client, moment = online
    original = client.get
    count = [0]

    def fail_second(kind, params):
        count[0] += 1
        if count[0] == 2:
            raise ValueError("fixture later page unavailable")
        return original(kind, params)

    client.get = fail_second
    with pytest.raises(ValueError, match="later page"):
        inputs._candles("futures", "BTCUSDT", "5m", moment[0])
    frame = pd.read_parquet(inputs._path("futures", "BTCUSDT", "5m"))
    assert len(frame) == 1000
    assert frame.first_received_at.eq(moment[0]).all()
    assert inputs.snapshot()["status"] == "BLOCKED"
