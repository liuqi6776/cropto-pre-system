"""Offline transport timing and safety: no service, real data or network writes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import sqlite3
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest
import requests

from tools.argon4_forward import online


NOW = pd.Timestamp("2026-11-06T03:00:05Z")


class SimulatedClock:
    def __init__(self, origin=NOW):
        self.origin = origin
        self.elapsed = 100.
        self.sleeps = []

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.elapsed += seconds

    def utc(self):
        return self.origin + pd.Timedelta(round((self.elapsed-100)*1_000_000_000), unit="ns")


@pytest.mark.parametrize("count", [17, 23])
@pytest.mark.parametrize("spacing", [1., .25])
def test_reused_session_serial_spacing_and_real_receipts(tmp_path, monkeypatch, spacing, count):
    clock = SimulatedClock()
    monkeypatch.setattr(online, "time", clock)
    monkeypatch.setattr(online, "_now", clock.utc)
    sessions, starts, payloads = [], [], []

    class Session:
        def __init__(self):
            self.cookies = requests.cookies.RequestsCookieJar()
            self.cookies.set("old_cookie", "fixture")
            self.proxies = {"https": "https://fixture.invalid"}
            self.closed = False
            sessions.append(self)

        def get(self, url, **kwargs):
            assert not self.closed and self.trust_env is False and self.auth is None
            assert not self.cookies and not self.proxies
            assert kwargs["allow_redirects"] is False and kwargs["stream"] is True
            assert kwargs["timeout"] == (4, 12)
            starts.append(clock.elapsed)
            raw = json.dumps({"serverTime": int(clock.utc().timestamp()*1000)}).encode()
            payloads.append(raw)
            owner = self
            class Response:
                status_code = 200
                headers = {}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def iter_content(self, size):
                    clock.elapsed += .05
                    owner.cookies.set("response_cookie", "must_not_propagate")
                    yield raw
            return Response()

        def close(self): self.closed = True

    monkeypatch.setattr(online.requests, "Session", Session)
    client = online.PublicInputsHTTP(tmp_path / "receipts.sqlite3", clock=clock.utc, min_spacing=spacing)
    receipts = [client.get("clock")[1] for _ in range(count)]
    assert len(sessions) == 1
    assert clock.elapsed-100 == pytest.approx((count-1)*spacing+.05)
    assert all(b-a >= spacing-1e-10 for a, b in zip(starts, starts[1:]))
    assert not sessions[0].cookies
    with sqlite3.connect(client.path) as conn:
        stored = conn.execute("SELECT requested_at, received_at, raw, sha256 FROM receipts ORDER BY id").fetchall()
    for receipt, row, raw in zip(receipts, stored, payloads):
        assert row == (receipt["requested_at"], receipt["received_at"], raw, hashlib.sha256(raw).hexdigest())
        assert pd.Timestamp(receipt["received_at"])-pd.Timestamp(receipt["requested_at"]) == pd.Timedelta(milliseconds=50)
        assert receipt["received_at"] == receipt["received_at_local"]
    client.close()
    assert sessions[0].closed and client._session is None


def test_transport_failure_has_no_automatic_retry_and_next_call_rebuilds_session(tmp_path, monkeypatch):
    sessions, calls = [], []
    class Session:
        def __init__(self):
            self.closed = False
            sessions.append(self)
        def get(self, url, **kwargs):
            calls.append(url)
            if len(sessions) == 1:
                raise requests.ConnectionError("fixture disconnect")
            class Response:
                status_code = 200
                headers = {}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def iter_content(self, size): yield b'{"serverTime": 1}'
            return Response()
        def close(self): self.closed = True
    monkeypatch.setattr(online.requests, "Session", Session)
    client = online.PublicInputsHTTP(tmp_path / "receipts.sqlite3", min_spacing=0)
    with pytest.raises(requests.ConnectionError): client.get("clock")
    assert len(calls) == 1 and sessions[0].closed
    assert not client.path.exists()  # no fictitious completed response
    assert client.get("clock")[0] == {"serverTime": 1}
    assert len(calls) == 2 and len(sessions) == 2
    client.close()
    client.close()
    assert sessions[1].closed


@pytest.mark.parametrize("status", [418, 429])
def test_fast_queue_waits_for_durable_backoff_and_restart_cannot_bypass(tmp_path, monkeypatch, status):
    now = pd.Timestamp("2026-11-06T03:00:05Z")
    raw = b'{"code": -1003, "msg": "fixture rate limited"}'
    pending_receipt, release_receipt = threading.Event(), threading.Event()
    unexpected = threading.Event()
    barrier = threading.Barrier(5)
    calls, sessions = [], []
    class Session:
        def __init__(self):
            self.closed = False
            sessions.append(self)
        def get(self, url, **kwargs):
            calls.append(url)
            if len(calls) > 1: unexpected.set()
            class Response:
                status_code = status
                headers = {"Retry-After": "120"}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def iter_content(self, size): yield raw
            return Response()
        def close(self): self.closed = True
    monkeypatch.setattr(online.requests, "Session", Session)
    path = tmp_path / "receipts.sqlite3"
    client = online.PublicInputsHTTP(path, clock=lambda: now, min_spacing=.25)
    save = client._save
    def blocked_save(receipt, body):
        pending_receipt.set()
        assert release_receipt.wait(5)
        save(receipt, body)
    monkeypatch.setattr(client, "_save", blocked_save)
    def get(kind):
        barrier.wait(5)
        with pytest.raises(ValueError) as error: client.get(kind)
        return str(error.value)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(get, k) for k in ("clock", "spot_clock", "clock", "spot_clock")]
        barrier.wait(5)
        try:
            assert pending_receipt.wait(5)
            assert not unexpected.wait(.2)
            assert len(calls) == 1
        finally:
            release_receipt.set()
        errors = [future.result(5) for future in futures]
    assert errors.count("PUBLIC_HTTP_STATUS_"+str(status)) == 1
    assert sum(e.startswith("PUBLIC_RATE_LIMIT_BACKOFF_UNTIL_") for e in errors) == 3
    restarted = online.PublicInputsHTTP(path, clock=lambda: now, min_spacing=.25)
    with pytest.raises(ValueError, match="PUBLIC_RATE_LIMIT_BACKOFF"):
        restarted.get("spot_clock")
    assert len(calls) == len(sessions) == 1 and sessions[0].closed
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT raw,http_status FROM receipts").fetchall() == [(raw, status)]
    shared = json.loads(path.with_name("binance_backoff.json").read_text())
    assert pd.Timestamp(shared["blocked_until_utc"]) == now+pd.Timedelta(seconds=135)


@pytest.mark.parametrize("spacing", [True, False, -.1, 0., .249, float("nan"), float("inf"), 61])
def test_runtime_spacing_is_explicit_and_bounded(tmp_path, spacing):
    with pytest.raises(ValueError, match="spacing"):
        online.OnlineInputs(tmp_path, http_min_spacing_seconds=spacing)


def test_legacy_default_and_explicit_incremental_spacing_and_optional_close(tmp_path):
    slow = online.OnlineInputs(tmp_path / "slow")
    fast = online.OnlineInputs(tmp_path / "fast", http_min_spacing_seconds=.25)
    assert slow.client.min_spacing == 1 and fast.client.min_spacing == .25
    assert slow.client._session is fast.client._session is None
    closed = []
    injected = online.OnlineInputs(tmp_path / "injected", client=SimpleNamespace(close=lambda: closed.append(True)))
    injected.close()
    assert closed == [True]
    online.OnlineInputs(tmp_path / "legacy", client=SimpleNamespace(get=lambda *_: None)).close()


@pytest.mark.parametrize("cutoff,count,hourly,four_hourly,daily", [
    (pd.Timestamp("2026-11-06T03:00:05Z"), 17, 8, 0, 0),
    (pd.Timestamp("2026-11-06T04:00:05Z"), 23, 8, 6, 0),
    (pd.Timestamp("2026-11-06T00:05:05Z"), 11, 0, 0, 2),
])
def test_fast_refresh_preserves_complete_due_source_set_and_observed_values(
        tmp_path, monkeypatch, cutoff, count, hourly, four_hourly, daily):
    # Reuse the existing complete offline provider; these are fixture candles,
    # never observations or an edit to a running collector's files.
    from test_argon4_forward_online import FixtureHTTP
    seed_time = cutoff-pd.Timedelta(minutes=5)
    provider = FixtureHTTP(lambda: seed_time)
    seed = online.OnlineInputs(tmp_path / "seed", client=provider, clock=lambda: seed_time)
    assert seed.refresh()["status"] == "READY"
    results, all_frames = {}, {}
    for label, spacing in (("legacy", 1.), ("fast", .25)):
        root = tmp_path / label
        shutil.copytree(seed.root, root / "features")
        clock = SimulatedClock(cutoff)
        monkeypatch.setattr(online, "time", clock)
        monkeypatch.setattr(online, "_now", clock.utc)
        fixture = FixtureHTTP(clock.utc)
        sessions = []
        class Session:
            def __init__(self): sessions.append(self)
            def get(self, url, **kwargs):
                parsed = urlsplit(url)
                kind = next(k for k, (base, path) in online.ENDPOINTS.items()
                            if base+path == parsed.scheme+"://"+parsed.netloc+parsed.path)
                params = {k: (int(v[0]) if k in ("startTime", "endTime", "limit", "start_timestamp", "end_timestamp") else v[0])
                          for k, v in parse_qs(parsed.query).items()}
                payload, _ = fixture.get(kind, params)
                raw = json.dumps(payload).encode()
                class Response:
                    status_code = 200
                    headers = {}
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def iter_content(self, size):
                        clock.elapsed += .05
                        yield raw
                return Response()
            def close(self): pass
        monkeypatch.setattr(online.requests, "Session", Session)
        collector = online.OnlineInputs(root, clock=clock.utc, http_min_spacing_seconds=spacing)
        result = collector.refresh(now=cutoff)
        assert result["status"] == "READY", result
        assert len(fixture.calls) == count and len(sessions) == 1
        assert sum(p.get("interval") == "1h" for _, p in fixture.calls) == hourly
        assert sum(p.get("interval") == "4h" for _, p in fixture.calls) == four_hourly
        assert sum(k == "dvol" for k, _ in fixture.calls) == daily
        assert sum(k == "funding" for k, _ in fixture.calls) == 4
        assert sum(p.get("interval") == "5m" for _, p in fixture.calls) == 4
        results[label] = result
        all_frames[label] = {}
        for path in collector.data_root.rglob("*.parquet"):
            frame = pd.read_parquet(path)
            all_frames[label][path.relative_to(collector.data_root).as_posix()] = frame.drop(columns=list(online.PROVENANCE), errors="ignore")
        collector.close()
    assert all_frames["legacy"].keys() == all_frames["fast"].keys()
    for name, frame in all_frames["legacy"].items():
        pd.testing.assert_frame_equal(frame, all_frames["fast"][name], obj=name)
    assert results["legacy"]["latest_closed_5m"] == results["fast"]["latest_closed_5m"]
    # Actual receive times differ with transport; they must not be overwritten
    # with the requested cutoff to claim artificial timing parity.
    assert pd.Timestamp(results["fast"]["refresh_completed_at"]) < pd.Timestamp(results["legacy"]["refresh_completed_at"])
    assert pd.Timestamp(results["fast"]["feature_available_at"]) > cutoff
