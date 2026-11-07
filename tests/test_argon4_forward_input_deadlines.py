"""Offline deadline/concurrency evidence; no production data or order access."""
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
import sqlite3
import threading
import time

import pandas as pd
import pytest

from tools.argon4_forward import online
from test_argon4_forward_online import FixtureHTTP


NOW = pd.Timestamp("2026-11-06T15:12:00Z")


def test_http_overlaps_network_with_worker_owned_reusable_sessions(tmp_path, monkeypatch):
    starts, sessions, owners = [], [], {}
    simultaneous, release = threading.Event(), threading.Event()
    count_lock = threading.Lock()
    class Session:
        def __init__(self):
            self.owner = threading.get_ident()
            self.closed = False
            sessions.append(self)
        def get(self, url, **kwargs):
            assert threading.get_ident() == self.owner and not self.closed
            assert 0 < kwargs["timeout"][0] <= 2 and 0 < kwargs["timeout"][1] <= 3.5
            with count_lock:
                starts.append(time.monotonic())
                owners[self.owner] = owners.get(self.owner, 0) + 1
                if len(starts) == 4:
                    simultaneous.set()
            class Response:
                status_code = 200
                headers = {}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def iter_content(self, size):
                    assert release.wait(2)
                    yield b'{"serverTime": 1}'
            return Response()
        def close(self): self.closed = True
    monkeypatch.setattr(online.requests, "Session", Session)
    client = online.PublicInputsHTTP(tmp_path / "receipts.sqlite3", min_spacing=.04)
    deadline = time.monotonic() + 3
    def twice():
        return [client.get("clock", deadline_monotonic=deadline, generation_id="test", cohort_id="c")[1]
                for _ in range(2)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(twice) for _ in range(4)]
        try:
            assert simultaneous.wait(1), "network waiting still serialized"
        finally:
            release.set()
        receipts = [receipt for future in futures for receipt in future.result(2)]
    assert len(sessions) == 4 and sorted(owners.values()) == [2, 2, 2, 2]
    assert all(b-a >= .035 for a, b in zip(starts, starts[1:]))
    assert all(r["generation_id"] == "test" and r["cohort_id"] == "c" for r in receipts)
    with sqlite3.connect(client.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM receipts WHERE generation_id='test' AND cohort_id='c'").fetchone()[0] == 8
    client.close()
    assert all(session.closed for session in sessions)


def test_deadline_before_admission_never_sends_and_receipt_clock_is_captured(tmp_path, monkeypatch):
    sent = []
    class Session:
        def get(self, url, **kwargs):
            sent.append(url)
            class Response:
                status_code = 200
                headers = {}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def iter_content(self, size):
                    client.clock = lambda: NOW + pd.Timedelta(days=1)
                    yield b'{"serverTime": 1}'
            return Response()
        def close(self): pass
    monkeypatch.setattr(online.requests, "Session", Session)
    client = online.PublicInputsHTTP(tmp_path / "receipts.sqlite3", clock=lambda: NOW, min_spacing=.25)
    receipt = client.get("clock", deadline_monotonic=time.monotonic()+1)[1]
    assert receipt["received_at"] == NOW.isoformat()
    with pytest.raises(online.InputDeadlineExceeded, match="BEFORE_SEND_SLOT"):
        client.get("clock", deadline_monotonic=time.monotonic()+.02)
    assert len(sent) == 1
    client.close()


def test_deadline_during_body_cannot_become_completed_receipt(tmp_path, monkeypatch):
    class Session:
        def get(self, url, **kwargs):
            class Response:
                status_code = 200
                headers = {}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def iter_content(self, size):
                    time.sleep(.06)
                    yield b'{"serverTime": 1}'
            return Response()
        def close(self): pass
    monkeypatch.setattr(online.requests, "Session", Session)
    client = online.PublicInputsHTTP(tmp_path / "receipts.sqlite3", min_spacing=0)
    with pytest.raises(online.InputDeadlineExceeded):
        client.get("clock", deadline_monotonic=time.monotonic()+.03)
    assert not client.path.exists()
    client.close()


def test_waiting_send_lock_respects_deadline_without_network(tmp_path, monkeypatch):
    def forbidden_session():
        pytest.fail("deadline-expired admission created a network session")
    monkeypatch.setattr(online.requests, "Session", forbidden_session)
    client = online.PublicInputsHTTP(tmp_path / "receipts.sqlite3", min_spacing=.25)
    client._request_lock.acquire()
    started = time.monotonic()
    try:
        with pytest.raises(online.InputDeadlineExceeded):
            client.get("clock", deadline_monotonic=started+.04)
    finally:
        client._request_lock.release()
        client.close()
    assert time.monotonic()-started < .3


def test_rate_limit_headers_survive_body_timeout_and_restart(tmp_path, monkeypatch):
    class Session:
        def get(self, url, **kwargs):
            class Response:
                status_code = 429
                headers = {"Retry-After": "120"}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def iter_content(self, size):
                    time.sleep(.08)
                    yield b'{"code": -1003}'
            return Response()
        def close(self): pass
    monkeypatch.setattr(online.requests, "Session", Session)
    path = tmp_path / "receipts.sqlite3"
    client = online.PublicInputsHTTP(path, clock=lambda: NOW, min_spacing=0)
    with pytest.raises(online.InputDeadlineExceeded):
        client.get("clock", deadline_monotonic=time.monotonic()+.04)
    evidence = json.loads(path.with_name("binance_backoff.json").read_text())
    assert evidence["evidence_type"] == "HTTP_STATUS_AND_HEADERS_ONLY"
    assert evidence["source_receipt_sha256"] is None
    assert pd.Timestamp(evidence["blocked_until_utc"]) == NOW+pd.Timedelta(seconds=135)
    restarted = online.PublicInputsHTTP(path, clock=lambda: NOW, min_spacing=0)
    with pytest.raises(ValueError, match="PUBLIC_RATE_LIMIT_BACKOFF"):
        restarted.get("spot_clock")
    client.close()
    restarted.close()


def _small_batch(tmp_path, monkeypatch):
    collector = online.OnlineInputs(tmp_path, clock=lambda: NOW, client=FixtureHTTP(lambda: NOW))
    monkeypatch.setattr(collector, "_check_clock", lambda: {"status": "VERIFIED", "received_at": NOW.isoformat()})
    monkeypatch.setattr(collector, "_funding", lambda *args: None)
    monkeypatch.setattr(collector, "_dvol", lambda *args: None)
    monkeypatch.setattr(collector, "_basis", lambda: None)
    monkeypatch.setattr(collector, "snapshot", lambda: {"status": "READY"})
    return collector


def test_deadline_returns_without_join_and_late_workers_never_publish(tmp_path, monkeypatch):
    collector = _small_batch(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    calls = []
    before = {}
    for symbol in online.SYMBOLS:
        path = collector._path("futures", symbol, "5m")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"observed": [1]}).to_parquet(path)
        before[path] = path.read_bytes()
    def candles(kind, symbol, interval, cutoff):
        if interval != "5m": return
        calls.append(threading.get_ident())
        if len(calls) == 4: entered.set()
        assert release.wait(3)
        collector._checked(symbol, {"received_at": NOW.isoformat(), "url": "fixture", "sha256": "a"*64})
        collector._publish_frame(pd.DataFrame({"observed": [2]}), collector._path(kind, symbol, interval))
    monkeypatch.setattr(collector, "_candles", candles)
    started = time.monotonic()
    result = collector.refresh(deadline_monotonic=started+.15, cohort_id="c1")
    assert time.monotonic()-started < .6
    assert entered.is_set() and result["status"] == "BLOCKED"
    assert "DEADLINE" in result["errors"][0]["error"]
    for _ in range(4):
        busy = collector.refresh(deadline_monotonic=time.monotonic()+.1)
        assert busy["errors"][0]["error"] == "INPUT_WORKERS_BUSY"
    assert len(calls) == 4 and len(collector._pool._threads) <= 4
    release.set()
    wait(collector._outstanding, timeout=2)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert collector._checks == {}
    monkeypatch.setattr(collector, "_candles", lambda *args: None)
    assert collector.refresh(deadline_monotonic=time.monotonic()+1, cohort_id="c2")["status"] == "READY"
    collector.close()


def test_atomic_publication_rechecks_generation_after_slow_serialization(tmp_path, monkeypatch):
    collector = _small_batch(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    path = tmp_path / "accepted.parquet"
    pd.DataFrame({"observed": [1]}).to_parquet(path)
    before = path.read_bytes()
    write = pd.DataFrame.to_parquet
    def slow_write(frame, target, *args, **kwargs):
        write(frame, target, *args, **kwargs)
        entered.set()
        assert release.wait(3)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", slow_write)
    def candles(kind, symbol, interval, cutoff):
        if symbol == "BTCUSDT" and interval == "5m":
            collector._publish_frame(pd.DataFrame({"observed": [2]}), path)
    monkeypatch.setattr(collector, "_candles", candles)
    result = collector.refresh(deadline_monotonic=time.monotonic()+.15)
    assert entered.is_set() and result["status"] == "BLOCKED"
    release.set()
    wait(collector._outstanding, timeout=2)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("accepted.parquet.*.tmp"))
    collector.close()


def test_funding_receipt_cache_expires_and_does_not_invent_event_schedule(tmp_path):
    moment = [NOW]
    client = FixtureHTTP(lambda: moment[0])
    collector = online.OnlineInputs(tmp_path, client=client, clock=lambda: moment[0])
    path = collector._path("funding", "BTCUSDT")
    collector._funding("BTCUSDT", moment[0])
    original = pd.read_parquet(path)
    client.calls.clear()
    moment[0] += pd.Timedelta(seconds=15)
    collector._funding("BTCUSDT", moment[0])
    assert not client.calls
    pd.testing.assert_frame_equal(original, pd.read_parquet(path))
    moment[0] += pd.Timedelta(seconds=46)
    collector._funding("BTCUSDT", moment[0])
    assert len(client.calls) == 1
    # A newly due event is fetched even inside the receipt TTL. Only actual
    # provider timestamps are appended; the observation grid is not filled.
    moment[0] = pd.Timestamp("2026-11-06T15:59:50Z")
    collector._funding("BTCUSDT", moment[0])
    client.calls.clear()
    moment[0] += pd.Timedelta(seconds=15)
    collector._funding("BTCUSDT", moment[0])
    assert len(client.calls) == 1
    events = pd.read_parquet(path)
    assert len(events) == len(original)+1 and events.date.iloc[-1] == pd.Timestamp("2026-11-06T16:00:00Z")
    assert events.first_received_at.iloc[-1] == moment[0]
    pd.testing.assert_frame_equal(original, events.iloc[:-1].reset_index(drop=True))
    collector.close()


def test_auxiliary_prefetch_has_no_five_minute_reads_or_future_cutoff(tmp_path, monkeypatch):
    collector = _small_batch(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(collector, "_candles", lambda kind, symbol, interval, cutoff: calls.append((interval, cutoff)))
    result = collector.prefetch_auxiliary(NOW, deadline_monotonic=time.monotonic()+1, cohort_id="next")
    assert result["status"] == "AUXILIARY_PREFETCH_COMPLETE"
    assert len(calls) == 14 and all(interval in ("1h", "4h") and cutoff == NOW for interval, cutoff in calls)
    assert collector.state == {} and not collector.state_path.exists()
    collector.close()
