"""Offline transport behavior: never contact a provider or a running account."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import threading

import pytest
import requests

from tools.argon4_forward import polymarket as pm
from tools.argon4_forward import polymarket_transport as pt


REAL_SESSION = requests.Session


class Response:
    def __init__(self, raw=b'{"observed":1}', status=200, action=None):
        self.raw_bytes, self.status_code, self.action = raw, status, action
        self.exited = False
        self.on_exit = lambda: None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exited = True
        self.on_exit()

    def iter_content(self, chunk_size):
        if self.action:
            self.action()
        if isinstance(self.raw_bytes, BaseException):
            raise self.raw_bytes
        yield self.raw_bytes


@pytest.fixture
def fake_http(monkeypatch):
    plans, sessions, calls = deque(), [], []
    guard = threading.Lock()

    class Session(REAL_SESSION):
        def __init__(self):
            super().__init__()
            self.closed_count = 0
            self.in_use = False
            sessions.append(self)

        def get(self, url, **kwargs):
            with guard:
                assert not self.in_use, "one Session was concurrently used"
                assert not self.closed_count
                self.in_use = True
                calls.append({"session": self, "url": url, "kwargs": kwargs,
                              "cookies": dict(self.cookies), "headers": dict(self.headers),
                              "params": dict(self.params), "auth": self.auth,
                              "proxies": dict(self.proxies), "trust_env": self.trust_env,
                              "cert": self.cert, "verify": self.verify,
                              "hooks": self.hooks})
                plan = plans.popleft() if plans else Response()
            try:
                response = plan(self, url) if callable(plan) else plan
                if isinstance(response, BaseException):
                    raise response
                response.on_exit = lambda: setattr(self, "in_use", False)
                return response
            except BaseException:
                self.in_use = False
                raise

        def close(self):
            assert not self.in_use, "closed an in-flight Session"
            self.closed_count += 1
            super().close()

    monkeypatch.setattr(pt.requests, "Session", Session)
    return plans, sessions, calls


def test_owned_connection_reuse_keeps_each_exact_receipt_independent(fake_http, monkeypatch):
    plans, sessions, calls = fake_http
    bodies = [b'{"observed": 1}\n', b'{ "observed":2 }']
    plans.extend(Response(body) for body in bodies)
    stamps = iter(f"2026-11-06T15:00:0{i}Z" for i in range(4))
    monkeypatch.setattr(pt, "_utc_now", lambda: next(stamps))
    with pt.PublicPolymarketTransport() as transport:
        first = transport.get("book", "111")
        second = transport.get("book", "222")
        assert len(sessions) == 1 and sessions[0].closed_count == 0
    assert sessions[0].closed_count == 1 and len(calls) == 2
    for index, (data, receipt, error) in enumerate((first, second)):
        assert data == {"observed": index + 1} and error is None
        assert receipt["raw_text"].encode() == bodies[index]
        assert receipt["sha256"] == hashlib.sha256(bodies[index]).hexdigest()
        assert receipt["body_complete"] is True and receipt["http_status"] == 200
        assert receipt["requested_at"] == f"2026-11-06T15:00:0{index*2}Z"
        assert receipt["received_at"] == f"2026-11-06T15:00:0{index*2+1}Z"
        assert receipt["transport_timing"]["basis"] == "MONOTONIC"
        assert all(value >= 0 for key, value in receipt["transport_timing"].items() if key != "basis")
    assert first[1] is not second[1]


def test_persistent_connections_never_carry_cookies_credentials_or_hooks(fake_http, monkeypatch):
    plans, sessions, calls = fake_http
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient.invalid:9999")
    def cookie_response(session, url):
        return Response(action=lambda: session.cookies.set("server_cookie", "must_not_persist"))
    plans.append(cookie_response)
    with pt.PublicPolymarketTransport() as transport:
        transport.get("book", "111")
        session = sessions[0]
        assert not session.cookies
        session.cookies.set("stale", "value")
        session.auth = ("unused", "unused")
        session.proxies["https"] = "http://ambient.invalid"
        session.headers["Authorization"] = "unused"
        session.params["api_key"] = "unused"
        session.cert = "unused"
        session.verify = False
        session.hooks = {"response": [lambda response: response]}
        transport.get("book", "222")
        adapter = session.adapters["https://"]
        assert adapter.max_retries.total == 0 and adapter._pool_maxsize == 1
        assert adapter._pool_block is True
    for call in calls:
        assert call["trust_env"] is False and call["auth"] is None
        assert call["cookies"] == call["proxies"] == call["params"] == {}
        assert call["headers"] == requests.utils.default_headers()
        assert "Authorization" not in call["headers"]
        assert call["hooks"] == {"response": []}
        assert call["cert"] is None and call["verify"] is True
        assert call["kwargs"]["allow_redirects"] is False
        assert call["kwargs"]["stream"] is True and call["kwargs"]["timeout"] == (2, 3)
        assert set(call["kwargs"]["headers"]) == {"Accept", "User-Agent"}


@pytest.mark.parametrize("status", [301, 302, 307, 401, 418, 429, 503])
def test_http_failure_has_original_receipt_and_never_redirects_or_retries(fake_http, status):
    plans, sessions, calls = fake_http
    raw = b'{"failure":true}'
    response = Response(raw, status)
    plans.append(response)
    with pt.PublicPolymarketTransport() as transport:
        data, receipt, error = transport.get("book", "111")
        assert data is None and error == f"HTTP_{status}"
        assert receipt["raw_text"].encode() == raw and receipt["body_complete"]
        assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
        assert len(calls) == 1 and response.exited and sessions[0].closed_count == 1
        transport.get("book", "111")
        assert len(calls) == 2 and len(sessions) == 2


@pytest.mark.parametrize("failure", [requests.ConnectionError("do not print credential-shaped error"),
                                    requests.ReadTimeout("do not print proxy error")])
def test_connection_failure_discards_only_failed_session_without_retry(fake_http, failure):
    plans, sessions, calls = fake_http
    plans.append(failure)
    with pt.PublicPolymarketTransport() as transport:
        data, receipt, error = transport.get("book", "111")
        assert data is None and error == type(failure).__name__
        assert receipt["raw_text"] is None and receipt["sha256"] is None
        assert receipt["body_complete"] is False and receipt["received_at"]
        assert len(calls) == 1 and sessions[0].closed_count == 1
        transport.get("book", "111")
        assert len(sessions) == 2


@pytest.mark.parametrize("raw,code,complete", [
    (b'{broken', "JSONDecodeError", True),
    (b'\xff', "UnicodeDecodeError", False),
    (b'x' * (pt.MAX_RESPONSE_BYTES + 1), "RESPONSE_LIMIT", False),
    (requests.exceptions.ChunkedEncodingError("partial"), "ChunkedEncodingError", False),
], ids=["invalid_json", "invalid_utf8", "oversized", "truncated"])
def test_bad_or_incomplete_body_does_not_fabricate_a_complete_receipt(fake_http, raw, code, complete):
    plans, sessions, calls = fake_http
    response = Response(raw)
    plans.append(response)
    with pt.PublicPolymarketTransport() as transport:
        data, receipt, error = transport.get("book", "111")
    assert data is None and error == code and receipt["body_complete"] is complete
    assert len(calls) == 1 and response.exited and sessions[0].closed_count == 1
    if not complete:
        assert receipt["raw_text"] is None and receipt["sha256"] is None


def test_body_deadline_keeps_six_second_limit(fake_http, monkeypatch):
    plans, sessions, calls = fake_http
    monotonic = [10.0]
    monkeypatch.setattr(pt.time, "monotonic", lambda: monotonic[0])
    plans.append(Response(action=lambda: monotonic.__setitem__(0, 16.001)))
    with pt.PublicPolymarketTransport() as transport:
        _, receipt, error = transport.get("book", "111")
    assert error == "RESPONSE_LIMIT" and not receipt["body_complete"]
    assert receipt["transport_timing"]["request_elapsed_seconds"] == pytest.approx(6.001)
    assert len(calls) == 1 and sessions[0].closed_count == 1


def test_monotonic_clock_receipt_brackets_only_network_and_body(fake_http, monkeypatch):
    plans, _, _ = fake_http
    clock = [100.0]
    monkeypatch.setattr(pt.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pt, "_utc_now", lambda: datetime.fromtimestamp(
        clock[0], timezone.utc).isoformat().replace("+00:00", "Z"))
    original_loads = pt.json.loads
    def delayed_parse(raw):
        clock[0] += 90
        return original_loads(raw)
    monkeypatch.setattr(pt.json, "loads", delayed_parse)
    def delayed_response(session, url):
        clock[0] = 102.0
        return Response(b'1793977200', action=lambda: clock.__setitem__(0, 105.0))
    plans.append(delayed_response)
    with pt.PublicPolymarketTransport() as transport:
        data, receipt, error = transport.get("time")
    assert data == 1793977200 and error is None
    assert receipt["requested_at"] == "1970-01-01T00:01:40Z"
    assert receipt["received_at"] == "1970-01-01T00:01:45Z"
    assert receipt["transport_timing"] == {"basis": "MONOTONIC", "queue_wait_seconds": 0,
        "request_elapsed_seconds": 5, "headers_elapsed_seconds": 2, "body_elapsed_seconds": 3}
    assert clock[0] == 195


def test_time_route_warms_same_clob_session_without_changing_raw_clock_evidence(fake_http):
    plans, sessions, calls = fake_http
    raw = b'1793977200\n'
    plans.append(Response(raw))
    with pt.PublicPolymarketTransport() as transport:
        data, receipt, error = transport.get("time")
        transport.get("book", "111")
    assert data == 1793977200 and error is None
    assert receipt["raw_text"].encode() == raw
    assert receipt["url"] == "https://clob.polymarket.com/time"
    assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(sessions) == 1
    assert calls[0]["kwargs"]["timeout"] == (2, 2)
    assert calls[0]["kwargs"]["headers"]["Cache-Control"] == "no-cache"


@pytest.mark.parametrize("length", [64, 65, 2000])
def test_clock_body_remains_strictly_smaller_than_sixty_four_bytes(fake_http, length):
    plans, _, calls = fake_http
    plans.append(Response(b'1' * length))
    with pt.PublicPolymarketTransport() as transport:
        _, receipt, error = transport.get("time")
    assert error == "RESPONSE_LIMIT" and receipt["body_complete"] is False
    assert len(calls) == 1


@pytest.mark.parametrize("kind,value", [("order", "1"), ("resolution", "1"),
    ("time", "1"), ("time", False), ("book", "https://evil.invalid"),
    ("book", "0"), ("book", "1&api_key=x"), ("market", "../../key")])
def test_allowlist_rejects_before_creating_session_or_network(fake_http, kind, value):
    _, sessions, calls = fake_http
    with pt.PublicPolymarketTransport() as transport, pytest.raises(ValueError):
        transport.get(kind, value)
    assert not sessions and not calls


@pytest.mark.parametrize("size", [0, 2, 5, True, 1.0, "4"])
def test_pool_size_is_fixed_and_bounded(size):
    with pytest.raises(ValueError):
        pt.PublicPolymarketTransport(size)


def test_four_sessions_are_exclusive_bounded_and_reused_across_worker_pools(fake_http):
    plans, sessions, calls = fake_http
    barrier = threading.Barrier(4)
    plans.extend(Response(action=lambda: barrier.wait(timeout=5)) for _ in range(8))
    with pt.PublicPolymarketTransport(4) as transport:
        for _ in range(2):
            with ThreadPoolExecutor(max_workers=4) as workers:
                results = list(workers.map(lambda token: transport.get("book", str(token)), range(1, 5)))
            assert all(result[2] is None for result in results)
            assert len(sessions) == 4
    assert len(calls) == 8 and all(session.closed_count == 1 for session in sessions)


def test_observer_lease_cannot_block_separate_execution_connection(fake_http):
    plans, sessions, calls = fake_http
    entered, release = threading.Event(), threading.Event()
    def block_body():
        entered.set()
        assert release.wait(5)
    plans.append(Response(action=block_body))
    observer, execution = pt.PublicPolymarketTransport(4), pt.PublicPolymarketTransport()
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            observed = workers.submit(observer.get, "book", "111")
            assert entered.wait(5)
            urgent = workers.submit(execution.get, "book", "222")
            try:
                assert urgent.result(timeout=2)[2] is None
                assert not observed.done()
            finally:
                release.set()
            assert observed.result(timeout=2)[2] is None
    finally:
        release.set()
        observer.close()
        execution.close()
    assert len(sessions) == 2 and calls[0]["session"] is not calls[1]["session"]


def test_close_waits_for_active_receipt_rejects_waiter_and_is_terminal(fake_http):
    plans, sessions, calls = fake_http
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    def block_body():
        entered.set()
        assert release.wait(5)
    response = Response(action=block_body)
    plans.append(response)
    transport = pt.PublicPolymarketTransport()
    def queued_request():
        waiting.set()
        return transport.get("book", "222")
    with ThreadPoolExecutor(max_workers=3) as workers:
        active = workers.submit(transport.get, "book", "111")
        assert entered.wait(5)
        queued = workers.submit(queued_request)
        assert waiting.wait(5)
        closed = workers.submit(transport.close)
        try:
            with pytest.raises(ValueError, match="CLOSED"):
                queued.result(timeout=2)
            assert not closed.done() and sessions[0].closed_count == 0
        finally:
            release.set()
        assert active.result(timeout=2)[2] is None
        closed.result(timeout=2)
    assert response.exited and sessions[0].closed_count == 1 and len(calls) == 1
    transport.close()
    with pytest.raises(ValueError, match="CLOSED"):
        transport.get("book", "111")
    assert sessions[0].closed_count == 1


def test_receipt_request_time_excludes_waiting_for_lease(fake_http, monkeypatch):
    plans, _, calls = fake_http
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    clock = ["2026-11-06T15:00:00Z"]
    monkeypatch.setattr(pt, "_utc_now", lambda: clock[0])
    def block_body():
        entered.set()
        assert release.wait(5)
    plans.append(Response(action=block_body))
    with pt.PublicPolymarketTransport() as transport, ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(transport.get, "book", "111")
        assert entered.wait(5)
        def second_request():
            waiting.set()
            return transport.get("book", "222")
        second = workers.submit(second_request)
        assert waiting.wait(5)
        clock[0] = "2026-11-06T15:00:03Z"
        release.set()
        first_result, second_result = first.result(timeout=2), second.result(timeout=2)
    assert first_result[1]["requested_at"] == "2026-11-06T15:00:00Z"
    assert second_result[1]["requested_at"] == "2026-11-06T15:00:03Z"
    assert len(calls) == 2


def test_default_cold_get_and_explicit_injection_remain_compatible(fake_http):
    _, sessions, _ = fake_http
    pm._get_public("book", "111")
    pm._get_public("book", "111")
    assert len(sessions) == 2 and all(session.closed_count == 1 for session in sessions)
    with pt.PublicPolymarketTransport() as transport:
        pm._get_public("book", "111", transport=transport)
        pm._get_public("book", "111", transport=transport)
        assert len(sessions) == 3 and sessions[-1].closed_count == 0


def test_collection_reuses_four_owned_sessions_and_preserves_exact_request_count(fake_http):
    plans, sessions, calls = fake_http
    plans.extend(Response(b'[]') for _ in range(28))
    now = datetime(2026, 11, 6, 15, 15, tzinfo=timezone.utc)
    with pt.PublicPolymarketTransport(4) as transport:
        results = [pm.collect_market_snapshot(now, transport=transport) for _ in range(2)]
    assert len(calls) == 28 and 1 <= len(sessions) <= 4
    assert all(session.closed_count == 1 for session in sessions)
    assert all(len(result["receipts"]) == 14 for result in results)
    assert all(result["market_data_status"] == "UNAVAILABLE" for result in results)


def test_collection_getter_keeps_old_two_argument_injection_without_network(fake_http):
    _, sessions, calls = fake_http
    def getter(kind, value):
        assert kind == "market"
        return [], {"received_at": "2026-11-06T15:15:00Z", "raw_text": "[]"}, None
    now = datetime(2026, 11, 6, 15, 15, tzinfo=timezone.utc)
    result = pm.collect_market_snapshot(now, getter=getter)
    assert len(result["receipts"]) == 14 and not sessions and not calls
    with pytest.raises(ValueError, match="choose one"):
        pm.collect_market_snapshot(now, getter=getter, transport=object())
