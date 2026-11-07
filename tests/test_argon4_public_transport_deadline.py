"""Budgeted public reads reject expiry without borrowing execution leases."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time

import pytest

from tools.argon4_forward import polymarket_transport as pt
from test_argon4_forward_polymarket_transport import fake_http, Response


@pytest.mark.parametrize('deadline', [True, float('inf'), float('nan'), '10'])
def test_invalid_deadline_does_not_create_session(fake_http, deadline):
    _, sessions, calls = fake_http
    with pt.PublicPolymarketTransport() as transport, pytest.raises(ValueError, match='deadline'):
        transport.get('time', deadline_monotonic=deadline)
    assert not sessions and not calls


def test_expired_budget_does_not_acquire_lease_or_invent_request_time(fake_http, monkeypatch):
    _, sessions, calls = fake_http
    monkeypatch.setattr(pt.time, 'monotonic', lambda: 100.)
    with pt.PublicPolymarketTransport() as transport:
        data, receipt, error = transport.get('time', deadline_monotonic=100.)
        assert transport._active == 0 and transport._available == [0]
    assert data is None and error == 'DEADLINE_EXCEEDED'
    assert receipt['requested_at'] is None and receipt['received_at'] is None
    assert not receipt['body_complete'] and not sessions and not calls


def test_lease_wait_obeys_budget_and_leaves_active_owner_untouched(fake_http):
    _, sessions, calls = fake_http
    with pt.PublicPolymarketTransport() as transport:
        index, _ = transport._acquire()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(transport.get, 'time', deadline_monotonic=time.monotonic()+.03)
                data, receipt, error = future.result(timeout=1)
            assert error == 'DEADLINE_EXCEEDED' and data is None
            assert transport._active == 1 and transport._available == []
            assert receipt['requested_at'] is None and receipt['received_at'] is None
            assert receipt['transport_timing']['queue_wait_seconds'] >= .02
        finally:
            transport._release(index)
    assert not sessions and not calls


def test_connect_and_initial_read_timeout_share_remaining_total(fake_http, monkeypatch):
    plans, _, calls = fake_http
    now = [100.]
    monkeypatch.setattr(pt.time, 'monotonic', lambda: now[0])
    raw = b'1794039600'
    plans.append(Response(raw, action=lambda: now.__setitem__(0, 100.1)))
    with pt.PublicPolymarketTransport() as transport:
        data, receipt, error = transport.get('time', deadline_monotonic=100.5)
    timeout = calls[0]['kwargs']['timeout']
    assert timeout.total == .5 and timeout.connect_timeout == .5
    assert data == 1794039600 and error is None
    assert receipt['sha256'] == hashlib.sha256(raw).hexdigest() and receipt['body_complete']


@pytest.mark.parametrize('phase', ['headers', 'body'])
def test_late_response_phase_is_closed_and_never_accepted(fake_http, monkeypatch, phase):
    plans, sessions, calls = fake_http
    now = [100.]
    monkeypatch.setattr(pt.time, 'monotonic', lambda: now[0])
    response = Response(b'1794039600', action=(lambda: now.__setitem__(0, 100.6)) if phase == 'body' else None)
    def headers(session, url):
        now[0] = 100.6
        return response
    plans.append(headers if phase == 'headers' else response)
    with pt.PublicPolymarketTransport() as transport:
        data, receipt, error = transport.get('time', deadline_monotonic=100.5)
    assert data is None and error == 'DEADLINE_EXCEEDED'
    assert not receipt['body_complete'] and receipt['sha256'] is None
    assert len(calls) == 1 and response.exited and sessions[0].closed_count == 1


def test_budget_is_rechecked_after_parsing_and_complete_late_bytes_are_retained(fake_http, monkeypatch):
    plans, sessions, _ = fake_http
    now = [100.]
    monkeypatch.setattr(pt.time, 'monotonic', lambda: now[0])
    raw = b'1794039600'
    plans.append(Response(raw))
    loads = pt.json.loads
    def slow_parse(value):
        now[0] = 100.6
        return loads(value)
    monkeypatch.setattr(pt.json, 'loads', slow_parse)
    with pt.PublicPolymarketTransport() as transport:
        data, receipt, error = transport.get('time', deadline_monotonic=100.5)
    assert data is None and error == 'DEADLINE_EXCEEDED'
    assert receipt['body_complete'] and receipt['raw_text'] == raw.decode()
    assert receipt['sha256'] == hashlib.sha256(raw).hexdigest()
    assert sessions[0].closed_count == 1


def test_session_setup_cannot_consume_budget_and_then_start_http(fake_http, monkeypatch):
    _, sessions, calls = fake_http
    now = [100.]
    monkeypatch.setattr(pt.time, 'monotonic', lambda: now[0])
    with pt.PublicPolymarketTransport() as transport:
        original = transport._session
        def slow_session(index):
            session = original(index)
            now[0] = 100.6
            return session
        monkeypatch.setattr(transport, '_session', slow_session)
        data, receipt, error = transport.get('time', deadline_monotonic=100.5)
    assert not calls and len(sessions) == 1 and sessions[0].closed_count == 1
    assert data is None and error == 'DEADLINE_EXCEEDED'
    assert receipt['requested_at'] is None and receipt['received_at'] is None


def test_return_from_response_cleanup_cannot_accept_expired_budget(fake_http, monkeypatch):
    plans, sessions, _ = fake_http
    now = [100.]
    monkeypatch.setattr(pt.time, 'monotonic', lambda: now[0])
    class SlowClose(Response):
        def __exit__(self, *args):
            super().__exit__(*args)
            now[0] = 100.6
    plans.append(SlowClose(b'1794039600'))
    with pt.PublicPolymarketTransport() as transport:
        data, receipt, error = transport.get('time', deadline_monotonic=100.5)
    assert data is None and error == 'DEADLINE_EXCEEDED'
    assert receipt['body_complete'] and sessions[0].closed_count == 1
