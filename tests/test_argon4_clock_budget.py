"""Six real clock samples remain mandatory; retries cannot weaken acceptance."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import hashlib
import json

import pytest

from tools.argon4_forward import clock_sync as cs

T = datetime(2026, 11, 7, 8, 20, tzinfo=timezone.utc)


@pytest.fixture
def clock_transport(monkeypatch):
    state = SimpleNamespace(monotonic=100., calls=[], bad_at=None, late_at=None, wall_shift=0.)
    state.specs = [(i+.1, i+.3) for i in range(6)] + [(6.5, 6.7), (6.9, 7.1)]
    monkeypatch.setattr(cs.time, 'monotonic', lambda: state.monotonic)
    class Stamp(datetime):
        @classmethod
        def now(cls, tz=None):
            return T+timedelta(seconds=state.monotonic-100+state.wall_shift)
    monkeypatch.setattr(cs, 'datetime', Stamp)
    def get(kind, **kwargs):
        index = len(state.calls)
        state.calls.append((kind, kwargs))
        start, end = state.specs[index]
        state.monotonic = 100+end
        raw = str(int((T+timedelta(seconds=start)).timestamp())+5)
        receipt = {'source': 'POLYMARKET_PUBLIC_READ_ONLY', 'url': cs.URL, 'method': 'GET',
                   'requested_at': (T+timedelta(seconds=start)).isoformat(),
                   'received_at': (T+timedelta(seconds=end)).isoformat(),
                   'body_complete': True, 'http_status': 200, 'raw_text': raw,
                   'sha256': hashlib.sha256(raw.encode()).hexdigest(),
                   'transport_timing': {'request_elapsed_seconds': end-start}}
        if index == state.late_at:
            state.monotonic = kwargs['deadline_monotonic']+.001
        if index == state.bad_at:
            receipt['body_complete'] = False
            return None, receipt, 'HTTP_429'
        return int(raw), receipt, None
    state.get = get
    return state


def test_opt_in_intersects_all_eight_actual_samples_and_keeps_hard_half_second(clock_transport):
    result = cs.calibrate(transport=clock_transport, retry_borderline=True, budget_monotonic=110.)
    assert result['status'] == 'CALIBRATED'
    assert .5 < result['borderline_retry']['initial_uncertainty_seconds'] <= .65
    assert result['uncertainty_seconds'] < .5
    assert result['borderline_retry']['extra_samples_collected'] == 2
    assert len(result['receipts']) == 8
    assert all(k == 'time' and kw == {'deadline_monotonic': 110.} for k, kw in clock_transport.calls)
    assert result['receipt']['samples'] == result['receipts']
    raw = json.dumps(result['receipts'], sort_keys=True, separators=(',', ':')).encode()
    assert result['receipt_sha256'] == hashlib.sha256(raw).hexdigest()
    assert result['offset_lower_seconds'] <= 5 <= result['offset_upper_seconds']
    assert cs.bounds_from_samples(result['receipts'])['uncertainty_seconds'] == result['uncertainty_seconds']


def test_default_does_not_retry_borderline_and_preserves_old_get_signature(clock_transport):
    result = cs.calibrate(transport=clock_transport)
    assert result['status'] == 'UNCERTAINTY_TOO_LARGE'
    assert len(clock_transport.calls) == 6
    assert all(kwargs == {} for _, kwargs in clock_transport.calls)
    assert 'borderline_retry' not in result


def test_opt_in_without_budget_records_reason_without_adding_requests(clock_transport):
    result = cs.calibrate(transport=clock_transport, retry_borderline=True)
    assert result['status'] == 'UNCERTAINTY_TOO_LARGE'
    assert result['borderline_retry']['status'] == 'NO_RETRY_BUDGET'
    assert result['borderline_retry']['extra_samples_collected'] == 0
    assert len(clock_transport.calls) == 6


def test_already_narrow_baseline_does_not_spend_extra_budget(clock_transport):
    clock_transport.specs = [(i*.4+.1, i*.4+.3) for i in range(6)]
    result = cs.calibrate(transport=clock_transport, retry_borderline=True, budget_monotonic=110.)
    assert result['status'] == 'CALIBRATED' and len(clock_transport.calls) == 6
    assert result['borderline_retry']['status'] == 'NOT_NEEDED'
    assert result['borderline_retry']['extra_samples_collected'] == 0


def test_budget_spent_after_baseline_prevents_first_extra_request(clock_transport, monkeypatch):
    original = cs.bounds_from_samples
    def finish_budget(rows):
        result = original(rows)
        clock_transport.monotonic = 110.
        return result
    monkeypatch.setattr(cs, 'bounds_from_samples', finish_budget)
    result = cs.calibrate(transport=clock_transport, retry_borderline=True, budget_monotonic=110.)
    assert len(clock_transport.calls) == 6 and len(result['receipts']) == 6
    assert result['status'] == 'UNCALIBRATED' and result['borderline_retry']['status'] == 'FAILED'
    assert result['borderline_retry']['extra_samples_collected'] == 0


def test_clock_step_between_baseline_and_extra_cannot_fake_narrower_bounds(clock_transport, monkeypatch):
    original = cs.bounds_from_samples
    def step_between_requests(rows):
        result = original(rows)
        clock_transport.wall_shift = .2
        return result
    monkeypatch.setattr(cs, 'bounds_from_samples', step_between_requests)
    result = cs.calibrate(transport=clock_transport, retry_borderline=True, budget_monotonic=110.)
    assert len(clock_transport.calls) == 6 and len(result['receipts']) == 6
    assert result['status'] == 'UNCALIBRATED' and result['borderline_retry']['status'] == 'FAILED'
    assert result['error_code'] == 'LOCAL_CLOCK_CHANGED_DURING_CALIBRATION'
    assert 'observed_at' not in result and 'calibrated_monotonic' not in result


@pytest.mark.parametrize('close_after_deadline', [False, True])
def test_budgeted_standalone_uses_owned_transport_and_checks_close(clock_transport, monkeypatch, close_after_deadline):
    from tools.argon4_forward import polymarket_transport as pt
    closed = []
    class Owned:
        def __enter__(self):
            return clock_transport
        def __exit__(self, *args):
            closed.append(True)
            if close_after_deadline:
                clock_transport.monotonic = 111.
    monkeypatch.setattr(pt, 'PublicPolymarketTransport', Owned)
    monkeypatch.setattr(cs, '_collect_samples', lambda *a: pytest.fail('budget bypassed public transport'))
    result = cs.calibrate(retry_borderline=True, budget_monotonic=110.)
    assert closed == [True] and len(clock_transport.calls) == 8
    assert result['status'] == ('UNCALIBRATED' if close_after_deadline else 'CALIBRATED')
    if close_after_deadline:
        assert 'observed_at' not in result and result['borderline_retry']['status'] == 'FAILED'


@pytest.mark.parametrize('uncertainty,expected_calls', [(.5, 6), (.500001, 8), (.65, 8), (.650001, 6)])
def test_retry_range_boundaries_never_raise_acceptance_limit(clock_transport, monkeypatch, uncertainty, expected_calls):
    monkeypatch.setattr(cs, 'bounds_from_samples', lambda rows: {
        'server_minus_local_seconds': 5., 'uncertainty_seconds': uncertainty,
        'offset_lower_seconds': 5.-uncertainty, 'offset_upper_seconds': 5.+uncertainty})
    result = cs.calibrate(transport=clock_transport, retry_borderline=True, budget_monotonic=110.)
    assert len(clock_transport.calls) == expected_calls
    assert result['status'] == ('CALIBRATED' if uncertainty <= .5 else 'UNCERTAINTY_TOO_LARGE')


def test_coarse_point_502_samples_may_still_fail_after_eight(clock_transport):
    clock_transport.specs = [(i+.1, i+.1) for i in range(8)]
    result = cs.calibrate(transport=clock_transport, retry_borderline=True, budget_monotonic=110.)
    assert len(clock_transport.calls) == 8
    assert result['uncertainty_seconds'] == pytest.approx(.502)
    assert result['status'] == 'UNCERTAINTY_TOO_LARGE'
    assert result['borderline_retry']['status'] == 'COMPLETED'


def test_budget_expiry_before_any_request_does_not_make_clock_fresh(clock_transport):
    result = cs.calibrate(transport=clock_transport, retry_borderline=True, budget_monotonic=100.)
    assert not clock_transport.calls and result['status'] == 'UNCALIBRATED'
    assert result['error_code'] == 'CLOCK_DEADLINE_EXCEEDED'
    assert 'observed_at' not in result and 'calibrated_monotonic' not in result


def test_seven_narrow_samples_are_not_accepted_if_last_extra_request_expires(clock_transport):
    result = cs.calibrate(transport=clock_transport, retry_borderline=True, budget_monotonic=106.8)
    assert len(clock_transport.calls) == 8 and len(result['receipts']) == 7
    assert cs.bounds_from_samples(result['receipts'])['uncertainty_seconds'] < .5
    assert result['status'] == 'UNCALIBRATED'
    assert result['borderline_retry']['status'] == 'FAILED'
    assert result['borderline_retry']['extra_samples_collected'] == 1
    assert result['failed_receipt']['body_complete'] is True
    assert 'observed_at' not in result and 'calibrated_monotonic' not in result


def test_failed_extra_sample_is_retained_without_retry_or_partial_success(clock_transport):
    clock_transport.bad_at = 7
    result = cs.calibrate(transport=clock_transport, retry_borderline=True, budget_monotonic=110.)
    assert len(clock_transport.calls) == 8 and len(result['receipts']) == 7
    assert result['status'] == 'UNCALIBRATED' and result['borderline_retry']['status'] == 'FAILED'
    assert result['failed_receipt']['body_complete'] is False
    assert 'observed_at' not in result


def test_inconsistent_additional_clock_does_not_discard_bad_evidence(clock_transport):
    original = clock_transport.get
    def get(kind, **kwargs):
        value, receipt, error = original(kind, **kwargs)
        if len(clock_transport.calls) == 8:
            value += 30
            receipt['raw_text'] = str(value)
            receipt['sha256'] = hashlib.sha256(str(value).encode()).hexdigest()
        return value, receipt, error
    transport = SimpleNamespace(get=get)
    result = cs.calibrate(transport=transport, retry_borderline=True, budget_monotonic=110.)
    assert len(result['receipts']) == 8 and result['status'] == 'UNCALIBRATED'
    assert result['borderline_retry']['status'] == 'FAILED'
    assert 'observed_at' not in result


@pytest.mark.parametrize('changed', ['url', 'method', 'source', 'sha256'])
def test_budgeted_clock_requires_real_route_and_byte_hash(clock_transport, changed):
    original = clock_transport.get
    def get(kind, **kwargs):
        value, receipt, error = original(kind, **kwargs)
        receipt[changed] = 'invalid'
        return value, receipt, error
    result = cs.calibrate(transport=SimpleNamespace(get=get), budget_monotonic=110.)
    assert len(clock_transport.calls) == 1 and result['status'] == 'UNCALIBRATED'
    assert result['failed_receipt'][changed] == 'invalid'


@pytest.mark.parametrize('budget', [True, float('nan'), float('inf'), '110'])
def test_invalid_deadlines_are_rejected_before_network(clock_transport, budget):
    with pytest.raises(ValueError, match='budget'):
        cs.calibrate(transport=clock_transport, budget_monotonic=budget)
    assert not clock_transport.calls


def test_opt_in_cannot_expand_non_six_sample_baseline(clock_transport):
    with pytest.raises(ValueError, match='six-sample'):
        cs.calibrate(8, transport=clock_transport, retry_borderline=True, budget_monotonic=110.)
    assert not clock_transport.calls
