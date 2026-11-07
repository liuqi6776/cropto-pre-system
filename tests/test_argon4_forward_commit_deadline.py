"""No-network acceptance deadlines with temporary accounts and real SQL locks."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest
import requests

from tools.argon4_forward.account import Account, AccountRejected, STREAMS, V5_SEMANTICS
from tools.argon4_forward.execution import PaperExecutor


BOUNDARY = datetime(2026, 11, 6, tzinfo=timezone.utc)


def at(seconds):
    return BOUNDARY + timedelta(seconds=seconds)


class Clock:
    def __init__(self, seconds=8):
        self.value = at(seconds)
        self.calls = []
    def __call__(self):
        self.calls.append(self.value)
        return self.value


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("network forbidden"))
    account = Account(tmp_path / "isolated.sqlite3", started_at=at(-1))
    policy = {"status": "CONFIRMED", "version": "offline-deadline-v5",
              "ranking_source": "explicit synthetic test fixture", "order": list(STREAMS),
              "waterfall_semantics": V5_SEMANTICS}
    cohort = account.ensure_cohort(BOUNDARY.isoformat(), policy, now=at(6))
    signal = {"signal_id": "synthetic-deadline-signal", "stream": "BTC_15m", "underlying": "BTC",
              "outcome": "UP", "market_id": "synthetic-btc", "model_id": "synthetic-model",
              "model_sha256": "a" * 64, "validation_status": "INPUTS_VALIDATED",
              "feature_available_at": at(5).isoformat(), "decision_at": at(6).isoformat(),
              "valid_until": at(20).isoformat(), "window_start": BOUNDARY.isoformat(),
              "window_end": at(900).isoformat()}
    book = {k: signal[k] for k in ("market_id", "underlying", "outcome", "window_start", "window_end")}
    book.update(kind="OBSERVED_PUBLIC_ORDERBOOK", source="https://fixture.invalid/public-book",
                observed_at=at(8).isoformat(), received_at=at(8).isoformat(),
                asks=[{"price": ".5", "size": "1000"}])
    fee = {"fee_model": "fixed_bps", "bps": 0, "label": "SIMULATED_ASSUMPTION",
           "source": "explicit fixture only", "observed_at": at(7).isoformat()}
    authorization = {"policy": "RANKED_CAPACITY_WATERFALL", "waterfall_semantics": V5_SEMANTICS,
                     "plan_id": cohort["plan_id"], "rank": 1, "signal_id": signal["signal_id"],
                     "market_id": signal["market_id"], "approved_stake_usd": 150,
                     "planned_at": at(7).isoformat(), "valid_until": at(20).isoformat(),
                     "policy_version": policy["version"], "preceding_allocations": [],
                     "cohort_id": BOUNDARY.isoformat(), "cohort_capacity_usd": 500}
    return SimpleNamespace(account=account, signal=signal, book=book, fee=fee,
                           authorization=authorization, policy=policy, clock=Clock())


def purchase(fixture, now=None):
    return fixture.account.place_paper_buy(fixture.signal, fixture.book, fixture.fee,
        stake=150, now=fixture.clock if now is None else now, allocation_authorization=fixture.authorization)


def logical_state(account):
    with sqlite3.connect(f"file:{account.path.as_posix()}?mode=ro", uri=True) as con:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {name: con.execute('SELECT * FROM "'+name+'" ORDER BY rowid').fetchall() for name in names}


def instrument_connections(monkeypatch, account, callback):
    original = account._connection
    class Proxy:
        def __init__(self, con): self.con = con
        def execute(self, sql, *args):
            callback("before", sql)
            result = self.con.execute(sql, *args)
            callback("after", sql)
            return result
        def __getattr__(self, name): return getattr(self.con, name)
    @contextmanager
    def connection():
        with original() as con:
            yield Proxy(con)
    monkeypatch.setattr(account, "_connection", connection)


@pytest.mark.parametrize("kind", ["fixed", "callable"])
def test_normal_fixed_or_live_clock_preserves_amount_and_accounting(prepared, kind):
    f = prepared
    result = purchase(f, at(8) if kind == "fixed" else f.clock)
    assert result["cost_usd"] == 150 and result["fee_usd"] == 0
    assert result["quantity"] == "300.00000000" and result["fill_at"] == at(8).isoformat()
    assert f.account.cohort_state(BOUNDARY.isoformat())["used_usd"] == 150
    assert f.account.overview(now=at(9))["cash"] == 9850
    assert result["live_enabled"] is False
    if kind == "callable": assert len(f.clock.calls) >= 4


def test_fill_time_is_actual_acceptance_after_computation_not_initial_stamp(prepared):
    f = prepared
    calls = []
    def ticking_clock():
        value = at(8 + len(calls) / 10)
        calls.append(value)
        return value
    result = purchase(f, ticking_clock)
    assert result["fill_at"] == at(8.2).isoformat()
    assert result["fill_at"] != calls[0].isoformat()
    assert calls[-1] == at(8.3)
    with sqlite3.connect(f.account.path) as con:
        assert con.execute("SELECT fill_at FROM orders").fetchone()[0] == result["fill_at"]
        payload = json.loads(con.execute("SELECT payload FROM events WHERE event_type='PAPER_IOC'").fetchone()[0])
        assert payload["fill_at"] == result["fill_at"]


def test_omitted_now_resamples_system_utc_instead_of_freezing_entry_time(prepared, monkeypatch):
    from tools.argon4_forward import account as account_module
    f = prepared
    original = account_module._now
    sampled = []
    def observed_now(value=None):
        if value is not None:
            return original(value)
        stamp = at(8 + len(sampled) / 10)
        sampled.append(stamp)
        return stamp
    monkeypatch.setattr(account_module, "_now", observed_now)
    result = f.account.place_paper_buy(f.signal, f.book, f.fee, stake=150,
                                      allocation_authorization=f.authorization)
    assert sampled == [at(8), at(8.1), at(8.2), at(8.3)]
    assert result["fill_at"] == at(8.2).isoformat()


def test_real_sql_write_lock_wait_crosses_deadline_and_leaves_no_trade_writes(prepared, monkeypatch):
    f = prepared
    before = logical_state(f.account)
    entering = threading.Event()
    instrument_connections(monkeypatch, f.account,
        lambda phase, sql: entering.set() if phase == "before" and sql == "BEGIN IMMEDIATE" else None)
    blocker = sqlite3.connect(f.account.path)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(purchase, f)
            try:
                assert entering.wait(5)
                assert not future.done()
                f.clock.value = at(21)
            finally:
                blocker.rollback()
            with pytest.raises(AccountRejected, match="expired"):
                future.result(5)
    finally:
        blocker.close()
    assert logical_state(f.account) == before


@pytest.mark.parametrize("point", ["consumed_depth", "final_snapshot"])
def test_computation_or_final_writes_overrun_rolls_back_every_trade_table(prepared, monkeypatch, point):
    f = prepared
    before = logical_state(f.account)
    seen = []
    def delay(phase, sql):
        if phase == "after":
            seen.append(sql)
            if point == "consumed_depth" and sql.startswith("INSERT INTO consumed_book_depth"):
                f.clock.value = at(21)
            if point == "final_snapshot" and sql.startswith("INSERT INTO equity_observations"):
                f.clock.value = at(21)
    instrument_connections(monkeypatch, f.account, delay)
    with pytest.raises(AccountRejected, match="expired"):
        purchase(f)
    assert any(sql.startswith("INSERT INTO signals") for sql in seen)
    assert any(sql.startswith("INSERT INTO consumed_book_depth") for sql in seen)
    if point == "final_snapshot":
        assert any(sql.startswith("INSERT INTO orders") for sql in seen)
        assert any(sql.startswith("UPDATE account SET cash_atoms") for sql in seen)
    assert logical_state(f.account) == before


@pytest.mark.parametrize("failure", ["authorization", "cohort", "book", "fee", "backwards", "clock_expired"])
def test_every_acceptance_guard_keeps_time_and_freshness_limits(prepared, monkeypatch, failure):
    f = prepared
    # Isolate each failure rather than masking it with an earlier expiry.
    if failure == "authorization": f.authorization["valid_until"] = at(8.5).isoformat()
    if failure == "cohort":
        f.signal["valid_until"] = f.authorization["valid_until"] = at(25).isoformat()
    if failure == "fee": f.fee["observed_at"] = at(-291).isoformat()
    before = logical_state(f.account)
    original = f.account._snapshot
    def delayed_snapshot(con, stamp):
        original(con, stamp)
        f.clock.value = {"authorization": at(9), "cohort": at(20), "book": at(13.001),
                         "fee": at(10), "backwards": at(7.9), "clock_expired": at(8)}[failure]
        if failure == "clock_expired": f.clock.failed = True
    monkeypatch.setattr(f.account, "_snapshot", delayed_snapshot)
    def now():
        if getattr(f.clock, "failed", False): raise ValueError("clock calibration expired")
        return f.clock()
    with pytest.raises(ValueError):
        purchase(f, now)
    assert logical_state(f.account) == before


def test_identical_late_retry_returns_old_fill_without_sampling_expired_clock(prepared):
    f = prepared
    result = purchase(f)
    before = logical_state(f.account)
    def invalid_clock():
        pytest.fail("an identical existing order needs no new acceptance clock")
    repeated = purchase(f, invalid_clock)
    assert repeated["duplicate"] is True and repeated["fill_at"] == result["fill_at"]
    assert logical_state(f.account) == before
    f.signal["model_id"] = "conflicting-model"
    with pytest.raises(AccountRejected, match="conflicting reuse"):
        purchase(f, invalid_clock)
    assert logical_state(f.account) == before


@pytest.mark.parametrize("semantics", ["legacy", "v5"])
def test_executor_passes_live_acceptance_clock_instead_of_captured_stamp(tmp_path, monkeypatch, semantics):
    from test_argon4_forward_execution import BOUNDARY as boundary, fixture_inputs, policy, v5_policy, clock
    account = Account(tmp_path / "executor-clock.sqlite3", started_at=boundary-timedelta(seconds=1))
    executor = PaperExecutor(account, v5_policy() if semantics == "v5" else policy())
    decisions, markets = fixture_inputs(capacity=100)
    current = {"now": boundary+timedelta(seconds=1)}
    original = account.place_paper_buy
    seen = []
    def overrun(*args, **kwargs):
        seen.append(callable(kwargs["now"]))
        current["now"] = boundary+timedelta(seconds=21)
        return original(*args, **kwargs)
    monkeypatch.setattr(account, "place_paper_buy", overrun)
    result = executor.process_cohort(boundary.isoformat(), decisions, markets,
                                    now=lambda: current["now"], clock=clock())
    assert seen == [True] and result["fills"] == [] and result["status"] == "BLOCKED"
    with sqlite3.connect(account.path) as con:
        for name in ("signals", "orders", "consumed_book_depth", "allocation_consumption"):
            assert con.execute("SELECT COUNT(*) FROM " + name).fetchone()[0] == 0
        assert con.execute("SELECT cash_atoms FROM account").fetchone()[0] == 10_000_000_000


@pytest.mark.parametrize("stale", ["book", "metadata", "calibration"])
def test_executor_acceptance_clock_preserves_stricter_current_observation_limits(prepared, stale):
    f = prepared
    executor = PaperExecutor(f.account, f.policy)
    calibration = {"status": "CALIBRATED", "source": "POLYMARKET_CLOB_SERVER_TIME",
                   "uncertainty_seconds": .1, "observed_at": at(7).isoformat()}
    if stale == "metadata": f.fee["observed_at"] = at(-82).isoformat()
    if stale == "calibration": calibration["observed_at"] = at(-52).isoformat()
    callback = executor._paper_acceptance_clock(f.clock, calibration, f.book, f.fee, f.signal)
    assert callback() == at(8)
    f.clock.value = at(10.001) if stale == "book" else at(8.001)
    with pytest.raises(AccountRejected): callback()
