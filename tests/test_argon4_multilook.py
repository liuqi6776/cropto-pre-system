"""Future-only, bounded repeated public book observations; no external orders."""
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import requests

from tools.argon4_forward.account import Account, AccountRejected, multi_look_settings
from tools.argon4_forward.execution import PaperExecutor
from test_argon4_forward_execution import BOUNDARY, ORDER, clock, fixture_inputs, no_signal, v5_policy


def v12_policy():
    return {**v5_policy(), "version": "isolated-multi-look-v12",
            "engineering": {"profile": "argon4-latency-v12"},
            "execution": {"multi_look_ioc": True, "max_price_rechecks": 3,
                "price_recheck_interval_seconds": 2.5, "minimum_recheck_remaining_seconds": 3.0,
                "last_recheck_start_after_boundary_seconds": 18.5}}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("network forbidden"))


@pytest.fixture
def case(tmp_path):
    account = Account(tmp_path / "isolated.sqlite3", started_at=BOUNDARY-timedelta(seconds=1))
    d, m = fixture_inputs(capacity=600)
    for decision in d.values():
        if decision.get("signal"):
            decision["signal"]["valid_until"] = (BOUNDARY+timedelta(seconds=20)).isoformat()
    return SimpleNamespace(account=account, executor=PaperExecutor(account, v12_policy()),
        decisions=d, markets=m, now=BOUNDARY+timedelta(seconds=9), calls=[], first_price=.52)


def book(case, market, side):
    case.calls.append((market["market_id"], side, case.now))
    case.now += timedelta(seconds=.05)
    value = deepcopy(market["books"][side])
    value.update(source_at=case.now.isoformat(), received_at=case.now.isoformat())
    value["asks"] = [{"price": case.first_price if market["market_id"] == case.markets[ORDER[0]]["market_id"] else .5,
                      "size": 1200}]
    return value


def execute(case, refresh=None):
    return case.executor.process_cohort(BOUNDARY, case.decisions, case.markets, now=lambda: case.now,
        clock=clock(BOUNDARY), book_refresh=refresh or (lambda market, side: book(case, market, side)))


def at_next(case, result):
    from datetime import datetime
    case.now = datetime.fromisoformat(result["next_retry_at"])


def test_price_miss_waits_without_cashflow_or_lower_rank_credit_then_fills(case):
    first = execute(case)
    assert first["status"] == "WAITING_PRICE_RECHECK"
    assert first["fills"] == [] and first["blocked"] == []
    assert first["cohort"]["streams"] == []
    assert first["stream_results"][ORDER[1]]["status"] == "UNREACHED"
    assert len(case.calls) == 1 and case.account.overview()["cash"] == 10000
    proof = case.account.book_look_state(BOUNDARY, ORDER[0])
    assert proof["state"] == "PRICE_LIMIT_NOT_MET" and proof["attempt"] == 0
    assert proof["receipt"]["orderbook"]["asks"][0]["price"] == .52
    case.now += timedelta(seconds=1)
    waiting = execute(case)
    assert waiting["status"] == "WAITING_PRICE_RECHECK" and len(case.calls) == 1
    at_next(case, first)
    case.first_price = .51
    result = execute(case)
    assert result["status"] == "PROCESSED"
    assert [f["cost_usd"] for f in result["fills"]] == [150, 70]
    assert result["fills"][0]["allocation_authorization"]["book_look_attempt"] == 1
    assert result["cohort"]["used_usd"] == 220
    assert len(case.calls) == 3
    repeat = execute(case)
    assert repeat["fills"] == [] and len(case.calls) == 3


def test_three_rechecks_then_terminal_price_skip_releases_lower_rank(case):
    for attempt in range(4):
        result = execute(case)
        proof = case.account.book_look_state(BOUNDARY, ORDER[0])
        assert proof["attempt"] == attempt
        if attempt < 3:
            assert result["status"] == "WAITING_PRICE_RECHECK"
            assert result["fills"] == []
            at_next(case, result)
    assert result["status"] == "PROCESSED"
    assert result["stream_results"][ORDER[0]]["status"] == "PRICE_LIMIT_NOT_MET"
    assert result["fills"][0]["underlying"] == "ETH"
    assert sum(x[0] == case.markets[ORDER[0]]["market_id"] for x in case.calls) == 4
    assert proof["next_retry_at"] is None
    terminal = result["cohort"]["streams"][0]["receipt"]
    assert terminal["orderbook"]["observed_at"] == proof["receipt"]["orderbook"]["observed_at"]
    with case.account._connection() as con:
        assert con.execute("SELECT COUNT(*) FROM cohort_book_looks WHERE stream=?", (ORDER[0],)).fetchone()[0] == 4


def test_last_recheck_may_start_at_18_point_5_and_fill_before_20(case):
    case.now = BOUNDARY + timedelta(seconds=15.95)
    result = execute(case)
    assert result["next_retry_at"] == (BOUNDARY+timedelta(seconds=18.5)).isoformat()
    at_next(case, result)
    case.first_price = .5
    result = execute(case)
    assert result["fills"] and result["status"] == "PROCESSED"
    assert all(f["fill_at"] < (BOUNDARY+timedelta(seconds=20)).isoformat() for f in result["fills"])


def test_missed_recheck_start_cannot_reuse_old_book_or_unlock_lower(case):
    result = execute(case)
    case.now = BOUNDARY + timedelta(seconds=18.500001)
    result = execute(case)
    assert result["status"] == "BLOCKED" and result["fills"] == []
    assert len(case.calls) == 1 and result["cohort"]["streams"] == []
    assert case.account.book_look_state(BOUNDARY, ORDER[0])["state"] == "FAILED"


@pytest.mark.parametrize("start", [16.1, 17.1, 19.0])
def test_insufficient_remaining_budget_records_only_observed_final_miss(case, start):
    case.now = BOUNDARY + timedelta(seconds=start)
    result = execute(case)
    assert result["status"] == "PROCESSED"
    assert result["stream_results"][ORDER[0]]["status"] == "PRICE_LIMIT_NOT_MET"
    assert sum(x[0] == case.markets[ORDER[0]]["market_id"] for x in case.calls) == 1


@pytest.mark.parametrize("fault", ["stale", "empty", "wrong_token", "missing_fee", "network", "cached_response", "past_deadline"])
def test_unknown_or_invalid_observation_is_terminal_error_not_price_retry(case, fault):
    def refresh(market, side):
        if fault == "network":
            raise TimeoutError("offline timeout")
        result = book(case, market, side)
        if fault == "stale":
            result["source_at"] = (case.now-timedelta(seconds=3)).isoformat()
        elif fault == "empty":
            result["asks"] = []
        elif fault == "wrong_token":
            result["asset_id"] = "wrong"
        elif fault == "missing_fee":
            market["fee"] = {}
        elif fault == "cached_response":
            result["received_at"] = (case.now-timedelta(seconds=.2)).isoformat()
            result["source_at"] = result["received_at"]
        elif fault == "past_deadline":
            case.now = BOUNDARY + timedelta(seconds=20)
        return result
    result = execute(case, refresh)
    assert result["status"] == "BLOCKED" and not result["fills"]
    assert case.account.book_look_state(BOUNDARY, ORDER[0])["state"] == "FAILED"
    count = len(case.calls)
    result = execute(case)
    assert not result["fills"] and len(case.calls) == count


def test_interrupted_http_claim_cannot_be_issued_twice_after_account_reopen(case):
    case.account.ensure_cohort(BOUNDARY, case.executor.policy, now=case.now)
    signal = {**case.decisions[ORDER[0]]["signal"], "market_id": case.markets[ORDER[0]]["market_id"]}
    claim = case.account.begin_book_look(BOUNDARY, ORDER[0], signal, now=case.now)
    assert claim["state"] == "REQUESTED"
    case.account = Account(case.account.path)
    case.executor = PaperExecutor(case.account, v12_policy())
    result = execute(case)
    assert result["status"] == "BLOCKED" and case.calls == []
    assert case.account.overview()["order_count"] == 0


def test_signal_identity_cannot_change_between_looks(case):
    result = execute(case)
    at_next(case, result)
    case.decisions[ORDER[0]]["signal"]["signal_id"] = "changed"
    result = execute(case)
    assert result["status"] == "BLOCKED" and len(case.calls) == 1


def test_pending_signal_cannot_be_replaced_by_no_signal(case):
    execute(case)
    case.decisions[ORDER[0]] = no_signal(ORDER[0])
    result = execute(case)
    assert result["status"] == "BLOCKED" and len(case.calls) == 1


def test_asks_at_limit_remain_eligible_with_fee_inside_budget(case):
    case.first_price = .515
    case.markets[ORDER[0]]["fee"] = {"status":"KNOWN_SCHEDULE", "fees_enabled":True,
                                     "rate":.07,"exponent":1,"taker_only":True}
    result = execute(case)
    fill = result["fills"][0]
    assert float(fill["fills"][0]["price"]) == .515 and fill["fee_usd"] > 0
    assert fill["cost_usd"] + fill["fee_usd"] == pytest.approx(150, abs=.000002)


def test_old_v11_policy_keeps_one_look_even_if_new_fields_are_present(case):
    case.executor.policy["engineering"]["profile"] = "argon4-latency-v11"
    result = execute(case)
    assert result["status"] == "PROCESSED"
    assert result["stream_results"][ORDER[0]]["status"] == "PRICE_LIMIT_NOT_MET"
    assert case.account.book_look_state(BOUNDARY, ORDER[0]) is None


@pytest.mark.parametrize("key,value", [("max_price_rechecks",4),("max_price_rechecks",True),
    ("price_recheck_interval_seconds",0),("minimum_recheck_remaining_seconds",0),
    ("last_recheck_start_after_boundary_seconds",19.9),("multi_look_ioc","true")])
def test_policy_rejects_unbounded_or_ambiguous_rechecks(key, value):
    policy = v12_policy()
    policy["execution"][key] = value
    with pytest.raises(AccountRejected):
        multi_look_settings(policy)


def test_paper_fill_requires_matching_durable_book_look(case, monkeypatch):
    case.first_price = .5
    original = case.account.place_paper_buy
    def alter(*args, **kwargs):
        kwargs["allocation_authorization"]["book_look_attempt"] = 99
        return original(*args, **kwargs)
    monkeypatch.setattr(case.account, "place_paper_buy", alter)
    result = execute(case)
    assert result["status"] == "BLOCKED" and not result["fills"]
    assert case.account.overview()["cash"] == 10000


def test_no_historical_cohort_can_start_or_refill(case):
    case.now = BOUNDARY+timedelta(seconds=20)
    result = execute(case)
    assert result["status"] == "BLOCKED" and case.calls == []
    assert case.account.book_look_state(BOUNDARY, ORDER[0]) is None


@pytest.mark.parametrize("spent,remaining", [(50, 50), (100, 0)])
def test_actual_partial_or_zero_fill_never_retries_or_unlocks_lower(case, spent, remaining):
    market = case.markets[ORDER[0]]
    observed = deepcopy(market["books"]["UP"])
    observed.update(source_at=case.now.isoformat(), received_at=case.now.isoformat(),
                    asks=[{"price": .5, "size": 200}])
    adapted, fee, _ = case.executor._adapt(market, "UP", observed, case.now, clock(BOUNDARY))
    prior = {**case.decisions[ORDER[0]]["signal"], "signal_id": "fixture-prior-consumption",
             "market_id": market["market_id"]}
    authorization = {"policy":"RANKED_CAPACITY_WATERFALL", "plan_id":"prior", "rank":1,
        "signal_id":prior["signal_id"], "market_id":market["market_id"], "approved_stake_usd":spent,
        "planned_at":case.now.isoformat(), "valid_until":prior["valid_until"]}
    case.account.place_paper_buy(prior, adapted, fee, now=case.now, stake=spent,
                                allocation_authorization=authorization)
    result = execute(case, lambda *args: observed)
    assert result["status"] == "PARTIAL_BLOCKED"
    assert result["fills"][0]["cost_usd"] == remaining
    assert result["stream_results"][ORDER[0]]["status"] == "PARTIAL"
    assert result["stream_results"][ORDER[1]]["status"] == "UNREACHED"
    case.executor = PaperExecutor(Account(case.account.path), v12_policy())
    case.now += timedelta(seconds=3)
    repeat = execute(case, lambda *args: pytest.fail("partial cannot retry"))
    assert repeat["status"] == "BLOCKED" and not repeat["fills"]


def test_begin_claim_rechecks_clock_after_database_wait(case):
    case.account.ensure_cohort(BOUNDARY, case.executor.policy, now=case.now)
    signal = {**case.decisions[ORDER[0]]["signal"], "market_id":case.markets[ORDER[0]]["market_id"]}
    times = iter([case.now, BOUNDARY+timedelta(seconds=20)])
    with pytest.raises(AccountRejected):
        case.account.begin_book_look(BOUNDARY, ORDER[0], signal, now=lambda: next(times))
    assert case.account.book_look_state(BOUNDARY, ORDER[0]) is None


def test_clock_overrun_after_claim_never_dispatches_http(case, monkeypatch):
    original = case.account.begin_book_look
    def delayed(*args, **kwargs):
        claimed = original(*args, **kwargs)
        case.now = BOUNDARY+timedelta(seconds=20)
        return claimed
    monkeypatch.setattr(case.account, "begin_book_look", delayed)
    result = execute(case, lambda *args: pytest.fail("expired claim cannot dispatch"))
    assert result["status"] == "BLOCKED" and not result["fills"]
    assert case.account.book_look_state(BOUNDARY, ORDER[0])["state"] == "FAILED"


def test_final_price_skip_cannot_commit_after_lock_wait_expiry(case):
    case.now = BOUNDARY + timedelta(seconds=19)
    case.account.ensure_cohort(BOUNDARY, case.executor.policy, now=case.now)
    market = case.markets[ORDER[0]]
    signal = {**case.decisions[ORDER[0]]["signal"], "market_id":market["market_id"]}
    claim = case.account.begin_book_look(BOUNDARY, ORDER[0], signal, now=case.now)
    observed = book(case, market, "UP")
    adapted, fee, _ = case.executor._adapt(market, "UP", observed, case.now, clock(BOUNDARY), allow_price_skip=True)
    receipt = {"source":"OBSERVED_PRICE_LIMIT_CHECK", "signal":signal, "orderbook":adapted, "fee_quote":fee}
    case.account.finish_book_look(BOUNDARY, ORDER[0], claim["attempt"], "PRICE_LIMIT_NOT_MET", receipt, now=case.now)
    times = iter([case.now, BOUNDARY+timedelta(seconds=20)])
    with pytest.raises(AccountRejected):
        case.account.record_cohort_skip(BOUNDARY, ORDER[0], "PRICE_LIMIT_NOT_MET", receipt, now=lambda: next(times))
    assert case.account.cohort_state(BOUNDARY)["streams"] == []
