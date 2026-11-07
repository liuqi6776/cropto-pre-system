from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from tools.argon4_forward.account import Account, AccountRejected, STREAMS, V5_SEMANTICS


START = datetime(2026, 11, 5, 0, 0, tzinfo=timezone.utc)
DECISION = START + timedelta(seconds=1)
FILL = START + timedelta(seconds=2)


def signal(identifier="s1", **changes):
    value = dict(signal_id=identifier, stream="BTC_15m", underlying="BTC", outcome="UP",
                 market_id="btc-15m-1", model_id="frozen-local-model-v1", model_sha256="a"*64,
                 validation_status="INPUTS_VALIDATED", feature_available_at=START.isoformat(),
                 decision_at=DECISION.isoformat(), valid_until=(START+timedelta(seconds=15)).isoformat(),
                 window_start=START.isoformat(), window_end=(START+timedelta(minutes=15)).isoformat())
    value.update(changes)
    return value


def book(sig=None, **changes):
    sig = sig or signal()
    value = {k: sig[k] for k in ("market_id", "underlying", "outcome", "window_start", "window_end")}
    value.update(kind="OBSERVED_PUBLIC_ORDERBOOK", source="https://venue.example/public/book",
                 observed_at=FILL.isoformat(), received_at=FILL.isoformat(),
                 asks=[dict(price="0.5", size="1000")])
    value.update(changes)
    return value


def fee(**changes):
    value = dict(fee_model="fixed_bps", bps=0, label="SIMULATED_ASSUMPTION",
                 source="explicit test-only cost assumption", observed_at=FILL.isoformat())
    value.update(changes)
    return value


def resolution(outcome="UP", **changes):
    end = START+timedelta(minutes=15)
    value = dict(kind="OBSERVED_OFFICIAL_RESOLUTION", market_id="btc-15m-1", outcome=outcome,
                 resolution_id="official-1", source="https://venue.example/public/resolution",
                 resolved_at=end.isoformat(), received_at=(end+timedelta(seconds=1)).isoformat())
    value.update(changes)
    return value


def buy(account, sig, observed, fees, **kwargs):
    authorization = dict(policy="RANKED_CAPACITY_WATERFALL", plan_id="ranked-plan-"+sig["signal_id"], rank=1,
                         signal_id=sig["signal_id"], market_id=sig["market_id"], approved_stake_usd=200,
                         planned_at=DECISION.isoformat(), valid_until=sig["valid_until"])
    kwargs.setdefault("stake", 200)
    return account.place_paper_buy(sig, observed, fees, allocation_authorization=authorization, **kwargs)


@pytest.fixture
def account(tmp_path):
    return Account(tmp_path/"forward.sqlite3", started_at=START)


def test_initial_capital_restart_never_resets(account):
    buy(account, signal(), book(), fee(), now=FILL)
    restart = Account(account.path, initial_capital=50000, started_at=START+timedelta(days=1))
    state = restart.overview(now=FILL)
    assert state["initial_capital"] == 10000
    assert state["started_at"] == START.isoformat()
    assert state["cash"] == 9800
    assert state["locked_cost"] == 200
    assert state["equity"] is None
    assert state["unrealized_pnl"] is None
    assert state["live_enabled"] is False


def test_balance_conservation_and_settlement_idempotence(account):
    order = buy(account, signal(), book(), fee(), now=FILL)
    again = buy(account, signal(), book(), fee(), now=FILL)
    assert again["duplicate"] is True
    assert account.overview(now=FILL)["order_count"] == 1
    assert order["quantity"] == "400.00000000"
    end = START+timedelta(minutes=16)
    result = account.settle("btc-15m-1", resolution(), now=end)
    assert result["payout_usd"] == 400
    assert result["realized_pnl_usd"] == 200
    assert account.settle("btc-15m-1", resolution(), now=end)["status"] == "ALREADY_SETTLED"
    state = account.overview(now=end)
    assert state["cash"] == state["equity"] == 10200
    assert state["realized_pnl"] == 200
    assert state["wins"] == 1 and state["losses"] == 0
    assert state["win_rate_pct"] == 100
    assert len(account.equity_history()) == 3


@pytest.mark.parametrize("changes", [
    {"decision_at": (START-timedelta(seconds=1)).isoformat()},
    {"decision_at": (FILL+timedelta(seconds=1)).isoformat()},
    {"feature_available_at": (FILL+timedelta(seconds=1)).isoformat()},
    {"valid_until": FILL.isoformat()},
    {"model_sha256": ""},
    {"validation_status": "WAITING_FOR_MODEL"},
    {"underlying": "ETH"},
    {"window_end": (START+timedelta(minutes=5)).isoformat()},
])
def test_invalid_signal_never_reaches_ledger(account, changes):
    with pytest.raises(AccountRejected):
        account.record_signal(signal(**changes), now=FILL)
    assert account.overview(now=FILL)["signal_count"] == 0
    assert len(account.ledger()) == 1


@pytest.mark.parametrize("changes", [
    {"market_id": "wrong-market"},
    {"underlying": "ETH"},
    {"outcome": "DOWN"},
    {"observed_at": (START-timedelta(seconds=10)).isoformat()},
    {"observed_at": (FILL+timedelta(seconds=1)).isoformat()},
    {"window_end": (START+timedelta(minutes=5)).isoformat()},
    {"kind": "SYNTHETIC_BOOK"},
])
def test_invalid_orderbook_rejected(account, changes):
    with pytest.raises(AccountRejected):
        buy(account, signal(), book(**changes), fee(), now=FILL)
    assert account.overview(now=FILL)["cash"] == 10000


def test_stale_book_and_same_instant_fill_rejected(account):
    with pytest.raises(AccountRejected):
        buy(account, signal(), book(), fee(), now=START+timedelta(seconds=8))
    with pytest.raises(AccountRejected):
        buy(account, signal(), book(), fee(), now=DECISION)


def test_sparse_book_does_not_invent_liquidity(account):
    observed = book(asks=[dict(price=.5, size=20), dict(price=.51, size=10), dict(price=.52, size=1000)])
    order = buy(account, signal(), observed, fee(), now=FILL)
    assert order["quantity"] == "30.00000000"
    assert order["cost_usd"] == pytest.approx(15.1)
    assert order["unfilled_budget_usd"] == pytest.approx(184.9)
    assert account.overview(now=FILL)["cash"] == pytest.approx(9984.9)


def test_same_observed_depth_cannot_be_filled_twice(account):
    observed = book(asks=[dict(price=.5, size=20)])
    first = buy(account, signal("s1"), observed, fee(), now=FILL)
    second = buy(account, signal("s2"), observed, fee(), now=FILL)
    assert first["quantity"] == "20.00000000"
    assert second["status"] == "NO_FILL"
    assert account.overview(now=FILL)["locked_cost"] == 10


def test_reobserved_identical_resolution_is_idempotent(account):
    buy(account, signal(), book(), fee(), now=FILL)
    stamp = START+timedelta(minutes=16)
    account.settle("btc-15m-1", resolution(), now=stamp)
    later = resolution(received_at=stamp.isoformat())
    assert account.settle("btc-15m-1", later, now=stamp)["status"] == "ALREADY_SETTLED"
    assert account.overview(now=stamp)["cash"] == 10200


def test_no_fill_consumes_no_cash_and_cannot_retry_signal_with_other_order(account):
    order = buy(account, signal(), book(asks=[]), fee(), now=FILL)
    assert order["status"] == "NO_FILL"
    assert account.overview(now=FILL)["cash"] == 10000
    with pytest.raises(AccountRejected):
        buy(account, signal(), book(), fee(), now=FILL, order_id="another")


def test_void_refunds_cost_and_fees_only_when_rule_is_explicit(account):
    buy(account, signal(), book(), fee(bps=100), now=FILL)
    now = START+timedelta(minutes=16)
    with pytest.raises(AccountRejected):
        account.settle("btc-15m-1", resolution("VOID"), now=now)
    account.settle("btc-15m-1", resolution("VOID", refund_rule="COST_AND_FEES"), now=now)
    state = account.overview(now=now)
    assert state["cash"] == 10000
    assert state["realized_pnl"] == 0
    assert state["voids"] == 1
    assert state["win_rate_pct"] is None


def test_resolution_must_be_official_after_contract_close(account):
    buy(account, signal(), book(), fee(), now=FILL)
    now = START+timedelta(minutes=16)
    with pytest.raises(AccountRejected):
        account.settle("btc-15m-1", resolution(kind="BINANCE_PRICE_PROXY"), now=now)
    with pytest.raises(AccountRejected):
        account.settle("btc-15m-1", resolution(resolved_at=FILL.isoformat()), now=now)
    account.settle("btc-15m-1", resolution("DOWN"), now=now)
    state = account.overview(now=now)
    assert state["cash"] == 9800
    assert state["losses"] == 1
    assert state["max_realized_drawdown_pct"] == -2
    with pytest.raises(AccountRejected):
        account.settle("btc-15m-1", resolution("UP"), now=now)


def test_fees_require_provenance_and_polymarket_curvature(account):
    with pytest.raises(AccountRejected):
        buy(account, signal(), book(), {}, now=FILL)
    q = dict(fee_model="polymarket_crypto", rate=.07, exponent=1, taker_only=True,
             market_id="btc-15m-1", outcome="UP", source_sha256="b"*64,
             source="https://venue.example/market-fees", observed_at=FILL.isoformat())
    result = buy(account, signal(), book(asks=[dict(price=.5, size=100)]), q, now=FILL)
    assert result["quantity"] == "100.00000000"
    assert result["cost_usd"] == 50
    assert result["fee_usd"] == 1.75
    assert account.overview(now=FILL)["cash"] == 9948.25


def test_exposure_and_stake_limits(account):
    buy(account, signal("s1"), book(), fee(), now=FILL)
    buy(account, signal("s2"), book(), fee(), now=FILL)
    with pytest.raises(AccountRejected):
        buy(account, signal("s3"), book(), fee(), now=FILL)
    assert account.overview(now=FILL)["locked_cost"] == 400
    with pytest.raises(AccountRejected):
        buy(account, signal("s4"), book(), fee(), now=FILL, stake=201)


def test_unallocated_signal_cannot_trade(account):
    with pytest.raises(AccountRejected, match="allocation authorization required"):
        account.place_paper_buy(signal(), book(), fee(), now=FILL)
    assert account.overview(now=FILL)["order_count"] == 0


def test_changed_retry_is_conflict_but_identical_late_retry_is_idempotent(account):
    buy(account, signal(), book(), fee(), now=FILL)
    with pytest.raises(AccountRejected, match="conflicting reuse"):
        buy(account, signal(), book(), fee(), now=FILL, stake=100)
    with pytest.raises(AccountRejected, match="conflicting reuse"):
        buy(account, signal(model_id="different-model"), book(), fee(), now=FILL)
    repeated = buy(account, signal(), book(), fee(), now=START+timedelta(days=1))
    assert repeated["duplicate"] is True
    assert account.overview(now=FILL)["order_count"] == 1


@pytest.mark.parametrize("changes", [dict(market_id="wrong-market"), dict(outcome="DOWN"),
                                     dict(exponent=2), dict(taker_only=False), dict(source_sha256="")])
def test_polymarket_fee_schedule_must_match_market_and_supported_formula(account, changes):
    q = dict(fee_model="polymarket_crypto", rate=.07, exponent=1, taker_only=True,
             market_id="btc-15m-1", outcome="UP", source_sha256="b"*64,
             source="https://venue.example/market-fees", observed_at=FILL.isoformat())
    q.update(changes)
    with pytest.raises(AccountRejected, match="unbound Polymarket"):
        buy(account, signal(), book(), q, now=FILL)


@pytest.mark.parametrize("value", ["private_key", "mnemonic", "my secret", "seed phrase", "api-key"])
def test_credential_like_input_is_never_persisted(account, value):
    with pytest.raises(AccountRejected):
        account.set_runtime_status("WAITING_FOR_MODEL", details={"note": value}, now=FILL)
    assert value.encode() not in account.path.read_bytes()


def test_heartbeat_and_audit_hash_chain(account):
    state = account.set_runtime_status("WAITING_FOR_MODEL", "Model artifact unavailable", {"observed_sources": 0}, now=FILL)
    assert state["runtime_status"] == "WAITING_FOR_MODEL"
    assert state["heartbeat_age_seconds"] == 0
    assert state["ready_streams"] == []
    account.record_signal(signal(), now=FILL)
    events = list(reversed(account.ledger()))
    assert events[1]["previous_hash"] == events[0]["event_hash"]
    json.dumps(state, allow_nan=False)


def v5_cohort(account):
    policy={"status":"CONFIRMED","version":"account-test-v5","ranking_source":"explicit test ranking",
            "order":list(STREAMS),"waterfall_semantics":V5_SEMANTICS}
    return account.ensure_cohort(START,policy,now=FILL)


def no_signal_receipt(stream="BTC_15m"):
    return {"source":"VALIDATED_MODEL_DECISION","prediction":{"stream":stream,"selected":False,
            "z_v4":.2,"threshold_frozen":1,"side":"UP","model_version":"test-model",
            "model_sha256":"a"*64,"manifest_sha256":"b"*64,"decision_time":DECISION.isoformat(),
            "feature_bar_open":(START-timedelta(minutes=5)).isoformat()}}


def test_v5_skip_has_audited_zero_spend_and_conflict_aware_idempotence(account):
    v5_cohort(account)
    receipt=no_signal_receipt()
    first=account.record_cohort_skip(START,"BTC_15m","NO_SIGNAL",receipt,now=FILL)
    again=account.record_cohort_skip(START,"BTC_15m","NO_SIGNAL",receipt,now=FILL+timedelta(seconds=1))
    assert first["used_usd"]==again["used_usd"]==0
    assert again["streams"][0]["state"]=="NO_SIGNAL"
    assert account.overview()["cash"]==10000
    assert len([e for e in account.ledger() if e["event_type"]=="COHORT_STREAM_SKIPPED"])==1
    changed=deepcopy(receipt);changed["prediction"]["z_v4"]=.3
    with pytest.raises(AccountRejected,match="conflicting"):
        account.record_cohort_skip(START,"BTC_15m","NO_SIGNAL",changed,now=FILL)


def test_v5_skips_cannot_bypass_expiry_or_rank_or_due_state(account):
    v5_cohort(account)
    with pytest.raises(AccountRejected,match="higher cohort"):
        account.record_cohort_skip(START,"ETH_15m","NO_SIGNAL",no_signal_receipt("ETH_15m"),now=FILL)
    with pytest.raises(AccountRejected,match="actually due"):
        account.record_cohort_skip(START,"BTC_15m","NOT_DUE",{"source":"UTC_COHORT_SCHEDULE"},now=FILL)
    with pytest.raises(AccountRejected,match="cohort unavailable"):
        account.record_cohort_skip(START,"BTC_15m","NO_SIGNAL",no_signal_receipt(),now=START+timedelta(seconds=20))
    assert account.cohort_state(START)["streams"]==[]


def test_v5_cannot_use_legacy_capacity_gate_as_skip_authority(account):
    v5_cohort(account)
    with pytest.raises(AccountRejected,match="explicit skip receipt"):
        account.record_cohort_gate(START,"BTC_15m","NO_SIGNAL",100,
            {"source":"test book","sha256":"c"*64,"observed_at":FILL.isoformat()},now=FILL)
    assert account.cohort_state(START)["streams"]==[]


def test_legacy_cohort_cannot_receive_v5_skip_state(account):
    policy={"status":"CONFIRMED","version":"legacy-test","ranking_source":"test","order":list(STREAMS)}
    account.ensure_cohort(START,policy,now=FILL)
    with pytest.raises(AccountRejected,match="not valid under"):
        account.record_cohort_skip(START,"BTC_15m","NO_SIGNAL",no_signal_receipt(),now=FILL)
    assert account.cohort_state(START)["streams"]==[]


def test_v5_direct_price_skip_requires_actual_valid_signal_and_price_evidence(account):
    v5_cohort(account)
    receipt={"source":"OBSERVED_PRICE_LIMIT_CHECK","signal":signal(),
             "orderbook":book(asks=[{"price":.6,"size":1000}]),"fee_quote":fee()}
    bad=deepcopy(receipt);bad["signal"]["model_sha256"]="bad"
    with pytest.raises(AccountRejected,match="SHA256"):
        account.record_cohort_skip(START,"BTC_15m","PRICE_LIMIT_NOT_MET",bad,now=FILL)
    bad=deepcopy(receipt);bad["orderbook"]["asks"]=[{"price":.5,"size":1}]
    with pytest.raises(AccountRejected,match="above the limit"):
        account.record_cohort_skip(START,"BTC_15m","PRICE_LIMIT_NOT_MET",bad,now=FILL)
    result=account.record_cohort_skip(START,"BTC_15m","PRICE_LIMIT_NOT_MET",receipt,now=FILL)
    assert result["streams"][0]["state"]=="PRICE_LIMIT_NOT_MET"
    assert account.overview()["order_count"]==0 and account.overview()["cash"]==10000


@pytest.mark.parametrize("change",["missing_semantics","stake_under_approved"])
def test_v5_direct_order_requires_exact_approved_budget_authority(account,change):
    cohort=v5_cohort(account)
    auth={"policy":"RANKED_CAPACITY_WATERFALL","waterfall_semantics":V5_SEMANTICS,
          "cohort_id":cohort["cohort_id"],"plan_id":cohort["plan_id"],"policy_version":cohort["policy_version"],
          "rank":1,"signal_id":"s1","market_id":"btc-15m-1","approved_stake_usd":150,
          "cohort_capacity_usd":500,"planned_at":DECISION.isoformat(),"valid_until":signal()["valid_until"]}
    stake=150
    if change=="missing_semantics":auth.pop("waterfall_semantics")
    else:stake=100
    with pytest.raises(AccountRejected,match="approved-budget"):
        account.place_paper_buy(signal(),book(),fee(),now=FILL,stake=stake,allocation_authorization=auth)
    assert account.overview()["order_count"]==0
    assert account.cohort_state(START)["used_usd"]==0


def staged_policy(enabled=None, effective=START, stage_id="three-stream-test"):
    order = list(STREAMS)
    allowed = set(enabled or ("ETH_15m", "ETH_5m", "SOL_5m"))
    return {"status": "CONFIRMED", "version": "stage-test-v1", "ranking_source": "test fixed ranking",
            "order": order, "waterfall_semantics": V5_SEMANTICS,
            "execution_stage": {"id": stage_id, "status": "CONFIRMED",
                                "enabled_streams": [s for s in order if s in allowed],
                                "effective_from": effective.isoformat()}}


def staged_signal(stream, boundary=START):
    underlying, seconds = STREAMS[stream]
    return signal("stage-"+stream+boundary.isoformat(), stream=stream, underlying=underlying,
                  market_id="market-"+stream+boundary.isoformat(),
                  feature_available_at=boundary.isoformat(), decision_at=(boundary+timedelta(seconds=1)).isoformat(),
                  valid_until=(boundary+timedelta(seconds=15)).isoformat(), window_start=boundary.isoformat(),
                  window_end=(boundary+timedelta(seconds=seconds)).isoformat())


def staged_buy(account, cohort, stream, stake=150):
    boundary = datetime.fromisoformat(cohort["cohort_id"])
    sig = staged_signal(stream, boundary)
    stamp = boundary+timedelta(seconds=2)
    auth = {"policy": "RANKED_CAPACITY_WATERFALL", "waterfall_semantics": V5_SEMANTICS,
            "cohort_id": cohort["cohort_id"], "plan_id": cohort["plan_id"],
            "policy_version": cohort["policy_version"], "rank": cohort["policy"]["order"].index(stream)+1,
            "signal_id": sig["signal_id"], "market_id": sig["market_id"], "approved_stake_usd": stake,
            "cohort_capacity_usd": 500, "planned_at": sig["decision_at"], "valid_until": sig["valid_until"]}
    return account.place_paper_buy(sig, book(sig, observed_at=stamp.isoformat(), received_at=stamp.isoformat()),
                                  fee(observed_at=stamp.isoformat()), now=stamp, stake=stake,
                                  allocation_authorization=auth)


def test_stage_disabled_tiers_are_atomic_zero_budget_and_restart_idempotent(account):
    policy = staged_policy()
    cohort = account.ensure_cohort(START, policy, now=FILL)
    disabled = [s for s in policy["order"] if s not in policy["execution_stage"]["enabled_streams"]]
    assert [s["stream"] for s in cohort["streams"]] == disabled
    assert len(cohort["streams"]) == 4
    assert all(s["state"] == "DISABLED_BY_STAGE" and s["capacity_usd"] == s["approved_usd"] == s["spent_usd"] == 0
               and s["signal_id"] is None and s["receipt"]["book_checked"] is False for s in cohort["streams"])
    assert cohort["used_usd"] == 0 and cohort["remaining_usd"] == 220
    events = account.ledger()
    assert len([e for e in events if e["event_type"] == "COHORT_STREAM_SKIPPED"]) == 4
    restarted = Account(account.path, initial_capital=20000)
    assert restarted.ensure_cohort(START, policy, now=FILL) == cohort
    assert restarted.ledger() == events and restarted.overview()["cash"] == 10000


def test_stage_disabled_tier_failure_rolls_back_entire_cohort(account, monkeypatch):
    original = account._event
    def fail_on_skip(con, kind, payload, now):
        if kind == "COHORT_STREAM_SKIPPED":
            raise RuntimeError("test transaction failure")
        original(con, kind, payload, now)
    monkeypatch.setattr(account, "_event", fail_on_skip)
    with pytest.raises(RuntimeError, match="transaction failure"):
        account.ensure_cohort(START, staged_policy(), now=FILL)
    assert account.cohort_state(START) is None
    assert len(account.ledger()) == 1


def test_stage_disabled_upper_does_not_block_enabled_lower_or_change_caps(account):
    cohort = account.ensure_cohort(START, staged_policy(), now=FILL)
    assert staged_buy(account, cohort, "ETH_15m")["cohort_stream_state"] == "FILLED_BUDGET"
    assert staged_buy(account, cohort, "ETH_5m", 70)["cohort_stream_state"] == "FILLED_BUDGET"
    with pytest.raises(AccountRejected, match="persistent cohort budget"):
        staged_buy(account, cohort, "SOL_5m", 1)
    assert account.cohort_state(START)["used_usd"] == 220
    assert account.overview()["cash"] == 9780 and account.overview()["locked_cost"] == 220


def test_stage_direct_buy_and_skip_cannot_reenable_disabled_stream(account):
    cohort = account.ensure_cohort(START, staged_policy(), now=FILL)
    with pytest.raises(AccountRejected, match="disabled"):
        staged_buy(account, cohort, "BTC_15m")
    # Removing the cohort from an otherwise valid legacy authorization cannot bypass it.
    with pytest.raises(AccountRejected, match="disabled"):
        buy(account, signal(), book(), fee(), now=FILL)
    sig = staged_signal("ETH_15m")
    with pytest.raises(AccountRejected, match="persisted decision cohort"):
        buy(account, sig, book(sig), fee(), now=FILL)
    with pytest.raises(AccountRejected, match="disabled"):
        account.record_cohort_skip(START, "BTC_15m", "NO_SIGNAL", no_signal_receipt(), now=FILL)
    with pytest.raises(AccountRejected, match="explicit validated skip"):
        account.record_cohort_skip(START, "ETH_15m", "DISABLED_BY_STAGE", {}, now=FILL)
    assert account.overview()["order_count"] == 0


@pytest.mark.parametrize("change", ["pending", "duplicate", "unknown", "wrong_order", "future", "legacy"])
def test_bad_stage_never_creates_cohort_or_partial_disabled_events(account, change):
    policy = staged_policy()
    spec = policy["execution_stage"]
    if change == "pending": spec["status"] = "PENDING"
    elif change == "duplicate": spec["enabled_streams"].append(spec["enabled_streams"][0])
    elif change == "unknown": spec["enabled_streams"].append("UNKNOWN_5m")
    elif change == "wrong_order": spec["enabled_streams"].reverse()
    elif change == "future": spec["effective_from"] = (START+timedelta(minutes=5)).isoformat()
    elif change == "legacy": policy.pop("waterfall_semantics")
    with pytest.raises(AccountRejected, match="stage"):
        account.ensure_cohort(START, policy, now=FILL)
    assert account.cohort_state(START) is None and len(account.ledger()) == 1


def test_stage_expansion_only_changes_future_cohort(account):
    original = account.ensure_cohort(START, staged_policy(), now=FILL)
    next_boundary = START+timedelta(minutes=15)
    expanded = staged_policy(list(STREAMS), effective=next_boundary, stage_id="seven-stream-test")
    with pytest.raises(AccountRejected, match="pre-activation"):
        account.ensure_cohort(START, expanded, now=FILL)
    changed_now = staged_policy(list(STREAMS), stage_id="conflicting-test")
    with pytest.raises(AccountRejected, match="cannot change"):
        account.ensure_cohort(START, changed_now, now=FILL)
    future = account.ensure_cohort(next_boundary, expanded, now=next_boundary+timedelta(seconds=2))
    assert future["streams"] == []
    assert staged_buy(account, future, "BTC_15m")["status"] == "OPEN"
    assert account.cohort_state(START) == original


def test_stage_buy_restriction_never_prevents_existing_official_settlement(account):
    buy(account, signal(), book(), fee(), now=FILL)
    boundary = START+timedelta(minutes=5)
    account.ensure_cohort(boundary, staged_policy(effective=boundary), now=boundary+timedelta(seconds=2))
    result = account.settle("btc-15m-1", resolution(), now=START+timedelta(minutes=16))
    assert result["status"] == "SETTLED" and result["realized_pnl_usd"] == 200
    assert account.overview()["cash"] == 10200 and account.overview()["locked_cost"] == 0


@pytest.mark.parametrize("eth_skip", [False, True])
def test_user_stage_order_requires_eth5_before_btc15_then_btc5(account, eth_skip):
    order = ["ETH_5m", "BTC_15m", "BTC_5m", "SOL_15m", "SOL_5m", "ETH_15m", "BNB_15m"]
    policy = staged_policy(order[:3])
    policy["order"] = order
    policy["execution_stage"]["enabled_streams"] = order[:3]
    cohort = account.ensure_cohort(START, policy, now=FILL)
    with pytest.raises(AccountRejected, match="higher cohort"):
        staged_buy(account, cohort, "BTC_15m")
    if eth_skip:
        account.record_cohort_skip(START, "ETH_5m", "NO_SIGNAL", no_signal_receipt("ETH_5m"), now=FILL)
        assert staged_buy(account, cohort, "BTC_15m")["cost_usd"] == 150
        assert staged_buy(account, cohort, "BTC_5m", 70)["cost_usd"] == 70
    else:
        assert staged_buy(account, cohort, "ETH_5m")["cost_usd"] == 150
        assert staged_buy(account, cohort, "BTC_15m", 70)["cost_usd"] == 70
        with pytest.raises(AccountRejected, match="persistent cohort budget"):
            staged_buy(account, cohort, "BTC_5m", 1)
    assert account.cohort_state(START)["used_usd"] == 220
