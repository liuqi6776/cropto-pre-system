from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from tools.argon4_forward.allocation import plan_waterfall, ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1
from tools.argon4_forward.account import Account, AccountRejected


NOW = datetime(2026, 11, 5, 0, 0, 1, tzinfo=timezone.utc)
ORDER = ["BTC_15m", "ETH_5m", "ETH_15m", "BTC_5m", "SOL_15m", "SOL_5m", "BNB_15m"]


def policy(**changes):
    result = dict(status="CONFIRMED", order=ORDER[:], ranking_source="user-confirmed frozen validation ranking", version="test-v1")
    result.update(changes)
    return result


def candidates(capacity=100):
    return [dict(stream=name, capacity_usd=capacity, capacity_source="https://venue.example/observed-book",
                 capacity_observed_at=NOW.isoformat(),
                 signal=dict(stream=name, signal_id="signal-"+name, market_id="market-"+name,
                             decision_at=NOW.isoformat(), valid_until=(NOW+timedelta(seconds=10)).isoformat()))
            for name in ORDER]


def plan(c=None, p=None, **kwargs):
    defaults = dict(equity=10000, cash=10000, open_risk=0, now=NOW)
    defaults.update(kwargs)
    return plan_waterfall(policy() if p is None else p, candidates() if c is None else c, **defaults)


def test_highest_capacity_not_full_keeps_next_locked_and_cash_unspent():
    result = plan(candidates(600))
    assert [(a["stream"], a["stake_usd"]) for a in result["allocations"]] == [(ORDER[0], 150)]
    assert result["unspent"] == 70
    assert result["preview"][1]["status"] == "LOCKED"


def test_covered_capacity_unlocks_next_under_shared_tick_budget():
    result = plan(candidates(100))
    assert [a["stake_usd"] for a in result["allocations"]] == [100, 100, 20]
    assert [a["rank"] for a in result["allocations"]] == [1, 2, 3]
    assert result["unspent"] == 0
    assert result["preview"][3]["status"] == "LOCKED"


def test_input_order_does_not_override_confirmed_ranking():
    result = plan(list(reversed(candidates(600))))
    assert result["allocations"][0]["stream"] == ORDER[0]
    assert len(result["allocations"]) == 1


@pytest.mark.parametrize("mutation", ["missing", "zero", "stale", "future", "unsourced"])
def test_unknown_capacity_never_counts_as_full(mutation):
    c = candidates()
    if mutation == "missing":
        c = c[1:]
    elif mutation == "zero":
        c[0]["capacity_usd"] = 0
    elif mutation == "stale":
        c[0]["capacity_observed_at"] = (NOW-timedelta(seconds=3)).isoformat()
    elif mutation == "future":
        c[0]["capacity_observed_at"] = (NOW+timedelta(seconds=1)).isoformat()
    else:
        c[0]["capacity_source"] = ""
    result = plan(c)
    assert result["allocations"] == []
    assert result["preview"][1]["status"] == "LOCKED"


def test_explicit_no_signal_may_skip_only_when_capacity_covered():
    c = candidates()
    c[0]["signal"] = None
    result = plan(c)
    assert result["allocations"][0]["stream"] == ORDER[1]
    c[0]["capacity_usd"] = 600
    assert plan(c)["allocations"] == []


@pytest.mark.parametrize("changes", [dict(status="PENDING"), dict(order=ORDER[:6]),
                                     dict(order=[ORDER[0]]*7), dict(ranking_source=""), dict(version="")])
def test_unconfirmed_or_incomplete_policy_never_authorizes(changes):
    result = plan(p=policy(**changes))
    assert result["allocations"] == []


def test_cash_and_open_risk_constrain_whole_tick_without_invention():
    result = plan(candidates(50), cash=75, open_risk=920)
    assert [a["stake_usd"] for a in result["allocations"]] == [50, 25]
    assert sum(a["stake_usd"] for a in result["allocations"]) == 75
    assert result["initial_tick_budget_usd"] == 75
    assert plan(candidates(), cash=0)["allocations"] == []
    assert plan(candidates(), open_risk=1000)["allocations"] == []


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), 20000])
def test_invalid_cash_is_not_filled_in(value):
    assert plan(cash=value)["allocations"] == []


def test_daily_loss_uses_observed_day_start_equity_and_stops_at_threshold():
    assert plan(day_pnl=-550, day_start_equity=10000)["reasons"] == ["DAILY_LOSS_LIMIT"]
    assert plan(day_pnl=-1)["reasons"] == ["DAY_START_EQUITY_REQUIRED"]
    assert plan(day_pnl=-549, day_start_equity=10000)["allocations"]


def test_four_losing_days_halves_single_budget_not_ranking():
    result = plan(candidates(600), losing_days=4)
    assert result["allocations"][0]["stake_usd"] == 75
    assert result["unspent"] == 145


def test_five_losing_days_require_explicit_cooldown_not_permanent_lock():
    assert plan(losing_days=5)["reasons"] == ["COOLDOWN_STATE_REQUIRED"]
    active = policy(cooldown_until=(NOW+timedelta(days=1)).isoformat())
    assert plan(p=active, losing_days=5)["reasons"] == ["COOLDOWN_ACTIVE"]
    expired = policy(cooldown_until=(NOW-timedelta(seconds=1)).isoformat())
    assert plan(p=expired, losing_days=5)["allocations"]


def test_hard_order_limit_does_not_unlock_uncaptured_capacity():
    result = plan(candidates(300), equity=100000, cash=100000)
    assert [a["stake_usd"] for a in result["allocations"]] == [200]
    assert result["preview"][1]["status"] == "LOCKED"


def test_authorization_is_short_lived_and_scoped_to_signal():
    result = plan()
    auth = result["allocations"][0]["allocation_authorization"]
    assert auth["policy"] == "RANKED_CAPACITY_WATERFALL"
    assert auth["signal_id"] == "signal-"+ORDER[0]
    assert auth["market_id"] == "market-"+ORDER[0]
    assert auth["planned_at"] == NOW.isoformat()
    assert auth["valid_until"] == (NOW+timedelta(seconds=2)).isoformat()


def test_future_or_expired_upper_signal_blocks_lower():
    c = candidates()
    c[0]["signal"]["decision_at"] = (NOW+timedelta(seconds=1)).isoformat()
    result = plan(c)
    assert result["allocations"] == []
    assert result["reasons"] == ["SIGNAL_INVALID"]


def test_missing_signal_status_is_not_treated_as_explicit_no_signal():
    c = candidates()
    del c[0]["signal"]
    assert plan(c)["reasons"] == ["SIGNAL_STATE_MISSING"]


def test_lower_rank_waits_for_actual_upper_fill_in_same_plan(tmp_path):
    account = Account(tmp_path/"ranked.sqlite3", started_at=NOW-timedelta(seconds=1))
    c = candidates(100)
    for row in c:
        sym, minutes = row["stream"].split("_")
        row["signal"].update(underlying=sym, outcome="UP", model_id="test-model", model_sha256="a"*64,
                             validation_status="INPUTS_VALIDATED", feature_available_at=(NOW-timedelta(seconds=1)).isoformat(),
                             window_start=(NOW-timedelta(seconds=1)).isoformat(),
                             window_end=(NOW-timedelta(seconds=1)+timedelta(minutes=int(minutes[:-1]))).isoformat())
    planned = plan(c)
    first, second = planned["allocations"][:2]
    stamp = NOW+timedelta(seconds=1)
    fee = dict(fee_model="fixed_bps", label="SIMULATED_ASSUMPTION", bps=0,
               source="explicit test-only assumption", observed_at=stamp.isoformat())

    def execute(allocation, size=200, auth=None):
        sig = next(row["signal"] for row in c if row["stream"] == allocation["stream"])
        book = {k: sig[k] for k in ("market_id", "underlying", "outcome", "window_start", "window_end")}
        book.update(kind="OBSERVED_PUBLIC_ORDERBOOK", source="https://venue.example/book",
                    observed_at=stamp.isoformat(), received_at=stamp.isoformat(), asks=[dict(price=.5, size=size)])
        return account.place_paper_buy(sig, book, fee, now=stamp, stake=allocation["stake_usd"],
                                      allocation_authorization=auth or allocation["allocation_authorization"])

    with pytest.raises(AccountRejected, match="has not actually filled"):
        execute(second)
    execute(first)
    wrong_plan = dict(second["allocation_authorization"], plan_id="different-plan")
    with pytest.raises(AccountRejected, match="has not actually filled"):
        execute(second, auth=wrong_plan)
    execute(second)
    assert account.overview(now=stamp)["locked_cost"] == 200


def test_partial_upper_fill_never_unlocks_lower(tmp_path):
    account = Account(tmp_path/"partial.sqlite3", started_at=NOW-timedelta(seconds=1))
    c = candidates(100)
    result = plan(c)
    stamp = NOW+timedelta(seconds=1)
    fee = dict(fee_model="fixed_bps", label="SIMULATED_ASSUMPTION", bps=0,
               source="explicit test-only assumption", observed_at=stamp.isoformat())
    for allocation in result["allocations"][:2]:
        sig = next(row["signal"] for row in c if row["stream"] == allocation["stream"])
        sym, minutes = allocation["stream"].split("_")
        sig.update(underlying=sym, outcome="UP", model_id="test-model", model_sha256="a"*64,
                   validation_status="INPUTS_VALIDATED", feature_available_at=(NOW-timedelta(seconds=1)).isoformat(),
                   window_start=(NOW-timedelta(seconds=1)).isoformat(),
                   window_end=(NOW-timedelta(seconds=1)+timedelta(minutes=int(minutes[:-1]))).isoformat())
        book = {k: sig[k] for k in ("market_id", "underlying", "outcome", "window_start", "window_end")}
        book.update(kind="OBSERVED_PUBLIC_ORDERBOOK", source="https://venue.example/book",
                    observed_at=stamp.isoformat(), received_at=stamp.isoformat(), asks=[dict(price=.5, size=20)])
        if allocation["rank"] == 1:
            account.place_paper_buy(sig, book, fee, now=stamp, stake=100,
                                   allocation_authorization=allocation["allocation_authorization"])
        else:
            with pytest.raises(AccountRejected, match="has not actually filled"):
                account.place_paper_buy(sig, book, fee, now=stamp, stake=100,
                                       allocation_authorization=allocation["allocation_authorization"])
    assert account.overview(now=stamp)["locked_cost"] == 10


def test_function_does_not_mutate_policy_candidate_or_account_inputs():
    c, p = candidates(), policy()
    original_c, original_p = deepcopy(c), deepcopy(p)
    plan(c, p)
    assert c == original_c and p == original_p


def v5_policy(**changes):
    return policy(waterfall_semantics=ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1, **changes)


def price_failure_candidate():
    row = candidates()[0]
    row.update(status="PRICE_LIMIT_NOT_MET", capacity_usd=0)
    row["signal"]["outcome"] = "UP"
    row["price_evidence"] = {"book": dict(
        kind="OBSERVED_PUBLIC_ORDERBOOK", source="https://clob.polymarket.com/book?token_id=123",
        market_id=row["signal"]["market_id"], outcome="UP", observed_at=NOW.isoformat(),
        received_at=NOW.isoformat(), asks=[dict(price=".60", size="200"), dict(price=".70", size="10")])}
    hash_price_evidence(row)
    return row


def hash_price_evidence(row):
    row["price_evidence"]["sha256"] = hashlib.sha256(json.dumps(
        row["price_evidence"]["book"], ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def test_v5_deep_book_plans_150_then_70_without_claiming_any_fill():
    result = plan(candidates(600), v5_policy())
    assert [(a["stream"], a["stake_usd"]) for a in result["allocations"]] == [(ORDER[0], 150), (ORDER[1], 70)]
    assert result["unspent"] == 0
    assert result["execution_confirmation_required"] is True
    assert result["preview"][0]["status"] == "PLANNED_ALLOCATION"
    assert result["preview"][0]["requires_actual_approved_fill"] is True
    assert result["allocations"][0]["capacity_usd"] == 600
    assert result["allocations"][1]["allocation_authorization"]["preceding_allocations"] == [
        dict(signal_id="signal-"+ORDER[0], rank=1, required_cost_usd=150)]
    assert result["allocations"][0]["allocation_authorization"]["waterfall_semantics"] == ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1


def test_v5_no_signal_needs_no_capacity_and_does_not_consume_budget():
    c = candidates(600)
    c[0] = dict(stream=ORDER[0], signal=None)
    result = plan(c, v5_policy())
    assert result["preview"][0]["status"] == "NO_SIGNAL"
    assert result["preview"][0]["capacity_usd"] is None
    assert result["preview"][0]["unlocks_next"] is True
    assert [(a["rank"], a["stake_usd"]) for a in result["allocations"]] == [(2, 150), (3, 70)]


def test_v5_all_explicit_no_signals_returns_no_eligible_signal():
    c = [dict(stream=s, signal=None) for s in ORDER]
    result = plan(c, v5_policy())
    assert result["status"] == "NO_ELIGIBLE_SIGNAL"
    assert result["blocked"] == []
    assert result["unspent"] == 220


@pytest.mark.parametrize("explicit_none", [False, True])
def test_v5_not_due_without_capacity_is_checked_against_utc_boundary(explicit_none):
    when = NOW + timedelta(minutes=5)
    c = candidates(600)
    c[0] = dict(stream=ORDER[0], status="NOT_DUE")
    if explicit_none:
        c[0]["signal"] = None
    for row in c[1:]:
        row["capacity_observed_at"] = when.isoformat()
        row["signal"].update(decision_at=when.isoformat(), valid_until=(when+timedelta(seconds=10)).isoformat())
    result = plan(c, v5_policy(), now=when)
    assert result["preview"][0]["status"] == "NOT_DUE"
    assert [a["stake_usd"] for a in result["allocations"]] == [150, 70]
    assert plan(c, v5_policy())["reasons"] == ["NOT_DUE_INVALID"]


@pytest.mark.parametrize("mutation", ["five_minute_stream", "contradictory_signal"])
def test_v5_not_due_does_not_hide_a_due_or_actionable_signal(mutation):
    c = candidates()
    p = v5_policy()
    c[0]["status"] = "NOT_DUE"
    if mutation == "five_minute_stream":
        p["order"] = [ORDER[1], ORDER[0], *ORDER[2:]]
        c[1] = dict(stream=ORDER[1], status="NOT_DUE")
    result = plan(c, p, now=NOW+timedelta(minutes=5))
    assert result["allocations"] == []
    assert result["reasons"] == ["NOT_DUE_INVALID"]


@pytest.mark.parametrize("mutation", ["missing", "zero", "stale", "future", "unsourced", "missing_signal", "unknown_status", "contradictory_no_signal"])
def test_v5_unknown_or_bad_input_still_blocks_lower_ranks(mutation):
    c = candidates(600)
    if mutation == "missing":
        c = c[1:]
    elif mutation == "zero":
        c[0]["capacity_usd"] = 0
    elif mutation == "stale":
        c[0]["capacity_observed_at"] = (NOW-timedelta(seconds=3)).isoformat()
    elif mutation == "future":
        c[0]["capacity_observed_at"] = (NOW+timedelta(seconds=1)).isoformat()
    elif mutation == "unsourced":
        c[0]["capacity_source"] = ""
    elif mutation == "missing_signal":
        del c[0]["signal"]
    elif mutation == "unknown_status":
        c[0]["status"] = "ERROR"
    else:
        c[0]["status"] = "NO_SIGNAL"
    result = plan(c, v5_policy())
    assert result["allocations"] == []
    assert result["preview"][1]["status"] == "LOCKED"


def test_v5_verified_price_failure_skips_without_consuming_cash():
    c = candidates(600)
    c[0] = price_failure_candidate()
    result = plan(c, v5_policy())
    assert result["preview"][0]["status"] == "PRICE_LIMIT_NOT_MET"
    assert result["preview"][0]["best_ask"] == .60
    assert result["preview"][0]["allocated_usd"] == 0
    assert [(a["rank"], a["stake_usd"]) for a in result["allocations"]] == [(2, 150), (3, 70)]
    assert result["unspent"] == 0


@pytest.mark.parametrize("mutation", ["missing", "hash", "stale", "future", "before_decision", "received_before_observed", "empty", "zero_quantity",
                                     "at_limit", "below_limit", "invalid_price", "invalid_size", "wrong_market", "wrong_side", "unsourced", "invalid_source", "expired_signal"])
def test_v5_price_failure_requires_consistent_fresh_nonempty_evidence(mutation):
    c = candidates()
    c[0] = row = price_failure_candidate()
    book = row["price_evidence"]["book"]
    if mutation == "missing":
        del row["price_evidence"]
    elif mutation == "hash":
        row["price_evidence"]["sha256"] = "0"*64
    elif mutation == "stale":
        book["observed_at"] = (NOW-timedelta(seconds=3)).isoformat()
    elif mutation == "future":
        book["received_at"] = (NOW+timedelta(seconds=1)).isoformat()
    elif mutation == "before_decision":
        book["observed_at"] = (NOW-timedelta(seconds=1)).isoformat()
    elif mutation == "received_before_observed":
        book["received_at"] = (NOW-timedelta(seconds=1)).isoformat()
    elif mutation == "empty":
        book["asks"] = []
    elif mutation == "zero_quantity":
        book["asks"] = [dict(price=".60", size="0")]
    elif mutation in ("at_limit", "below_limit", "invalid_price"):
        book["asks"][0]["price"] = {"at_limit":".515", "below_limit":".50", "invalid_price":"NaN"}[mutation]
    elif mutation == "invalid_size":
        book["asks"][0]["size"] = "-1"
    elif mutation == "wrong_market":
        book["market_id"] = "other-market"
    elif mutation == "wrong_side":
        book["outcome"] = "DOWN"
    elif mutation == "unsourced":
        book["source"] = ""
    elif mutation == "invalid_source":
        book["source"] = "https://"
    else:
        row["signal"]["valid_until"] = NOW.isoformat()
    if mutation not in ("missing", "hash"):
        hash_price_evidence(row)
    result = plan(c, v5_policy())
    assert result["allocations"] == []
    assert result["preview"][1]["status"] == "LOCKED"
    assert result["reasons"] == (["SIGNAL_INVALID"] if mutation == "expired_signal" else ["PRICE_EVIDENCE_INVALID"])


def test_v5_preserves_cash_risk_daily_loss_cooldown_and_half_budget_limits():
    assert [a["stake_usd"] for a in plan(candidates(600), v5_policy(), cash=75)["allocations"]] == [75]
    assert [a["stake_usd"] for a in plan(candidates(600), v5_policy(), open_risk=920)["allocations"]] == [80]
    assert [a["stake_usd"] for a in plan(candidates(600), v5_policy(), losing_days=4)["allocations"]] == [75, 75, 70]
    assert plan(p=v5_policy(), day_pnl=-550, day_start_equity=10000)["reasons"] == ["DAILY_LOSS_LIMIT"]
    assert plan(p=v5_policy(), losing_days=5)["reasons"] == ["COOLDOWN_STATE_REQUIRED"]
    active = v5_policy(cooldown_until=(NOW+timedelta(days=1)).isoformat())
    assert plan(p=active, losing_days=5)["reasons"] == ["COOLDOWN_ACTIVE"]
    capped = plan(candidates(600), v5_policy(), equity=100000, cash=100000)
    assert [a["stake_usd"] for a in capped["allocations"]] == [200]*5
    assert sum(a["stake_usd"] for a in capped["allocations"]) == capped["initial_tick_budget_usd"] == 1000


def test_v5_rank_budget_and_input_immutability_and_legacy_compatibility():
    c, p = list(reversed(candidates(600))), v5_policy()
    original_c, original_p = deepcopy(c), deepcopy(p)
    result = plan(c, p)
    assert [a["stream"] for a in result["allocations"]] == ORDER[:2]
    assert c == original_c and p == original_p
    legacy = plan(candidates(600))
    assert [a["stake_usd"] for a in legacy["allocations"]] == [150]
    assert legacy["unspent"] == 70
    assert legacy["plan_id"] != result["plan_id"]
    assert plan(p=policy(waterfall_semantics="UNKNOWN"))["reasons"] == ["UNSUPPORTED_WATERFALL_SEMANTICS"]


def stage_policy(enabled=None, **changes):
    value = {"id": "three-stream-test", "status": "CONFIRMED",
             "enabled_streams": enabled or [ORDER[1], ORDER[3], ORDER[5]],
             "effective_from": NOW.replace(second=0).isoformat()}
    value.update(changes)
    return v5_policy(execution_stage=value)


def test_stage_preview_ignores_disabled_books_and_retains_full_relative_ranking():
    p = stage_policy()
    enabled = p["execution_stage"]["enabled_streams"]
    only_enabled = [c for c in reversed(candidates(600)) if c["stream"] in enabled]
    result = plan(only_enabled, p)
    assert [(a["stream"], a["stake_usd"]) for a in result["allocations"]] == [(enabled[0], 150), (enabled[1], 70)]
    assert [a["rank"] for a in result["allocations"]] == [2, 4]
    disabled = [row for row in result["preview"] if row["stream"] not in enabled]
    assert len(disabled) == 4
    assert all(row["status"] == "DISABLED_BY_STAGE" and row["allocated_usd"] == 0
               and row["book_checked"] is False for row in disabled)
    assert result["allocations"][1]["allocation_authorization"]["preceding_allocations"] == [
        {"signal_id": "signal-"+enabled[0], "rank": 2, "required_cost_usd": 150}]


def test_stage_preview_unknown_enabled_signal_blocks_but_disabled_still_displayed():
    p = stage_policy()
    result = plan([], p)
    assert result["allocations"] == [] and result["reasons"] == ["SIGNAL_STATE_MISSING"]
    assert all(row["status"] == "DISABLED_BY_STAGE" for row in result["preview"]
               if row["stream"] not in p["execution_stage"]["enabled_streams"])


@pytest.mark.parametrize("changes", [{"status": "PENDING"}, {"enabled_streams": [ORDER[1], ORDER[1]]},
    {"enabled_streams": ["UNKNOWN_5m"]}, {"enabled_streams": [ORDER[3], ORDER[1]]},
    {"effective_from": (NOW.replace(second=0)+timedelta(minutes=5)).isoformat()}])
def test_stage_preview_unconfirmed_invalid_or_future_stage_never_authorizes(changes):
    result = plan(p=stage_policy(**changes))
    assert result["allocations"] == [] and result["reasons"] == ["INVALID_EXECUTION_STAGE"]


def test_stage_preview_identity_binds_allowlist_and_future_activation():
    first = plan(p=stage_policy())
    second = plan(p=stage_policy([ORDER[0], ORDER[1], ORDER[3]], id="different-stage"))
    assert first["plan_id"] != second["plan_id"]
    assert first["allocations"][0]["allocation_authorization"]["execution_stage"]["id"] == "three-stream-test"
    assert plan(p=stage_policy([ORDER[0], ORDER[1], ORDER[3]]))["allocations"][0]["stream"] == ORDER[0]


@pytest.mark.parametrize("case, expected", [
    ("deep", [("ETH_5m", 150), ("BTC_15m", 70)]),
    ("eth_skip", [("BTC_15m", 150), ("BTC_5m", 70)]),
    ("finite_depth", [("ETH_5m", 60), ("BTC_15m", 60), ("BTC_5m", 100)]),
])
def test_user_stage_order_eth5_btc15_btc5_uses_remaining_budget(case, expected):
    order = ["ETH_5m", "BTC_15m", "BTC_5m", "SOL_15m", "SOL_5m", "ETH_15m", "BNB_15m"]
    p = stage_policy(order[:3])
    p["order"] = order
    c = [row for row in reversed(candidates(600)) if row["stream"] in order[:3]]
    if case == "eth_skip":
        next(row for row in c if row["stream"] == "ETH_5m")["signal"] = None
    elif case == "finite_depth":
        for row in c:
            if row["stream"] in order[:2]:
                row["capacity_usd"] = 60
    result = plan(c, p)
    assert [(row["stream"], row["stake_usd"]) for row in result["allocations"]] == expected
    assert all(row["status"] == "DISABLED_BY_STAGE" for row in result["preview"][3:])
