"""Capital-sharing and causal cashflow tests for post-hoc assumptions only."""
from decimal import Decimal,ROUND_DOWN

import numpy as np
import pandas as pd
import pytest

from tools.argon4_forward import gen4_portfolio_scenario as s


@pytest.fixture
def policy():
    return {**s.POLICY_NUMBERS,"status":"CONFIRMED","live_enabled":False,
            "waterfall_semantics":"ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1",
            "order":["BTC_15m","ETH_5m","SOL_15m","SOL_5m","BTC_5m","ETH_15m","BNB_15m"]}


def predictions(entries,phase="fixture"):
    rows=[]
    for raw_time,selected in entries.items():
        time=pd.Timestamp(raw_time)
        for stream,minutes in s.STREAM_MINUTES.items():
            if time.value%(minutes*60*1_000_000_000):
                continue
            active=stream in selected
            side,ret=selected.get(stream,("UP",.001))
            score=(2. if side=="UP" else -2.) if active else .1
            finite=np.isfinite(ret)
            for variant in s.VARIANTS:
                rows.append({"stream":stream,"phase":phase,"variant":variant,"decision_close_utc":time,
                             "score":score,"threshold":1.,"selected":active,"side":side if active else "UP",
                             "forward_return":ret,"settled":bool(active and finite),
                             "win":bool(active and finite and ((side=="UP")==(ret>=0)))})
    return pd.DataFrame(rows)


def simulate(frame,policy,**kwargs):
    return s.simulate_portfolio(frame,policy,"CURRENT_FROZEN",include_ledger=True,**kwargs)


def test_one_shared_tick_budget_and_rank_priority(policy):
    all_active={name:("UP",.01) for name in s.STREAM_MINUTES}
    result=simulate(predictions({"2026-01-01T00:00:00Z":all_active}),policy)
    assert [r["stream"] for r in result["scenario_entries"]]==["BTC_15m","ETH_5m"]
    assert [r["approved_usd"] for r in result["scenario_entries"]]==[150,70]
    assert result["maximum_open_risk_usd"]<=220
    assert result["end_of_decision_window"]["cash_usd"]==9780
    assert result["end_of_decision_window"]["open_cost_usd"]==220
    assert result["cash_plus_open_cost_equals_initial_plus_realized"]


def test_coin_limit_blocks_remaining_lower_streams(policy):
    times=pd.date_range("2026-01-01",periods=5,freq="15min",tz="UTC")
    all_active={name:("UP",.01) for name in s.STREAM_MINUTES}
    result=simulate(predictions({t:all_active for t in times}),policy,settlement_delay_minutes=60)
    assert result["maximum_underlying_open_risk_usd"]["BTC"]==500
    assert result["maximum_open_risk_usd"]==880
    assert result["end_of_decision_window"]["open_cost_usd"]==880
    assert result["blocked_cohorts_or_streams"]["CURRENT_STREAM_CASH_OR_RISK_EXHAUSTED"]==1
    assert not any(r["decision_utc"]==times[-1].isoformat() for r in result["scenario_entries"])


def test_aggregate_risk_and_overlapping_positions_share_one_limit(policy):
    times=pd.date_range("2026-01-01",periods=8,freq="5min",tz="UTC")
    chosen=[{"ETH_5m":("UP",.01)}]*3
    chosen+=[{"ETH_5m":("UP",.01),"BTC_5m":("UP",.01)}]
    chosen+=[{"BTC_5m":("UP",.01)}]*3
    chosen+=[{"SOL_5m":("UP",.01)}]
    result=simulate(predictions(dict(zip(times,chosen))),policy,settlement_delay_minutes=60)
    assert result["maximum_open_risk_usd"]==1000
    assert result["end_of_decision_window"]["open_cost_usd"]==1000
    assert result["end_of_decision_window"]["cash_usd"]==9000
    assert result["maximum_underlying_open_risk_usd"]["BTC"]==500
    assert result["maximum_underlying_open_risk_usd"]["ETH"]==500
    assert result["blocked_cohorts_or_streams"]["NO_RISK_BUDGET"]==1


def test_delayed_settlement_does_not_release_future_cash_or_losses_early(policy):
    frame=predictions({"2026-01-01T00:00:00Z":{"ETH_5m":("UP",-.01)},
                       "2026-01-01T00:05:00Z":{"ETH_5m":("UP",-.01)}})
    immediate=simulate(frame,policy,settlement_delay_minutes=0)
    delayed=simulate(frame,policy,settlement_delay_minutes=60)
    assert immediate["scenario_entries"][1]["approved_usd"]==147.75
    assert delayed["scenario_entries"][1]["approved_usd"]==150
    assert immediate["end_of_decision_window"]["realized_balance_usd"]==9850
    assert delayed["end_of_decision_window"]["realized_balance_usd"]==10000
    assert immediate["realized_balance_usd"]==9702.25
    assert delayed["realized_balance_usd"]==9700
    assert pd.Timestamp(delayed["settlement_drain_until"])==pd.Timestamp("2026-01-01T01:10:00Z")


def test_daily_loss_stop_uses_realized_day_start_balance(policy):
    times=pd.date_range("2026-01-01",periods=9,freq="5min",tz="UTC")
    result=simulate(predictions({t:{"ETH_5m":("UP",-.01)} for t in times}),policy)
    assert result["entries"]==4
    assert result["blocked_cohorts_or_streams"]["DAILY_LOSS_LIMIT"]==5
    assert -600<result["realized_pnl_usd"]<=-550
    assert result["max_realized_balance_drawdown_usd"]==-result["realized_pnl_usd"]


def test_all_same_time_settlements_precede_next_budget_and_drawdown_order_is_declared(policy):
    frame=predictions({"2026-01-01T00:00:00Z":{"ETH_5m":("UP",.01),"SOL_5m":("UP",-.01)},
                       "2026-01-01T00:05:00Z":{"ETH_5m":("UP",.01)}})
    result=simulate(frame,policy)
    first,second,third=result["scenario_entries"]
    basis=Decimal(10000)+Decimal(str(first["realized_pnl_usd"]))+Decimal(str(second["realized_pnl_usd"]))
    expected=(basis*Decimal(".015")).quantize(Decimal(".000001"),rounding=ROUND_DOWN)
    assert third["approved_usd"]==float(expected)
    assert result["max_realized_balance_drawdown_usd"]==70
    assert "ENTRY_NUMBER" in result["assumptions"]["simultaneous_settlements"]


def test_four_losing_days_half_size_fifth_requires_cooldown_then_zero_day_resets(policy):
    times=pd.date_range("2026-01-01",periods=7,freq="D",tz="UTC")
    result=simulate(predictions({t:{"ETH_5m":("UP",-.01)} for t in times}),policy)
    records={r["decision_utc"]:r for r in result["scenario_entries"]}
    fifth=records[times[4].isoformat()]
    assert fifth["preceding_losing_calendar_days"]==4
    assert fifth["approved_usd"]==float((Decimal(str(fifth["realized_equity_basis_usd"]))*Decimal(".0075")).quantize(Decimal(".000001"),rounding=ROUND_DOWN))
    assert times[5].isoformat() not in records
    assert records[times[6].isoformat()]["preceding_losing_calendar_days"]==0
    assert result["blocked_cohorts_or_streams"]["COOLDOWN_STATE_REQUIRED"]==1


@pytest.mark.parametrize("side,wins,losses",[("UP",1,0),("DOWN",0,1)])
def test_direction_proxy_ties_go_up(policy,side,wins,losses):
    result=simulate(predictions({"2026-01-01T00:00:00Z":{"ETH_5m":(side,0.)}}),policy)
    assert result["proxy_wins"]==wins and result["proxy_losses"]==losses
    assert result["entries"]==1 and result["open_cost_usd"]==0
    assert (result["realized_pnl_usd"]>0)==(side=="UP")


def test_missing_future_label_enters_then_keeps_risk_without_invented_loss(policy):
    result=simulate(predictions({"2026-01-01T00:00:00Z":{"ETH_5m":("UP",np.nan)}}),policy)
    assert result["entries"]==1 and result["unresolved_entries"]==1
    assert result["proxy_settled_entries"]==result["proxy_losses"]==result["proxy_wins"]==0
    assert result["end_cash_usd"]==9850 and result["open_cost_usd"]==150
    assert result["realized_balance_usd"]==10000 and result["realized_pnl_usd"]==0
    assert result["scenario_entries"][0]["scheduled_proxy_settlement_utc"] is None


def test_unknown_outcome_cannot_affect_decisions_before_its_maturity(policy):
    entries={"2026-01-01T00:00:00Z":{"BTC_15m":("UP",.01)},
             "2026-01-01T00:05:00Z":{"ETH_5m":("UP",.01)}}
    finite=simulate(predictions(entries),policy)
    entries["2026-01-01T00:00:00Z"]["BTC_15m"]=("UP",np.nan)
    unknown=simulate(predictions(entries),policy)
    assert finite["end_of_decision_window"]==unknown["end_of_decision_window"]
    assert [r["approved_usd"] for r in finite["scenario_entries"]]==[r["approved_usd"] for r in unknown["scenario_entries"]]


def test_absent_due_row_is_unknown_not_a_no_signal_skip(policy):
    frame=predictions({"2026-01-01T00:00:00Z":{"SOL_5m":("UP",.01)},"2026-01-01T00:05:00Z":{}})
    frame=frame.loc[~((frame.stream=="ETH_5m")&(frame.decision_close_utc==pd.Timestamp("2026-01-01T00:00:00Z")))]
    result=simulate(frame,policy)
    assert result["entries"]==0
    assert result["blocked_cohorts_or_streams"]["MISSING_DUE_DECISION"]==1


def test_dataframe_order_cannot_bypass_frozen_rank(policy):
    frame=predictions({"2026-01-01T00:00:00Z":{name:("UP",.01) for name in s.STREAM_MINUTES}})
    first=simulate(frame,policy)
    shuffled=simulate(frame.sample(frac=1,random_state=7),policy)
    assert first["scenario_entries"]==shuffled["scenario_entries"]
    assert first["policy_order"]==policy["order"]


@pytest.mark.parametrize("rate",s.FEE_RATES)
def test_fast_fill_rounding_matches_account_step_loop(rate):
    price=Decimal(".515"); fee=Decimal(str(rate))*price*(1-price)
    for budget in (1,2,3,17,1001,999999,70_000_000,150_000_000,200_000_000):
        qty=(Decimal(budget)/s.MICRO/(price+fee)).quantize(s.QUANTITY_STEP,rounding=ROUND_DOWN)
        while qty>0:
            cost=s._atoms(qty*price); fees=s._atoms(qty*fee)
            if cost+fees<=budget:
                break
            qty-=s.QUANTITY_STEP
        expected=(qty,cost,fees) if qty>0 else (Decimal(0),0,0)
        assert s._ideal_fill(budget,Decimal(str(rate)))==expected


def test_risk_changes_and_unregistered_arms_rejected(policy):
    frame=predictions({"2026-01-01T00:00:00Z":{}})
    changed=dict(policy,max_open_risk_usd=2000)
    with pytest.raises(s.ScenarioError,match="numerical risk"):
        simulate(frame,changed)
    with pytest.raises(s.ScenarioError,match="fixed post-hoc"):
        s.simulate_portfolio(frame,policy,"TAIL_BASE_RESET")
    with pytest.raises(s.ScenarioError,match="absent cooldown"):
        simulate(frame,dict(policy,cooldown_until="2026-01-02T00:00:00Z"))


def test_fixed_36_cases_never_flip_preregistered_selection(policy):
    frame=predictions({"2026-01-01T00:00:00Z":{"ETH_5m":("UP",.01)}})
    result=s.run_portfolio_scenarios({"historical":frame,"recent":frame},policy)
    assert result["scenario_count"]==36
    assert result["preregistered_model_screen"]=="DO_NOT_PROMOTE"
    assert result["cannot_override_preregistered_model_screen"] and result["actual_account_returns"] is False
    assert all(r["end_cash_usd"]+r["open_cost_usd"]==pytest.approx(r["realized_balance_usd"]) for r in result["scenarios"])
    assert all(r["maximum_open_risk_usd"]<=1000 for r in result["scenarios"])

