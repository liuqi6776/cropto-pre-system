"""Post-hoc hypothetical capital scenarios, never historical venue execution.

Enough depth at .515, instantaneous entry and a chosen deterministic settlement
delay are assumptions. No account, execution adapter, network or receipt API is
imported. This module cannot change the preregistered DO_NOT_PROMOTE decision.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN
import heapq
import math

import numpy as np
import pandas as pd

from .gen4_evaluation import STREAM_MINUTES, validate_predictions

CLASSIFICATION = "POST_HOC_SENSITIVITY_NOT_MODEL_SELECTION"
VARIANTS = ("CURRENT_FROZEN", "CURRENT_RESET", "TAIL_CURRENT_RESET")
FEE_RATES = (0.0, .07, .10)
DELAYS = (0, 60)
MICRO = Decimal(1_000_000)
QUANTITY_STEP = Decimal(".00000001")
PRICE = Decimal(".515")
POLICY_NUMBERS = {
    "initial_capital_usd": 10000, "single_stream_equity_fraction": .015,
    "tick_equity_fraction": .022, "max_tick_budget_usd": 1200, "max_order_usd": 200,
    "max_open_risk_usd": 1000, "max_underlying_open_risk_usd": 500,
    "max_buy_price": .515, "daily_loss_fraction": .055,
}


class ScenarioError(ValueError):
    pass


def _atoms(value):
    return int((Decimal(str(value))*MICRO).to_integral_value(rounding=ROUND_CEILING))


def _usd(value):
    return float(Decimal(value)/MICRO)


def _validate_policy(policy):
    if (policy.get("status") != "CONFIRMED"
            or policy.get("waterfall_semantics") != "ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1"
            or len(policy.get("order", [])) != 7 or set(policy.get("order", [])) != set(STREAM_MINUTES)
            or policy.get("live_enabled") is not False):
        raise ScenarioError("confirmed seven-stream paper V5 policy required")
    if any(Decimal(str(policy.get(k))) != Decimal(str(v)) for k,v in POLICY_NUMBERS.items()):
        raise ScenarioError("scenario must retain the frozen numerical risk policy")
    if policy.get("cooldown_until") is not None:
        raise ScenarioError("registered scenario assumes the current absent cooldown configuration")


def _ideal_fill(budget, rate):
    fee_per_share=rate*PRICE*(1-PRICE)
    quantity=(Decimal(budget)/MICRO/(PRICE+fee_per_share)).quantize(QUANTITY_STEP,rounding=ROUND_DOWN)
    while quantity>0:
        cost=_atoms(quantity*PRICE)
        fee=_atoms(quantity*fee_per_share)
        if cost+fee<=budget:
            return quantity,cost,fee
        # Jump to the next currency-rounding breakpoint, equivalent to the
        # account's one-share-step loop without hundreds of Decimal iterations.
        breakpoints=[Decimal(cost-1)/MICRO/PRICE]
        if fee_per_share>0:
            breakpoints.append(Decimal(fee-1)/MICRO/fee_per_share)
        quantity=max(Decimal(0),min(quantity-QUANTITY_STEP,max(breakpoints).quantize(QUANTITY_STEP,rounding=ROUND_DOWN)))
    return Decimal(0),0,0


def _prepare(predictions,variant):
    arm=predictions.loc[predictions.variant==variant]
    if arm.empty or arm.phase.nunique()!=1 or set(arm.stream)!=set(STREAM_MINUTES):
        raise ScenarioError("one complete seven-stream phase per scenario required")
    timeline=defaultdict(dict)
    for row in arm[["decision_close_utc","stream","selected","side","forward_return"]].itertuples(index=False,name=None):
        time,stream,selected,side,forward_return=row
        timeline[time][stream]=(bool(selected),side,float(forward_return))
    return sorted(timeline.items()),str(arm.phase.iloc[0])


def _simulate(timeline,phase,policy,variant,rate,delay,include_ledger=False):
    initial=10_000_000_000
    cash,realized,peak=initial,0,initial
    max_dd=0
    max_dd_fraction=0.
    positions={}
    settlement_queue=[]
    settlement_daily=defaultdict(int)
    ledger=[]
    next_id=0
    blocked=Counter()
    cohort_count=0
    total_approved=0
    maximum_risk=0
    max_coin=defaultdict(int)
    total_entries=0
    total_wins=0
    total_losses=0
    conservation_checks=0
    per_stream={s:{"entries":0,"wins":0,"losses":0,"unresolved_entries":0,
                   "approved_atoms":0,"spent_atoms":0,"cost_atoms":0,"fee_atoms":0,
                   "no_signal_skips":0,"not_due_skips":0,"reached_signals":0} for s in policy["order"]}

    def balances():
        return {"cash_usd":_usd(cash),"realized_balance_usd":_usd(initial+realized),
                "open_cost_usd":_usd(sum(p["spent"] for p in positions.values())),
                "realized_pnl_usd":_usd(realized)}

    def enforce():
        nonlocal conservation_checks,maximum_risk
        risk=sum(p["spent"] for p in positions.values())
        if cash<0 or cash+risk!=initial+realized or risk>1_000_000_000:
            raise ScenarioError("scenario cash conservation or aggregate risk failed")
        maximum_risk=max(maximum_risk,risk)
        for coin in ("BTC","ETH","SOL","BNB"):
            amount=sum(p["spent"] for p in positions.values() if p["coin"]==coin)
            if amount>500_000_000:
                raise ScenarioError("scenario underlying risk failed")
            max_coin[coin]=max(max_coin[coin],amount)
        conservation_checks+=1

    def settle_through(time):
        nonlocal cash,realized,peak,max_dd,max_dd_fraction,total_wins,total_losses
        while settlement_queue and settlement_queue[0][0]<=time:
            settled_at,identifier=heapq.heappop(settlement_queue)
            position=positions.pop(identifier)
            payout=int((position["quantity"]*MICRO).to_integral_value(rounding=ROUND_DOWN)) if position["win"] else 0
            pnl=payout-position["spent"]
            cash+=payout
            realized+=pnl
            settlement_daily[settled_at.date()]+=pnl
            stream=position["stream"]
            if position["win"]:
                total_wins+=1
                per_stream[stream]["wins"]+=1
            else:
                total_losses+=1
                per_stream[stream]["losses"]+=1
            if include_ledger:
                position["record"].update(proxy_settled_at=settled_at.isoformat(),proxy_payout_usd=_usd(payout),
                                          realized_pnl_usd=_usd(pnl),state="PROXY_WIN" if position["win"] else "PROXY_LOSS")
            peak=max(peak,initial+realized)
            drawdown=peak-initial-realized
            max_dd=max(max_dd,drawdown)
            max_dd_fraction=max(max_dd_fraction,drawdown/peak if peak else 0.)
            enforce()

    for boundary,decisions in timeline:
        settle_through(boundary)
        cohort_count+=1
        equity=initial+realized
        today_pnl=settlement_daily.get(boundary.date(),0)
        day_start_equity=equity-today_pnl
        streak=0
        day=boundary.date()-timedelta(days=1)
        while settlement_daily.get(day,0)<0:
            streak+=1
            day-=timedelta(days=1)
        reason=None
        if day_start_equity<=0 or today_pnl<=-day_start_equity*.055:
            reason="DAILY_LOSS_LIMIT"
        if streak>=5:
            reason="COOLDOWN_STATE_REQUIRED"
        risk=sum(p["spent"] for p in positions.values())
        budget=max(0,min(int(Decimal(equity)*Decimal(".022")),1_200_000_000,cash,1_000_000_000-risk))
        single=max(0,min(200_000_000,int(Decimal(equity)*Decimal(".015")*(Decimal(".5") if streak>=4 else 1))))
        if budget==0:
            reason=reason or "NO_RISK_BUDGET"
        if reason:
            blocked[reason]+=1
            continue
        used=0
        for rank,stream in enumerate(policy["order"],1):
            summary=per_stream[stream]
            due=boundary.value%(STREAM_MINUTES[stream]*60*1_000_000_000)==0
            if not due:
                summary["not_due_skips"]+=1
                continue
            if stream not in decisions:
                blocked["MISSING_DUE_DECISION"]+=1
                break
            selected,side,forward_return=decisions[stream]
            if not selected:
                summary["no_signal_skips"]+=1
                continue
            summary["reached_signals"]+=1
            if budget-used<=0:
                blocked["COHORT_BUDGET_EXHAUSTED"]+=1
                break
            coin=stream.split("_",1)[0]
            risk=sum(p["spent"] for p in positions.values())
            coin_risk=sum(p["spent"] for p in positions.values() if p["coin"]==coin)
            approved=min(single,budget-used,cash,200_000_000,1_000_000_000-risk,500_000_000-coin_risk)
            if approved<=0:
                blocked["CURRENT_STREAM_CASH_OR_RISK_EXHAUSTED"]+=1
                break
            quantity,cost,fee=_ideal_fill(approved,rate)
            spent=cost+fee
            total_approved+=approved
            summary["approved_atoms"]+=approved
            if quantity<=0:
                blocked["ZERO_QUANTITY_PARTIAL"]+=1
                break
            if spent>approved:
                raise ScenarioError("scenario spent more than its approved target")
            cash-=spent
            used+=spent
            next_id+=1
            total_entries+=1
            summary["entries"]+=1
            summary["spent_atoms"]+=spent
            summary["cost_atoms"]+=cost
            summary["fee_atoms"]+=fee
            maturity=boundary+pd.Timedelta(minutes=STREAM_MINUTES[stream]+delay)
            finite=math.isfinite(forward_return)
            win=(side=="UP")==(forward_return>=0) if finite else None
            record={"scenario_entry_number":next_id,"stream":stream,"rank":rank,"decision_utc":boundary.isoformat(),
                    "side":side,"assumed_price":float(PRICE),"quantity":str(quantity),"approved_usd":_usd(approved),
                    "spent_usd":_usd(spent),"cost_usd":_usd(cost),"fee_usd":_usd(fee),
                    "scheduled_proxy_settlement_utc":maturity.isoformat() if finite else None,
                    "state":"PENDING_PROXY_SETTLEMENT" if finite else "UNRESOLVED_LABEL",
                    "realized_equity_basis_usd":_usd(equity),"single_stream_limit_usd":_usd(single),
                    "cohort_budget_usd":_usd(budget),"preceding_losing_calendar_days":streak}
            positions[next_id]={"stream":stream,"coin":coin,"spent":spent,"quantity":quantity,"win":win,"record":record}
            if finite:
                heapq.heappush(settlement_queue,(maturity,next_id))
            if include_ledger:
                ledger.append(record)
            enforce()
            if spent+2<approved:
                blocked["APPROVED_TARGET_PARTIAL"]+=1
                break
        if used>budget:
            raise ScenarioError("scenario cohort budget exceeded")
    last_decision=timeline[-1][0]
    end_of_decisions=balances()
    drain_until=max((entry[0] for entry in settlement_queue),default=last_decision)
    settle_through(drain_until)
    enforce()
    for position in positions.values():
        per_stream[position["stream"]]["unresolved_entries"]+=1
    for values in per_stream.values():
        for key in ("approved","spent","cost","fee"):
            values[key+"_usd"]=_usd(values.pop(key+"_atoms"))
    output={"classification":CLASSIFICATION,"phase":phase,"variant":variant,"fee_rate":float(rate),
            "settlement_delay_minutes":delay,"initial_capital_usd":10000.,
            "first_decision_utc":timeline[0][0].isoformat(),"last_decision_utc":last_decision.isoformat(),
            "settlement_drain_until":drain_until.isoformat(),"end_of_decision_window":end_of_decisions,
            "end_cash_usd":_usd(cash),"realized_balance_usd":_usd(initial+realized),
            "open_cost_usd":_usd(sum(p["spent"] for p in positions.values())),"realized_pnl_usd":_usd(realized),
            "max_realized_balance_drawdown_usd":_usd(max_dd),"max_realized_balance_drawdown_fraction":max_dd_fraction,
            "entries":total_entries,"proxy_settled_entries":total_wins+total_losses,"proxy_wins":total_wins,
            "proxy_losses":total_losses,"unresolved_entries":len(positions),"cohorts":cohort_count,
            "approved_budget_total_usd":_usd(total_approved),"per_stream":per_stream,
            "blocked_cohorts_or_streams":dict(blocked),"maximum_open_risk_usd":_usd(maximum_risk),
            "maximum_underlying_open_risk_usd":{coin:_usd(value) for coin,value in max_coin.items()},
            "conservation_checks":conservation_checks,"cash_plus_open_cost_equals_initial_plus_realized":True,
            "policy_order":list(policy["order"]),"production_published":False,"live_enabled":False,
            "cannot_override_preregistered_model_screen":True,"preregistered_model_screen":"DO_NOT_PROMOTE",
            "assumptions":{"price":float(PRICE),"depth":"SUFFICIENT_AT_PRICE_ASSUMED_NOT_OBSERVED",
                "entry_timing":"INSTANTANEOUS_AT_DECISION_BOUNDARY_NO_REAL_20_SECOND_DEADLINE",
                "settlement":"BINANCE_DIRECTION_PROXY_PLUS_ASSUMED_DELAY_NOT_OFFICIAL_EVENTS",
                "simultaneous_events":"DUE_SETTLEMENTS_BEFORE_NEW_DECISION_COHORT",
                "simultaneous_settlements":"ORDER_BY_SCENARIO_ENTRY_NUMBER; DRAWDOWN_IS_EVENT_LEVEL_IN_THAT_ORDER_NOT_DAILY_OR_MARK_TO_MARKET",
                "final_cash":"STOP_NEW_ENTRIES_AT_LAST_DECISION_THEN_DRAIN_FINITE_LABEL_SETTLEMENTS",
                "unknown_labels":"ENTER_IF_SELECTED_RETAIN_OPEN_COST_FOREVER_NO_INVENTED_LOSS",
                "missing":"REAL_BOOKS_PRICE_FAILURES_SLIPPAGE_CONTRACT_AVAILABILITY_EXECUTION_LATENCY",
                "valuation":"REALIZED_BALANCE_AND_PAID_OPEN_COST_NOT_MARK_TO_MARKET"}}
    if include_ledger:
        output["scenario_entries"]=ledger
    return output


def simulate_portfolio(predictions,policy,variant,fee_rate=.07,settlement_delay_minutes=0,*,include_ledger=False):
    _validate_policy(policy)
    if variant not in VARIANTS or fee_rate not in FEE_RATES or settlement_delay_minutes not in DELAYS:
        raise ScenarioError("only the fixed post-hoc arms, fees and delays are allowed")
    relevant=predictions.loc[predictions.variant.isin(VARIANTS)]
    valid=validate_predictions(relevant)
    timeline,phase=_prepare(valid,variant)
    return _simulate(timeline,phase,policy,variant,Decimal(str(fee_rate)),settlement_delay_minutes,include_ledger)


def run_portfolio_scenarios(predictions_by_window,policy,*,progress=None):
    _validate_policy(policy)
    if set(predictions_by_window)!={"historical","recent"}:
        raise ScenarioError("both fixed study windows required")
    results=[]
    for window in ("historical","recent"):
        raw=predictions_by_window[window]
        valid=validate_predictions(raw.loc[raw.variant.isin(VARIANTS)])
        if set(valid.variant)!=set(VARIANTS):
            raise ScenarioError("all three fixed variants required")
        for variant in VARIANTS:
            timeline,phase=_prepare(valid,variant)
            for rate in FEE_RATES:
                for delay in DELAYS:
                    if progress:
                        progress({"window":window,"variant":variant,"fee_rate":rate,"settlement_delay_minutes":delay})
                    result=_simulate(timeline,phase,policy,variant,Decimal(str(rate)),delay)
                    results.append({"window":window,**result})
    return {"classification":CLASSIFICATION,"preregistered_model_screen":"DO_NOT_PROMOTE",
            "cannot_override_preregistered_model_screen":True,"actual_account_returns":False,
            "live_enabled":False,"production_published":False,"scenario_count":len(results),"scenarios":results}

