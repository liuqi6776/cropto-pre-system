"""Pure, fail-closed previews for versioned ranked allocation policies.

Capacity is the all-in USD amount visible in a fresh observed venue ask book at
the configured acceptable price. It is the IOC simulation's depth constraint,
not promised real liquidity or an estimate of passive-maker capacity. The
separate 1.5%-of-equity stream budget is the account's risk constraint.
The $200 ceiling is explicitly per stream per decision cohort, not an invitation
to split several $200 orders. This pure function is a preview; recurring execution
must use execution.PaperExecutor and Account's persisted decision-cohort budget.
One authorization per stream is issued per plan. Legacy policies retain the
observed-depth gate for historical interpretation. The explicitly selected v5
semantics skip known absent/not-due signals and plan only the risk-approved
amount before considering the next eligible stream. Planning a lower allocation
does not certify an upper fill: the account must verify actual spending first.
No fills, model validation, rankings, or historical performance are invented here.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import hashlib
import json
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

from .account import AccountRejected, _policy_risk_limits
from .stage import DISABLED, StageRejected, stage_profile


STREAMS = {"BTC_15m", "ETH_15m", "BTC_5m", "ETH_5m", "SOL_15m", "SOL_5m", "BNB_15m"}
ALL_STREAMS = {*STREAMS, "XRP_15m"}
ATOM = Decimal("0.000001")
ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1 = "ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1"
PRICE_LIMIT = Decimal(".515")


def _number(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not an amount")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("finite amount required")
    return result


def _time(value):
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timezone-aware time required")
    return result.astimezone(timezone.utc)


def _money(value):
    return value.quantize(ATOM, rounding=ROUND_DOWN)


def _signal_times(sig, name, stamp):
    decision = _time(sig.get("decision_at"))
    expires = _time(sig.get("valid_until"))
    if (sig.get("stream") != name or not isinstance(sig.get("signal_id"), str) or not sig["signal_id"]
            or not isinstance(sig.get("market_id"), str) or not sig["market_id"]
            or not decision <= stamp < expires):
        raise ValueError("invalid signal")
    return decision, expires


def _price_limit_evidence(candidate, sig, stamp):
    """Validate a supplied observation, without claiming independent HTTP proof.

    The evidence digest hashes the complete supplied book, not Account's
    separately normalized/aggregated book identity. Execution revalidates the
    official contract, fee schedule, clock and actual book independently.
    """
    evidence = candidate.get("price_evidence")
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get("book"), Mapping):
        raise ValueError("price evidence required")
    book = evidence["book"]
    encoded = json.dumps(book, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    if evidence.get("sha256") != digest:
        raise ValueError("price evidence digest mismatch")
    source = book.get("source")
    parsed_source = urlsplit(source) if isinstance(source, str) else None
    if (book.get("kind") != "OBSERVED_PUBLIC_ORDERBOOK"
            or parsed_source is None or parsed_source.scheme != "https" or not parsed_source.hostname
            or parsed_source.username is not None or parsed_source.password is not None or parsed_source.fragment
            or book.get("market_id") != sig["market_id"]
            or sig.get("outcome") not in ("UP", "DOWN") or book.get("outcome") != sig["outcome"]):
        raise ValueError("price evidence identity invalid")
    observed, received = _time(book.get("observed_at")), _time(book.get("received_at"))
    if not _time(sig["decision_at"]) <= observed <= received <= stamp or not 0 <= (stamp-observed).total_seconds() <= 2:
        raise ValueError("price evidence timing invalid")
    asks = book.get("asks")
    if not isinstance(asks, list) or not asks:
        raise ValueError("nonempty ask observation required")
    positive_prices = []
    for ask in asks:
        price, size = _number(ask.get("price")), _number(ask.get("size"))
        if not 0 < price < 1 or size < 0:
            raise ValueError("invalid ask level")
        if size > 0:
            positive_prices.append(price)
    if not positive_prices or min(positive_prices) <= PRICE_LIMIT:
        raise ValueError("observed positive ask prices do not establish price failure")
    return digest, min(positive_prices)


def plan_waterfall(policy, candidates, equity, cash, open_risk, now, day_pnl=0,
                   day_start_equity=None, losing_days=0):
    """Return a preview and narrowly scoped, local paper-order authorizations.

    ``policy`` requires status=CONFIRMED, complete unique ``order`` (7 stream
    names), nonempty ranking_source and version. With >=5 losing days the caller
    must supply an explicit ``cooldown_until``; expiry releases the time lock
    without starting a fresh one. The >=4-day half-budget remains until the caller
    resets the observed losing-days count. Negative day PnL requires an observed
    positive day_start_equity; the 5.5% daily loss lock uses that fixed base.

    Each actionable candidate requires stream, capacity_usd, capacity_source,
    capacity_observed_at and signal. Capacity timestamps may be at most 2 seconds
    old. Legacy policies require this capacity even for explicit signal=None.
    With waterfall_semantics=ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1, signal=None
    skips without capacity. status=NOT_DUE may omit signal only for a 15m stream
    at a non-15m five-minute boundary. Unknown signal state still blocks.

    V5 status=PRICE_LIMIT_NOT_MET requires a valid signal plus price_evidence:
    {book: {kind: OBSERVED_PUBLIC_ORDERBOOK, source, market_id, outcome,
    observed_at, received_at, asks: [{price, size}, ...]}, sha256: <canonical
    complete-book digest>}. Canonical JSON uses UTF-8, sorted keys, compact
    separators, ensure_ascii=False and allow_nan=False. Every positive-size ask
    must exceed .515. Empty/zero/stale/unverified capacity alone never skips.
    This supplied evidence is checked for consistency; it is not independently
    retrieved or authenticated by the pure preview.

    Output authorization is compatible with account.Account.place_paper_buy.
    The call is pure: it does not debit cash or certify that any fill occurred.
    """
    result = {
        "status": "BLOCKED", "allocations": [], "blocked": [], "reasons": [],
        "preview": [], "unspent": 0.0, "initial_tick_budget_usd": 0.0,
        "policy_version": policy.get("version") if isinstance(policy, Mapping) else None,
        "ranking_source": policy.get("ranking_source") if isinstance(policy, Mapping) else None,
        "live_enabled": False,
        "limits": {"single_stream_equity_fraction": .015, "tick_equity_fraction": .022,
                   "tick_hard_cap_usd": 1200, "order_hard_cap_usd": 200,
                   "stream_cohort_hard_cap_usd": 200,
                   "open_risk_hard_cap_usd": 1000, "daily_loss_fraction": .055,
                   "capacity_max_age_seconds": 2,
                   "max_buy_price": float(PRICE_LIMIT),
                   "open_risk_limit_scope": "EXPLICIT_LOCAL_SAFETY_ASSUMPTION"},
    }

    def block(code, reason, stream=None, rank=None):
        result["blocked"].append({"code": code, "reason": reason, "stream": stream, "rank": rank})
        if code not in result["reasons"]:
            result["reasons"].append(code)

    if not isinstance(policy, Mapping) or policy.get("status") != "CONFIRMED":
        result["status"] = "WAITING_FOR_POLICY"
        block("POLICY_NOT_CONFIRMED", "Ranking policy has not been explicitly confirmed")
        return result
    order = policy.get("order")
    valid_streams = STREAMS if isinstance(order, (tuple, list)) and len(order) == 7 else (
        ALL_STREAMS if isinstance(order, (tuple, list)) and len(order) == 8 else None)
    if (valid_streams is None
            or any(not isinstance(s, str) for s in order) or set(order) != valid_streams
            or not isinstance(policy.get("ranking_source"), str) or not policy["ranking_source"].strip()
            or not isinstance(policy.get("version"), str) or not policy["version"].strip()):
        block("INVALID_RANKING_POLICY", "A versioned complete ranking and its source are required")
        return result
    semantics = policy.get("waterfall_semantics")
    if semantics not in (None, ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1):
        block("UNSUPPORTED_WATERFALL_SEMANTICS", "Unknown allocation semantics cannot be inferred")
        return result
    eligible_budget = semantics == ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1
    if eligible_budget:
        result["waterfall_semantics"] = semantics
        result["execution_confirmation_required"] = True
    try:
        stamp = _time(now)
        eq, available_cash, risk, day_result = map(_number, (equity, cash, open_risk, day_pnl))
        if eq <= 0 or available_cash < 0 or risk < 0 or available_cash > eq:
            raise ValueError("invalid account amounts")
        if isinstance(losing_days, bool) or not isinstance(losing_days, int) or losing_days < 0:
            raise ValueError("invalid losing-day count")
    except (ValueError, TypeError, InvalidOperation):
        block("INVALID_ACCOUNT_STATE", "Observed finite equity, cash, open risk and losing-day count are required")
        return result
    result["planned_at"] = stamp.isoformat()
    boundary = datetime.fromtimestamp(int(stamp.timestamp()) // 300 * 300, timezone.utc)
    try:
        stage = stage_profile(policy, boundary)
    except StageRejected:
        block("INVALID_EXECUTION_STAGE", "Execution stage must be confirmed, active and a unique ranked subset")
        return result
    if stage["configured"]:
        result["execution_stage"] = stage
    try:
        limits = _policy_risk_limits(policy)
    except AccountRejected:
        block("INVALID_RISK_LIMITS", "Policy risk limits are invalid or out of bounds")
        return result

    def _num_out(d):
        return int(d) if d == int(d) else float(d)

    result["limits"].update(
        single_stream_equity_fraction=float(limits["single_stream_equity_fraction"]),
        tick_equity_fraction=float(limits["tick_equity_fraction"]),
        tick_hard_cap_usd=_num_out(limits["max_tick_budget_usd"]),
        order_hard_cap_usd=_num_out(limits["max_order_usd"]),
        stream_cohort_hard_cap_usd=_num_out(limits["max_order_usd"]),
        open_risk_hard_cap_usd=_num_out(limits["max_open_risk_usd"]),
        underlying_open_risk_hard_cap_usd=_num_out(limits["max_underlying_open_risk_usd"]),
        daily_loss_fraction=float(limits["daily_loss_fraction"]),
    )
    tick_budget = _money(min(eq * limits["tick_equity_fraction"], limits["max_tick_budget_usd"], available_cash,
                             max(Decimal(0), limits["max_open_risk_usd"] - risk)))
    result["initial_tick_budget_usd"] = result["unspent"] = float(tick_budget)
    single_budget = _money(eq * limits["single_stream_equity_fraction"] * (Decimal(".5") if losing_days >= 4 else Decimal(1)))
    effective_single = min(single_budget, limits["max_order_usd"])
    result["single_stream_budget_usd"] = float(single_budget)
    result["effective_single_stream_budget_usd"] = float(effective_single)
    if day_result < 0:
        try:
            day_base = _number(day_start_equity)
            if day_base <= 0:
                raise ValueError("invalid day-start equity")
        except (ValueError, TypeError, InvalidOperation):
            block("DAY_START_EQUITY_REQUIRED", "Daily loss guard requires the observed day-start equity")
            return result
        max_daily_loss = day_base * limits["daily_loss_fraction"]
        if -day_result >= max_daily_loss:
            block("DAILY_LOSS_LIMIT", "Observed daily loss reached 5.5% of day-start equity")
            return result
        if policy.get("cap_budget_by_daily_loss_room", True):
            realized_room = _money(max(Decimal(0), max_daily_loss + day_result))
            remaining_room = _money(max(Decimal(0), max_daily_loss + day_result - risk))
            if realized_room <= 0:
                block("DAILY_LOSS_LIMIT", "Observed daily loss reached 5.5% of day-start equity")
                return result
            tick_budget = min(tick_budget, remaining_room)
            effective_single = min(effective_single, remaining_room)
            result["initial_tick_budget_usd"] = result["unspent"] = float(tick_budget)
            result["effective_single_stream_budget_usd"] = float(effective_single)
    if policy.get("cooldown_until") is not None or losing_days >= 5:
        try:
            cooldown_until = _time(policy.get("cooldown_until"))
        except (ValueError, TypeError):
            block("COOLDOWN_STATE_REQUIRED", "Five losing days require an explicit cooldown deadline")
            return result
        result["cooldown_until"] = cooldown_until.isoformat()
        if stamp < cooldown_until:
            block("COOLDOWN_ACTIVE", "The explicitly recorded cooldown has not ended")
            return result
    if tick_budget <= 0:
        block("NO_RISK_BUDGET", "Cash or the total open-risk limit leaves no allocation budget")
        return result
    by_stream = {}
    try:
        if isinstance(candidates, Mapping):
            entries = [dict(value, stream=key) for key, value in candidates.items()]
        elif isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            entries = list(candidates)
        else:
            raise ValueError("invalid candidates")
        for candidate in entries:
            name = candidate.get("stream")
            if name not in valid_streams or name in by_stream:
                raise ValueError("duplicate or unknown candidate")
            by_stream[name] = candidate
    except (ValueError, TypeError, AttributeError):
        block("INVALID_CANDIDATES", "Candidates must be unique known stream objects")
        return result
    identity = {"version": policy["version"], "order": order,
                "now": stamp.isoformat(), "equity": str(eq), "cash": str(available_cash),
                "open_risk": str(risk)}
    if eligible_budget:
        identity["waterfall_semantics"] = semantics
    if stage["configured"]:
        identity["execution_stage"] = stage
    seed = json.dumps(identity, sort_keys=True).encode()
    plan_id = "waterfall:" + hashlib.sha256(seed).hexdigest()
    result["plan_id"] = plan_id
    remaining = tick_budget
    locked_by = None
    preceding_allocations = []
    for rank, name in enumerate(order, 1):
        preview = {"stream": name, "rank": rank, "status": "LOCKED", "capacity_usd": None,
                   "allocated_usd": 0.0, "unlocks_next": False}
        result["preview"].append(preview)
        if name in stage["disabled_streams"]:
            preview.update(status=DISABLED, stage_id=stage["id"], unlocks_next=True,
                           book_checked=False, model_evaluated=False)
            continue
        if locked_by:
            preview["reason"] = "UPPER_RANK_NOT_FILLED"
            preview["locked_by"] = locked_by
            continue
        if remaining <= 0:
            preview["reason"] = "TICK_BUDGET_EXHAUSTED"
            locked_by = name
            block("TICK_BUDGET_EXHAUSTED", "The shared per-tick budget is fully allocated", name, rank)
            continue
        candidate = by_stream.get(name)
        if not isinstance(candidate, Mapping):
            code = "SIGNAL_STATE_MISSING" if eligible_budget else "CAPACITY_MISSING"
            preview["reason"] = code
            locked_by = name
            block(code, "Missing upper-rank candidate cannot be treated as eligible or skipped", name, rank)
            continue
        if eligible_budget:
            state = candidate.get("status")
            sig = candidate.get("signal")
            if state not in (None, "SIGNAL", "NO_SIGNAL", "NOT_DUE", "PRICE_LIMIT_NOT_MET"):
                preview.update(status="SIGNAL_UNKNOWN", reason="SIGNAL_STATE_INVALID")
                locked_by = name
                block("SIGNAL_STATE_INVALID", "Unknown candidate state cannot unlock a lower rank", name, rank)
                continue
            if state == "NOT_DUE":
                boundary = int(stamp.timestamp()) // 300 * 300
                if not name.endswith("_15m") or boundary % 900 == 0 or sig is not None:
                    preview.update(status="SIGNAL_INVALID", reason="NOT_DUE_INVALID")
                    locked_by = name
                    block("NOT_DUE_INVALID", "Not-due state conflicts with the UTC boundary or signal", name, rank)
                else:
                    preview.update(status="NOT_DUE", unlocks_next=True)
                continue
            if "signal" not in candidate:
                preview.update(status="SIGNAL_UNKNOWN", reason="SIGNAL_STATE_MISSING")
                locked_by = name
                block("SIGNAL_STATE_MISSING", "Unknown upper-rank signal state cannot be treated as no signal", name, rank)
                continue
            if sig is None and state in (None, "NO_SIGNAL"):
                preview.update(status="NO_SIGNAL", unlocks_next=True)
                continue
            try:
                if state == "NO_SIGNAL":
                    raise ValueError("signal contradicts absent state")
                _signal_times(sig, name, stamp)
            except (ValueError, TypeError, AttributeError):
                preview.update(status="SIGNAL_INVALID", reason="SIGNAL_INVALID")
                locked_by = name
                block("SIGNAL_INVALID", "Upper-rank signal identity or timing is invalid", name, rank)
                continue
            if state == "PRICE_LIMIT_NOT_MET":
                try:
                    evidence_hash, best_ask = _price_limit_evidence(candidate, sig, stamp)
                except (ValueError, TypeError, AttributeError, InvalidOperation, OverflowError):
                    preview.update(status="PRICE_UNVERIFIED", reason="PRICE_EVIDENCE_INVALID")
                    locked_by = name
                    block("PRICE_EVIDENCE_INVALID", "A fresh, consistent nonempty ask observation must prove the price failure", name, rank)
                else:
                    preview.update(status="PRICE_LIMIT_NOT_MET", unlocks_next=True,
                                   best_ask=float(best_ask), price_evidence_sha256=evidence_hash)
                continue
        try:
            capacity = _number(candidate.get("capacity_usd"))
            observed = _time(candidate.get("capacity_observed_at"))
            age = (stamp - observed).total_seconds()
            source = candidate.get("capacity_source")
            if capacity <= 0:
                raise ValueError("zero capacity is not filled capacity")
            if not isinstance(source, str) or not source.strip() or not 0 <= age <= 2:
                raise ValueError("invalid capacity provenance")
        except (ValueError, TypeError, InvalidOperation):
            preview["reason"] = "CAPACITY_UNVERIFIED"
            locked_by = name
            block("CAPACITY_UNVERIFIED", "A positive, sourced, fresh observed capacity is required", name, rank)
            continue
        preview.update(capacity_usd=float(capacity), capacity_source=source,
                       capacity_observed_at=observed.isoformat(), status="UNLOCKED")
        stream_max_order = (limits.get("stream_max_order_usd") or {}).get(name, limits["max_order_usd"])
        effective_stream_single = min(effective_single, stream_max_order)
        can_cover_capacity = effective_stream_single >= capacity
        if "signal" not in candidate:
            preview.update(status="SIGNAL_UNKNOWN", reason="SIGNAL_STATE_MISSING")
            locked_by = name
            block("SIGNAL_STATE_MISSING", "Unknown upper-rank signal state cannot be treated as no signal", name, rank)
            continue
        sig = candidate.get("signal")
        if sig is None:
            preview["status"] = "NO_SIGNAL"
            preview["unlocks_next"] = can_cover_capacity
            if not can_cover_capacity:
                preview["reason"] = "UPPER_CAPACITY_NOT_COVERED"
                locked_by = name
                block("UPPER_CAPACITY_NOT_COVERED", "Upper-rank capacity exceeds its effective per-stream budget", name, rank)
            continue
        try:
            decision, expires = _signal_times(sig, name, stamp)
        except (ValueError, TypeError, AttributeError):
            preview.update(status="SIGNAL_INVALID", reason="SIGNAL_INVALID")
            locked_by = name
            block("SIGNAL_INVALID", "Upper-rank signal identity or timing is invalid", name, rank)
            continue
        approved = _money(min(capacity, effective_stream_single, remaining))
        if approved <= 0:
            preview.update(status="NO_ALLOCATION", reason="BELOW_CURRENCY_PRECISION")
            locked_by = name
            block("BELOW_CURRENCY_PRECISION", "Capacity is below supported currency precision", name, rank)
            continue
        validity = min(expires, stamp + timedelta(seconds=2))
        authorization = {"policy": "RANKED_CAPACITY_WATERFALL", "plan_id": plan_id, "rank": rank,
                         "signal_id": sig["signal_id"], "market_id": sig["market_id"],
                         "approved_stake_usd": float(approved), "planned_at": stamp.isoformat(),
                         "valid_until": validity.isoformat(), "ranking_source": policy["ranking_source"],
                         "policy_version": policy["version"],
                         "preceding_allocations": [dict(p) for p in preceding_allocations]}
        if eligible_budget:
            authorization["waterfall_semantics"] = semantics
        if stage["configured"]:
            authorization["execution_stage"] = stage
        result["allocations"].append({"stream": name, "rank": rank, "signal_id": sig["signal_id"],
                                      "market_id": sig["market_id"], "stake_usd": float(approved),
                                      "capacity_usd": float(capacity), "allocation_authorization": authorization})
        remaining -= approved
        preceding_allocations.append({"signal_id": sig["signal_id"], "rank": rank,
                                      "required_cost_usd": float(approved)})
        planned_requirement_met = eligible_budget or approved >= capacity
        preview.update(status="PLANNED_ALLOCATION" if eligible_budget else "ALLOCATED",
                       allocated_usd=float(approved), unlocks_next=planned_requirement_met)
        if eligible_budget:
            preview["requires_actual_approved_fill"] = True
        if not planned_requirement_met:
            preview["reason"] = "UPPER_CAPACITY_NOT_COVERED"
            locked_by = name
            block("UPPER_CAPACITY_NOT_COVERED", "Upper-rank capacity remains unallocated; lower ranks stay locked", name, rank)
    result["unspent"] = float(remaining)
    result["status"] = "PLANNED" if result["allocations"] else ("BLOCKED" if result["blocked"] else "NO_ELIGIBLE_SIGNAL")
    return result
