"""Classify already-open paid cost for display only; never release risk or cash.

An expired contract remains in Account.overview().positions until an observed
official resolution is applied by the account. This module only partitions
that same locked cost. It has no ledger, exchange, settlement or policy access.
"""
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import math
import re
import time

from .clock_sync import SOURCE, project_local

ACTIVE = "ACTIVE_WINDOW"
PENDING = "AWAITING_OFFICIAL_RESOLUTION"
UNKNOWN = "UNKNOWN"
_FIELDS = {ACTIVE: "active_window_cost_usd", PENDING: "awaiting_resolution_cost_usd",
           UNKNOWN: "unknown_cost_usd"}
_MICRO = Decimal(1_000_000)


class RiskViewError(ValueError):
    """The supplied overview cannot be reconciled without inventing amounts."""


def _utc(value):
    stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("aware timestamp required")
    return stamp.astimezone(timezone.utc)


def _atoms(value):
    if isinstance(value, bool):
        raise RiskViewError("invalid locked cost amount")
    try:
        amount = Decimal(str(value))
        scaled = amount * _MICRO
        if (not amount.is_finite() or amount < 0 or not scaled.is_finite()
                or scaled != scaled.to_integral_value() or scaled > 9_007_199_254_740_991):
            raise RiskViewError("invalid locked cost amount")
        return int(scaled)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise RiskViewError("invalid locked cost amount") from exc


def _usd(value):
    return float(Decimal(value) / _MICRO)


def _bounds(clock, now):
    """Validate one local instant against the existing calibration contract.

    ``now`` is an optional *local* aware instant for deterministic callers. It
    never substitutes for calibration and must agree with its monotonic age.
    """
    if not isinstance(clock, Mapping) or clock.get("status") != "CALIBRATED":
        return None, "CLOCK_UNAVAILABLE"
    try:
        offset, uncertainty, calibrated = (clock[k] for k in
            ("server_minus_local_seconds", "uncertainty_seconds", "calibrated_monotonic"))
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (offset, uncertainty, calibrated)):
            raise ValueError("invalid numeric calibration")
        if (clock.get("source") != SOURCE or not 0 <= uncertainty <= .5
                or not re.fullmatch(r"[0-9a-f]{64}", str(clock.get("receipt_sha256", "")))):
            raise ValueError("unverified calibration")
        age = time.monotonic() - calibrated
        if not 0 <= age <= 60:
            return None, "CLOCK_EXPIRED_OR_MONOTONIC_INVALID"
        local = _utc(now) if now is not None else datetime.now(timezone.utc)
        expected = _utc(clock["calibrated_at_local"]) + timedelta(seconds=age)
        if abs((local - expected).total_seconds()) > .05:
            return None, "LOCAL_CLOCK_CHANGED"
        lower, upper = project_local(local, clock, upper=False), project_local(local, clock, upper=True)
        if lower > upper:
            raise ValueError("inverted clock bounds")
        return (lower, upper), None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "CLOCK_INVALID"


def split_open_risk(overview, clock=None, now=None):
    """Return a non-mutating display partition of the account's open paid cost.

    No clock means UNKNOWN, even if ``now`` or overview.as_of is present. An end
    inside the calibrated interval is also UNKNOWN; it is not rounded into an
    expired or still-active bucket. Invalid monetary input raises instead of
    reporting a misleading sum. Invalid window timestamps remain UNKNOWN.
    """
    if not isinstance(overview, Mapping) or not isinstance(overview.get("positions"), list):
        raise RiskViewError("account overview with open positions required")
    if "locked_cost" not in overview:
        raise RiskViewError("account locked cost required")
    locked = _atoms(overview["locked_cost"])
    bounds, clock_reason = _bounds(clock, now)
    totals = {ACTIVE: 0, PENDING: 0, UNKNOWN: 0}
    counts = {ACTIVE: 0, PENDING: 0, UNKNOWN: 0}
    by_coin, positions, seen = {}, [], set()
    for p in overview["positions"]:
        if not isinstance(p, Mapping):
            raise RiskViewError("invalid open position")
        order, coin = p.get("order_id"), p.get("underlying")
        if not isinstance(order, str) or not order or order in seen:
            raise RiskViewError("open position identity missing or duplicated")
        if not isinstance(coin, str) or not re.fullmatch(r"[A-Z0-9]{2,12}", coin):
            raise RiskViewError("open position underlying invalid")
        if "cost_usd" not in p or "fee_usd" not in p:
            raise RiskViewError("open position paid cost and fee required")
        seen.add(order)
        cost, fees = _atoms(p["cost_usd"]), _atoms(p["fee_usd"])
        paid = cost + fees
        state, reason = UNKNOWN, clock_reason
        end_text = None
        try:
            end = _utc(p["window_end"])
            end_text = end.isoformat()
        except (KeyError, TypeError, ValueError, OverflowError):
            reason = "WINDOW_END_INVALID"
        else:
            if bounds is not None:
                lower, upper = bounds
                if end <= lower:
                    state, reason = PENDING, "WINDOW_ENDED_OFFICIAL_RESOLUTION_NOT_APPLIED"
                elif end > upper:
                    state, reason = ACTIVE, "WINDOW_HAS_NOT_ENDED"
                else:
                    reason = "WINDOW_END_OVERLAPS_CLOCK_UNCERTAINTY"
        totals[state] += paid
        counts[state] += 1
        bucket = by_coin.setdefault(coin, {ACTIVE: 0, PENDING: 0, UNKNOWN: 0, "count": 0})
        bucket[state] += paid
        bucket["count"] += 1
        positions.append({"order_id": order, "underlying": coin, "window_end": end_text,
                          "cost_usd": _usd(cost), "fee_usd": _usd(fees), "locked_cost_usd": _usd(paid),
                          "risk_state": state, "reason": reason, "risk_released_usd": 0.0,
                          "reusable_cash_usd": 0.0})
    if sum(totals.values()) != locked:
        raise RiskViewError("open positions do not reconcile to account locked cost")
    return {"schema_version": 1, "classification": "DISPLAY_ONLY_NOT_ALLOCATION",
            "status": "CLASSIFIED" if counts[UNKNOWN] == 0 else "PARTIAL_UNKNOWN" if counts[ACTIVE]+counts[PENDING] else "UNKNOWN",
            "clock_status": "CALIBRATED" if bounds is not None else "UNAVAILABLE", "clock_reason": clock_reason,
            "time_bounds": {"lower": bounds[0].isoformat(), "upper": bounds[1].isoformat(),
                            "source": clock["source"], "receipt_sha256": clock["receipt_sha256"]} if bounds else None,
            "total_locked_cost_usd": _usd(locked), **{_FIELDS[k]: _usd(v) for k, v in totals.items()},
            "counts": counts, "positions": positions,
            "by_underlying": {coin: {"total_locked_cost_usd": _usd(sum(v[k] for k in _FIELDS)),
                                      **{_FIELDS[k]: _usd(v[k]) for k in _FIELDS}, "position_count": v["count"]}
                              for coin, v in sorted(by_coin.items())},
            "reconciled": True, "locked_cost_atoms": locked, "risk_released_usd": 0.0,
            "reusable_cash_from_unsettled_usd": 0.0, "awaiting_resolution_releases_risk": False,
            "limitations": ["所有未结算成本（含手续费）继续占用总风险与单币风险额度。",
                            "窗口到期不等于已收到官方结算，待结算资金不可再次使用。",
                            "分类仅用于展示，不修改账本、资金上限或下单预算。"]}
