"""Exact Decimal order math, tick-size alignment, capital scaling, and calibrated UTC deadline checks.

Implements Section 4.2 & Section 5 of the ARGON4 External Live Executor Specification:
  - live_stake_usd = live_scale_factor * approved_stake_usd
  - limit_price: 0.51 (when tick_size == "0.01") or 0.515 (when tick_size == "0.001")
  - fee_per_share = rate * P_limit * (1 - P_limit)
  - size_shares = floor_0.01(live_stake_usd / (P_limit + fee_per_share))
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Mapping

SHARE_STEP = Decimal("0.01")
USD_MICRO = Decimal("0.000001")
MAX_BUY_PRICE_CEILING = Decimal("0.515")
MIN_BUY_PRICE_FLOOR = Decimal("0.37")
HARD_WINDOW_DEADLINE_SECONDS = 20.0

TICK_SIZE_TO_LIMIT_PRICE: dict[str, Decimal] = {
    "0.01": Decimal("0.51"),
    "0.001": Decimal("0.515"),
}


def to_decimal(value: Any, name: str = "value") -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Invalid numeric {name}: {value!r}")
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric {name}: {value!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"Non-finite numeric {name}: {value!r}")
    return dec


def parse_utc_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise ValueError(f"Expected ISO timestamp or datetime, got {type(value).__name__}")
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"Timezone-aware UTC timestamp required: {value!r}")
    return dt.astimezone(timezone.utc)


def format_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def project_to_calibrated_utc(
    local_utc: datetime | None = None,
    clock_projection: Mapping[str, Any] | None = None,
) -> datetime:
    """Project host UTC time into upstream's calibrated Polymarket CLOB UTC upper-bound clock.

    Why: Upstream `decision_at`, `fill_at`, and `valid_until` are stamped using
    `local_utc + server_minus_local_seconds + uncertainty_seconds`. Using the same
    projection ensures `current_utc < valid_until` and `detection_lag_ms` are aligned.
    """
    base = parse_utc_timestamp(local_utc) if local_utc is not None else datetime.now(timezone.utc)
    if not isinstance(clock_projection, Mapping):
        return base
    if clock_projection.get("status") != "CALIBRATED":
        return base
    try:
        offset = to_decimal(clock_projection.get("server_minus_local_seconds", 0), "server_minus_local_seconds")
        uncertainty = to_decimal(clock_projection.get("uncertainty_seconds", 0), "uncertainty_seconds")
        if uncertainty < 0 or uncertainty > Decimal("2.0"):
            return base
        shift_seconds = float(offset + uncertainty)
        return base + timedelta(seconds=shift_seconds)
    except (ValueError, TypeError):
        return base


def validate_clock_projection(
    clock_projection: Mapping[str, Any] | None = None,
    *,
    state_clock: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Validate optional upstream clock projection (`CALIBRATED` with finite offset & uncertainty in [0, 2.0]s)."""
    proj = clock_projection if clock_projection is not None else state_clock
    if proj is None:
        return None
    if not isinstance(proj, Mapping):
        raise ValueError("Invalid clock_projection: expected mapping")
    status = proj.get("status")
    if status != "CALIBRATED":
        raise ValueError(f"Uncalibrated clock_projection status: {status!r}")
    if "server_minus_local_seconds" not in proj or proj.get("server_minus_local_seconds") is None:
        raise ValueError("Missing server_minus_local_seconds in clock_projection")
    if "uncertainty_seconds" not in proj or proj.get("uncertainty_seconds") is None:
        raise ValueError("Missing uncertainty_seconds in clock_projection")
    to_decimal(proj.get("server_minus_local_seconds"), "server_minus_local_seconds")
    uncertainty = to_decimal(proj.get("uncertainty_seconds"), "uncertainty_seconds")
    if uncertainty < Decimal("0") or uncertainty > Decimal("2.0"):
        raise ValueError(f"Clock projection uncertainty out of bounds [0, 2.0]: {uncertainty}")
    return proj


def validate_and_project_utc(
    local_utc: datetime | str | None = None,
    clock_projection: Mapping[str, Any] | None = None,
    *,
    state_clock: Mapping[str, Any] | None = None,
) -> datetime:
    """Validate `clock_projection` (failing closed if provided and invalid/uncalibrated) and project `local_utc`."""
    base = parse_utc_timestamp(local_utc) if local_utc is not None else datetime.now(timezone.utc)
    proj = validate_clock_projection(clock_projection, state_clock=state_clock)
    if proj is None:
        return base
    return project_to_calibrated_utc(base, proj)


def derive_effective_order_deadline(
    *,
    cohort_id: Any,
    sig: Mapping[str, Any] | None,
    fill: Mapping[str, Any] | None,
    signal_id: str,
    state_last_cohort: Any = None,
    hard_cap_seconds: float = HARD_WINDOW_DEADLINE_SECONDS,
) -> dict[str, Any]:
    """Derive and fail-closed validate window boundaries, matching decision evidence, and `effective_valid_until`.

    Enforces:
      - Matching decision signal (`sig["signal_id"] == signal_id`).
      - Required timezone-aware UTC timestamps (`cohort_id`, `window_start`, `window_end`,
        `decision_at`, `valid_until`, `fill_at`).
      - Cross-window consistency across `cohort_id`, `state.last_cohort`, `signal`,
        `allocation_authorization`, and `venue_market`.
      - Signal `valid_until` MUST satisfy `window_start < valid_until <= window_start + hard_cap_seconds`
        (20s hard cap); signals with `valid_until > window_start + 20s` are REJECTED, never clamped/submitted.
    """
    sig_id_clean = str(signal_id or "").strip()
    if not sig_id_clean:
        raise ValueError("Missing signal_id")
    if not isinstance(sig, Mapping) or str(sig.get("signal_id") or "").strip() != sig_id_clean:
        raise ValueError("Missing matching decision signal evidence")
    if not isinstance(fill, Mapping):
        raise ValueError("Missing fill evidence")
    if cohort_id is None or not str(cohort_id).strip():
        raise ValueError("Missing cohort_id")
    for req_key in ("window_start", "window_end", "decision_at", "valid_until"):
        val = sig.get(req_key)
        if val is None or (isinstance(val, str) and not val.strip()):
            raise ValueError(f"Missing signal {req_key}")
    fill_at_raw = fill.get("fill_at")
    if fill_at_raw is None or (isinstance(fill_at_raw, str) and not fill_at_raw.strip()):
        raise ValueError("Missing fill fill_at")

    cohort_dt = parse_utc_timestamp(cohort_id)
    w_start_dt = parse_utc_timestamp(sig["window_start"])
    w_end_dt = parse_utc_timestamp(sig["window_end"])
    decision_dt = parse_utc_timestamp(sig["decision_at"])
    fill_dt = parse_utc_timestamp(fill_at_raw)
    sig_valid_dt = parse_utc_timestamp(sig["valid_until"])

    if w_start_dt != cohort_dt:
        raise ValueError(f"Cross-window mismatch: signal.window_start={w_start_dt.isoformat()} != cohort_id={cohort_dt.isoformat()}")
    if state_last_cohort is not None and str(state_last_cohort).strip():
        last_cohort_dt = parse_utc_timestamp(state_last_cohort)
        if last_cohort_dt != w_start_dt:
            raise ValueError(
                f"Cross-window mismatch: state.last_cohort={last_cohort_dt.isoformat()} != window_start={w_start_dt.isoformat()}"
            )
    venue_mkt = fill.get("venue_market") if isinstance(fill.get("venue_market"), Mapping) else {}
    if venue_mkt.get("window_start") is not None and str(venue_mkt.get("window_start")).strip():
        if parse_utc_timestamp(venue_mkt["window_start"]) != w_start_dt:
            raise ValueError("Cross-window mismatch: venue_market.window_start != signal.window_start")
    if venue_mkt.get("window_end") is not None and str(venue_mkt.get("window_end")).strip():
        if parse_utc_timestamp(venue_mkt["window_end"]) != w_end_dt:
            raise ValueError("Cross-window mismatch: venue_market.window_end != signal.window_end")
    auth = fill.get("allocation_authorization") if isinstance(fill.get("allocation_authorization"), Mapping) else {}
    if auth.get("cohort_id") is not None and str(auth.get("cohort_id")).strip():
        if parse_utc_timestamp(auth["cohort_id"]) != w_start_dt:
            raise ValueError("Cross-window mismatch: allocation_authorization.cohort_id != signal.window_start")

    dur_sec = (w_end_dt - w_start_dt).total_seconds()
    if not (60.0 <= dur_sec <= 3600.0):
        raise ValueError(f"Invalid market window duration: {dur_sec}s")

    cap_s = float(hard_cap_seconds)
    if not (0.0 < cap_s <= HARD_WINDOW_DEADLINE_SECONDS):
        raise ValueError(f"Invalid hard_cap_seconds: {cap_s}")
    hard_cap_dt = w_start_dt + timedelta(seconds=cap_s)

    if not (w_start_dt <= decision_dt < hard_cap_dt):
        raise ValueError("decision_at outside [window_start, window_start + hard_cap)")
    if not (decision_dt <= fill_dt < hard_cap_dt):
        raise ValueError("fill_at outside [decision_at, window_start + hard_cap)")
    if not (w_start_dt < sig_valid_dt <= hard_cap_dt):
        raise ValueError("signal valid_until outside (window_start, window_start + 20s]")
    if not (decision_dt < sig_valid_dt):
        raise ValueError("signal valid_until must be strictly after decision_at")

    effective_valid_dt = min(sig_valid_dt, hard_cap_dt)
    return {
        "window_start_dt": w_start_dt,
        "window_end_dt": w_end_dt,
        "decision_dt": decision_dt,
        "fill_dt": fill_dt,
        "effective_valid_dt": effective_valid_dt,
        "effective_valid_iso": format_utc_iso(effective_valid_dt),
    }


def check_hard_deadline(
    current_utc: datetime | str,
    valid_until: datetime | str,
) -> tuple[bool, float]:
    """Check the hard abort gate: if `current_utc >= valid_until`, abort immediately.

    Returns:
        (is_within_deadline, remaining_seconds)
    """
    now_dt = parse_utc_timestamp(current_utc)
    deadline_dt = parse_utc_timestamp(valid_until)
    remaining = (deadline_dt - now_dt).total_seconds()
    return (now_dt < deadline_dt), remaining


def align_limit_price(
    tick_size: str | float | Decimal,
    custom_mapping: Mapping[str, Any] | None = None,
) -> Decimal:
    """Align the buy limit price (`0.515` ceiling) to the market's `tick_size`.

    - If `tick_size == "0.01"` -> `0.51`
    - If `tick_size == "0.001"` -> `0.515`
    """
    tick_dec = to_decimal(tick_size, "tick_size").normalize()
    tick_key = format(tick_dec, "f")
    mapping = TICK_SIZE_TO_LIMIT_PRICE
    if isinstance(custom_mapping, Mapping) and tick_key in custom_mapping:
        price = to_decimal(custom_mapping[tick_key], "limit_price")
    elif tick_key in mapping:
        price = mapping[tick_key]
    else:
        raise ValueError(f"Unsupported tick_size: {tick_size!r} (expected '0.01' or '0.001')")

    if not (MIN_BUY_PRICE_FLOOR <= price <= MAX_BUY_PRICE_CEILING):
        raise ValueError(f"Aligned limit_price {price} outside allowed range [{MIN_BUY_PRICE_FLOOR}, {MAX_BUY_PRICE_CEILING}]")
    return price


def compute_fee_per_share(rate: Any, price: Any) -> Decimal:
    """Compute Polymarket crypto taker fee per share: `rate * P * (1 - P)`."""
    r = to_decimal(rate, "fee_rate")
    p = to_decimal(price, "price")
    if not (Decimal("0") <= r <= Decimal("1")):
        raise ValueError(f"Fee rate out of bounds [0, 1]: {r}")
    if not (Decimal("0") < p < Decimal("1")):
        raise ValueError(f"Price out of bounds (0, 1): {p}")
    return r * p * (Decimal("1") - p)


def compute_live_order_sizing(
    approved_stake_usd: Any,
    live_scale_factor: Any,
    fee_rate: Any,
    tick_size: str = "0.01",
    quota_cap_usd: Any = None,
    tick_size_limit_prices: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute fee-inclusive share quantity and spend caps for a Polymarket FAK order.

    Formulas (Spec 4.2):
      - `live_stake_usd = live_scale_factor * approved_stake_usd` (capped by remaining live risk quota if provided)
      - `fee_per_share = rate * P_limit * (1 - P_limit)`
      - `size_shares = floor_0.01(live_stake_usd / (P_limit + fee_per_share))`
    """
    approved_dec = to_decimal(approved_stake_usd, "approved_stake_usd")
    scale_dec = to_decimal(live_scale_factor, "live_scale_factor")
    if approved_dec <= 0:
        raise ValueError(f"approved_stake_usd must be positive, got {approved_dec}")
    if not (Decimal("0") < scale_dec <= Decimal("10")):
        raise ValueError(f"live_scale_factor out of bounds: {scale_dec}")

    scaled_target = (scale_dec * approved_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
    if quota_cap_usd is not None:
        cap_dec = max(Decimal("0"), to_decimal(quota_cap_usd, "quota_cap_usd"))
        live_stake = min(scaled_target, cap_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
    else:
        live_stake = scaled_target

    limit_price = align_limit_price(tick_size, custom_mapping=tick_size_limit_prices)
    fee_per_share = compute_fee_per_share(fee_rate, limit_price)
    unit_cost_with_fee = limit_price + fee_per_share

    if live_stake <= 0:
        size_shares = Decimal("0.00")
    else:
        size_shares = (live_stake / unit_cost_with_fee).quantize(SHARE_STEP, rounding=ROUND_DOWN)
        while size_shares > 0 and (size_shares * unit_cost_with_fee) > live_stake:
            size_shares = (size_shares - SHARE_STEP).quantize(SHARE_STEP, rounding=ROUND_DOWN)

    est_cost_usdc = (size_shares * limit_price).quantize(USD_MICRO, rounding=ROUND_DOWN)
    est_fee_usdc = (size_shares * fee_per_share).quantize(USD_MICRO, rounding=ROUND_DOWN)
    est_total_spent = (est_cost_usdc + est_fee_usdc).quantize(USD_MICRO, rounding=ROUND_DOWN)
    unfilled_stake = max(Decimal("0"), (live_stake - est_total_spent).quantize(USD_MICRO, rounding=ROUND_DOWN))

    return {
        "approved_stake_usd": float(approved_dec),
        "live_scale_factor": float(scale_dec),
        "scaled_paper_target_usd": float(scaled_target),
        "requested_stake_usd": float(live_stake),
        "tick_size": str(to_decimal(tick_size, "tick_size").normalize()),
        "limit_price": float(limit_price),
        "limit_price_dec": limit_price,
        "fee_rate": float(to_decimal(fee_rate, "fee_rate")),
        "fee_per_share": float(fee_per_share),
        "fee_per_share_dec": fee_per_share,
        "size_shares": float(size_shares),
        "size_shares_dec": size_shares,
        "requested_quantity": format(size_shares, ".2f"),
        "est_cost_usdc": float(est_cost_usdc),
        "est_fee_usdc": float(est_fee_usdc),
        "est_total_spent_usdc": float(est_total_spent),
        "est_unfilled_stake_usdc": float(unfilled_stake),
    }


def compute_paper_benchmark_metrics(fill: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and normalize paper benchmark metrics from an upstream `execution.fills` item."""
    auth = fill.get("allocation_authorization") or {}
    approved_stake = float(to_decimal(auth.get("approved_stake_usd", 0), "approved_stake_usd"))
    qty_dec = to_decimal(fill.get("quantity", "0"), "quantity")
    cost_dec = to_decimal(fill.get("cost_usd", 0), "cost_usd")
    fee_dec = to_decimal(fill.get("fee_usd", 0), "fee_usd")
    unfilled_dec = to_decimal(fill.get("unfilled_budget_usd", 0), "unfilled_budget_usd")

    if qty_dec > 0:
        avg_price = float((cost_dec / qty_dec).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))
    else:
        avg_price = 0.0

    return {
        "approved_stake_usd": approved_stake,
        "filled_quantity": str(fill.get("quantity", "0")),
        "avg_fill_price": avg_price,
        "cost_usd": float(cost_dec),
        "fee_usd": float(fee_dec),
        "unfilled_budget_usd": float(unfilled_dec),
        "book_observed_at": fill.get("book_observed_at"),
    }


def compute_diff_vs_paper(
    paper_benchmark: Mapping[str, Any],
    live_execution: Mapping[str, Any],
    paper_status: str = "OPEN",
) -> dict[str, Any]:
    """Compute the 1-to-1 reconciliation diff object (`diff_vs_paper`) per Spec Section 5.1."""
    scale = to_decimal(live_execution.get("live_scale_factor", 0.1), "live_scale_factor")
    paper_approved = to_decimal(paper_benchmark.get("approved_stake_usd", 0), "approved_stake_usd")
    paper_spent = to_decimal(paper_benchmark.get("cost_usd", 0), "cost_usd") + to_decimal(
        paper_benchmark.get("fee_usd", 0), "fee_usd"
    )
    scaled_paper_target = (scale * paper_approved).quantize(USD_MICRO, rounding=ROUND_DOWN)
    scaled_paper_spent = (scale * paper_spent).quantize(USD_MICRO, rounding=ROUND_DOWN)

    live_spent = to_decimal(live_execution.get("total_spent_usdc", 0), "total_spent_usdc")
    if scaled_paper_spent > 0:
        fill_ratio = float((live_spent / scaled_paper_spent).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))
    elif scaled_paper_target > 0:
        fill_ratio = float((live_spent / scaled_paper_target).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))
    else:
        fill_ratio = 0.0

    paper_avg_price = to_decimal(paper_benchmark.get("avg_fill_price", 0), "avg_fill_price")
    live_avg_price = to_decimal(live_execution.get("avg_fill_price", 0) or 0, "live_avg_fill_price")
    if paper_avg_price > 0 and live_avg_price > 0:
        slippage_bps = float(((live_avg_price - paper_avg_price) / paper_avg_price * Decimal("10000")).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN
        ))
    else:
        slippage_bps = 0.0

    spent_diff = float((live_spent - scaled_paper_spent).quantize(USD_MICRO, rounding=ROUND_DOWN))
    clob_status = str(live_execution.get("clob_status", "")).upper()
    live_filled_qty = to_decimal(live_execution.get("filled_quantity", "0") or "0", "filled_quantity")
    status_match = (paper_status == "OPEN") == (
        clob_status in ("MATCHED", "FILLED", "SHADOW_MATCHED") and live_filled_qty > 0
    )

    return {
        "scaled_paper_target_usd": float(scaled_paper_target),
        "fill_ratio_vs_scaled_paper": fill_ratio,
        "avg_price_slippage_bps": slippage_bps,
        "total_spent_diff_vs_scaled_paper_usd": spent_diff,
        "status_match": bool(status_match),
    }
