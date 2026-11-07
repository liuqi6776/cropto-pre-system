"""Offline, paired evaluation of binary-direction research predictions.

No account, network, order book, or live-model APIs are used here. Payouts are
fixed-price scenarios on Binance direction labels, not historical venue fills
or official Polymarket settlements. All output is returned to the caller.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import pandas as pd


STREAM_MINUTES = {"BTC_5m": 5, "ETH_5m": 5, "SOL_5m": 5,
                  "BTC_15m": 15, "ETH_15m": 15, "SOL_15m": 15, "BNB_15m": 15}
REQUIRED = ("stream", "variant", "decision_close_utc", "score", "threshold",
            "selected", "side", "forward_return", "settled", "win", "phase")
KEY = ["stream", "decision_close_utc"]
SPEND_USD = 200.0
BUY_PRICE = .515
DEFAULT_FEE_RATES = (0.0, .07, .10)
PRIMARY_PAIR = ("TAIL_CURRENT_RESET", "CURRENT_RESET")
SECONDARY_PAIR = ("TAIL_CURRENT_RESET", "CURRENT_FROZEN")
DIAGNOSTIC_VARIANTS = frozenset(("DOC_BASE_RESET", "TAIL_BASE_RESET", "TAIL_CURRENT_FROZEN"))
CLASSIFICATION = "OFFLINE_FIXED_PRICE_DIRECTION_PAYOUT_PROXY_NOT_EXECUTED_TRADES"
LIMITATION = ("Fixed $200 total outlay at price .515; no historical order books, "
              "liquidity, capital allocation or official venue settlements. "
              "Drawdown is a USD proxy-PnL peak-to-trough difference, not a $10,000 account curve.")


class EvaluationError(ValueError):
    """Prediction records do not satisfy the frozen evaluation contract."""


def _numeric(series, name, allow_nan=False):
    if pd.api.types.is_bool_dtype(series.dtype) or (
            not pd.api.types.is_numeric_dtype(series.dtype)
            and any(isinstance(value, (bool, np.bool_)) for value in series)):
        raise EvaluationError(f"BOOLEAN_NUMERIC:{name}")
    try:
        values = series.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"INVALID_NUMERIC:{name}") from exc
    if np.isinf(values).any() or (not allow_nan and np.isnan(values).any()):
        raise EvaluationError(f"NONFINITE:{name}")
    return values


def _booleans(series, name):
    if series.isna().any() or (not pd.api.types.is_bool_dtype(series.dtype)
            and not all(isinstance(value, (bool, np.bool_)) for value in series)):
        raise EvaluationError(f"BOOLEAN_REQUIRED:{name}")
    return series.to_numpy(dtype=bool)


def validate_predictions(predictions):
    """Return a validated copy with UTC window-end/day and label availability.

    ``settled`` means selected AND finite return, matching the existing model
    export. ``win`` is false for unselected/unsettled rows. Ties resolve UP.
    Every variant in one phase must contain the same complete key set and the
    exact same underlying return labels, including the locations of NaNs.
    Different phases are not pooled or required to share their time axes.
    """
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise EvaluationError("NONEMPTY_DATAFRAME_REQUIRED")
    if any(name not in predictions for name in REQUIRED):
        raise EvaluationError("REQUIRED_COLUMNS_MISSING")
    if "split" in predictions:
        if not set(predictions["split"]).issubset({"test", "recent"}):
            raise EvaluationError("VALIDATION_OR_UNKNOWN_SPLIT_NOT_FOR_PAYOUT_EVALUATION")
        if predictions.groupby("phase")["split"].nunique().gt(1).any():
            raise EvaluationError("MIXED_EVALUATION_SPLITS")
    # We only rebind columns below; no shared input arrays are edited in place.
    frame = predictions.loc[:, list(REQUIRED)].copy(deep=False)
    for name in ("stream", "variant", "phase"):
        if not all(isinstance(value, str) and value.strip() for value in pd.unique(frame[name])):
            raise EvaluationError(f"INVALID_IDENTITY:{name}")
    if not set(frame["stream"]).issubset(STREAM_MINUTES):
        raise EvaluationError("UNKNOWN_STREAM")
    try:
        if not isinstance(frame["decision_close_utc"].dtype, pd.DatetimeTZDtype):
            for value in pd.unique(frame["decision_close_utc"]):
                stamp = pd.Timestamp(value)
                if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() is None:
                    raise EvaluationError("AWARE_DECISION_TIME_REQUIRED")
        frame["decision_close_utc"] = pd.to_datetime(frame["decision_close_utc"], utc=True, errors="raise").astype("datetime64[ns, UTC]")
        if frame["decision_close_utc"].isna().any():
            raise EvaluationError("AWARE_DECISION_TIME_REQUIRED")
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError("INVALID_DECISION_TIME") from exc
    if frame.duplicated(["phase", "variant", *KEY]).any():
        raise EvaluationError("DUPLICATE_PREDICTION")
    horizon = frame["stream"].map(STREAM_MINUTES).to_numpy(dtype=np.int64)
    # Exact UTC stream boundaries; a 15m prediction cannot be relabelled as 5m.
    ns = frame["decision_close_utc"].astype("int64").to_numpy()
    if np.any(ns % (horizon * 60 * 1_000_000_000)):
        raise EvaluationError("MISALIGNED_DECISION_BOUNDARY")
    score = _numeric(frame["score"], "score")
    threshold = _numeric(frame["threshold"], "threshold")
    if np.any(threshold <= 0):
        raise EvaluationError("POSITIVE_THRESHOLD_REQUIRED")
    selected = _booleans(frame["selected"], "selected")
    settled = _booleans(frame["settled"], "settled")
    win = _booleans(frame["win"], "win")
    returns = _numeric(frame["forward_return"], "forward_return", allow_nan=True)
    if not np.array_equal(selected, np.abs(score) >= threshold):
        raise EvaluationError("SELECTION_SCORE_MISMATCH")
    if not np.array_equal(frame["side"].to_numpy(), np.where(score > 0, "UP", "DOWN")):
        raise EvaluationError("SIDE_SCORE_MISMATCH")
    available = np.isfinite(returns)
    if not np.array_equal(settled, selected & available):
        raise EvaluationError("SETTLED_SELECTION_MISMATCH")
    expected_win = settled & ((score > 0) == (returns >= 0))
    if not np.array_equal(win, expected_win):
        raise EvaluationError("WIN_LABEL_MISMATCH")
    for name, values in (("score", score), ("threshold", threshold), ("selected", selected),
                         ("settled", settled), ("win", win), ("forward_return", returns)):
        frame[name] = values
    frame["label_available"] = available
    frame["window_end_utc"] = frame["decision_close_utc"] + pd.to_timedelta(horizon, unit="min")
    frame["settlement_day_utc"] = frame["window_end_utc"].dt.floor("D")
    frame = frame.sort_values(["phase", "variant", *KEY], kind="stable").reset_index(drop=True)
    for _, phase in frame.groupby("phase", sort=True):
        reference = None
        for _, arm in phase.groupby("variant", sort=True):
            axis = arm.set_index(KEY).sort_index()
            if reference is None:
                reference = axis
            elif not reference.index.equals(axis.index):
                raise EvaluationError("VARIANT_TIME_AXIS_MISMATCH")
            elif not np.array_equal(reference["forward_return"].to_numpy(),
                                    axis["forward_return"].to_numpy(), equal_nan=True):
                raise EvaluationError("VARIANT_LABEL_MISMATCH")
    return frame


def _fee_rate(value):
    if isinstance(value, (bool, np.bool_)):
        raise EvaluationError("INVALID_FEE_RATE")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError("INVALID_FEE_RATE") from exc
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise EvaluationError("INVALID_FEE_RATE")
    return rate


def payout_assumptions(fee_rate):
    """Describe fixed *total* spend: price cost plus scenario fee is $200."""
    rate = _fee_rate(fee_rate)
    fee_per_share = rate * BUY_PRICE * (1-BUY_PRICE)
    quantity = SPEND_USD / (BUY_PRICE + fee_per_share)
    return {"fee_rate": rate, "total_spend_per_matured_signal_usd": SPEND_USD,
            "buy_price": BUY_PRICE, "fee_per_share": fee_per_share, "quantity": quantity,
            "price_cost_usd": quantity*BUY_PRICE, "fee_usd": quantity*fee_per_share,
            "win_payout_usd": quantity, "loss_payout_usd": 0.0,
            "classification": CLASSIFICATION, "limitation": LIMITATION,
            "tie_rule": "UP_IF_FORWARD_RETURN_EQUALS_ZERO", "official_settlement": False}


def _with_pnl(frame, fee_rate):
    assumptions = payout_assumptions(fee_rate)
    out = frame.copy(deep=False)
    out["proxy_pnl_usd"] = np.where(out["settled"],
        np.where(out["win"], assumptions["quantity"]-SPEND_USD, -SPEND_USD), 0.0)
    return out


def _drawdown(values):
    cumulative = np.r_[0.0, np.cumsum(np.asarray(values, dtype=np.float64))]
    return float(np.max(np.maximum.accumulate(cumulative)-cumulative))


def _metrics(frame):
    selected, settled = frame["selected"], frame["settled"]
    n = int(settled.sum())
    wins = int(frame["win"].sum())
    total = math.fsum(frame["proxy_pnl_usd"])
    by_close = frame.groupby("window_end_utc", sort=True)["proxy_pnl_usd"].sum()
    by_day = frame.groupby("settlement_day_utc", sort=True)["proxy_pnl_usd"].sum()
    return {"prediction_rows": len(frame), "total_signals": int(selected.sum()), "matured": n,
            "unmatured": int((selected & ~settled).sum()), "wins": wins, "losses": n-wins,
            "ties": int((settled & (frame["forward_return"] == 0)).sum()),
            "ties_are_subset_of_wins_or_losses": True,
            "raw_win_rate_pct": 100*wins/n if n else None,
            "net_proxy_pnl_usd": total, "modeled_matured_outlay_usd": SPEND_USD*n,
            "pending_nominal_outlay_usd": SPEND_USD*int((selected & ~settled).sum()),
            "mean_proxy_pnl_per_200_usd": total/n if n else None,
            "proxy_return_on_matured_outlay_pct": 100*total/(SPEND_USD*n) if n else None,
            "max_drawdown_usd": _drawdown(by_close),
            "max_daily_drawdown_usd": _drawdown(by_day),
            "drawdown_clock": "AGGREGATE_EQUAL_WINDOW_ENDS_THEN_CHRONOLOGICAL_PNL_STARTING_AT_ZERO",
            "observed_settlement_days": len(by_day)}


def summarize_predictions(predictions, fee_rates=DEFAULT_FEE_RATES):
    """Return JSON-safe per-phase, per-stream and calendar scenario summaries.

    Calendar/day grouping uses window END, not signal decision/start. No phase
    is mixed with another. Calendar rows aggregate all streams in the variant;
    daily rows additionally retain each stream. No result is an account NAV.
    """
    return _summarize_validated(validate_predictions(predictions), fee_rates)


def _summarize_validated(frame, fee_rates):
    if not isinstance(fee_rates, Sequence) or isinstance(fee_rates, (str, bytes)) or not fee_rates:
        raise EvaluationError("FEE_SCENARIOS_REQUIRED")
    rates = [_fee_rate(rate) for rate in fee_rates]
    if len(set(rates)) != len(rates):
        raise EvaluationError("DUPLICATE_FEE_SCENARIO")
    result = {"classification": CLASSIFICATION, "limitation": LIMITATION,
              "assumptions": [payout_assumptions(rate) for rate in rates], "phases": {}}
    for phase_name, phase in frame.groupby("phase", sort=True):
        output = {"overall": [], "per_stream": [], "daily_pnl": [],
                  "calendar": {"year": [], "quarter": [], "month": []}}
        for variant, arm_base in phase.groupby("variant", sort=True):
            for rate in rates:
                arm = _with_pnl(arm_base, rate)
                identity = {"phase": phase_name, "variant": variant, "fee_rate": rate,
                            "diagnostic_only": variant in DIAGNOSTIC_VARIANTS}
                output["overall"].append({**identity, **_metrics(arm)})
                groups = [("ALL", arm), *list(arm.groupby("stream", sort=True))]
                for stream, rows in groups:
                    if stream != "ALL":
                        output["per_stream"].append({**identity, "stream": stream, **_metrics(rows)})
                    for date, daily in rows.groupby("settlement_day_utc", sort=True):
                        output["daily_pnl"].append({**identity, "stream": stream,
                            "date_utc": date.date().isoformat(), "total_signals": int(daily["selected"].sum()),
                            "matured": int(daily["settled"].sum()), "wins": int(daily["win"].sum()),
                            "unmatured": int((daily["selected"] & ~daily["settled"]).sum()),
                            "net_proxy_pnl_usd": float(daily["proxy_pnl_usd"].sum())})
                end = arm["window_end_utc"]
                periods = {"year": end.dt.year.astype(str), "month": end.dt.strftime("%Y-%m"),
                           "quarter": end.dt.year.astype(str)+"Q"+end.dt.quarter.astype(str)}
                for scale, labels in periods.items():
                    for period, rows in arm.groupby(labels, sort=True):
                        output["calendar"][scale].append({**identity, "period": period, **_metrics(rows)})
        result["phases"][phase_name] = output
    return result


def circular_day_blocks(n_days, n_bootstrap=2000, block_days=7, seed=20261106):
    """Indices for jointly resampling all stream/arm daily totals in each draw."""
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0
           for value in (n_days, n_bootstrap, block_days)):
        raise EvaluationError("POSITIVE_BOOTSTRAP_SIZES_REQUIRED")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise EvaluationError("INVALID_BOOTSTRAP_SEED")
    effective = min(int(block_days), int(n_days))
    count = math.ceil(n_days/effective)
    starts = np.random.default_rng(seed).integers(0, n_days, size=(n_bootstrap, count))
    return ((starts[..., None]+np.arange(effective)) % n_days).reshape(n_bootstrap, -1)[:, :n_days]


def _interval(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return ([float(v) for v in np.quantile(finite, [.025, .975])] if len(finite) else None, len(finite))


def compare_variants(predictions, new="TAIL_CURRENT_RESET", old="CURRENT_RESET", fee_rate=.07,
                     n_bootstrap=2000, seed=20261106, block_days=7):
    """Compare a pair on a common phase/axis using paired circular day blocks.

    Blocks sample whole calendar days jointly across every coin and both arms.
    Fewer than 28 days, or missing interior observation days, suppress intervals;
    six days with an effective six-day block cannot establish significance.
    This function reports evidence only and never chooses/promotes a model.
    """
    return _compare_validated(validate_predictions(predictions), new, old, fee_rate,
                              n_bootstrap, seed, block_days)


def _compare_validated(frame, new, old, fee_rate, n_bootstrap=2000, seed=20261106, block_days=7):
    if frame["phase"].nunique() != 1:
        raise EvaluationError("COMPARE_ONE_PHASE_AT_A_TIME")
    if new == old or new not in set(frame["variant"]) or old not in set(frame["variant"]):
        raise EvaluationError("DISTINCT_EXISTING_COMPARISON_VARIANTS_REQUIRED")
    rate = _fee_rate(fee_rate)
    newer = _with_pnl(frame[frame["variant"] == new], rate).set_index(KEY).sort_index()
    older = _with_pnl(frame[frame["variant"] == old], rate).set_index(KEY).sort_index()
    new_metrics, old_metrics = _metrics(newer), _metrics(older)
    selection = {}
    common = newer["selected"] & older["selected"]
    for name, mask in (("common", common), ("old_only", older["selected"] & ~newer["selected"]),
                       ("new_only", newer["selected"] & ~older["selected"])):
        selection[name] = {"signals": int(mask.sum()),
                           "label_available": int(newer.loc[mask, "label_available"].sum()),
                           "old": _metrics(older.loc[mask]), "new": _metrics(newer.loc[mask])}
    dates = pd.date_range(newer["settlement_day_utc"].min(), newer["settlement_day_utc"].max(), freq="D")
    arrays = {}
    for name, arm in (("new", newer), ("old", older)):
        daily = arm.groupby("settlement_day_utc", sort=True).agg(
            pnl=("proxy_pnl_usd", "sum"), wins=("win", "sum"), matured=("settled", "sum"),
            rows=("selected", "size"))
        daily = daily.reindex(dates)  # Missing observation days remain NaN.
        arrays[name] = daily
    daily_delta = arrays["new"]["pnl"]-arrays["old"]["pnl"]
    n_days = len(dates)
    indices = circular_day_blocks(n_days, n_bootstrap, block_days, seed)
    missing_days = [date.date().isoformat() for date in dates[arrays["new"]["rows"].isna()]]
    delta_wr = (new_metrics["raw_win_rate_pct"]-old_metrics["raw_win_rate_pct"]
                if new_metrics["raw_win_rate_pct"] is not None and old_metrics["raw_win_rate_pct"] is not None else None)
    role = ("PRIMARY" if (new, old) == PRIMARY_PAIR else "SECONDARY_EXISTING_FROZEN"
            if (new, old) == SECONDARY_PAIR else "DIAGNOSTIC_ONLY")
    result = {"classification": CLASSIFICATION, "limitation": LIMITATION,
              "phase": str(frame["phase"].iloc[0]), "new": new, "old": old, "comparison_role": role,
              "eligible_for_winner_selection": role != "DIAGNOSTIC_ONLY",
              "promotion_decision": "NOT_MADE_BY_EVALUATOR", "fee_assumptions": payout_assumptions(rate),
              "new_metrics": new_metrics, "old_metrics": old_metrics, "selection_overlap": selection,
              "point_estimate": {"delta_total_proxy_pnl_usd": new_metrics["net_proxy_pnl_usd"]-old_metrics["net_proxy_pnl_usd"],
                                  "mean_observed_day_delta_usd": float(daily_delta.mean()),
                                  "raw_win_rate_difference_pp": delta_wr},
              "daily_paired": [{"date_utc": date.date().isoformat(),
                                "new_proxy_pnl_usd": float(arrays["new"].loc[date, "pnl"]) if date not in dates[arrays["new"]["rows"].isna()] else None,
                                "old_proxy_pnl_usd": float(arrays["old"].loc[date, "pnl"]) if date not in dates[arrays["old"]["rows"].isna()] else None,
                                "delta_proxy_pnl_usd": float(daily_delta.loc[date]) if pd.notna(daily_delta.loc[date]) else None}
                               for date in dates],
              "bootstrap": {"method": "PAIRED_ALL_STREAMS_CIRCULAR_MOVING_CALENDAR_DAY_BLOCKS",
                            "block_days_requested": int(block_days), "block_days_effective": min(block_days, n_days),
                            "n_days": n_days, "minimum_days_for_interpretation": 28,
                            "missing_observation_days": missing_days, "n_bootstrap_requested": int(n_bootstrap),
                            "seed": int(seed), "confidence_level": .95,
                            "n_bootstrap_run": 0, "delta_total_proxy_pnl_usd_ci95": None,
                            "mean_day_delta_usd_ci95": None, "raw_win_rate_difference_pp_ci95": None}}
    bootstrap = result["bootstrap"]
    if n_days < 28:
        bootstrap.update(status="insufficient_days", interpretation="Short samples do not support significance; a full-sample circular block yields degenerate totals.")
    elif missing_days:
        bootstrap.update(status="missing_observation_days", interpretation="Unobserved calendar days were not filled with zero PnL.")
    else:
        totals = daily_delta.to_numpy()[indices].sum(axis=1)
        new_w = arrays["new"]["wins"].to_numpy()[indices].sum(axis=1)
        old_w = arrays["old"]["wins"].to_numpy()[indices].sum(axis=1)
        new_n = arrays["new"]["matured"].to_numpy()[indices].sum(axis=1)
        old_n = arrays["old"]["matured"].to_numpy()[indices].sum(axis=1)
        wr_deltas = np.full(n_bootstrap, np.nan)
        usable = (new_n > 0) & (old_n > 0)
        wr_deltas[usable] = 100*(new_w[usable]/new_n[usable]-old_w[usable]/old_n[usable])
        total_ci, _ = _interval(totals)
        mean_ci, _ = _interval(totals/n_days)
        wr_ci, valid_wr = _interval(wr_deltas)
        bootstrap.update(status="computed", n_bootstrap_run=int(n_bootstrap),
                         delta_total_proxy_pnl_usd_ci95=total_ci, mean_day_delta_usd_ci95=mean_ci,
                         raw_win_rate_difference_pp_ci95=wr_ci, win_rate_valid_replicates=valid_wr,
                         interpretation="Paired historical uncertainty only; does not remove prior model-selection exposure or establish executable profit.")
    return result


def evaluate_predictions(predictions, fee_rates=DEFAULT_FEE_RATES, **bootstrap_options):
    """Convenience wrapper: summaries plus the two specified current-model pairs.

    The caller writes reports. DOC_BASE_RESET and TAIL_BASE_RESET remain summary
    diagnostics; this wrapper never selects them as upgrade winners.
    """
    frame = validate_predictions(predictions)
    summaries = _summarize_validated(frame, fee_rates)
    comparisons = []
    for _, phase in frame.groupby("phase", sort=True):
        variants = set(phase["variant"])
        for new, old in (PRIMARY_PAIR, SECONDARY_PAIR):
            if new in variants and old in variants:
                for rate in fee_rates:
                    comparisons.append(_compare_validated(phase, new, old, rate, **bootstrap_options))
    return {"summary": summaries, "comparisons": comparisons,
            "promotion_decision": "NOT_MADE_BY_EVALUATOR"}
