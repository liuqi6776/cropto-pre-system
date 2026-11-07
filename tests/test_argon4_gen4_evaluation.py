"""Synthetic offline fixtures; no market retrieval, accounts or saved models."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from tools.argon4_forward.gen4_evaluation import (
    BUY_PRICE, SPEND_USD, EvaluationError, circular_day_blocks, compare_variants,
    evaluate_predictions, payout_assumptions, summarize_predictions, validate_predictions,
)


OLD, NEW = "CURRENT_RESET", "TAIL_CURRENT_RESET"


def row(variant=OLD, stream="BTC_5m", stamp="2026-01-01T00:00:00Z", score=2., ret=.01, phase="oos_2026"):
    selected = abs(score) >= 1.
    settled = selected and bool(np.isfinite(ret))
    return dict(stream=stream, variant=variant, decision_close_utc=stamp, score=score, threshold=1.,
                selected=selected, side="UP" if score > 0 else "DOWN", forward_return=ret,
                settled=settled, win=settled and ((score > 0) == (ret >= 0)), phase=phase)


def pair(days=35, streams=("BTC_5m",), opposite=False):
    records = []
    for day in range(days):
        stamp = (pd.Timestamp("2026-01-01T00:00:00Z")+pd.Timedelta(days=day)).isoformat()
        for j, stream in enumerate(streams):
            old_score = -2. if opposite and j % 2 == 0 else 2.
            new_score = -old_score if opposite else (2. if day % 3 else -2.)
            ret = .01 if opposite or day % 4 else -.01
            records.extend([row(OLD, stream, stamp, old_score, ret), row(NEW, stream, stamp, new_score, ret)])
    return pd.DataFrame(records)


def overall(summary, variant=OLD, rate=0.):
    return next(r for r in summary["phases"]["oos_2026"]["overall"] if r["variant"] == variant and r["fee_rate"] == rate)


@pytest.mark.parametrize("rate", [0., .07, .10])
def test_fee_is_inside_the_fixed_200_total_spend_and_never_double_charged(rate):
    a = payout_assumptions(rate)
    expected = 200/(.515+rate*.515*(1-.515))
    assert a["quantity"] == pytest.approx(expected)
    assert a["price_cost_usd"]+a["fee_usd"] == pytest.approx(200)
    frame = pd.DataFrame([row(ret=.01), row(stamp="2026-01-01T00:05:00Z", ret=-.01)])
    result = overall(summarize_predictions(frame), rate=rate)
    assert result["net_proxy_pnl_usd"] == pytest.approx(expected-400)
    assert result["modeled_matured_outlay_usd"] == 400
    assert result["mean_proxy_pnl_per_200_usd"] == pytest.approx((expected-400)/2)
    assert result["proxy_return_on_matured_outlay_pct"] == pytest.approx(100*(expected-400)/400)


@pytest.mark.parametrize("rate", [-.01, 1.01, float("nan"), float("inf"), True, None])
def test_invalid_fee_scenario_is_rejected(rate):
    with pytest.raises(EvaluationError, match="INVALID_FEE_RATE"):
        payout_assumptions(rate)


def test_ties_are_retained_up_wins_down_loses_and_pending_is_not_a_loss():
    records = [row(ret=0.), row(stamp="2026-01-01T00:05:00Z", score=-2., ret=0.),
               row(stamp="2026-01-01T00:10:00Z", ret=np.nan),
               row(stamp="2026-01-01T00:15:00Z", score=.5, ret=-.01)]
    metrics = overall(summarize_predictions(pd.DataFrame(records)))
    assert (metrics["total_signals"], metrics["matured"], metrics["unmatured"]) == (3, 2, 1)
    assert (metrics["wins"], metrics["losses"], metrics["ties"]) == (1, 1, 2)
    assert metrics["raw_win_rate_pct"] == 50
    assert metrics["net_proxy_pnl_usd"] == pytest.approx(200/.515-400)
    assert metrics["pending_nominal_outlay_usd"] == 200


def test_no_matured_signals_has_no_invented_win_rate_or_loss():
    metrics = overall(summarize_predictions(pd.DataFrame([row(ret=np.nan)])))
    assert metrics["net_proxy_pnl_usd"] == metrics["losses"] == 0
    assert metrics["raw_win_rate_pct"] is None
    assert metrics["mean_proxy_pnl_per_200_usd"] is None


@pytest.mark.parametrize("field,value,error", [
    ("score", np.nan, "NONFINITE"), ("score", np.inf, "NONFINITE"),
    ("threshold", 0., "POSITIVE_THRESHOLD"), ("threshold", -1., "POSITIVE_THRESHOLD"),
    ("selected", False, "SELECTION_SCORE"), ("selected", 1, "BOOLEAN_REQUIRED"),
    ("side", "DOWN", "SIDE_SCORE"), ("settled", False, "SETTLED_SELECTION"),
    ("win", False, "WIN_LABEL"), ("forward_return", np.inf, "NONFINITE"),
    ("stream", "BNB_5m", "UNKNOWN_STREAM"),
    ("decision_close_utc", "2026-01-01T00:00:00", "INVALID_DECISION_TIME"),
    ("decision_close_utc", "2026-01-01T00:01:00Z", "MISALIGNED_DECISION_BOUNDARY"),
])
def test_prediction_contract_rejects_bad_rows(field, value, error):
    record = row()
    record[field] = value
    with pytest.raises(EvaluationError, match=error):
        validate_predictions(pd.DataFrame([record]))


def test_duplicate_axis_and_inconsistent_underlying_labels_are_rejected():
    frame = pair(2)
    with pytest.raises(EvaluationError, match="DUPLICATE_PREDICTION"):
        validate_predictions(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    with pytest.raises(EvaluationError, match="VARIANT_TIME_AXIS_MISMATCH"):
        validate_predictions(frame.iloc[:-1])
    modified = frame.copy()
    i = modified.index[modified.variant == NEW][0]
    modified.loc[i, "forward_return"] *= 2  # Same sign preserves win; actual label is still different.
    with pytest.raises(EvaluationError, match="VARIANT_LABEL_MISMATCH"):
        validate_predictions(modified)


def test_maturity_flag_is_selection_dependent_but_label_nan_locations_must_match():
    records = [row(OLD, score=.5), row(NEW, score=2.)]
    checked = validate_predictions(pd.DataFrame(records))
    assert checked["settled"].tolist() == [False, True]
    records[0] = row(OLD, score=.5, ret=np.nan)
    with pytest.raises(EvaluationError, match="VARIANT_LABEL_MISMATCH"):
        validate_predictions(pd.DataFrame(records))


def test_5m_15m_cashflows_belong_to_window_end_day_month_quarter_and_year():
    frame = pd.DataFrame([row(stream="BTC_5m", stamp="2026-12-31T23:55:00Z"),
                          row(stream="ETH_15m", stamp="2026-12-31T23:45:00Z")])
    result = summarize_predictions(frame, (0.,))["phases"]["oos_2026"]
    aggregate = [r for r in result["daily_pnl"] if r["stream"] == "ALL"]
    assert len(aggregate) == 1 and aggregate[0]["date_utc"] == "2027-01-01"
    assert result["calendar"]["year"][0]["period"] == "2027"
    assert result["calendar"]["quarter"][0]["period"] == "2027Q1"
    assert result["calendar"]["month"][0]["period"] == "2027-01"
    with pytest.raises(EvaluationError, match="MISALIGNED"):
        validate_predictions(pd.DataFrame([row(stream="BTC_15m", stamp="2026-01-01T00:05:00Z")]))


def test_max_drawdown_is_pnl_peak_to_trough_usd_without_a_fictitious_10k_account():
    records = [row(stamp="2026-01-01T00:00:00Z", ret=.01),
               row(stamp="2026-01-02T00:00:00Z", ret=-.01),
               row(stamp="2026-01-03T00:00:00Z", ret=-.01)]
    metrics = overall(summarize_predictions(pd.DataFrame(records)))
    assert metrics["max_drawdown_usd"] == metrics["max_daily_drawdown_usd"] == 400
    assert "equity" not in metrics
    assert "max_drawdown_pct" not in metrics
    simultaneous = pd.DataFrame([row(stream="BTC_5m", ret=.01), row(stream="ETH_5m", ret=-.01)])
    assert overall(summarize_predictions(simultaneous))["max_drawdown_usd"] == pytest.approx(400-200/.515)


def test_common_and_exclusive_signal_sets_preserve_their_own_win_rates():
    records = []
    specs = [(2., 2., .01), (2., .5, -.01), (.5, -2., -.01), (2., 2., np.nan)]
    for i, (old_score, new_score, ret) in enumerate(specs):
        stamp = (pd.Timestamp("2026-01-01T00:00:00Z")+pd.Timedelta(minutes=5*i)).isoformat()
        records.extend([row(OLD, stamp=stamp, score=old_score, ret=ret), row(NEW, stamp=stamp, score=new_score, ret=ret)])
    result = compare_variants(pd.DataFrame(records))
    overlap = result["selection_overlap"]
    assert [overlap[name]["signals"] for name in ("common", "old_only", "new_only")] == [2, 1, 1]
    assert overlap["common"]["label_available"] == 1
    assert overlap["common"]["old"]["raw_win_rate_pct"] == overlap["common"]["new"]["raw_win_rate_pct"] == 100
    assert overlap["old_only"]["old"]["raw_win_rate_pct"] == 0
    assert overlap["new_only"]["new"]["raw_win_rate_pct"] == 100


def test_circular_indices_are_whole_seven_day_blocks_and_reproducible():
    indices = circular_day_blocks(35, 2000, 7, 20261106)
    assert indices.shape == (2000, 35)
    assert np.array_equal(indices, circular_day_blocks(35))
    blocks = indices.reshape(2000, 5, 7)
    assert np.all(np.diff(blocks, axis=2) % 35 == 1)
    assert np.any(blocks[:, :, -1] < blocks[:, :, 0])  # Genuine circular wrapping occurs.


def test_joint_days_preserve_cross_stream_cancellation_not_iid_trade_resampling():
    result = compare_variants(pair(35, streams=("BTC_5m", "ETH_5m"), opposite=True))
    boot = result["bootstrap"]
    assert boot["status"] == "computed"
    assert boot["n_bootstrap_run"] == 2000
    assert boot["delta_total_proxy_pnl_usd_ci95"] == [0., 0.]
    assert boot["mean_day_delta_usd_ci95"] == [0., 0.]
    assert boot["raw_win_rate_difference_pp_ci95"] == [0., 0.]
    assert result["point_estimate"]["delta_total_proxy_pnl_usd"] == 0


def test_bootstrap_interval_matches_the_frozen_paired_day_sampling_algorithm():
    result = compare_variants(pair(35), fee_rate=0.)
    daily = np.array([d["delta_proxy_pnl_usd"] for d in result["daily_paired"]])
    indices = circular_day_blocks(35)
    totals = daily[indices].sum(axis=1)
    assert result["bootstrap"]["delta_total_proxy_pnl_usd_ci95"] == pytest.approx(np.quantile(totals, [.025, .975]))
    assert result["bootstrap"]["mean_day_delta_usd_ci95"] == pytest.approx(np.quantile(totals/35, [.025, .975]))


def test_six_day_sample_is_explicitly_insufficient_and_has_no_significance_interval():
    result = compare_variants(pair(6))
    boot = result["bootstrap"]
    assert boot["status"] == "insufficient_days"
    assert boot["block_days_effective"] == 6
    assert boot["minimum_days_for_interpretation"] == 28
    assert boot["delta_total_proxy_pnl_usd_ci95"] is None
    assert boot["raw_win_rate_difference_pp_ci95"] is None
    assert boot["n_bootstrap_run"] == 0


def test_missing_calendar_day_is_not_filled_as_zero_profit():
    frame = pair(35)
    frame = frame[~frame["decision_close_utc"].str.startswith("2026-01-12")]
    result = compare_variants(frame)
    assert result["bootstrap"]["status"] == "missing_observation_days"
    assert result["bootstrap"]["missing_observation_days"] == ["2026-01-12"]
    missing = next(d for d in result["daily_paired"] if d["date_utc"] == "2026-01-12")
    assert missing["new_proxy_pnl_usd"] is None and missing["delta_proxy_pnl_usd"] is None


def test_phases_are_summarized_separately_and_must_not_be_pooled_in_comparison():
    frame = pair(2)
    frame2 = frame.copy()
    frame2["phase"] = "deployment_q4"
    combined = pd.concat([frame, frame2], ignore_index=True)
    result = summarize_predictions(combined)
    assert set(result["phases"]) == {"oos_2026", "deployment_q4"}
    with pytest.raises(EvaluationError, match="COMPARE_ONE_PHASE"):
        compare_variants(combined)


def test_validation_scores_cannot_be_silently_included_in_payout_results():
    frame = pair(2)
    frame["split"] = "validation"
    with pytest.raises(EvaluationError, match="VALIDATION_OR_UNKNOWN_SPLIT"):
        summarize_predictions(frame)
    frame["split"] = ["test", "recent", "test", "recent"]
    with pytest.raises(EvaluationError, match="MIXED_EVALUATION_SPLITS"):
        compare_variants(frame)


def test_wrapper_preserves_diagnostic_only_arms_and_does_not_promote_models():
    frame = pair(2)
    extras = []
    for variant in ("CURRENT_FROZEN", "DOC_BASE_RESET", "TAIL_BASE_RESET", "TAIL_CURRENT_FROZEN"):
        extra = frame[frame.variant == OLD].copy()
        extra["variant"] = variant
        extras.append(extra)
    frame = pd.concat([frame, *extras], ignore_index=True)
    original = deepcopy(frame)
    result = evaluate_predictions(frame)
    assert len(result["comparisons"]) == 6  # Two specified pairs, three fee scenarios.
    assert {c["comparison_role"] for c in result["comparisons"]} == {"PRIMARY", "SECONDARY_EXISTING_FROZEN"}
    assert result["promotion_decision"] == "NOT_MADE_BY_EVALUATOR"
    diagnostics = [r for r in result["summary"]["phases"]["oos_2026"]["overall"] if r["variant"] in ("DOC_BASE_RESET", "TAIL_BASE_RESET", "TAIL_CURRENT_FROZEN")]
    assert all(r["diagnostic_only"] for r in diagnostics)
    explicit_diagnostic = compare_variants(frame, "TAIL_BASE_RESET", "DOC_BASE_RESET")
    assert explicit_diagnostic["comparison_role"] == "DIAGNOSTIC_ONLY"
    assert explicit_diagnostic["eligible_for_winner_selection"] is False
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("kwargs", [dict(n_days=0), dict(n_days=4, n_bootstrap=0), dict(n_days=4, block_days=0),
                                    dict(n_days=4, seed=-1), dict(n_days=True)])
def test_invalid_resampling_parameters_are_rejected(kwargs):
    with pytest.raises(EvaluationError):
        circular_day_blocks(**kwargs)
