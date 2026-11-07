"""Phase-2 comparison identity and capital-adapter contract tests."""
from copy import deepcopy

import pandas as pd
import pytest

from scripts import evaluate_argon4_gen4_sparse_candidate as e


def frame(days=35, sparse_wins=True):
    rows = []
    for day in range(days):
        stamp = pd.Timestamp("2026-01-01T00:00:00Z")+pd.Timedelta(days=day)
        for stream in e.portfolio.STREAM_MINUTES:
            for variant in e.VARIANTS:
                win = variant == "TAIL_CURRENT_RESET" or (variant == e.PRIMARY and sparse_wins)
                rows.append({"stream": stream, "phase": "oos_2026", "variant": variant,
                             "decision_close_utc": stamp, "score": 2. if win else -2., "threshold": 1.,
                             "selected": True, "side": "UP" if win else "DOWN", "forward_return": .01,
                             "settled": True, "win": win})
    return pd.DataFrame(rows)


def test_primary_comparison_does_not_accidentally_select_successful_tail_arm():
    result = e.evaluate(frame(sparse_wins=False))
    assert len(result["comparisons"]) == 9
    assert all(row["new"] == e.PRIMARY for row in result["comparisons"])
    assert e.comparison(result, "CURRENT_RESET")["point_estimate"]["delta_total_proxy_pnl_usd"] == 0
    assert e.comparison(result, "TAIL_CURRENT_RESET")["point_estimate"]["delta_total_proxy_pnl_usd"] < 0
    table = e.compact_table(result, "oos_2026", list(e.portfolio.STREAM_MINUTES))
    assert all(row["candidate_wr_pct"] == 0 and row["tail_wr_pct"] == 100 for row in table)
    verdict = e.research_screen(result, result, table)
    assert verdict["status"] == "DO_NOT_PROMOTE"
    assert not verdict["production_published"] and not verdict["live_enabled"]


def test_statistical_pass_is_pending_review_never_an_automatic_live_or_paper_publish():
    result = e.evaluate(frame())
    table = e.compact_table(result, "oos_2026", list(e.portfolio.STREAM_MINUTES))
    verdict = e.research_screen(result, result, table)
    assert verdict["status"] == "RESEARCH_SCREEN_PASSED_PENDING_SOFTWARE_AND_OPERATIONAL_REVIEW"
    assert verdict["actual_incremental_10000_account_profit_usd"] is None
    assert verdict["untouched_holdout"] is False
    assert not verdict["production_published"] and not verdict["live_enabled"]
    diag = e.comparison(result, "TAIL_CURRENT_RESET")
    assert diag["comparison_role"] == "DIAGNOSTIC_INCREMENT_OVER_TAIL"
    assert diag["eligible_for_winner_selection"] is False


def test_short_positive_sample_cannot_pass_the_interval_condition():
    result = e.evaluate(frame(days=6))
    table = e.compact_table(result, "oos_2026", list(e.portfolio.STREAM_MINUTES))
    verdict = e.research_screen(result, result, table)
    assert "historical_paired_ci_lower_positive" in verdict["failed_conditions"]
    assert verdict["recent_interval_status"] == "insufficient_days"


def test_missing_arm_or_duplicate_phase_is_rejected():
    data = frame(days=1)
    with pytest.raises(ValueError, match="four-arm"):
        e.evaluate(data.loc[data.variant != "TAIL_CURRENT_RESET"])
    duplicate = data.copy()
    duplicate["phase"] = "deployment_q4"
    with pytest.raises(ValueError, match="four-arm"):
        e.evaluate(pd.concat([data, duplicate]))


def test_capital_adapter_uses_sparse_rows_and_cannot_determine_promotion():
    policy = {**e.portfolio.POLICY_NUMBERS, "status": "CONFIRMED", "live_enabled": False,
              "waterfall_semantics": "ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1",
              "order": ["BTC_15m", "ETH_5m", "SOL_15m", "SOL_5m", "BTC_5m", "ETH_15m", "BNB_15m"]}
    spec = {"capital_sensitivity": {"variants": list(e.VARIANTS), "fee_rates": [0., .07, .10],
                                   "settlement_delay_minutes": [0, 60], "eligible_for_model_selection": False}}
    data = frame(days=1, sparse_wins=False)
    before = data.copy(deep=True)
    old_variants = e.portfolio.VARIANTS
    result = e.capital_scenarios({"historical": data, "recent": data}, policy, spec)
    assert result["scenario_count"] == 48
    sparse = [r for r in result["scenarios"] if r["variant"] == e.PRIMARY]
    tail = [r for r in result["scenarios"] if r["variant"] == "TAIL_CURRENT_RESET"]
    assert len(sparse) == len(tail) == 12
    assert all(r["realized_balance_usd"] == 9780 for r in sparse)
    assert all(r["realized_balance_usd"] > 10000 for r in tail)
    assert all(r["preregistered_model_screen"] == "NOT_DETERMINED_BY_CAPITAL_SCENARIOS" for r in result["scenarios"])
    assert not result["actual_account_returns"] and not result["eligible_for_model_selection"]
    assert old_variants == e.portfolio.VARIANTS
    pd.testing.assert_frame_equal(data, before)
    changed = deepcopy(spec)
    changed["capital_sensitivity"]["eligible_for_model_selection"] = True
    with pytest.raises(ValueError, match="sensitivity"):
        e.capital_scenarios({"historical": data}, policy, changed)


def test_user_stage_skips_disabled_higher_rows_without_mutating_model_predictions():
    policy = {**e.portfolio.POLICY_NUMBERS, "status": "CONFIRMED", "live_enabled": False,
              "waterfall_semantics": "ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1",
              "order": ["ETH_5m", "BTC_15m", "BTC_5m", "SOL_15m", "SOL_5m", "ETH_15m", "BNB_15m"]}
    spec = {"capital_sensitivity": {"variants": list(e.VARIANTS), "fee_rates": [0., .07, .10],
                                   "settlement_delay_minutes": [0, 60], "eligible_for_model_selection": False}}
    data = frame(days=1)
    first_two = data.stream.isin(["ETH_5m", "BTC_15m"])
    data.loc[first_two, ["score", "selected", "settled", "win", "side"]] = [0., False, False, False, "DOWN"]
    original = data.copy(deep=True)
    result = e.capital_scenarios({"historical": data}, policy, spec, enabled_streams=["ETH_5m", "BTC_15m", "BTC_5m"])
    assert result["scenario_count"] == 24
    for case in result["scenarios"]:
        assert case["entries"] == 1
        assert case["per_stream"]["BTC_5m"]["entries"] == 1
        assert all(case["per_stream"][s]["entries"] == 0 for s in ["SOL_15m", "SOL_5m", "ETH_15m", "BNB_15m"])
    pd.testing.assert_frame_equal(data, original)


def test_stage_coverage_is_two_of_three_and_seven_stream_screen_remains_distinct():
    data = frame()
    data = data.loc[data.stream.isin(["ETH_5m", "BTC_15m", "BTC_5m"])]
    result = e.evaluate(data)
    table = e.compact_table(result, "oos_2026", ["ETH_5m", "BTC_15m", "BTC_5m"])
    assert e.research_screen(result, result, table, required_nonnegative=2)["failed_conditions"] == []
    assert e.research_screen(result, result, table)["status"] == "DO_NOT_PROMOTE"
