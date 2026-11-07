"""Evaluate the frozen local TabNet adaptation; no production/account writes."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from threadpoolctl import threadpool_limits
from tools.argon4_forward import gen4_evaluation as metrics
from tools.argon4_forward import gen4_portfolio_scenario as portfolio

REPORT = ROOT / "reports/argon4_gen4_sparse_candidate_20261106"
PRIMARY = "SPARSE_ADAPTIVE_RESET"
CONTROLS = ("CURRENT_RESET", "CURRENT_FROZEN", "TAIL_CURRENT_RESET")
VARIANTS = ("CURRENT_FROZEN", "CURRENT_RESET", "TAIL_CURRENT_RESET", PRIMARY)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def evaluate(frame):
    """Use the established arithmetic without modifying its frozen tail study."""
    valid = metrics.validate_predictions(frame)
    if set(valid.variant) != set(VARIANTS) or valid.phase.nunique() != 1:
        raise ValueError("one complete fixed four-arm phase required")
    comparisons = []
    for old in CONTROLS:
        for rate in metrics.DEFAULT_FEE_RATES:
            item = metrics._compare_validated(valid, PRIMARY, old, rate)
            item["comparison_role"] = ("PRIMARY" if old == "CURRENT_RESET" else
                                       "SECONDARY_EXISTING_FROZEN" if old == "CURRENT_FROZEN" else
                                       "DIAGNOSTIC_INCREMENT_OVER_TAIL")
            item["eligible_for_winner_selection"] = old != "TAIL_CURRENT_RESET"
            comparisons.append(item)
    return {"summary": metrics._summarize_validated(valid, metrics.DEFAULT_FEE_RATES),
            "comparisons": comparisons, "promotion_decision": "NOT_MADE_BY_EVALUATOR"}


def comparison(result, old):
    return next(row for row in result["comparisons"] if row["new"] == PRIMARY
                and row["old"] == old and row["fee_assumptions"]["fee_rate"] == .07)


def compact_table(result, phase, order):
    lookup = {(r["stream"], r["variant"], r["fee_rate"]): r
              for r in result["summary"]["phases"][phase]["per_stream"]}
    rows = []
    for stream in order:
        old, reset, tail, new = [lookup[(stream, v, .07)] for v in VARIANTS]
        rows.append({"stream": stream, "phase": phase,
                     "current_signals": old["total_signals"], "candidate_signals": new["total_signals"],
                     "current_wr_pct": old["raw_win_rate_pct"], "candidate_wr_pct": new["raw_win_rate_pct"],
                     "tail_wr_pct": tail["raw_win_rate_pct"],
                     "current_fee07_proxy_pnl_usd": old["net_proxy_pnl_usd"],
                     "candidate_fee07_proxy_pnl_usd": new["net_proxy_pnl_usd"],
                     "tail_fee07_proxy_pnl_usd": tail["net_proxy_pnl_usd"],
                     "delta_vs_current_fee07_usd": new["net_proxy_pnl_usd"]-old["net_proxy_pnl_usd"],
                     "delta_vs_reset_fee07_usd": new["net_proxy_pnl_usd"]-reset["net_proxy_pnl_usd"],
                     "delta_vs_tail_fee07_usd": new["net_proxy_pnl_usd"]-tail["net_proxy_pnl_usd"]})
    return rows


def research_screen(historical, recent, table, required_nonnegative=4):
    primary, actual = [comparison(historical, x) for x in CONTROLS[:2]]
    recent_primary, recent_actual = [comparison(recent, x) for x in CONTROLS[:2]]
    interval = primary["bootstrap"]["delta_total_proxy_pnl_usd_ci95"]
    conditions = {
        "historical_positive_vs_reset": primary["point_estimate"]["delta_total_proxy_pnl_usd"] > 0,
        "historical_positive_vs_current": actual["point_estimate"]["delta_total_proxy_pnl_usd"] > 0,
        "historical_paired_ci_lower_positive": interval is not None and interval[0] > 0,
        f"at_least_{required_nonnegative}_streams_nonnegative_vs_reset": sum(r["delta_vs_reset_fee07_usd"] >= 0 for r in table) >= required_nonnegative,
        "historical_daily_drawdown_not_worse_than_current": (
            actual["new_metrics"]["max_daily_drawdown_usd"] <= actual["old_metrics"]["max_daily_drawdown_usd"]),
        "recent_positive_vs_reset": recent_primary["point_estimate"]["delta_total_proxy_pnl_usd"] > 0,
        "recent_positive_vs_current": recent_actual["point_estimate"]["delta_total_proxy_pnl_usd"] > 0,
    }
    return {"status": "RESEARCH_SCREEN_PASSED_PENDING_SOFTWARE_AND_OPERATIONAL_REVIEW" if all(conditions.values()) else "DO_NOT_PROMOTE",
            "conditions": conditions, "failed_conditions": [k for k,v in conditions.items() if not v],
            "classification": "LOCAL_TRAINABLE_SPARSE_TABNET_ADAPTATION_NOT_AUTHOR_REPRODUCTION",
            "previous_tail_only_result_known_before_phase2": True,
            "historical_period_previously_examined": True, "untouched_holdout": False,
            "recent_interval_status": recent_primary["bootstrap"]["status"],
            "primary_proxy_fee_rate": .07, "production_published": False, "live_enabled": False,
            "strategy_effectiveness_gate": "NOT_CERTIFIED_MISSING_DECLARED_PROTOCOL_AND_HISTORICAL_EXECUTION_EVIDENCE",
            "actual_incremental_10000_account_profit_usd": None,
            "reason_actual_account_profit_unknown": "No historical venue orderbooks, executable fills or official settlement joins; capital scenarios assume these inputs."}


def verify_identity(snapshot, prereg, run, candidate):
    if digest(ROOT/"config/argon4_forward_policy.json") != snapshot["running_policy_sha256"]:
        raise ValueError("production policy changed")
    if digest(ROOT/snapshot["baseline_code"]["path"]) != snapshot["baseline_code"]["sha256"]:
        raise ValueError("production model code changed")
    protected = {**snapshot["baseline_artifacts"], **prereg["frozen_tail_artifacts"]}
    for name, expected in protected.items():
        if digest(ROOT/name) != expected:
            raise ValueError("baseline or tail artifact changed: " + name)
    for item in run["completed"]:
        stream, phase = item["stream"], item["phase"]
        manifest = candidate.MODEL_ROOT/phase/f"{stream}.manifest.json"
        if digest(manifest) != item["manifest_sha256"]:
            raise ValueError("candidate manifest differs from training receipt")
        ref = read(manifest)["baseline_reference"]
        oldroot = Path("research/argon4_local_v4_20261106")/phase
        if (ref["stream"] != stream or ref["phase"] != phase
                or ref["manifest_sha256"] != snapshot["baseline_artifacts"][str(oldroot/f"{stream}.manifest.json")]
                or ref["model_sha256"] != snapshot["baseline_artifacts"][str(oldroot/f"{stream}.joblib")]):
            raise ValueError("candidate does not reference the frozen production baseline")


def capital_scenarios(frames, policy, prereg, *, enabled_streams=None):
    spec = prereg["capital_sensitivity"]
    if (spec["variants"] != list(VARIANTS) or spec["fee_rates"] != [0., .07, .10]
            or spec["settlement_delay_minutes"] != [0, 60]
            or spec["eligible_for_model_selection"] is not False):
        raise ValueError("frozen capital sensitivity specification required")
    portfolio._validate_policy(policy)
    if enabled_streams is not None and (not enabled_streams or len(set(enabled_streams)) != len(enabled_streams)
                                        or not set(enabled_streams).issubset(portfolio.STREAM_MINUTES)):
        raise ValueError("valid nonempty stage stream subset required")
    results = []
    for window, frame in frames.items():
        valid = metrics.validate_predictions(frame)
        if set(valid.variant) != set(VARIANTS):
            raise ValueError("all four capital scenario arms required")
        for variant in VARIANTS:
            timeline, phase = portfolio._prepare(valid, variant)
            if enabled_streams is not None:
                # Gate eligibility in the allocator timeline; do not alter stored
                # model scores, labels, threshold decisions or prediction files.
                timeline = [(stamp, {stream: (selected and stream in enabled_streams, side, ret)
                                     for stream, (selected, side, ret) in cohort.items()})
                            for stamp, cohort in timeline]
            for rate in spec["fee_rates"]:
                for delay in spec["settlement_delay_minutes"]:
                    item = portfolio._simulate(timeline, phase, policy, variant, Decimal(str(rate)), delay)
                    # Arithmetic is shared; phase 1's verdict is not a phase 2 verdict.
                    item.update(classification="PREREGISTERED_CAPITAL_SENSITIVITY_NOT_MODEL_SELECTION",
                                preregistered_model_screen="NOT_DETERMINED_BY_CAPITAL_SCENARIOS",
                                enabled_streams=list(enabled_streams) if enabled_streams is not None else list(policy["order"]))
                    results.append({"window": window, **item})
    return {"classification": "PREREGISTERED_CAPITAL_SENSITIVITY_NOT_MODEL_SELECTION",
            "scenario_count": len(results), "scenarios": results,
            "eligible_for_model_selection": False, "actual_account_returns": False,
            "production_published": False, "live_enabled": False,
            "arithmetic_source_sha256": digest(portfolio.__file__)}


def main():
    from tools.argon4_forward import gen4_sparse_candidate as candidate
    receipt = candidate.validate_preregistration(REPORT/"preregistration.json")
    prereg, run = receipt["document"], read(REPORT/"run.json")
    if receipt["sha256"] != run["provenance"]["preregistration_sha256"]:
        raise ValueError("preregistration changed after training")
    expected = {(s,p) for s in candidate.frozen.STREAMS for p in candidate.frozen.FOLDS}
    if run["status"] != "COMPLETED" or {(r["stream"],r["phase"]) for r in run["completed"]} != expected:
        raise ValueError("fourteen complete candidate packages required")
    snapshot = read(REPORT/"baseline_snapshot.json")
    verify_identity(snapshot, prereg, run, candidate)
    historical_parts = []
    pins = {(r["stream"],r["phase"]):r["manifest_sha256"] for r in run["completed"]}
    for item in run["completed"]:
        if item["phase"] == "oos_2026":
            if item["controls_parity"]["status"] != "PASS" or "test" not in item["controls_parity"]["splits"]:
                raise ValueError("baseline cache parity required")
            source = item["outputs"]["test"]
            if digest(source["path"]) != source["sha256"]:
                raise ValueError("historical prediction bytes changed")
            historical_parts.append(pd.read_parquet(source["path"]))
    historical_frame = pd.concat(historical_parts, ignore_index=True)
    del historical_parts
    print("EVALUATE_SPARSE_HISTORICAL", flush=True)
    with threadpool_limits(limits=2):
        historical = evaluate(historical_frame)
    write(REPORT/"historical_evaluation.json", historical)
    spec = prereg["recent_research"]
    manifest_path = ROOT/spec["snapshot_manifest"]
    if digest(manifest_path) != spec["sha256"]:
        raise ValueError("recent snapshot changed")
    data_root = manifest_path.parent/"raw_data"
    for item in read(manifest_path)["files"]:
        if digest(data_root/item["relative_path"]) != item["copy_sha256"]:
            raise ValueError("recent input bytes changed")
    recent_parts, recent_receipts = [], []
    for item in run["completed"]:
        if item["phase"] != "deployment_q4":
            continue
        if item["controls_parity"]["status"] != "PASS" or "recent" not in item["controls_parity"]["splits"]:
            raise ValueError("recent baseline control parity required")
        source = item["outputs"]["recent"]
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("recent prediction bytes changed")
        rows = pd.read_parquet(source["path"])
        if rows.empty or rows.decision_close_utc.min() < pd.Timestamp(spec["decision_start_utc"]) or rows.decision_close_utc.max() > pd.Timestamp(spec["decision_end_utc"]):
            raise ValueError("recent predictions outside fixed evaluation window")
        recent_receipts.append({"stream": item["stream"], **source,
                                "manifest_sha256": pins[(item["stream"],"deployment_q4")]})
        recent_parts.append(rows)
    with threadpool_limits(limits=2):
        recent_frame = pd.concat(recent_parts, ignore_index=True)
        recent = evaluate(recent_frame)
    write(REPORT/"recent_evaluation.json", recent)
    order = snapshot["running_policy"]["order"]
    table = {"historical": compact_table(historical, "oos_2026", order),
             "recent": compact_table(recent, "deployment_q4", order),
             "assumptions": prereg["metrics"], "actual_account_returns": False}
    write(REPORT/"comparison_table.json", table)
    for window in ("historical", "recent"):
        pd.DataFrame(table[window]).to_csv(REPORT/f"{window}_seven_stream_comparison.csv", index=False, encoding="utf-8-sig")
    assessment = research_screen(historical, recent, table["historical"])
    print("SPARSE_CAPITAL_SENSITIVITY", flush=True)
    with threadpool_limits(limits=2):
        capital = capital_scenarios({"historical": historical_frame, "recent": recent_frame}, snapshot["running_policy"], prereg)
    write(REPORT/"portfolio_scenario.json", capital)
    stage = prereg["user_stage1"]
    enabled = stage["enabled_streams"]
    if (enabled != ["ETH_5m", "BTC_15m", "BTC_5m"]
            or stage["full_policy_order"] != ["ETH_5m", "BTC_15m", "BTC_5m", "SOL_15m", "SOL_5m", "ETH_15m", "BNB_15m"]
            or stage["primary_scope"] != "USER_SPECIFIED_BEFORE_PHASE2_TRAINING"
            or stage["required_nonnegative_streams"] != 2):
        raise ValueError("exact user-specified three-stream stage registration required")
    stage_frames = {"historical": historical_frame.loc[historical_frame.stream.isin(enabled)],
                    "recent": recent_frame.loc[recent_frame.stream.isin(enabled)]}
    stage_policy = {**snapshot["running_policy"], "order": stage["full_policy_order"]}
    with threadpool_limits(limits=2):
        stage_historical, stage_recent = [evaluate(stage_frames[w]) for w in ("historical", "recent")]
        stage_capital = capital_scenarios({"historical": historical_frame, "recent": recent_frame},
                                         stage_policy, prereg, enabled_streams=enabled)
    stage_order = list(enabled)
    stage_table = {"historical": compact_table(stage_historical, "oos_2026", stage_order),
                   "recent": compact_table(stage_recent, "deployment_q4", stage_order)}
    stage_assessment = research_screen(stage_historical, stage_recent, stage_table["historical"], required_nonnegative=2)
    stage_assessment.update(primary_scope="USER_SPECIFIED_THREE_STREAM_STAGE", enabled_streams=stage_order,
                            seven_stream_diagnostic_status=assessment["status"],
                            target_capital_usd=stage.get("target_capital_usd"),
                            speed_objective="COMPARE_REALIZED_PROXY_GROWTH_WITH_FROZEN_RISK_LIMITS_NOT_HIGHER_LEVERAGE")
    write(REPORT/"seven_stream_research_assessment.json", assessment)
    write(REPORT/"stage1_historical_evaluation.json", stage_historical)
    write(REPORT/"stage1_recent_evaluation.json", stage_recent)
    write(REPORT/"stage1_comparison_table.json", stage_table)
    write(REPORT/"stage1_portfolio_scenario.json", stage_capital)
    assessment = stage_assessment
    verify_identity(snapshot, prereg, run, candidate)
    if digest(REPORT/"preregistration.json") != receipt["sha256"]:
        raise ValueError("preregistration changed during evaluation")
    assessment.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      preregistration_sha256=receipt["sha256"], evaluation_script_sha256=digest(__file__),
                      evaluation_module_sha256=digest(metrics.__file__), recent_prediction_receipts=recent_receipts,
                      baseline_policy_and_artifacts_unchanged=True,
                      historical_evaluation_sha256=digest(REPORT/"historical_evaluation.json"),
                      recent_evaluation_sha256=digest(REPORT/"recent_evaluation.json"),
                      capital_sensitivity_sha256=digest(REPORT/"portfolio_scenario.json"),
                      stage1_historical_evaluation_sha256=digest(REPORT/"stage1_historical_evaluation.json"),
                      stage1_recent_evaluation_sha256=digest(REPORT/"stage1_recent_evaluation.json"),
                      stage1_capital_sensitivity_sha256=digest(REPORT/"stage1_portfolio_scenario.json"),
                      software_checks="PENDING_FINAL_WORKSPACE_VALIDATION")
    write(REPORT/"research_assessment.json", assessment)
    print(json.dumps({"status": assessment["status"], "failed_conditions": assessment["failed_conditions"],
                      "stage1_historical_vs_current": comparison(stage_historical, "CURRENT_FROZEN")["point_estimate"],
                      "stage1_historical_vs_reset": comparison(stage_historical, "CURRENT_RESET")["point_estimate"],
                      "stage1_recent_vs_current": comparison(stage_recent, "CURRENT_FROZEN")["point_estimate"]}), flush=True)


if __name__ == "__main__":
    main()
