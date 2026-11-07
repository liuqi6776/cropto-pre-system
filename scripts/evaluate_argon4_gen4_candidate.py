"""Evaluate frozen local research candidates without any production write path."""
from __future__ import annotations

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from threadpoolctl import threadpool_limits
from tools.argon4_forward import gen4_candidate as candidate
from tools.argon4_forward import gen4_evaluation as evaluation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def verify_baseline(snapshot, run):
    if digest(ROOT/"config/argon4_forward_policy.json") != snapshot["running_policy_sha256"]:
        raise ValueError("production policy changed during research")
    if digest(ROOT/snapshot["baseline_code"]["path"]) != snapshot["baseline_code"]["sha256"]:
        raise ValueError("production model code changed during research")
    for name, expected_hash in snapshot["baseline_artifacts"].items():
        if digest(ROOT/name) != expected_hash:
            raise ValueError("baseline artifact changed")
    checked = []
    for item in run["completed"]:
        stream, phase = item["stream"], item["phase"]
        path = candidate.MODEL_ROOT/phase/f"{stream}.manifest.json"
        if digest(path) != item["manifest_sha256"]:
            raise ValueError("candidate identity differs from training receipt")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        reference = manifest["baseline_reference"]
        base_root = Path("research/argon4_local_v4_20261106")/phase
        if (reference["stream"] != stream or reference["phase"] != phase
                or reference["manifest_sha256"] != snapshot["baseline_artifacts"][str(base_root/f"{stream}.manifest.json")]
                or reference["model_sha256"] != snapshot["baseline_artifacts"][str(base_root/f"{stream}.joblib")]):
            raise ValueError("candidate baseline is not the preregistered baseline")
        checked.append({"stream":stream,"phase":phase,"baseline_reference_matches_snapshot":True})
    return checked


def find_comparison(result, old, rate=.07):
    return next(row for row in result["comparisons"]
                if row["old"] == old and row["fee_assumptions"]["fee_rate"] == rate)


def compact_table(result, phase, order):
    records = result["summary"]["phases"][phase]["per_stream"]
    lookup = {(r["stream"], r["variant"], r["fee_rate"]): r for r in records}
    output = []
    for stream in order:
        old = lookup[(stream, "CURRENT_FROZEN", .07)]
        reset = lookup[(stream, "CURRENT_RESET", .07)]
        new = lookup[(stream, "TAIL_CURRENT_RESET", .07)]
        old0 = lookup[(stream, "CURRENT_FROZEN", 0.0)]
        new0 = lookup[(stream, "TAIL_CURRENT_RESET", 0.0)]
        output.append({"stream": stream, "phase": phase,
                       "current_signals": old["total_signals"], "candidate_signals": new["total_signals"],
                       "current_wr_pct": old["raw_win_rate_pct"], "candidate_wr_pct": new["raw_win_rate_pct"],
                       "delta_wr_pp": (new["raw_win_rate_pct"]-old["raw_win_rate_pct"]
                                       if old["raw_win_rate_pct"] is not None and new["raw_win_rate_pct"] is not None else None),
                       "current_zero_fee_proxy_pnl_usd": old0["net_proxy_pnl_usd"],
                       "candidate_zero_fee_proxy_pnl_usd": new0["net_proxy_pnl_usd"],
                       "current_fee07_proxy_pnl_usd": old["net_proxy_pnl_usd"],
                       "candidate_fee07_proxy_pnl_usd": new["net_proxy_pnl_usd"],
                       "delta_vs_current_fee07_usd": new["net_proxy_pnl_usd"]-old["net_proxy_pnl_usd"],
                       "reset_control_signals": reset["total_signals"],
                       "reset_control_wr_pct": reset["raw_win_rate_pct"],
                       "delta_vs_reset_fee07_usd": new["net_proxy_pnl_usd"]-reset["net_proxy_pnl_usd"]})
    return output


def research_screen(historical, recent, table):
    primary = find_comparison(historical, "CURRENT_RESET")
    actual = find_comparison(historical, "CURRENT_FROZEN")
    recent_primary = find_comparison(recent, "CURRENT_RESET")
    recent_actual = find_comparison(recent, "CURRENT_FROZEN")
    interval = primary["bootstrap"]["delta_total_proxy_pnl_usd_ci95"]
    conditions = {
        "historical_positive_vs_reset": primary["point_estimate"]["delta_total_proxy_pnl_usd"] > 0,
        "historical_positive_vs_current": actual["point_estimate"]["delta_total_proxy_pnl_usd"] > 0,
        "historical_paired_ci_lower_positive": interval is not None and interval[0] > 0,
        "at_least_four_streams_nonnegative_vs_reset": sum(r["delta_vs_reset_fee07_usd"] >= 0 for r in table) >= 4,
        "historical_daily_drawdown_not_worse_than_current": (
            actual["new_metrics"]["max_daily_drawdown_usd"] <= actual["old_metrics"]["max_daily_drawdown_usd"]),
        "recent_positive_vs_reset": recent_primary["point_estimate"]["delta_total_proxy_pnl_usd"] > 0,
        "recent_positive_vs_current": recent_actual["point_estimate"]["delta_total_proxy_pnl_usd"] > 0,
    }
    return {"status": "RESEARCH_SCREEN_PASSED_PENDING_SOFTWARE_AND_OPERATIONAL_REVIEW" if all(conditions.values()) else "DO_NOT_PROMOTE",
            "conditions": conditions, "failed_conditions": [k for k,v in conditions.items() if not v],
            "recent_interval_status": recent_primary["bootstrap"]["status"],
            "primary_proxy_fee_rate": .07, "production_published": False, "live_enabled": False,
            "strategy_effectiveness_gate": "NOT_CERTIFIED_MISSING_DECLARED_PROTOCOL_AND_HISTORICAL_EXECUTION_EVIDENCE",
            "actual_incremental_10000_account_profit_usd": None,
            "reason_actual_account_profit_unknown": "Historical venue orderbooks, executable fills and official settlement joins are absent; fixed-stake direction proxies are not an account backtest."}


def main():
    report = candidate.REPORT_ROOT
    prereg_receipt = candidate.validate_preregistration(report/"preregistration.json")
    prereg = prereg_receipt["document"]
    run = json.loads((report/"run.json").read_text(encoding="utf-8"))
    if prereg_receipt["sha256"] != run["provenance"]["preregistration_sha256"]:
        raise ValueError("preregistration changed since candidate training")
    expected = {(s,p) for s in candidate.frozen.STREAMS for p in candidate.frozen.FOLDS}
    if run["status"] != "COMPLETED" or {(r["stream"],r["phase"]) for r in run["completed"]} != expected:
        raise ValueError("all fourteen frozen candidate packages must complete before evaluation")
    snapshot = json.loads((report/"baseline_snapshot.json").read_text(encoding="utf-8"))
    verify_baseline(snapshot, run)
    pins = {(r["stream"],r["phase"]):r["manifest_sha256"] for r in run["completed"]}
    historical_parts = []
    for item in run["completed"]:
        if item["phase"] != "oos_2026":
            continue
        if item["baseline_cache_check"]["status"] != "PASS":
            raise ValueError("baseline cache parity not established")
        source = item["outputs"]["test"]
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("historical prediction bytes changed")
        historical_parts.append(pd.read_parquet(source["path"]))
    print("EVALUATE_HISTORICAL", flush=True)
    with threadpool_limits(limits=2):
        historical = evaluation.evaluate_predictions(pd.concat(historical_parts, ignore_index=True))
    write_json(report/"historical_evaluation.json", historical)
    del historical_parts
    gc.collect()

    recent_spec = prereg["recent_research"]
    manifest_path = ROOT/recent_spec["snapshot_manifest"]
    if digest(manifest_path) != recent_spec["sha256"]:
        raise ValueError("recent snapshot manifest changed")
    recent_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = manifest_path.parent/"raw_data"
    for item in recent_manifest["files"]:
        if digest(data_root/item["relative_path"]) != item["copy_sha256"]:
            raise ValueError("recent observed snapshot bytes changed")
    recent_parts, recent_receipts = [], []
    with threadpool_limits(limits=2):
        for symbol in candidate.frozen.SYMBOLS:
            print("RECENT_FEATURES "+symbol, flush=True)
            # This is retrospective research, retaining actual receipt fields.
            # Never invoke a historical decision timestamp to bypass reception guards.
            frames = candidate.frozen.build_feature_frames(data_root, symbol, as_of=None)
            first_decision = pd.Timestamp(recent_spec["decision_start_utc"])
            if first_decision-frames["index"][0] < pd.Timedelta(days=15):
                raise ValueError("recent study requires the preregistered fifteen-day warmup")
            for stream, (coin, _, _) in candidate.frozen.STREAMS.items():
                if coin != symbol:
                    continue
                bundle = candidate.load_candidate_bundle(stream, "deployment_q4", pins[(stream,"deployment_q4")])
                rows = candidate.evaluate_rows(bundle, frames, "recent", start=recent_spec["decision_start_utc"], end=recent_spec["decision_end_utc"])
                if rows.empty:
                    raise ValueError("empty recent evaluation")
                path = report/"deployment_q4"/f"{stream}_recent_predictions.parquet"
                rows.to_parquet(path, index=False)
                recent_receipts.append({"stream":stream,"path":str(path),"sha256":digest(path),"rows":len(rows),"manifest_sha256":pins[(stream,"deployment_q4")]})
                recent_parts.append(rows)
                del bundle
            del frames
            gc.collect()
        print("EVALUATE_RECENT", flush=True)
        recent = evaluation.evaluate_predictions(pd.concat(recent_parts, ignore_index=True))
    write_json(report/"recent_evaluation.json", recent)
    historical_table = compact_table(historical, "oos_2026", snapshot["running_policy"]["order"])
    recent_table = compact_table(recent, "deployment_q4", snapshot["running_policy"]["order"])
    table = {"historical":historical_table,"recent":recent_table,"assumptions":prereg["metrics"],"account_returns":False}
    write_json(report/"comparison_table.json", table)
    pd.DataFrame(historical_table).to_csv(report/"historical_seven_stream_comparison.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(recent_table).to_csv(report/"recent_seven_stream_comparison.csv",index=False,encoding="utf-8-sig")
    assessment = research_screen(historical, recent, historical_table)
    baseline_reference_checks = verify_baseline(snapshot, run)
    if digest(report/"preregistration.json") != prereg_receipt["sha256"]:
        raise ValueError("preregistration changed during evaluation")
    assessment.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      preregistration_sha256=digest(report/"preregistration.json"),
                      evaluation_script_sha256=digest(__file__),
                      evaluation_module_sha256=digest(evaluation.__file__),
                      recent_prediction_receipts=recent_receipts,
                      baseline_policy_and_artifacts_unchanged=True,
                      baseline_reference_checks=baseline_reference_checks,
                      historical_evaluation_sha256=digest(report/"historical_evaluation.json"),
                      recent_evaluation_sha256=digest(report/"recent_evaluation.json"),
                      software_checks="PENDING_FINAL_WORKSPACE_VALIDATION")
    write_json(report/"research_assessment.json", assessment)
    print(json.dumps({"status":assessment["status"],"failed_conditions":assessment["failed_conditions"],
                      "historical_vs_current":find_comparison(historical,"CURRENT_FROZEN")["point_estimate"],
                      "historical_vs_reset":find_comparison(historical,"CURRENT_RESET")["point_estimate"],
                      "recent_vs_current":find_comparison(recent,"CURRENT_FROZEN")["point_estimate"]},ensure_ascii=True),flush=True)


if __name__ == "__main__":
    main()
