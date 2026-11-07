"""Latency verification evidence, with continuous paper runtime left active."""
from __future__ import annotations
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "reports/argon4_mark_revision_repair_20261106"
spec = importlib.util.spec_from_file_location("repair_verification_helpers", PREVIOUS / "audit.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
read, metadata, now = repair.read, repair.metadata, repair.prior.now
RUNTIME = repair.RUNTIME


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def policy_contract():
    old = read(OUT / "policy_before.json")
    current = read(ROOT / "config/argon4_forward_policy.json")
    allowed = {"version", "previous_execution_version", "execution_change_authorized_at", "execution_change_note", "input_http_min_spacing_seconds"}
    changed = {key for key in set(old) | set(current) if old.get(key) != current.get(key)}
    good = (changed == allowed and current.get("input_http_min_spacing_seconds") == .25
        and current["previous_execution_version"] == old["version"] and current["version"].endswith("-v8")
        and all(old.get(key) == current.get(key) for key in (set(old) | set(current)) - allowed)
        and current["live_enabled"] is False)
    return {"valid": good, "allowed_changed_fields": sorted(allowed), "actual_changed_fields": sorted(changed),
        "policy_sha256": metadata(ROOT / "config/argon4_forward_policy.json")["sha256"],
        "original_policy_sha256": metadata(OUT / "policy_before.json")["sha256"],
        "risk_limits": {key: current[key] for key in repair.RISK_KEYS}, "order": current["order"],
        "execution_stage": current["execution_stage"], "min_decision_delay_seconds": current.get("min_decision_delay_seconds"),
        "input_revision_policy": current["input_revision_policy"], "deployment_manifest_pins": current["deployment_manifest_sha256"]}


def source_list():
    paths = set(read(OUT / "source_list_prepared.json")["source"])
    for folder, pattern in [("tools/argon4_forward", "*.py"), ("scripts", "*argon4_forward*"),
                            ("tests", "test_argon4*.py"), ("docs", "ARGON4*.md")]:
        paths.update(path.relative_to(ROOT).as_posix() for path in (ROOT / folder).glob(pattern) if path.is_file())
    return sorted(paths)


# Only redirect the imported report helper's output and report contract. No
# production module, source file, live account or runtime configuration changes.
repair.OUT = OUT
repair.policy_contract = policy_contract
repair.current_source_list = source_list


def snapshot(name):
    repair.snapshot(name)


def connect(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("BEGIN")
    return db


def collect_cycles(name="live_latency_observations"):
    baseline = read(OUT / "baseline_summary.json")
    policy = read(ROOT / "config/argon4_forward_policy.json")
    db = connect(RUNTIME / "observations.sqlite3")
    try:
        rows = [dict(row) for row in db.execute("SELECT id,kind,saved_at,payload FROM snapshots WHERE saved_at>=? AND kind IN ('cycle_latency','model_inputs','prediction_decisions','paper_execution','execution_clock') ORDER BY id", (baseline["completed_at_utc"],))]
    finally:
        db.close()
    for row in rows:
        row["payload"] = json.loads(row["payload"])
    receipt_db = connect(RUNTIME / "features/http_receipts.sqlite3")
    try:
        receipts = [dict(row) for row in receipt_db.execute("SELECT id,url,requested_at,received_at,sha256,http_status,time_basis,clock_evidence_sha256 FROM receipts WHERE requested_at>=? ORDER BY id", (baseline["completed_at_utc"],))]
    finally:
        receipt_db.close()
    cycles = []
    for row in rows:
        timing = row["payload"]
        if row["kind"] != "cycle_latency" or timing.get("policy_version") != policy["version"]:
            continue
        cohort = repair.stamp(timing["cohort"])
        own = [value for value in receipts if cohort <= repair.stamp(value["requested_at"]) < cohort + repair.timedelta(minutes=5)]
        inputs = [value for value in rows if value["kind"] == "model_inputs" and value["payload"].get("requested_cutoff") and cohort <= repair.stamp(value["payload"]["requested_cutoff"]) < cohort + repair.timedelta(minutes=5)]
        preds = [value for value in rows if value["kind"] == "prediction_decisions" and value["payload"].get("cohort") == timing["cohort"]]
        executions = [value for value in rows if value["kind"] == "paper_execution" and value["payload"].get("cohort") == timing["cohort"]]
        actual_hourly_requests = any("interval=1h" in value["url"] for value in own)
        ready_processed = (timing.get("status") == "PROCESSED" and timing.get("completed_after_boundary_seconds", 999) < 20
            and inputs and inputs[-1]["payload"].get("status") == "READY" and preds and executions
            and executions[-1]["payload"].get("result", {}).get("status") == "PROCESSED")
        success = cohort.minute == 0 and actual_hourly_requests and len(own) >= 17 and ready_processed
        cycles.append({"observation_id": row["id"], "saved_at": row["saved_at"], "timing": timing,
            "http_request_count": len(own), "http_receipts": own, "input_snapshots": inputs,
            "prediction_snapshots": preds, "execution_snapshots": executions,
            "real_ready_processed_window_completed_within_20_seconds": bool(ready_processed),
            "real_hourly_window_completed_within_20_seconds": bool(success)})
    activated = read(OUT / "policy_activation.json")["activated_at_utc"]
    clock_failures = [row for row in rows if row["kind"] == "execution_clock" and row["saved_at"] >= activated
        and row["payload"].get("status") not in ("CALIBRATED", "VERIFIED")]
    result = {"captured_at_utc": now(), "policy_version": policy["version"], "cycles": cycles,
        "persisted_clock_failures_since_activation": clock_failures,
        "successful_real_hourly_cohorts": [value["timing"]["cohort"] for value in cycles if value["real_hourly_window_completed_within_20_seconds"]],
        "hourly_window_requirement_met": any(value["real_hourly_window_completed_within_20_seconds"] for value in cycles),
        "successful_real_ready_processed_cohorts": [value["timing"]["cohort"] for value in cycles if value["real_ready_processed_window_completed_within_20_seconds"]],
        "ready_processed_window_requirement_met": any(value["real_ready_processed_window_completed_within_20_seconds"] for value in cycles),
        "acceptance_scope_update": "Root clarified during the 13:05 provider backoff: at least one real future READY/PROCESSED window under 20 seconds is required; a successful 14:00 full hourly cycle is supplementary and alone does not justify an extra wait. The 13:00 hourly collection measurement remains a failed-input whole-cycle observation, not a completed execution.",
        "baseline_path": str(OUT / "timing_baseline.json"),
        "limitations": "Real runtime observations only, not synthetic benchmark timing. Missing cycles remain missing. Network/funding/four-hour requests are not imputed. Provider and local clock bounds remain in saved evidence. Intermittent network/calibration failures persist, so no claim that every future window will complete. A null state.timing without a pending cycle can be a retained consequence of an earlier clock failure; it is not independently evidence of a new decision-window failure."}
    write(name + ".json", result)
    print(json.dumps({"cycles": len(cycles), "successful_real_hourly_cohorts": result["successful_real_hourly_cohorts"]}))
    return result


def finalize():
    checks_dir = OUT / "workspace_checks/final"
    first = read(OUT / "final_chain_start_snapshot.json")
    ready = read(OUT / "final_pytest_start_snapshot.json")
    end_path = OUT / ("final_after_focused_snapshot.json" if (OUT / "final_after_focused_snapshot.json").exists() else "final_pytest_end_snapshot.json")
    end = read(end_path)
    old_summary = read(PREVIOUS / "summary.json")
    old_tests = old_summary["pytest"]["current"]
    tests = repair.prior.parse_pytest(checks_dir / "05_pytest_full.log")
    historic = set(old_summary["pytest"]["baseline"]["failed"])
    known_environment = set(old_summary["pytest"]["comparison"]["new_failures"])
    comparison = {"new_failures_vs_previous_full": sorted(set(tests["failed"]) - set(old_tests["failed"])),
        "no_longer_failing_vs_previous_full": sorted(set(old_tests["failed"]) - set(tests["failed"])),
        "historic_17_failed_nodes": sorted(historic & set(tests["failed"])),
        "known_environment_failed_nodes": sorted(known_environment & set(tests["failed"])),
        "other_failed_nodes": sorted(set(tests["failed"]) - historic - known_environment),
        "passed_count_change_vs_previous_full": tests["counts"].get("passed", 0) - old_tests["counts"].get("passed", 0),
        "historic_failure_comparison": [{"node": node, "baseline": "FAILED", "current": "FAILED" if node in tests["failed"] else "NOT_IN_FAILURE_LIST"} for node in sorted(historic)]}
    integrity = {}
    for group in ("source", "models", "authority"):
        base = ready if group == "authority" else first
        changes = {key: [path for path, value in base[group].items() if end[group].get(path, {}).get(key) != value[key]] for key in ("sha256", "mtime_ns")}
        additions = sorted(set(end[group]) - set(base[group]))
        integrity[group] = {"count": len(end[group]), "start_count": len(base[group]), "changes": changes,
            "added_paths": additions, "stable": not any(changes.values()) and not additions}
    original_models = read(OUT / "production_models_before.json")
    models_preserved = all(end["models"][path]["sha256"] == row["sha256"] for path,row in original_models.items())
    registry_before, registry_after = read(OUT / "mark_registry_before.json"), end["dynamic_runtime"]["mark_registry"]
    refs = [row for row in registry_after["research_references"] if row.get("dataset_id") in repair.ONLINE_MARK_IDS]
    registry_valid = (repair.registry_core(registry_before) == repair.registry_core(registry_after)
        and {row["dataset_id"] for row in refs} == repair.ONLINE_MARK_IDS
        and all(row.get("installed") is False and row.get("research_only") is True and row.get("gate_effect") == "NONE" for row in refs))
    ledger = repair.account_audit(read(OUT / "account_before.json"), end["account"], read(ROOT / "config/argon4_forward_policy.json"))
    revisions = read(OUT / "runtime_revision_audit.json")
    cutovers = []
    for revision in revisions["audits"]:
        effective = repair.stamp(revision["view_effective_at"])
        skipped = effective.replace(minute=effective.minute // 5 * 5, second=0, microsecond=0).isoformat()
        cutovers.append({"revision_id": revision["revision_id"], "view_effective_at": revision["view_effective_at"],
            "skipped_cohort": skipped, "absent_from_new_ledger_cohorts": skipped not in ledger["new_cohorts"]})
    ledger["revision_cutovers"] = cutovers
    if not all(row["absent_from_new_ledger_cohorts"] for row in cutovers):
        ledger["valid"] = False
        ledger["issues"].append("revision_cutover_cohort_was_not_skipped")
    stages = [{"stage": stage, "exit_code": int((checks_dir / (stage + ".exit.txt")).read_text(encoding="utf-8-sig").strip()),
        "log": str(checks_dir / (stage + ".log")), "log_sha256": metadata(checks_dir / (stage + ".log"))["sha256"]}
        for stage in ("01_fetch_verify", "02_import", "03_inventory", "04_reports", "05_pytest_full")]
    live = collect_cycles()
    result = {"status": "COMPLETED_PENDING_REVIEW", "completed_at_utc": now(), "stages": stages,
        "final_snapshot": str(end_path), "final_snapshot_captured_at_utc": end["captured_at_utc"],
        "pytest": {"previous_full": old_tests, "current": tests, "comparison": comparison}, "integrity": integrity,
        "policy_contract": end["policy_contract"], "original_42_models_preserved": models_preserved,
        "dynamic_registry_valid": registry_valid, "canonical_and_non_online_registry_content_unchanged": repair.registry_core(registry_before) == repair.registry_core(registry_after),
        "account_audit": ledger, "real_latency_path": str(OUT / "live_latency_observations.json"),
        "runtime_revision_audit": {"path": str(OUT / "runtime_revision_audit.json"),
            "sha256": metadata(OUT / "runtime_revision_audit.json")["sha256"], "status": revisions["status"],
            "revision_ids": [row["revision_id"] for row in revisions["audits"]]},
        "real_latency_sha256": metadata(OUT / "live_latency_observations.json")["sha256"],
        "hourly_window_requirement_met": live["hourly_window_requirement_met"], "successful_real_hourly_cohorts": live["successful_real_hourly_cohorts"],
        "ready_processed_window_requirement_met": live["ready_processed_window_requirement_met"],
        "successful_real_ready_processed_cohorts": live["successful_real_ready_processed_cohorts"],
        "aborted_prior_run": {"path": str(OUT / "aborted_full_run.json"), "sha256": metadata(OUT / "aborted_full_run.json")["sha256"], "not_a_completed_test_result": True},
        "runtime_final": end["dynamic_runtime"], "full_suite_green": not tests["failed"] and not tests["errors"],
        "source_verification": "54 inherited source identities remain unverified; no source/conversion/readiness gate relaxed.",
        "limitations": "Full-suite raw counts are retained. Runtime authority updates may cause existing closure tests to reject; precise unchanged-node rechecks are reported separately. No profitability/live-trading certification."}
    result["protected_files_policy_and_account_valid"] = all(value["stable"] for value in integrity.values()) and models_preserved and registry_valid and ledger["valid"] and end["policy_contract"]["valid"] and revisions["status"] == "PASS"
    focused_path = checks_dir / "focused_rerun_summary.json"
    focused_valid = not comparison["known_environment_failed_nodes"]
    if focused_path.exists():
        focused = read(focused_path)
        focused_valid = (focused["status"] == "COMPLETED"
            and {row["node"] for row in focused["results"]} == set(comparison["known_environment_failed_nodes"])
            and all(row["exit_code"] == 0 and not row["authority_changes"]
                and metadata(Path(row["log"]))["sha256"] == row["log_sha256"] for row in focused["results"]))
        result["focused_environment_followup"] = {"path": str(focused_path), "sha256": metadata(focused_path)["sha256"],
            "valid": focused_valid, "results": focused["results"],
            "reason": "The same four historical research tests snapshot the entire mark registry at import. Authorized online research-reference updates change that hash during the full run. Each exact unchanged node was rerun in a fresh process after the real revision commit; no source, fixture, guard or registry was altered."}
    successful = [row["timing"] for row in live["cycles"] if row["real_ready_processed_window_completed_within_20_seconds"]]
    result["observed_successful_window_performance"] = {"sample_count": len(successful),
        "cohorts": [row["cohort"] for row in successful],
        "completed_after_boundary_seconds": [row["completed_after_boundary_seconds"] for row in successful],
        "input_refresh_seconds": [row.get("input_refresh_seconds") for row in successful],
        "not_completed_cycles_retained_in_live_report": [row["timing"]["cohort"] for row in live["cycles"] if not row["real_ready_processed_window_completed_within_20_seconds"]],
        "scope": "Observed completed windows, not a guarantee or a replacement for failed/blocked-window evidence. Network/calibration and provider revisions remain limitations."}
    result["latency_and_deadline_validation_passed"] = (result["protected_files_policy_and_account_valid"]
        and live["ready_processed_window_requirement_met"] and focused_valid and not comparison["other_failed_nodes"]
        and not tests["errors"] and tests["summary"] is not None and [row["exit_code"] for row in stages[:4]] == [1, 3, 0, 0])
    if result["latency_and_deadline_validation_passed"]:
        result["status"] = "LATENCY_AND_DEADLINE_CHECKS_PASSED_FULL_SUITE_NOT_GREEN"
    result["raw_full_suite_result_retained"] = True
    write("account_audit.json", ledger)
    write("failure_comparison.json", result["pytest"])
    write("summary.json", result)
    print(json.dumps({"status": result["status"], "pytest": tests["summary"], "comparison": comparison,
        "protected_files_policy_and_account_valid": result["protected_files_policy_and_account_valid"],
        "hourly_window_requirement_met": result["hourly_window_requirement_met"], "account_issues": ledger["issues"]}))


if __name__ == "__main__":
    if sys.argv[1] == "finalize":
        finalize()
    elif sys.argv[1] == "collect_cycles":
        collect_cycles()
    else:
        snapshot(sys.argv[1])
