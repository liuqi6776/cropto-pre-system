"""Client-latency validation evidence; only this report directory is writable."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REPORT = OUT.parent
PREVIOUS = ROOT / "reports/argon4_latency_20261106"
spec = importlib.util.spec_from_file_location("external_pilot_prior_audit", ROOT / "reports/argon4_mark_revision_repair_20261106/audit.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
read, metadata, now = repair.read, repair.metadata, repair.prior.now
SOURCE_ID = "argon4_pmxt_external_replay_pilot_20261106"


def write(name, value):
    path = OUT / name
    if path.exists():
        raise RuntimeError("do not overwrite verification evidence: " + name)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_list():
    paths = set(read(PREVIOUS / "final_after_focused_snapshot.json")["source"])
    for folder, pattern in [("tools/argon4_forward", "*.py"), ("scripts", "*argon4_forward*"),
                            ("tests", "test_argon4*.py"), ("docs", "ARGON4*.md")]:
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).glob(pattern) if p.is_file())
    paths.update(p.relative_to(ROOT).as_posix() for p in REPORT.rglob("*.py")
                 if "pytest_temp_" not in str(p) and "pytest_tmp_" not in str(p) and "__pycache__" not in str(p))
    return sorted(paths)


def source_registration():
    registry = read(ROOT / "archive/catalog/dataset_registry.json")
    rows = [r for r in registry["datasets"] if r.get("id") == SOURCE_ID]
    valid = len(rows) == 1 and all(rows[0].get(k) is False for k in ("convertible", "default_selected", "label_source_verified"))
    valid = valid and rows[0].get("verification_status") == "EXTERNAL_REFERENCE_ONLY" and rows[0].get("local_copy") is None
    return {"valid": bool(valid), "entry": rows[0] if len(rows) == 1 else rows,
            "canonical_conversion_authorized": False, "production_input_source": False}


def policy_contract():
    path = ROOT / "config/argon4_forward_policy.json"
    old = read(OUT / "policy_before.json")
    value = read(path)
    allowed = {"version", "previous_execution_version", "execution_change_authorized_at", "execution_change_note"}
    changed = {key for key in set(old) | set(value) if old.get(key) != value.get(key)}
    activation_pending = not changed
    activated = (changed == allowed and value.get("previous_execution_version") == old["version"]
                 and value.get("version", "").endswith("-v9"))
    unchanged_contract = all(old.get(k) == value.get(k) for k in (set(old) | set(value)) - allowed)
    return {"valid": (activation_pending or activated) and unchanged_contract and value.get("live_enabled") is False,
            "activation_pending": activation_pending, "v9_activated": activated,
            "changed_fields": sorted(changed), "allowed_changed_fields": sorted(allowed),
            "original_policy_sha256": metadata(OUT / "policy_before.json")["sha256"],
            "sha256": metadata(path)["sha256"], "non_metadata_contract_unchanged": unchanged_contract,
            "order": value["order"], "execution_stage": value["execution_stage"],
            "risk": {k: value[k] for k in repair.RISK_KEYS},
            "deployment_manifest_pins": value["deployment_manifest_sha256"], "live_enabled": value["live_enabled"]}


def snapshot(name):
    models = read(PREVIOUS / "production_models_before.json")
    value = {"captured_at_utc": now(), "source": {p: metadata(ROOT / p) for p in source_list()},
             "models": {p: metadata(ROOT / p) for p in models},
             "authority": {p: metadata(ROOT / p) for p in repair.AUTHORITY},
             "policy_contract": policy_contract(), "source_registration": source_registration(),
             "account": repair.prior.account_snapshot(), "dynamic_runtime": repair.dynamic_snapshot(),
             "scope": "No pause: observed online inputs, four research-only mark references, future paper orders and official settlements remain dynamic."}
    write(name + ".json", value)
    print(json.dumps({"snapshot": name, "source_count": len(value["source"]), "model_count": len(value["models"]),
                      "policy_valid": value["policy_contract"]["valid"], "account_counts": value["account"]["row_counts"],
                      "runtime_status": value["dynamic_runtime"]["pipeline"]["status"]}))
    return value


def prepare():
    if (OUT / "policy_before.json").exists():
        raise RuntimeError("baseline already exists")
    (OUT / "policy_before.json").write_bytes((ROOT / "reports/argon4_external_backtest_20261106/workspace_checks/policy_before.json").read_bytes())
    value = snapshot("preparation_baseline")
    prior = read(PREVIOUS / "final_after_focused_snapshot.json")
    source_changed = [p for p, m in prior["source"].items() if value["source"].get(p, {}).get("sha256") != m["sha256"]]
    summary = {"status": "PREPARED_WAITING_FOR_EXPLICIT_CLIENT_LATENCY_FREEZE",
               "captured_at_utc": value["captured_at_utc"], "source_count": len(value["source"]),
               "old_source_changed_since_previous_final": source_changed,
               "added_source_paths": sorted(set(value["source"]) - set(prior["source"])),
               "models_count": len(value["models"]),
               "model_hashes_match_previous_final": all(value["models"][p]["sha256"] == m["sha256"] for p, m in prior["models"].items()),
               "policy_unchanged_since_previous_final": value["policy_contract"]["sha256"] == prior["source"]["config/argon4_forward_policy.json"]["sha256"],
               "registration": value["source_registration"], "account_counts": value["account"]["row_counts"],
               "old_event_prefix_length": len(value["account"]["tables"]["events"]),
               "account_funds_conserved": value["account"]["funds_conserved"],
               "invalid_account_event_hashes": value["account"]["invalid_event_hashes"],
               "runtime": value["dynamic_runtime"]["pipeline"], "service": value["dynamic_runtime"]["service"],
               "verification_pause_exists": value["dynamic_runtime"]["verification_pause_exists"],
               "baseline_test_result": read(PREVIOUS / "summary.json")["pytest"]["current"],
               "full_chain_started": False, "note": "External pilot remains registered but research is paused. v9 may change exactly four metadata fields; all strategy/risk/model fields remain frozen. Refresh sources and capture chain-start at explicit freeze. Preparation performs no production mutation or control."}
    write("baseline_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "captured_at_utc", "source_count", "models_count", "old_source_changed_since_previous_final", "model_hashes_match_previous_final", "policy_unchanged_since_previous_final", "old_event_prefix_length", "account_funds_conserved", "verification_pause_exists")}))


def account_audit(before, after, policy):
    # The inherited cash-flow checker independently validates each new official
    # settlement, raw receipt, lifecycle transition and account/equity amount.
    # Its coarse equality check also flags a legitimate old OPEN order's status
    # update. Permit only that exact immutable-fields-preserved transition here.
    value = repair.account_audit(before, after, policy)
    newer = {r["order_id"]: r for r in after["tables"]["orders"]}
    status_only_valid = True
    transitions = []
    for old in before["tables"]["orders"]:
        new = newer.get(old["order_id"])
        if new is None or any(new.get(k) != v for k, v in old.items() if k != "status"):
            status_only_valid = False
        elif new["status"] != old["status"]:
            if old["status"] != "OPEN" or new["status"] not in ("WON", "LOST"):
                status_only_valid = False
            transitions.append({"order_id": old["order_id"], "before": old["status"], "after": new["status"]})
    if status_only_valid:
        value["issues"] = [x for x in value["issues"] if x != "historical_record_changed:orders"]
    value["valid"] = not value["issues"]
    value["old_order_lifecycle_changes"] = transitions
    return value


def finalize():
    start = read(OUT / "chain_start_snapshot.json")
    end_path = OUT / ("final_after_focused_snapshot.json" if (OUT / "final_after_focused_snapshot.json").exists() else "pytest_end_snapshot.json")
    end = read(end_path)
    baseline = read(OUT / "preparation_baseline.json")
    previous = read(PREVIOUS / "summary.json")
    tests = repair.prior.parse_pytest(OUT / "05_pytest_full.log")
    historical = set(previous["pytest"]["comparison"]["historic_17_failed_nodes"])
    environmental = set(previous["pytest"]["comparison"]["known_environment_failed_nodes"])
    failed = set(tests["failed"])
    comparison = {"historical_failed_nodes": sorted(failed & historical),
                  "known_dynamic_registry_failed_nodes": sorted(failed & environmental),
                  "other_failed_nodes": sorted(failed - historical - environmental),
                  "new_failures_vs_previous_full": sorted(failed - set(previous["pytest"]["current"]["failed"])),
                  "historical_17_comparison": [{"node": x, "current": "FAILED" if x in failed else "NOT_IN_FAILURE_LIST"} for x in sorted(historical)]}
    integrity = {}
    for group in ("source", "models", "authority"):
        changes = {key: [p for p, v in start[group].items() if end[group].get(p, {}).get(key) != v[key]] for key in ("sha256", "mtime_ns")}
        added = sorted(set(end[group]) - set(start[group]))
        integrity[group] = {"count": len(end[group]), "changes": changes, "added_paths": added,
                            "stable": not any(changes.values()) and not added}
    dynamic_valid = repair.registry_core(baseline["dynamic_runtime"]["mark_registry"]) == repair.registry_core(end["dynamic_runtime"]["mark_registry"])
    ledger = account_audit(baseline["account"], end["account"], read(ROOT / "config/argon4_forward_policy.json"))
    stages = [{"stage": s, "exit_code": int((OUT / (s + ".exit.txt")).read_text().strip()),
               "log": str(OUT / (s + ".log")), "log_sha256": metadata(OUT / (s + ".log"))["sha256"]}
              for s in ("01_fetch_verify", "02_import", "03_inventory", "04_reports", "05_pytest_full")]
    result = {"status": "COMPLETED_PENDING_ROOT_REVIEW", "completed_at_utc": now(), "stages": stages,
              "pytest": {"previous_full": previous["pytest"]["current"], "current": tests, "comparison": comparison},
              "integrity": integrity, "policy_contract": end["policy_contract"], "account_audit": ledger,
              "canonical_and_non_online_mark_registry_content_unchanged": dynamic_valid,
              "dynamic_mark_publication_consistent_at_final_read": end["dynamic_runtime"]["publication_consistent_at_read"],
              "source_registration": end["source_registration"], "final_snapshot": str(end_path),
              "full_suite_green": not failed and not tests["counts"].get("errors", 0),
              "limits": "Software verification and public-client latency do not prove strategy effectiveness. Existing source-label and historical execution gates are unchanged. Full raw result is never replaced by selective reruns."}
    focused = OUT / "focused_rerun_summary.json"
    if focused.exists():
        result["focused_followup"] = {"evidence": str(focused), "sha256": metadata(focused)["sha256"], "result": read(focused)}
    write("summary.json", result)
    print(json.dumps({"summary": str(OUT / "summary.json"), "counts": tests["counts"], "integrity": integrity,
                      "account_valid": ledger["valid"], "new_other_failures": comparison["other_failed_nodes"]}))


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "snapshot":
        snapshot(sys.argv[2])
    elif sys.argv[1] == "finalize":
        finalize()
    else:
        raise ValueError("unsupported audit action")
