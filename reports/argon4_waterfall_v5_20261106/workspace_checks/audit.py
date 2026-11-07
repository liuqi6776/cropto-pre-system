"""Read-only integrity checks; writes only this validation report directory."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REPORT = OUT.parent
PRIOR = ROOT / "reports/argon4_local_v4_20261106/workspace_checks/final"
AUTHORITY = ["archive/catalog/dataset_registry.json", "archive/catalog/mark_candle_registry.json",
             "source_data/MANIFEST.json", "source_data/staging/core4_p0_20260925/partition_bindings_v1.json"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def metadata(path):
    stat = path.stat()
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mtime_ns": stat.st_mtime_ns,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), "bytes": stat.st_size}


def numeric(value, prefix=""):
    if isinstance(value, dict):
        return {key: number for name, item in value.items() for key, number in numeric(item, prefix + "/" + name).items()}
    if isinstance(value, list):
        return {key: number for index, item in enumerate(value) for key, number in numeric(item, prefix + "/" + str(index)).items()}
    return {prefix: value} if isinstance(value, (float, int)) and not isinstance(value, bool) else {}


def policy_audit():
    before = read(REPORT / "pre_change_policy.json")
    current = read(ROOT / "config/argon4_forward_policy.json")
    old_numbers, new_numbers = numeric(before), numeric(current)
    return {"old_version": before["version"], "current_version": current["version"],
            "waterfall_semantics": current.get("waterfall_semantics"),
            "order_unchanged": before["order"] == current["order"], "order": current["order"],
            "deployment_manifest_pins_unchanged": before["deployment_manifest_sha256"] == current["deployment_manifest_sha256"],
            "changed_existing_numeric_paths": [key for key, value in old_numbers.items() if new_numbers.get(key) != value],
            "added_numeric_paths": [key for key in new_numbers if key not in old_numbers],
            "live_enabled": current.get("live_enabled"), "paper_trading_active": current.get("paper_trading_active")}


def historical_account():
    def connect(path):
        con = sqlite3.connect("file:" + path.resolve().as_posix() + "?mode=ro", uri=True)
        con.execute("BEGIN")
        return con
    before = connect(REPORT / "pre_change_account.sqlite3")
    current = connect(ROOT / "paper_runtime/argon4_forward_20261105/account.sqlite3")
    tables = [row[0] for row in before.execute("SELECT name FROM sqlite_master WHERE type='table'")
              if row[0] not in ("runtime", "sqlite_sequence")]
    results = {}
    try:
        for table in tables:
            columns = [row[1] for row in before.execute('PRAGMA table_info("' + table + '")')]
            select = 'SELECT ' + ','.join('"' + column + '"' for column in columns) + ' FROM "' + table + '"'
            def serialized(con):
                return sorted(json.dumps(list(row), ensure_ascii=False, separators=(",", ":")) for row in con.execute(select))
            old_rows, new_rows = serialized(before), serialized(current)
            digest = lambda rows: hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            results[table] = {"backup_rows": len(old_rows), "current_rows": len(new_rows),
                              "identical_old_columns": old_rows == new_rows,
                              "backup_sha256": digest(old_rows), "current_sha256": digest(new_rows)}
    finally:
        before.close()
        current.close()
    return {"comparison": "ALL_OLD_COLUMNS_EXACT_ROWS_WITHIN_READ_ONLY_TRANSACTION",
            "excluded_mutable_heartbeat_table": "runtime", "tables": results,
            "all_historical_rows_unchanged": all(row["identical_old_columns"] for row in results.values())}


def snapshot(name):
    previous = read(PRIOR / "pytest_start_snapshot.json")
    sources = sorted(set(previous["source"]) | {"scripts/start_argon4_forward.ps1", "scripts/stop_argon4_forward.ps1",
             "docs/ARGON4_WATERFALL_V5_20261106.md", "docs/ARGON4_LOCAL_MODEL_CHANGE_20261106.md"})
    initial = read(REPORT / "before_snapshot.json")["hashes"]
    model_paths = sorted(Path(path).as_posix() for path in initial if path.startswith("research"))
    unchanged = {Path(path).as_posix(): value for path, value in initial.items()
                 if path.startswith("research") or Path(path).as_posix() == "tools/argon4_forward/model.py"}
    result = {"recorded_at": now(),
              "source": {path: metadata(ROOT / path) for path in sources},
              "models": {path: metadata(ROOT / path) for path in model_paths},
              "authority": {path: metadata(ROOT / path) for path in AUTHORITY},
              "model_and_training_code_changes_from_pre_change": [path for path, sha in unchanged.items() if metadata(ROOT / path)["sha256"] != sha],
              "policy_audit": policy_audit(), "historical_account": historical_account(),
              "verification_pause": metadata(ROOT / "paper_runtime/argon4_forward_20261105/verification_pause.json")}
    write(name + ".json", result)
    print(json.dumps({"snapshot": name, "counts": {key: len(result[key]) for key in ("source", "models", "authority")},
                      "unchanged_model_code": not result["model_and_training_code_changes_from_pre_change"],
                      "unchanged_account": result["historical_account"]["all_historical_rows_unchanged"],
                      "policy": result["policy_audit"]}, ensure_ascii=False))


def parse_pytest(path):
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    summaries = [line.strip("= \r") for line in content.splitlines()
                 if re.search(r"\d+ passed", line) and re.search(r" in \d+\.\d+s", line)]
    summary = summaries[-1] if summaries else None
    return {"log": path.relative_to(ROOT).as_posix(), "summary": summary,
            "counts": {key: int(value) for value, key in re.findall(r"(\d+) (failed|passed|skipped|warnings|errors)", summary or "")},
            "failed": sorted(set(re.findall(r"^FAILED (\S+)", content, re.M))),
            "errors": sorted(set(re.findall(r"^ERROR (\S+)", content, re.M)))}


def finalize():
    first, start, end = (read(OUT / (name + ".json")) for name in ("chain_start_snapshot", "pytest_start_snapshot", "pytest_end_snapshot"))
    baseline = parse_pytest(PRIOR / "05_pytest_full.log")
    original = parse_pytest(ROOT / "reports/argon4_forward_20261106_v2/pytest_full.log")
    current = parse_pytest(OUT / "05_pytest_full.log")
    comparisons = {name: {"new_failures": sorted(set(current["failed"]) - set(prior["failed"])),
                         "no_longer_failing": sorted(set(prior["failed"]) - set(current["failed"])),
                         "passed_count_change": current["counts"].get("passed", 0) - prior["counts"].get("passed", 0)}
                   for name, prior in (("last_final", baseline), ("original", original))}
    integrity = {}
    for group in ("source", "models", "authority"):
        base = start if group == "authority" else first
        changes = {kind + "_changes": [path for path, value in base[group].items() if end[group].get(path, {}).get(kind) != value[kind]]
                   for kind in ("sha256", "mtime_ns")}
        integrity[group] = {"count": len(end[group]), **changes, "stable": not any(changes.values())}
    stage_specs = [("01_fetch_verify", "fetch_local_source_copies --verify"), ("02_import", "import_perp_data"),
                   ("03_inventory", "build_inventory"), ("04_reports", "build_reports"), ("05_pytest_full", "pytest tests -q")]
    stages = [{"order": index, "stage": label, "exit_code": int((OUT / (stem + ".exit.txt")).read_text(encoding="utf-8-sig").strip()),
               "log": stem + ".log", "sha256": metadata(OUT / (stem + ".log"))["sha256"]}
              for index, (stem, label) in enumerate(stage_specs, 1)]
    policy = end["policy_audit"]
    may_resume = (all(item["stable"] for item in integrity.values())
                  and not end["model_and_training_code_changes_from_pre_change"]
                  and end["historical_account"]["all_historical_rows_unchanged"]
                  and policy["order_unchanged"] and policy["deployment_manifest_pins_unchanged"]
                  and not policy["changed_existing_numeric_paths"] and policy["live_enabled"] is False
                  and not comparisons["last_final"]["new_failures"] and not current["errors"]
                  and current["summary"] is not None and first["verification_pause"] == end["verification_pause"])
    summary = {"completed_at_utc": now(), "scope": "WATERFALL_V5_FINAL_FROZEN_SOURCE_VALIDATION",
               "status": "COMPLETED_WITH_INHERITED_FAILURES" if may_resume else "REQUIRES_REVIEW",
               "preflight_stages": stages[:4], "pytest_stage": stages[4],
               "pytest": {"runs": {"original": original, "last_final": baseline, "waterfall_v5": current}, "comparisons": comparisons,
                          "failure_set_identical_to_last_final": current["failed"] == baseline["failed"]},
               "integrity": integrity, "policy_audit": policy,
               "model_and_training_code_changes_from_pre_change": end["model_and_training_code_changes_from_pre_change"],
               "historical_account": end["historical_account"],
               "required_generation_authority_changes": [path for path in AUTHORITY if first["authority"][path] != start["authority"][path]],
               "verification_pause_unchanged": first["verification_pause"] == end["verification_pause"],
               "source_verification": "54 inherited unverified source identities remain; conversion refuses them. Canonical readiness gate is not promoted.",
               "limitations": "Software consistency verification, not strategy effectiveness or authorization for real trading. Existing test failures are retained and listed.",
               "runtime_pause_may_be_removed": may_resume,
               "snapshots": ["chain_start_snapshot.json", "pytest_start_snapshot.json", "pytest_end_snapshot.json"],
               "changes_by_check_agent": "Required generated manifests/inventory/reports and verification artifacts only. No business source, policy, model, UI, or live account edits."}
    write("failure_comparison.json", summary["pytest"])
    write("summary.json", summary)
    print(json.dumps({"summary": str(OUT / "summary.json"), "pytest": current["summary"], "integrity": integrity,
                      "new_failures": comparisons["last_final"]["new_failures"], "resume": may_resume}, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "finalize":
        finalize()
    else:
        snapshot(sys.argv[1])
