"""Attach transparent focused recheck evidence without replacing full results."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    focused = read(OUT / "focused_rerun_summary.json")
    neutral_path = OUT / "longcycle_neutral_retest_3a9d5dff46b746fab3c57faed2cbac77.json"
    neutral = read(neutral_path)
    cases = []
    for row in focused["results"]:
        log = Path(row["log"])
        assert row["exit_code"] == 0 and "1 passed" in log.read_text(encoding="utf-8-sig")
        assert sha(log) == row["log_sha256"] and not row["authority_changes"]
        cases.append({"node": row["node"], "classification": "EXPECTED_GUARD_TRIGGERED_BY_CONCURRENT_DYNAMIC_RESEARCH_REGISTRY_UPDATES",
            "full_run_error": "authoritative registry/manifest/binding closure changed since backend import",
            "guard_location": "tools/perptools/research_pipeline_v1.py:117",
            "guard_scope_location": "tools/perptools/research_pipeline_v1.py:95",
            "explanation": "The module captures the entire mark registry hash at import. During the long full suite, the authorized online observer updates four research-only references. All three other authority files and all non-online registry content remained unchanged. The code-closure guard did not fail. Each exact original node passed in a fresh process with all authority hashes unchanged across that run.",
            "focused_retest_exit_code": row["exit_code"], "focused_retest_log": str(log), "focused_retest_log_sha256": row["log_sha256"],
            "fresh_process": True, "authority_before": row["before_authority"], "authority_after": row["after_authority"],
            "business_code_fixture_guards_and_registry_not_modified": True})
    neutral_log = ROOT / neutral["log_path"]
    assert neutral["exit_code"] == 0 and neutral["source_bytes_and_mtime_unchanged"]
    assert sha(neutral_log) == neutral["log_sha256"] and "1 passed" in neutral_log.read_text(encoding="utf-8-sig")
    cases.append({"node": neutral["node"], "classification": "PREEXISTING_TEST_FIXTURE_PARENT_PATH_SUBSTRING_DEPENDENCE",
        "full_run_error": "KeyError: ['base_volume'] not in index", "fixture_location": "tests/test_longcycle_signal_cases_luna_v1.py:173",
        "explanation": "The fake parquet reader identifies mark bars via 'mark' in str(path), including every ancestor directory. This run's basetemp is inside argon4_mark_revision_repair_20261106, so even a trade_1h filename is misclassified and base_volume is removed. The exact unchanged test passed with a fresh neutral temporary directory; only the temporary path changed.",
        "original_basetemp": neutral["original_basetemp"], "focused_retest_exit_code": neutral["exit_code"],
        "focused_retest_log": str(neutral_log), "focused_retest_log_sha256": neutral["log_sha256"],
        "independent_agent_receipt": str(neutral_path), "independent_agent_receipt_sha256": sha(neutral_path),
        "business_code_fixture_guards_and_registry_not_modified": True, "source_before": neutral["source_before"], "source_after": neutral["source_after"]})
    raw = (OUT / "05_pytest_full.log").read_text(encoding="utf-8-sig")
    sections = re.split(r"(?m)^_+ (test_[^\r\n]*?) _+\s*$", raw)
    names = {row["node"].split("::")[-1] for row in cases}
    traces = [sections[index] + "\n" + sections[index + 1] for index in range(1, len(sections) - 1, 2) if sections[index] in names]
    assert len(traces) == 5
    trace_path = OUT / "new_failure_traces.txt"
    trace_path.write_text("\n\n".join(traces), encoding="utf-8")
    value = {"status": "FIVE_NEW_FAILURES_EXPLAINED_AND_EXACT_NODES_PASSED_UNCHANGED",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(), "full_suite_original_result": "22 failed, 5747 passed, 3 skipped, 277 warnings in 2365.93s (0:39:25)",
        "full_suite_log": str(OUT / "05_pytest_full.log"), "full_suite_log_sha256": sha(OUT / "05_pytest_full.log"),
        "full_suite_result_not_replaced": True, "inherited_failure_count": 17, "focused_node_count": len(cases), "cases": cases,
        "trace_file": str(trace_path), "trace_sha256": sha(trace_path),
        "focused_four_receipt": str(OUT / "focused_rerun_summary.json"), "focused_four_receipt_sha256": sha(OUT / "focused_rerun_summary.json"),
        "source_changes": "No business code, fixture, registry guard, model, policy or runtime service was altered for these rechecks.",
        "limitations": "The full repository remains non-green. The 17 inherited failures were not fixed. Future full-suite runs with active online writers need explicitly isolated static research fixtures; the existing guard should not be weakened. The path-sensitive fake reader remains unchanged."}
    (OUT / "new_failure_diagnosis.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "cases": len(cases), "traces": len(traces)}))


if __name__ == "__main__":
    main()
