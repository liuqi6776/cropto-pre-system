"""Extract historical observed timing only; no collection or model calls."""
from __future__ import annotations
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNTIME = ROOT / "paper_runtime/argon4_forward_20261105"
START = datetime(2026, 11, 6, 11, 5, tzinfo=timezone.utc)


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def cohort(value):
    value = utc(value) if isinstance(value, str) else value
    return value.replace(minute=value.minute // 5 * 5, second=0, microsecond=0).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def connection(path):
    result = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    result.row_factory = sqlite3.Row
    result.execute("BEGIN")
    return result


def main():
    baseline = read(OUT / "baseline_summary.json")
    end = utc(baseline["account_captured_at_utc"])
    account = read(OUT / "account_before.json")
    with connection(RUNTIME / "observations.sqlite3") as db:
        rows = [dict(row) for row in db.execute("SELECT id,kind,saved_at,payload FROM snapshots WHERE saved_at>=? AND saved_at<=? AND kind IN ('model_inputs','prediction_decisions','paper_execution','execution_clock') ORDER BY id", (START.isoformat(), end.isoformat()))]
    for row in rows:
        row["payload"] = json.loads(row["payload"])
    write("timing_observation_snapshots_before.json", rows)
    with connection(RUNTIME / "features/http_receipts.sqlite3") as db:
        receipts = [dict(row) for row in db.execute("SELECT id,url,requested_at,received_at,sha256,http_status,requested_at_local,received_at_local,time_basis,clock_evidence_sha256 FROM receipts WHERE requested_at>=? AND requested_at<=? ORDER BY id", (START.isoformat(), end.isoformat()))]
    for row in receipts:
        parts = urlsplit(row["url"])
        query = parse_qs(parts.query)
        row["endpoint"] = parts.path
        row["symbol"] = query.get("symbol", [None])[0]
        row["interval"] = query.get("interval", [None])[0]
        first, last = row["requested_at_local"] or row["requested_at"], row["received_at_local"] or row["received_at"]
        row["observed_wall_request_seconds"] = (utc(last) - utc(first)).total_seconds()
    write("timing_http_receipts_before.json", receipts)
    input_by, predictions, executions = {}, {}, {}
    for row in rows:
        value = row["payload"]
        if row["kind"] == "model_inputs":
            key = cohort(value.get("requested_cutoff") or value.get("as_of") or row["saved_at"])
            input_by.setdefault(key, []).append(row)
        elif row["kind"] == "prediction_decisions":
            predictions.setdefault(value["cohort"], []).append(row)
        elif row["kind"] == "paper_execution":
            executions.setdefault(value["cohort"], []).append(row)
    windows = []
    point = START
    ledger_cohorts = {row["cohort_id"] for row in account["tables"]["decision_cohorts"]}
    while point <= utc(cohort(end)):
        key = point.isoformat()
        inputs, preds, executions_own = input_by.get(key, []), predictions.get(key, []), executions.get(key, [])
        selected = inputs[-1] if inputs else None
        input_value = selected["payload"] if selected else {}
        own_receipts = [row for row in receipts if cohort(row["requested_at"]) == key]
        pred, execution = preds[-1] if preds else None, executions_own[-1] if executions_own else None
        ready_at = input_value.get("refresh_completed_at")
        ready_delay = (utc(ready_at) - point).total_seconds() if ready_at else None
        request_at = input_value.get("requested_cutoff")
        execution_at = execution["payload"].get("completed_at") if execution else None
        result = execution["payload"].get("result", {}) if execution else {}
        status = "EXECUTION_RECORDED" if execution else ("INPUT_LATE_NO_EXECUTION" if ready_delay is not None and ready_delay > 20 else "NO_EXECUTION_RECORD")
        decisions = {stream: {"status": value.get("status"), "decision_time": value.get("prediction", {}).get("decision_time"),
            "model_evaluated": "prediction" in value, "selected": value.get("prediction", {}).get("selected")}
            for stream, value in (pred["payload"]["decisions"].items() if pred else [])}
        endpoint_counts = {}
        for receipt in own_receipts:
            label = receipt["endpoint"] + (":" + receipt["interval"] if receipt["interval"] else "")
            endpoint_counts[label] = endpoint_counts.get(label, 0) + 1
        windows.append({"cohort": key, "observed_status": status, "input_snapshot_ids": [row["id"] for row in inputs],
            "input_status": input_value.get("status"), "input_requested_at": request_at, "input_ready_at": ready_at,
            "input_start_seconds_after_boundary": (utc(request_at) - point).total_seconds() if request_at else None,
            "input_ready_seconds_after_boundary": ready_delay,
            "refresh_wall_seconds_calibrated_timestamps": (utc(ready_at) - utc(request_at)).total_seconds() if ready_at and request_at else None,
            "input_errors": input_value.get("errors"), "input_gaps": input_value.get("gaps"),
            "latest_closed_5m": input_value.get("latest_closed_5m"), "input_clock_sha256": input_value.get("clock_evidence_sha256"),
            "prediction_snapshot_ids": [row["id"] for row in preds], "decisions": decisions,
            "input_save_to_prediction_save_local_wall_seconds": (utc(pred["saved_at"]) - utc(selected["saved_at"])).total_seconds() if pred and selected else None,
            "execution_snapshot_ids": [row["id"] for row in executions_own], "execution_completed_at": execution_at,
            "execution_completed_seconds_after_boundary": (utc(execution_at) - point).total_seconds() if execution_at else None,
            "execution_result_status": result.get("status"), "execution_blocked": result.get("blocked"), "execution_fills": result.get("fills"),
            "ledger_cohort_exists": key in ledger_cohorts,
            "prediction_save_to_execution_save_local_wall_seconds": (utc(execution["saved_at"]) - utc(pred["saved_at"])).total_seconds() if execution and pred else None,
            "http_receipt_ids": [row["id"] for row in own_receipts], "http_request_count": len(own_receipts),
            "http_endpoint_counts": endpoint_counts,
            "http_serial_request_wall_seconds_sum": sum(row["observed_wall_request_seconds"] for row in own_receipts)})
        point += timedelta(minutes=5)
    values = sorted(row["input_ready_seconds_after_boundary"] for row in windows if row["input_ready_seconds_after_boundary"] is not None)
    stats = {"min": values[0], "median": (values[(len(values)-1)//2] + values[len(values)//2])/2, "p95_nearest_rank": values[math.ceil(.95*len(values))-1], "max": values[-1]} if values else {}
    result = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "baseline_cutoff_utc": end.isoformat(),
        "start_inclusive": START.isoformat(), "window_count": len(windows), "windows": windows,
        "input_ready_latency_seconds": stats, "missing_execution_windows": [row["cohort"] for row in windows if not row["execution_snapshot_ids"]],
        "late_input_windows": [row["cohort"] for row in windows if row["input_ready_seconds_after_boundary"] is not None and row["input_ready_seconds_after_boundary"] > 20],
        "limitations": "Only persisted observations are reported; missing predictions do not prove they were never attempted. Calibrated timestamps retain their clock receipt IDs. Local saved_at differences are wall-clock timings, not monotonic CPU profiles. No missing data, signals, fills or timing samples are invented.",
        "sources": [str(OUT / "timing_observation_snapshots_before.json"), str(OUT / "timing_http_receipts_before.json"), str(OUT / "account_before.json")]}
    write("timing_baseline.json", result)
    print(json.dumps({key: result[key] for key in ("window_count", "input_ready_latency_seconds", "missing_execution_windows", "late_input_windows")}))


if __name__ == "__main__":
    main()
