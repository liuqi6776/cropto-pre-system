"""Read-only runtime baseline capture; writes only this report directory."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNTIME = ROOT / "paper_runtime/argon4_forward_20261105"
PRIOR = ROOT / "reports/argon4_gen4_candidate_20261106/workspace_checks"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meta(path):
    info = path.stat()
    return {"path": str(path), "sha256": sha(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def save(name, value):
    path = OUT / name
    if path.exists():
        raise FileExistsError("Baseline evidence cannot be overwritten: " + name)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_stable(source, name):
    target = OUT / name
    if target.exists():
        raise FileExistsError(name)
    for _ in range(3):
        before = meta(source)
        body = source.read_bytes()
        after = meta(source)
        if before == after and hashlib.sha256(body).hexdigest() == before["sha256"]:
            target.write_bytes(body)
            return {"source": before, "copy": meta(target), "source_before_after_identical": True}
    raise RuntimeError("Source changed during all bounded snapshot attempts")


def main():
    spec = importlib.util.spec_from_file_location("previous_readonly_audit", PRIOR / "audit.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    account = previous.account_snapshot()
    save("account_before.json", account)
    policy_copy = copy_stable(ROOT / "config/argon4_forward_policy.json", "policy_before.json")
    mark_copy = copy_stable(RUNTIME / "features/raw_data/auxiliary/ETHUSDT_1h_mark.parquet", "ETHUSDT_1h_mark_before.parquet")
    registry_copy = copy_stable(ROOT / "archive/catalog/mark_candle_registry.json", "mark_registry_before.json")
    frozen = read(PRIOR / "pytest_end_snapshot.json")
    models = {path: meta(ROOT / path) for path in frozen["original_models"]}
    save("production_models_before.json", models)
    manifest_changes = [path for path, value in models.items() if value["sha256"] != frozen["original_models"][path]["sha256"]]
    policy = read(OUT / "policy_before.json")
    state = read(RUNTIME / "pipeline_state.json")
    service = read(RUNTIME / "service.json")
    snapshot = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                "service": service, "verification_pause_exists": (RUNTIME / "verification_pause.json").exists(),
                "pipeline": state, "online_state": read(RUNTIME / "features/online_state.json"),
                "logs": {name: meta(RUNTIME / name) for name in ("server.stdout.log", "server.stderr.log")}}
    save("runtime_before.json", snapshot)
    con = sqlite3.connect("file:" + (RUNTIME / "features/http_receipts.sqlite3").as_posix() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    receipts = []
    for row in con.execute("SELECT * FROM receipts WHERE url LIKE '%markPriceKlines%' AND url LIKE '%symbol=ETHUSDT%' AND url LIKE '%interval=1h%' ORDER BY id DESC LIMIT 12"):
        value = dict(row)
        raw = value.pop("raw")
        raw = bytes(raw) if not isinstance(raw, str) else raw.encode()
        value.update(raw_text=raw.decode(), body_sha256_verified=hashlib.sha256(raw).hexdigest() == value["sha256"])
        receipts.append(value)
    con.close()
    save("eth_mark_receipts_before.json", receipts)
    source = sorted(set(frozen["source"]) | {
        path.relative_to(ROOT).as_posix()
        for folder, pattern in (("tools/argon4_forward", "*.py"), ("tests", "test_argon4_forward*.py"),
                                ("docs", "ARGON4*MARK*REVISION*.md"))
        for path in (ROOT / folder).glob(pattern)
    })
    save("source_list_prepared.json", {"source": source,
         "scope": "Preliminary list only; source hashes must be captured after explicit source freeze.",
         "dynamic_runtime": ["online raw inputs", "mark registry research references", "account new events/cohorts/orders", "service metadata and heartbeat"]})
    risk_keys = ["initial_capital_usd", "single_stream_equity_fraction", "tick_equity_fraction", "max_tick_budget_usd",
                 "max_order_usd", "max_open_risk_usd", "max_underlying_open_risk_usd", "max_buy_price", "daily_loss_fraction"]
    summary = {"status": "BASELINE_PREPARED_NO_VALIDATION_CHAIN_STARTED",
               "captured_at_utc": datetime.now(timezone.utc).isoformat(),
               "previous_test_baseline": {"path": str(PRIOR / "summary.json"), "sha256": sha(PRIOR / "summary.json"),
                                          "counts": read(PRIOR / "summary.json")["pytest"]["current"]["counts"]},
               "account_counts": account["row_counts"], "account": account["tables"]["account"][0],
               "open_order_count": sum(row["status"] == "OPEN" for row in account["tables"]["orders"]),
               "funds_conserved": account["funds_conserved"], "invalid_event_hashes": account["invalid_event_hashes"],
               "policy_snapshot": policy_copy, "risk_limits": {key: policy[key] for key in risk_keys},
               "deployment_manifest_pins": policy["deployment_manifest_sha256"],
               "model_file_count": len(models), "model_changes_since_previous_full_checks": manifest_changes,
               "old_eth_mark_snapshot": mark_copy, "old_mark_registry_snapshot": registry_copy,
               "service": service, "pipeline_status": state["status"], "last_good_cohort": state.get("last_cohort"),
               "last_attempt_cohort": state.get("last_attempt_cohort"), "source_list_count": len(source),
               "live_enabled": policy["live_enabled"], "mutation_scope": "New report directory only. No service, configuration, source, model, online data or account mutation.",
               "validation_rule": "After explicit freeze, business code/model/risk are immutable. Legitimate live observations, versioned input repair, mark research references and new paper activity are separately audited, never falsely treated as immutable runtime files."}
    save("baseline_summary.json", summary)
    print(json.dumps({"status": summary["status"], "account_counts": summary["account_counts"],
                      "policy_sha256": policy_copy["source"]["sha256"], "model_count": len(models),
                      "model_changes": manifest_changes, "eth_mark_sha256": mark_copy["source"]["sha256"],
                      "service": service, "source_count": len(source), "summary_path": str(OUT / "baseline_summary.json")}))


if __name__ == "__main__":
    main()
