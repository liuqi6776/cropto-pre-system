"""Read existing public receipt metadata only; never invokes a data client."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    start = "2026-11-06T12:50:00+00:00"
    quote_path = ROOT / "altcoin_short_system/verified_v2/observations/quote_receipts_20261106.jsonl"
    with quote_path.open("rb") as handle:
        size = quote_path.stat().st_size
        offset = max(0, size-32*1024*1024)
        handle.seek(offset)
        if offset:
            handle.readline()
        lines = handle.read().splitlines()
    quotes, partial = [], 0
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            partial += 1
            continue
        if row["received_at"] >= start:
            quotes.append({key: row.get(key) for key in ("received_at", "source", "sha256", "kind")})
    raw_dir = ROOT / "altcoin_short_system/verified_v2/official_v3/raw"
    receipts = {}
    for path in raw_dir.glob("*.receipt.json"):
        if path.stat().st_mtime < datetime.fromisoformat(start).timestamp():
            continue
        row = read(path)
        if row.get("retrieved_at", "") >= start:
            key = (row.get("source"), row.get("retrieved_at"), row.get("sha256"))
            receipts.setdefault(key, {"receipt_files": [], **{name: row.get(name) for name in ("source", "retrieved_at", "sha256", "http_status", "method")}})["receipt_files"].append(str(path))
    state_path = ROOT / "altcoin_short_system/verified_v2/official_v3/paper/state.json"
    state = read(state_path)
    local_db = ROOT / "paper_runtime/argon4_forward_20261105/features/http_receipts.sqlite3"
    db = sqlite3.connect(local_db.resolve().as_uri()+"?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = [dict(row) for row in db.execute("SELECT id,url,requested_at,received_at,http_status,sha256 FROM receipts WHERE requested_at>=? ORDER BY id", ("2026-11-06T12:58:00+00:00",))]
    denial = dict(db.execute("SELECT * FROM receipts WHERE id=1234").fetchone())
    raw = denial["raw"].encode() if isinstance(denial["raw"], str) else denial["raw"]
    denial_sha_valid = hashlib.sha256(raw).hexdigest() == denial["sha256"]
    body = json.loads(raw)
    db.close()
    sources = ["tests/test_argon4_forward_online.py", "tests/test_argon4_forward_online_transport.py", "tests/test_argon4_forward_pipeline.py",
        "tests/test_argon4_forward_stage.py", "tests/test_argon4_forward_revisions.py", "tests/test_argon4_forward_mark_revisions.py", "tests/test_argon4_forward_latency_registry.py",
        "tests/test_argon4_clean_room.py", "tests/test_altcoin_verified_v2.py", "tests/test_altcoin_verified_v3.py", "tests/test_argon_paper_qa_rate_limit.py",
        "tests/test_binance_noncore_acquisition_v1.py", "tests/test_binance_noncore_acquisition_v2.py", "tests/test_binance_funding_exact_v1.py", "tests/test_noncore_capital_flows_acquisition_v1.py"]
    result = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "scope_start_utc": start,
        "local_argon4_receipts_since_1258": rows, "local_418_receipt_id": 1234, "local_418_raw_sha_matches": denial_sha_valid,
        "local_418_provider_code": body.get("code"), "local_backoff": read(ROOT / "paper_runtime/argon4_forward_20261105/features/binance_backoff.json"),
        "other_monitor_quote_log": str(quote_path), "quote_log_bytes_at_read": size,
        "read_tail_max_bytes": 32*1024*1024, "partial_tail_lines_not_counted": partial,
        "other_monitor_successful_quote_receipts": quotes,
        "other_monitor_successes_per_minute": dict(sorted(Counter(row["received_at"][:16] for row in quotes).items())),
        "other_monitor_distinct_official_successes": list(receipts.values()),
        "other_monitor_official_endpoints": dict(Counter(urlsplit(row["source"]).path for row in receipts.values())),
        "other_monitor_state_public_health_only": {key: state.get(key) for key in ("status", "error", "funding_error", "features_asof", "last_funding_check_ms")},
        "test_source_review": {"files": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources},
            "findings": ["ARGON4 online and transport tests replace Session with offline fixtures; constructor-only tests never refresh.",
                "Pipeline, stage, revision and latency-registry tests forbid requests Session.request.",
                "Early literal Binance URLs outside these tests are rejected order-route cases or simulated HTTPError values.",
                "Legacy rate-limit test named real_request executes a local callback only.",
                "Inspected Binance archive/funding acquisition tests inject fake opener/get/download callbacks."],
            "limits": "Static review of direct network-relevant tests is not a process network trace. The -q full log has no per-node timestamps, so the exact case executing at 13:05 cannot be inferred from the progress percentage; uninspected import side effects cannot be ruled out."},
        "causality_limit": "The other monitor's existence or successful requests do not establish that it caused the denial, nor that its curl route shares the same external IP as this requests Session. Failed other-monitor requests are not fully logged; rate/egress attribution remains unproven.",
        "actions": "Only existing local files and read-only SQLite were read; no new external request, backoff edit, service control or other-application mutation."}
    (OUT / "network_scope_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"quote_receipt_count_since_1250": len(quotes), "quote_latest": quotes[-1]["received_at"] if quotes else None,
        "other_official_success_count_since_1250": len(receipts), "other_endpoints": result["other_monitor_official_endpoints"],
        "local_418_raw_sha_matches": denial_sha_valid, "other_health": result["other_monitor_state_public_health_only"]}))


if __name__ == "__main__":
    main()
