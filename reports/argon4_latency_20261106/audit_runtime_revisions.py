"""Read-only audit of authorized input-view revisions during latency checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from urllib.parse import urlsplit, parse_qs

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REVISION_ROOT = ROOT / "paper_runtime/argon4_forward_20261105/features/revisions"
SPEC = {("futures", "5m"): ("intraday", "/fapi/v1/klines", 300000), ("mark", "1h"): ("auxiliary", "/fapi/v1/markPriceKlines", 3600000)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def audit_one(path):
    manifest = read(path)
    folder = path.parent
    checks = {"committed": manifest["status"] == "COMMITTED", "prepared_hash": sha(folder / "prepared.json") == manifest["prepared_sha256"],
        "before_file_set_matches_after": set(manifest["before_files"]) == set(manifest["after_files"]),
        "before_archive_all_hashes": all(sha(folder / "raw_data" / rel) == digest for rel,digest in manifest["before_files"].items()),
        "after_view_all_hashes": all(sha(folder / ("raw_data" if manifest["before_files"][rel] == digest else "proposed_raw_data") / rel) == digest for rel,digest in manifest["after_files"].items()),
        "sqlite_backup_hashes": all(sha(Path(row["snapshot_path"])) == row["sha256"] for row in manifest["sqlite_backups"]),
        "future_only_and_skip_cutover": manifest["working_view_for_future_cohorts_only"] is True and manifest["current_cohort_must_be_skipped"] is True and manifest["next_cohort_requires_fresh_validation"] is True,
        "no_model_or_source_edit": manifest["model_weights_modified_by_this_operation"] is False and manifest["collector_model_strategy_source_modified"] is False}
    details = []
    db = sqlite3.connect((folder / "receipts.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    try:
        for index, change in enumerate(manifest["changes"]):
            kind, interval, symbol = change["kind"], change["interval"], change["symbol"]
            prefix = f"change_{index}_"
            allowed = (kind, interval) in SPEC and symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
            checks[prefix + "whitelisted_identity"] = allowed
            if not allowed:
                continue
            directory, endpoint, step = SPEC[(kind, interval)]
            relative = f"{directory}/{symbol}_{interval}_{kind}.parquet"
            before = pd.read_parquet(folder / "raw_data" / relative)
            after = pd.read_parquet(folder / "proposed_raw_data" / relative)
            checks[prefix + "no_insert_or_retimestamp"] = (len(before) == len(after) and before.index.equals(after.index)
                and before.columns.equals(after.columns) and before[["date", "open_time", "close_time"]].equals(after[["date", "open_time", "close_time"]]))
            selected_before = before.loc[pd.to_datetime(before["date"], utc=True) == pd.Timestamp(change["bar_open"])]
            selected_after = after.loc[pd.to_datetime(after["date"], utc=True) == pd.Timestamp(change["bar_open"])]
            checks[prefix + "one_matching_row"] = len(selected_before) == len(selected_after) == 1
            checks[prefix + "declared_values_match_archived_rows"] = (len(selected_before) == len(selected_after) == 1
                and all(float(selected_before.iloc[0][key]) == float(value) for key,value in change["before"].items())
                and all(float(selected_after.iloc[0][key]) == float(value) for key,value in change["after"].items()))
            references = [("original", change["original_receipt"])] + [("confirmation", ref) for ref in change["confirmation_receipts"]]
            verified = []
            for role, ref in references:
                row = dict(db.execute("SELECT * FROM receipts WHERE id=?", (ref["id"],)).fetchone())
                raw = row["raw"].encode() if isinstance(row["raw"], str) else row["raw"]
                raw_sha = hashlib.sha256(raw).hexdigest()
                parts = urlsplit(row["url"])
                query = parse_qs(parts.query)
                target = [bar for bar in json.loads(raw) if bar[0] == ref["raw_bar"][0]]
                valid = (row["http_status"] == 200 and raw_sha == row["sha256"] == ref["sha256"]
                    and row["received_at"] == ref["received_at"] and row["url"] == ref["url"]
                    and parts.scheme == "https" and parts.netloc == "fapi.binance.com" and parts.path == endpoint
                    and query.get("symbol") == [symbol] and query.get("interval") == [interval]
                    and target == [ref["raw_bar"]] and ref["raw_bar"][6] == ref["raw_bar"][0] + step - 1
                    and ref["raw_bar"][6] / 1000 < utc(row["received_at"]).timestamp())
                checks[prefix + f"receipt_{ref['id']}_official_raw_hash_closed_bar"] = valid
                verified.append({"role": role, "id": ref["id"], "source": row["url"], "received_at": row["received_at"], "sha256": raw_sha, "http_status": row["http_status"], "time_basis": row["time_basis"]})
            confirmations = change["confirmation_receipts"]
            spacing = (utc(confirmations[1]["received_at"]) - utc(confirmations[0]["received_at"])).total_seconds()
            checks[prefix + "independent_equal_confirmations_60_seconds"] = (confirmations[0]["id"] != confirmations[1]["id"] and spacing >= 60 and confirmations[0]["raw_bar"] == confirmations[1]["raw_bar"])
            details.append({"change": change, "receipt_spacing_seconds": spacing, "verified_receipts": verified})
    finally:
        db.close()
    registry = manifest.get("mark_registry_update")
    if registry:
        checks["registry_archives_hashes"] = (sha(folder / registry["before_snapshot"]) == registry["before_sha256"] and sha(folder / registry["proposed_snapshot"]) == registry["after_sha256"])
        old, new = read(folder / registry["before_snapshot"]), read(folder / registry["proposed_snapshot"])
        affected = {row["dataset_id"] for row in registry["updates"]}
        def core(value):
            value = dict(value)
            value["research_references"] = [row for row in value["research_references"] if row["dataset_id"] not in affected]
            return value
        checks["canonical_and_unaffected_registry_unchanged"] = core(old) == core(new)
        checks["affected_registry_research_only"] = all(row.get("installed") is False and row.get("research_only") is True and row.get("gate_effect") == "NONE" for row in new["research_references"] if row["dataset_id"] in affected)
    return {"status": "PASS" if all(checks.values()) else "REQUIRES_REVIEW", "revision_id": manifest["revision_id"],
        "manifest_path": str(path), "manifest_sha256": sha(path), "view_effective_at": manifest["view_effective_at"],
        "checks": checks, "details": details, "changed_files": [rel for rel in manifest["before_files"] if manifest["before_files"][rel] != manifest["after_files"][rel]]}


def main():
    baseline = read(OUT / "baseline_summary.json")
    cutoff = utc(baseline["completed_at_utc"])
    selected = [path for path in REVISION_ROOT.glob("*/manifest.json") if utc(read(path)["view_effective_at"]) >= cutoff]
    audits = [audit_one(path) for path in sorted(selected)]
    marker = read(REVISION_ROOT / "view_state.json")
    marker_matches = marker.get("status") == "COMMITTED" and any(value["revision_id"] == marker.get("revision_id") and value["manifest_sha256"] == marker.get("manifest_sha256") for value in audits)
    status = "NO_NEW_COMMITTED_REVISION" if not audits else ("PASS" if all(value["status"] == "PASS" for value in audits) and marker_matches else "REQUIRES_REVIEW")
    result = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
        "baseline_after_utc": cutoff.isoformat(), "audits": audits, "current_marker": marker, "current_marker_matches_latest_audited_commit": marker_matches,
        "scope": "Read-only immutable archived views and receipts. Historical account/order preservation is independently verified by the final account audit. No reconstruction or replay of historical decisions."}
    (OUT / "runtime_revision_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "revisions": len(audits), "failed_checks": {value["revision_id"]: [key for key,ok in value["checks"].items() if not ok] for value in audits}}))


if __name__ == "__main__":
    main()
