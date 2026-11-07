"""Independent read-only validation of the committed repair archive."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main():
    parent = read(OUT / "restart_and_revision_commit.json")
    path = Path(parent["manifest_path"])
    archive = path.parent
    manifest = read(path)
    checks = {}
    checks["manifest_matches_commit_marker"] = sha(path) == parent["marker"]["manifest_sha256"]
    checks["prepared_archive_hash_matches"] = sha(archive / "prepared.json") == manifest["prepared_sha256"]
    checks["all_24_old_file_archives_match"] = len(manifest["before_files"]) == 24 and all(
        sha(archive / "raw_data" / relative) == digest for relative, digest in manifest["before_files"].items())
    checks["all_24_after_view_file_hashes_match_archives"] = len(manifest["after_files"]) == 24 and all(
        sha(archive / ("raw_data" if manifest["before_files"][relative] == digest else "proposed_raw_data") / relative) == digest
        for relative, digest in manifest["after_files"].items())
    checks["sqlite_backups_match"] = all(sha(Path(row["snapshot_path"])) == row["sha256"] for row in manifest["sqlite_backups"])
    changed_files = [name for name in manifest["before_files"] if manifest["before_files"][name] != manifest["after_files"][name]]
    checks["only_ETH_mark_1h_file_changed"] = changed_files == ["auxiliary/ETHUSDT_1h_mark.parquet"]
    checks["exact_one_declared_closed_bar_revision"] = len(manifest["changes"]) == 1
    change = manifest["changes"][0]
    checks["declared_identity_matches"] = (change["symbol"], change["kind"], change["interval"], change["bar_open"]) == (
        "ETHUSDT", "mark", "1h", "2026-11-06T08:00:00+00:00")
    checks["declared_value_change_only_close_count"] = change["changed_price_fields"] == ["close", "count"]
    old_path = archive / "raw_data" / changed_files[0]
    new_path = archive / "proposed_raw_data" / changed_files[0]
    before, after = pd.read_parquet(old_path), pd.read_parquet(new_path)
    equal = before.eq(after) | (before.isna() & after.isna())
    row_indices = equal.index[~equal.all(axis=1)].tolist()
    different_columns = [column for column in before if not equal[column].all()]
    checks["old_ETH_file_matches_independent_pre_repair_backup"] = sha(old_path) == sha(OUT / "ETHUSDT_1h_mark_before.parquet")
    checks["same_row_count_index_and_schema_no_inserts"] = (len(before) == len(after) and before.index.equals(after.index)
        and before.columns.equals(after.columns) and before.dtypes.equals(after.dtypes))
    checks["one_real_row_changed"] = len(row_indices) == 1
    checks["only_values_and_receipt_provenance_changed"] = set(different_columns) == {
        "close", "count", "first_received_at", "receipt_sha256", "receipt_time_basis", "clock_evidence_sha256"}
    checks["changed_row_uses_original_bar_timestamp"] = len(row_indices) == 1 and str(pd.Timestamp(before.loc[row_indices[0], "date"])) == "2026-11-06 08:00:00+00:00"
    receipts = []
    with sqlite3.connect((archive / "receipts.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for role, reference in [("original", change["original_receipt"])] + [("confirmation", x) for x in change["confirmation_receipts"]]:
            row = dict(connection.execute("SELECT * FROM receipts WHERE id=?", (reference["id"],)).fetchone())
            raw = row["raw"]
            raw = raw.encode() if isinstance(raw, str) else raw
            raw_sha = hashlib.sha256(raw).hexdigest()
            url = urlsplit(row["url"])
            query = parse_qs(url.query)
            selected = [bar for bar in json.loads(raw) if bar[0] == reference["raw_bar"][0]]
            valid = (raw_sha == row["sha256"] == reference["sha256"] and row["http_status"] == 200
                and row["received_at"] == reference["received_at"] and row["url"] == reference["url"]
                and url.scheme == "https" and url.netloc == "fapi.binance.com" and url.path == "/fapi/v1/markPriceKlines"
                and query.get("symbol") == ["ETHUSDT"] and query.get("interval") == ["1h"]
                and selected == [reference["raw_bar"]] and reference["raw_bar"][6] == reference["raw_bar"][0] + 3_599_999
                and reference["raw_bar"][6] / 1000 < utc(row["received_at"]).timestamp())
            checks[f"archived_official_receipt_{reference['id']}_valid"] = valid
            receipts.append({"role": role, "id": row["id"], "source": row["url"], "http_status": row["http_status"],
                             "received_at": row["received_at"], "sha256": raw_sha, "bar": selected})
    confirmations = change["confirmation_receipts"]
    spacing = (utc(confirmations[1]["received_at"]) - utc(confirmations[0]["received_at"])).total_seconds()
    checks["two_independent_matching_receipts_at_least_60_seconds"] = (
        confirmations[0]["id"] != confirmations[1]["id"] and spacing >= 60 and confirmations[0]["raw_bar"] == confirmations[1]["raw_bar"])
    registry = manifest["mark_registry_update"]
    checks["registry_archive_hashes_match"] = (sha(archive / registry["before_snapshot"]) == registry["before_sha256"]
        and sha(archive / registry["proposed_snapshot"]) == registry["after_sha256"])
    before_reg, after_reg = (read(archive / registry[key]) for key in ("before_snapshot", "proposed_snapshot"))
    target = "argon4_local_v4_online_mark_ETHUSDT_20261106"
    def without_target(value):
        value = dict(value)
        value["research_references"] = [x for x in value["research_references"] if x["dataset_id"] != target]
        return value
    checks["canonical_and_other_registry_content_preserved"] = without_target(before_reg) == without_target(after_reg)
    target_ref = next(x for x in after_reg["research_references"] if x["dataset_id"] == target)
    checks["repair_ref_hash_and_research_only_gate"] = (target_ref["sha256"] == sha(new_path) and target_ref["installed"] is False
        and target_ref["research_only"] is True and target_ref["gate_effect"] == "NONE")
    checks["future_only_skip_cutover_cohort"] = (manifest["working_view_for_future_cohorts_only"] is True
        and manifest["current_cohort_must_be_skipped"] is True and manifest["next_cohort_requires_fresh_validation"] is True
        and manifest["model_weights_modified_by_this_operation"] is False and manifest["collector_model_strategy_source_modified"] is False)
    result = {"status": "PASS" if all(checks.values()) else "REQUIRES_REVIEW", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(path), "manifest_sha256": sha(path), "parent_commit_receipt_sha256": sha(OUT / "restart_and_revision_commit.json"),
        "view_effective_at": manifest["view_effective_at"], "checks": checks, "changed_files": changed_files,
        "old_rows": len(before), "new_rows": len(after), "changed_columns": different_columns,
        "declared_change": change, "receipt_spacing_seconds": spacing, "independently_verified_archived_receipts": receipts,
        "scope": "Existing immutable repair archives only; no collection, model prediction, source mutation, account mutation or replay."}
    (OUT / "revision_archive_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [k for k,v in checks.items() if not v], "receipt_spacing_seconds": spacing}))


if __name__ == "__main__":
    main()
