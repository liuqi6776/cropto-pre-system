"""Archive and publish only verified online research-mark registry revisions.

No discovery of a writable registry, market requests, canonical installation or
account changes. The caller supplies the registry and holds the collector lock;
its PREPARED marker must remain in force until this publication succeeds.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://fapi.binance.com/fapi/v1/markPriceKlines"
SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"}
SNAPSHOTS = ("mark_registry_before.json", "mark_registry_proposed.json")
APPEND_PROOF_FILE = "mark_append_baseline_proofs.json"


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("MARK_REGISTRY_INVALID_DIGEST")
    return value


def _path(value):
    path = Path(value).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("MARK_REGISTRY_SYMLINK")
    return path.resolve()


def _write_new(path, payload):
    with Path(path).open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _frame_metadata(frame):
    required = {"date", "open", "high", "low", "close", "volume", "first_received_at",
                "receipt_sha256", "receipt_time_basis", "clock_evidence_sha256"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("MARK_REGISTRY_FRAME_SCHEMA")
    dates, received = pd.DatetimeIndex(frame.date), pd.DatetimeIndex(frame.first_received_at)
    if dates.tz is None or received.tz is None:
        raise ValueError("MARK_REGISTRY_FRAME_TIME")
    dates, received = dates.as_unit("ns"), received.as_unit("ns")
    if (dates.hasnans or received.hasnans
            or dates.has_duplicates or not dates.is_monotonic_increasing
            or np.any(dates.asi8 % pd.Timedelta(hours=1).value)
            or np.any(received.asi8 < dates.asi8 + pd.Timedelta(hours=1).value)):
        raise ValueError("MARK_REGISTRY_FRAME_TIME")
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if (not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any()
            or (values[:, 1] < values[:, :4].max(axis=1)).any()
            or (values[:, 2] > values[:, :4].min(axis=1)).any()):
        raise ValueError("MARK_REGISTRY_FRAME_PRICES")
    digests = sorted({_digest(x) for x in frame.receipt_sha256})
    for basis, clock in zip(frame.receipt_time_basis, frame.clock_evidence_sha256):
        if basis not in ("SYSTEM_UTC", "CALIBRATED_UTC_UPPER_BOUND"):
            raise ValueError("MARK_REGISTRY_RECEIPT_CLOCK")
        if basis == "CALIBRATED_UTC_UPPER_BOUND":
            _digest(clock)
    return {"rows": len(frame), "first_utc": dates.min().tz_convert("UTC").isoformat(),
            "last_utc": dates.max().tz_convert("UTC").isoformat(), "source_receipt_sha256": digests}


def prove_append_baseline(entry, raw_root, symbol, before_sha256, archive, *, validation_at=None):
    """Read-only proof of a stale registration caused solely by observed append.

    Read the current baseline from its immutable PREPARED snapshot, not from a
    live file that may already be replaced during recovery. A matching old hash
    must be bound to a committed revision manifest. New rows are checked against
    the archived original HTTP receipts, never the proposed revised values.
    """
    from .revisions import _read_cache, _receipts, _public_receipt, _time, PRICE_FIELDS

    error = "MARK_REGISTRY_REFERENCE_IDENTITY_APPEND_"
    raw_root, archive = _path(raw_root), _path(archive)
    revisions = _path(raw_root.parent / "revisions")
    if symbol not in SYMBOLS or archive.parent != revisions:
        raise ValueError(error + "SCOPE")
    relative = f"auxiliary/{symbol}_1h_mark.parquet"
    registered_sha, before_sha = _digest(entry.get("sha256")), _digest(before_sha256)
    prepared_path = _path(archive / "prepared.json")
    if not prepared_path.is_file():
        raise ValueError(error + "PREPARED_IDENTITY")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    stamp = _time(prepared["validation_at"] if validation_at is None else validation_at)
    if (stamp != _time(prepared["validation_at"]) or prepared.get("before_files", {}).get(relative) != before_sha
            or _path(prepared["data_root"]) != raw_root or _path(prepared["archive_path"]) != archive):
        raise ValueError(error + "PREPARED_IDENTITY")
    baseline_path = _path(archive / "raw_data" / relative)
    if not baseline_path.is_relative_to(archive) or _sha(baseline_path) != before_sha:
        raise ValueError(error + "BASELINE_HASH")
    baseline = _read_cache(baseline_path, 3_600_000)
    _frame_metadata(baseline)
    old_path, binding = None, None
    for folder in sorted(revisions.iterdir()):
        if not folder.is_dir() or folder == archive:
            continue
        for view, field in (("raw_data", "before_files"), ("proposed_raw_data", "after_files")):
            candidate = folder / view / relative
            if not candidate.is_file():
                continue
            candidate = _path(candidate)
            if not candidate.is_relative_to(revisions) or _sha(candidate) != registered_sha:
                continue
            manifest_path, old_prepared_path = _path(folder / "manifest.json"), _path(folder / "prepared.json")
            if not manifest_path.is_file() or not old_prepared_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            old_prepared = json.loads(old_prepared_path.read_text(encoding="utf-8"))
            if (manifest.get("status") != "COMMITTED" or old_prepared.get("status") != "PREPARED"
                    or manifest.get("revision_id") != folder.name or old_prepared.get("revision_id") != folder.name
                    or manifest.get(field, {}).get(relative) != registered_sha
                    or manifest.get("prepared_sha256") != _sha(old_prepared_path)
                    or _path(manifest["archive_path"]) != _path(folder)
                    or _path(old_prepared["archive_path"]) != _path(folder)
                    or _path(manifest["data_root"]) != raw_root or _path(old_prepared["data_root"]) != raw_root
                    or _time(manifest["view_effective_at"]) > stamp
                    or (view == "raw_data" and old_prepared.get("before_files", {}).get(relative) != registered_sha)):
                continue
            old_path = candidate
            binding = {"path": str(candidate), "sha256": registered_sha, "view": view,
                       "manifest_path": str(manifest_path), "manifest_sha256": _sha(manifest_path),
                       "manifest_hash_field": field, "prepared_path": str(old_prepared_path),
                       "prepared_sha256": _sha(old_prepared_path)}
            break
        if old_path is not None:
            break
    if old_path is None:
        raise ValueError(error + "COMMITTED_OLD_SNAPSHOT_REQUIRED")
    old = _read_cache(old_path, 3_600_000)
    _frame_metadata(old)
    if len(baseline) <= len(old):
        raise ValueError(error + "NOT_STRICT_APPEND")
    try:
        pd.testing.assert_frame_equal(old, baseline.iloc[:len(old)], check_exact=True,
                                      check_dtype=True, check_index_type=True, check_column_type=True)
        # Floating bit patterns (including signed zero) must also be preserved.
        for column in old:
            if pd.api.types.is_float_dtype(old[column].dtype):
                if old[column].to_numpy().tobytes() != baseline[column].iloc[:len(old)].to_numpy().tobytes():
                    raise AssertionError("float bits changed")
    except AssertionError as exc:
        raise ValueError(error + "OLD_ROWS_CHANGED") from exc
    added = baseline.iloc[len(old):]
    dates = pd.DatetimeIndex(baseline.date).as_unit("ns")
    if (np.any(np.diff(dates.asi8) != pd.Timedelta(hours=1).value)
            or added.date.min() <= old.date.max()
            or (added.date + pd.Timedelta(hours=1) > stamp).any()):
        raise ValueError(error + "NEW_ROWS_NOT_CLOSED_CONTIGUOUS_LATER")
    receipt_path = _path(archive / "receipts.sqlite3")
    if not receipt_path.is_relative_to(archive):
        raise ValueError(error + "RECEIPT_SCOPE")
    evidence = []
    for _, row in added.iterrows():
        opened = int(_time(row.date).value // 1_000_000)
        observations = _receipts(receipt_path, symbol, stamp, [opened], "mark", "1h",
                                original_identity=(str(row.receipt_sha256), _time(row.first_received_at).isoformat()))[opened]
        matching = [r for r in observations if r["sha256"] == row.receipt_sha256
                    and _time(r["received_at"]) == _time(row.first_received_at)]
        if len(matching) != 1:
            raise ValueError(error + "UNIQUE_ORIGINAL_RECEIPT_REQUIRED")
        original = matching[0]
        clock_hash = None if pd.isna(row.clock_evidence_sha256) else str(row.clock_evidence_sha256)
        if (any(float(row[field]) != float(original["numbers"][i]) for i, field in PRICE_FIELDS.items())
                or row.receipt_time_basis != original["time_basis"]
                or clock_hash != original["clock_evidence_sha256"]):
            raise ValueError(error + "RAW_RECEIPT_VALUE_OR_PROVENANCE")
        evidence.append({"bar_open": _time(row.date).isoformat(), "original_receipt": _public_receipt(original)})
    if (_sha(old_path) != registered_sha or _sha(baseline_path) != before_sha
            or _sha(binding["manifest_path"]) != binding["manifest_sha256"]
            or _sha(binding["prepared_path"]) != binding["prepared_sha256"]):
        raise ValueError(error + "EVIDENCE_CHANGED")
    return {"schema_version": 1, "symbol": symbol, "relative_path": relative,
            "registered_sha256": registered_sha, "before_sha256": before_sha,
            "validation_at": stamp.isoformat(), "old_snapshot": binding,
            "before_snapshot_path": str(baseline_path), "receipts_snapshot_path": str(receipt_path),
            "old_rows": len(old), "before_rows": len(baseline), "appended_rows": len(added),
            "columns": list(baseline.columns), "dtypes": [str(x) for x in baseline.dtypes],
            "comparison": "EXACT_PREFIX_ALL_COLUMNS_DTYPES_VALUES_AND_FLOAT_BITS",
            "added_observations": evidence}


def prepare_registry_update(registry_path, raw_root, plans, after_files, archive):
    """Write immutable before/proposed snapshots, leaving the registry untouched."""
    selected = [p for p in plans if p.get("kind") == "mark" and p.get("interval") == "1h"]
    if not selected:
        raise ValueError("MARK_REGISTRY_NO_MARK_PLANS")
    registry, raw_root, archive = map(_path, (registry_path, raw_root, archive))
    if not registry.is_file() or not raw_root.is_dir() or not archive.is_dir():
        raise ValueError("MARK_REGISTRY_REQUIRED_PATH_MISSING")
    if registry.is_relative_to(archive) or registry.is_relative_to(raw_root):
        raise ValueError("MARK_REGISTRY_PATH_OVERLAP")
    original = registry.read_bytes()
    before_sha = hashlib.sha256(original).hexdigest()
    document = json.loads(original)
    if not isinstance(document, dict) or not isinstance(document.get("research_references"), list):
        raise ValueError("MARK_REGISTRY_RESEARCH_REFERENCES_REQUIRED")
    proposed = deepcopy(document)
    references = proposed["research_references"]
    if any(not isinstance(entry, dict) for entry in references):
        raise ValueError("MARK_REGISTRY_REFERENCE_SCHEMA")
    changed, seen, append_proofs = [], set(), []
    for plan in selected:
        symbol = plan["symbol"]
        relative = f"auxiliary/{symbol}_1h_mark.parquet"
        if symbol not in SYMBOLS or symbol in seen or plan["relative_path"] != relative:
            raise ValueError("MARK_REGISTRY_PLAN_IDENTITY")
        seen.add(symbol)
        target = _path(plan["path"])
        if target != _path(raw_root / relative) or not target.is_relative_to(raw_root):
            raise ValueError("MARK_REGISTRY_PLAN_PATH")
        before = _digest(plan["before_sha256"])
        after = _digest(after_files[relative])
        if _sha(target) != before:
            raise ValueError("MARK_REGISTRY_OLD_FILE_CHANGED")
        proposed_raw = _path(archive / "proposed_raw_data" / relative)
        if not proposed_raw.is_relative_to(archive) or _sha(proposed_raw) != after:
            raise ValueError("MARK_REGISTRY_PROPOSED_FILE_MISMATCH")
        frame = pd.read_parquet(proposed_raw)
        try:
            pd.testing.assert_frame_equal(frame, plan["frame"], check_exact=True)
        except AssertionError as exc:
            raise ValueError("MARK_REGISTRY_PLAN_FRAME_MISMATCH") from exc
        metadata = _frame_metadata(frame)
        matches = []
        for entry in references:
            local = entry.get("local_path")
            if isinstance(local, str) and _path(Path(local) if Path(local).is_absolute() else ROOT / local) == target:
                matches.append(entry)
        if len(matches) != 1:
            raise ValueError("MARK_REGISTRY_UNIQUE_REFERENCE_REQUIRED")
        entry = matches[0]
        identifier = entry.get("dataset_id")
        if (not isinstance(identifier, str)
                or re.fullmatch(r"argon4_local_v4_online_mark_" + symbol + r"_\d{8}", identifier) is None
                or sum(e.get("dataset_id") == identifier for e in references) != 1
                or entry.get("pair") != symbol[:-4]+"/USDT:USDT"
                or entry.get("timeframe") != "1h" or entry.get("candle_type") != "mark"
                or entry.get("source_endpoint") != ENDPOINT
                or entry.get("verification_status") != "VERIFIED_PERPETUAL_LABEL"
                or entry.get("installed") is not False or entry.get("research_only") is not True
                or entry.get("gate_effect") != "NONE"):
            raise ValueError("MARK_REGISTRY_REFERENCE_IDENTITY")
        append_proof = None
        if entry.get("sha256") != before:
            append_proof = prove_append_baseline(entry, raw_root, symbol, before, archive)
            append_proofs.append(append_proof)
        entry.update(metadata, sha256=after)
        entry["revision_provenance"] = {
            "revision_id": archive.name, "manifest_path": str(archive / "manifest.json"),
            "before_sha256": before, "after_sha256": after,
            "source_endpoint": ENDPOINT, "future_cohorts_only": True,
            "confirmation_policy": "TWO_INDEPENDENT_MATCHING_RAW_RECEIPTS_AT_LEAST_60_SECONDS_APART",
        }
        if append_proof is not None:
            entry["revision_provenance"]["append_baseline_proof"] = append_proof
        changed.append({"dataset_id": identifier, "symbol": symbol, "relative_path": relative,
                        "before_sha256": before, "after_sha256": after})
    encoded = json.dumps(proposed, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    # Validate everything before either write-once archive file is created.
    if any((archive / name).exists() for name in (*SNAPSHOTS, APPEND_PROOF_FILE)):
        raise ValueError("MARK_REGISTRY_SNAPSHOT_ALREADY_EXISTS")
    if _sha(registry) != before_sha:
        raise ValueError("MARK_REGISTRY_CHANGED_DURING_PREPARATION")
    _write_new(archive / SNAPSHOTS[0], original)
    _write_new(archive / SNAPSHOTS[1], encoded)
    receipt = {"schema_version": 1, "registry_path": str(registry), "archive_path": str(archive),
            "before_sha256": before_sha, "after_sha256": hashlib.sha256(encoded).hexdigest(),
            "before_snapshot": SNAPSHOTS[0], "proposed_snapshot": SNAPSHOTS[1],
            "updates": changed, "canonical_entries_modified": False}
    if append_proofs:
        payload = json.dumps(append_proofs, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        _write_new(archive / APPEND_PROOF_FILE, payload)
        receipt.update(append_baseline_proofs=append_proofs,
                       append_baseline_proof_archive={"filename": APPEND_PROOF_FILE,
                           "sha256": hashlib.sha256(payload).hexdigest()})
    return receipt


def verify_registry_archive(folder, receipt):
    """Verify immutable evidence only; later valid registry appends are allowed."""
    if (not isinstance(folder, (str, Path)) or not isinstance(receipt, dict)
            or not all(isinstance(receipt.get(k), str) for k in
                       ("archive_path", "before_sha256", "after_sha256", "registry_path"))):
        raise ValueError("MARK_REGISTRY_ARCHIVE_IDENTITY")
    folder = _path(folder)
    if (receipt.get("schema_version") != 1
            or _path(receipt["archive_path"]) != folder
            or receipt.get("before_snapshot") != SNAPSHOTS[0]
            or receipt.get("proposed_snapshot") != SNAPSHOTS[1]
            or receipt.get("canonical_entries_modified") is not False):
        raise ValueError("MARK_REGISTRY_ARCHIVE_IDENTITY")
    for key, name in (("before_sha256", SNAPSHOTS[0]), ("after_sha256", SNAPSHOTS[1])):
        if _sha(_path(folder / name)) != _digest(receipt[key]):
            raise ValueError("MARK_REGISTRY_ARCHIVE_HASH_MISMATCH")
    proofs, proof_archive = receipt.get("append_baseline_proofs"), receipt.get("append_baseline_proof_archive")
    if proofs is not None or proof_archive is not None:
        if (not isinstance(proofs, list) or not proofs or not isinstance(proof_archive, dict)
                or proof_archive.get("filename") != APPEND_PROOF_FILE
                or _sha(_path(folder / APPEND_PROOF_FILE)) != _digest(proof_archive.get("sha256"))
                or json.loads((folder / APPEND_PROOF_FILE).read_text(encoding="utf-8")) != proofs):
            raise ValueError("MARK_REGISTRY_APPEND_PROOF_ARCHIVE_MISMATCH")
    elif (folder / APPEND_PROOF_FILE).exists():
        raise ValueError("MARK_REGISTRY_UNDECLARED_APPEND_PROOF")
    return {"status": "VERIFIED", "before_sha256": receipt["before_sha256"],
            "after_sha256": receipt["after_sha256"]}


def publish_registry_update(receipt):
    """Publish within the caller's PREPARED barrier; refuse a changed registry."""
    verify_registry_archive(receipt.get("archive_path") if isinstance(receipt, dict) else None, receipt)
    archive = _path(receipt["archive_path"])
    registry = _path(receipt["registry_path"])
    if registry.is_relative_to(archive) or _sha(registry) != receipt["before_sha256"]:
        raise ValueError("MARK_REGISTRY_CHANGED_BEFORE_PUBLICATION")
    payload = (archive / receipt["proposed_snapshot"]).read_bytes()
    temporary = registry.with_name(registry.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        _write_new(temporary, payload)
        if _sha(registry) != receipt["before_sha256"]:
            raise ValueError("MARK_REGISTRY_CHANGED_BEFORE_PUBLICATION")
        os.replace(temporary, registry)
        if _sha(registry) != receipt["after_sha256"]:
            raise ValueError("MARK_REGISTRY_PUBLISHED_HASH_MISMATCH")
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"status": "PUBLISHED", "sha256": receipt["after_sha256"]}
