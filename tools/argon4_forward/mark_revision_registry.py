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
SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}
SNAPSHOTS = ("mark_registry_before.json", "mark_registry_proposed.json")


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
    changed, seen = [], set()
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
            pd.testing.assert_frame_equal(frame, plan["frame"])
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
                or entry.get("gate_effect") != "NONE" or entry.get("sha256") != before):
            raise ValueError("MARK_REGISTRY_REFERENCE_IDENTITY")
        entry.update(metadata, sha256=after)
        entry["revision_provenance"] = {
            "revision_id": archive.name, "manifest_path": str(archive / "manifest.json"),
            "before_sha256": before, "after_sha256": after,
            "source_endpoint": ENDPOINT, "future_cohorts_only": True,
            "confirmation_policy": "TWO_INDEPENDENT_MATCHING_RAW_RECEIPTS_AT_LEAST_60_SECONDS_APART",
        }
        changed.append({"dataset_id": identifier, "symbol": symbol, "relative_path": relative,
                        "before_sha256": before, "after_sha256": after})
    encoded = json.dumps(proposed, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    # Validate everything before either write-once archive file is created.
    if any((archive / name).exists() for name in SNAPSHOTS):
        raise ValueError("MARK_REGISTRY_SNAPSHOT_ALREADY_EXISTS")
    if _sha(registry) != before_sha:
        raise ValueError("MARK_REGISTRY_CHANGED_DURING_PREPARATION")
    _write_new(archive / SNAPSHOTS[0], original)
    _write_new(archive / SNAPSHOTS[1], encoded)
    return {"schema_version": 1, "registry_path": str(registry), "archive_path": str(archive),
            "before_sha256": before_sha, "after_sha256": hashlib.sha256(encoded).hexdigest(),
            "before_snapshot": SNAPSHOTS[0], "proposed_snapshot": SNAPSHOTS[1],
            "updates": changed, "canonical_entries_modified": False}


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
