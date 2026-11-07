"""Evidence-backed versions of observed 5m futures and 1h mark working views.

No network, model changes, orders, funding corrections or synthetic bars. The
caller is the single managed input writer, holds OnlineInputs._lock, and has
finished refresh. Every recovery skips that decision cohort; the next cohort
must refresh and validate inputs again. Historical replay must select the old
archived view, never the current view before its recorded effective time.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import numpy as np
import pandas as pd

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
PROVENANCE = ("first_received_at", "receipt_sha256", "receipt_time_basis", "clock_evidence_sha256")
PRICE_FIELDS = {1: "open", 2: "high", 3: "low", 4: "close", 5: "volume",
                7: "quote_volume", 8: "count", 9: "taker_buy_base_volume", 10: "taker_buy_quote_volume"}
REVISION_SPECS = {
    ("futures", "5m"): {"endpoint": "/fapi/v1/klines", "directory": "intraday", "step_ms": 300000},
    ("mark", "1h"): {"endpoint": "/fapi/v1/markPriceKlines", "directory": "auxiliary", "step_ms": 3600000},
}
SAFE_ERROR = "REVISION_VIEW_NOT_COMMITTED"
_RECOVERY_LOCK = threading.Lock()


class RevisionSafetyError(ValueError):
    pass


def _time(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError("INVALID_RECEIPT_TIME")
    return stamp.tz_convert("UTC")


def _now(value):
    return _time(value() if callable(value) else value if value is not None else datetime.now(timezone.utc))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_json(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _new_json(path, value):
    with Path(path).open("xb") as handle:
        handle.write(_bytes_json(value))
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    _new_json(temporary, value)
    os.replace(temporary, path)


def _copy_new(source, target):
    """Archive destinations are write-once, even if a path already exists."""
    with Path(source).open("rb") as original, Path(target).open("xb") as copy:
        shutil.copyfileobj(original, copy)
        copy.flush()
        os.fsync(copy.fileno())


def _paths(runtime):
    runtime = Path(runtime).resolve()
    features = runtime / "features"
    return runtime, features, features / "raw_data", features / "revisions"


def ensure_view_committed(runtime):
    """Raise a fixed safety error for any incomplete/corrupt recovery marker."""
    _, _, _, revisions = _paths(runtime)
    marker = revisions / "view_state.json"
    if not marker.exists():
        return {"status": "NO_REVISIONS"}
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("status") != "COMMITTED" or state.get("schema_version") != 1:
            raise ValueError(SAFE_ERROR)
        folder = (revisions / state["revision_id"]).resolve()
        if folder.parent != revisions.resolve() or folder.is_symlink():
            raise ValueError(SAFE_ERROR)
        manifest = folder / "manifest.json"
        if _sha(manifest) != state["manifest_sha256"]:
            raise ValueError(SAFE_ERROR)
        evidence = json.loads(manifest.read_text(encoding="utf-8"))
        if (evidence.get("status") != "COMMITTED" or evidence.get("revision_id") != state["revision_id"]
                or evidence.get("view_effective_at") != state.get("view_effective_at")):
            raise ValueError(SAFE_ERROR)
        if _inventory(folder / "raw_data") != evidence["before_files"]:
            raise ValueError(SAFE_ERROR)
        if _sha(folder / "prepared.json") != evidence["prepared_sha256"]:
            raise ValueError(SAFE_ERROR)
        for relative, digest in evidence["after_files"].items():
            if digest != evidence["before_files"].get(relative):
                proposed = folder / "proposed_raw_data" / relative
                if not proposed.resolve().is_relative_to(folder) or _sha(proposed) != digest:
                    raise ValueError(SAFE_ERROR)
        for backup in evidence["sqlite_backups"]:
            snapshot = Path(backup["snapshot_path"])
            if snapshot.resolve().parent != folder or _sha(snapshot) != backup["sha256"]:
                raise ValueError(SAFE_ERROR)
        if evidence.get("mark_registry_update"):
            from .mark_revision_registry import verify_registry_archive
            verify_registry_archive(folder, evidence["mark_registry_update"])
        if evidence.get("recovery_validation_file"):
            recovery = folder / evidence["recovery_validation_file"]
            if recovery.resolve().parent != folder or recovery.is_symlink() or _sha(recovery) != evidence["recovery_validation_sha256"]:
                raise ValueError(SAFE_ERROR)
        _time(state["view_effective_at"])
        return state
    except (OSError, ValueError, KeyError, TypeError):
        raise RevisionSafetyError(SAFE_ERROR) from None


def _error_sources(errors):
    if not isinstance(errors, list) or not errors:
        return []
    sources = []
    for error in errors:
        if not isinstance(error, dict):
            raise ValueError("UNSUPPORTED_REVISION_ERROR")
        parts = str(error.get("source", "")).split(":")
        if (len(parts) != 3 or (parts[0], parts[2]) not in REVISION_SPECS or parts[1] not in SYMBOLS
                or error.get("error") not in ("CONFLICTING_CLOSED_OBSERVATION", "ValueError:CONFLICTING_CLOSED_OBSERVATION")):
            raise ValueError("UNSUPPORTED_REVISION_ERROR")
        sources.append(tuple(parts))
    return sorted(set(sources))


def _inventory(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("RAW_VIEW_MISSING_OR_SYMLINK")
    paths = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("RAW_VIEW_SYMLINK")
        if path.is_file():
            paths.append(path)
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(paths)}


def _read_cache(path, step_ms=300000):
    frame = pd.read_parquet(path)
    required = {"date", "open_time", "close_time", *PRICE_FIELDS.values(), *PROVENANCE}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("REVISION_CACHE_SCHEMA")
    dates = pd.to_datetime(frame.date, utc=True)
    if dates.duplicated().any() or not dates.is_monotonic_increasing or dates.isna().any():
        raise ValueError("REVISION_CACHE_TIMELINE")
    if not np.isfinite(frame[list(PRICE_FIELDS.values())].to_numpy(dtype=float)).all():
        raise ValueError("REVISION_CACHE_NONFINITE")
    if not pd.to_datetime(frame.open_time, utc=True).equals(dates):
        raise ValueError("REVISION_CACHE_OPEN_TIME")
    if (dates.astype("int64") % (step_ms * 1_000_000)).any():
        raise ValueError("REVISION_CACHE_GRID")
    expected_close = dates + pd.Timedelta(milliseconds=step_ms-1)
    if not pd.to_datetime(frame.close_time, utc=True).equals(expected_close):
        raise ValueError("REVISION_CACHE_CLOSE_TIME")
    return frame


def _query(url, symbol, kind="futures", interval="5m"):
    spec = REVISION_SPECS[(kind, interval)]
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "fapi.binance.com" or parsed.path != spec["endpoint"] or parsed.fragment:
        raise ValueError("UNSUPPORTED_RECEIPT_ENDPOINT")
    params = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    if set(params) != {"symbol", "interval", "startTime", "endTime", "limit"} or any(len(v) != 1 for v in params.values()):
        raise ValueError("UNSUPPORTED_RECEIPT_PARAMETERS")
    params = {k: v[0] for k, v in params.items()}
    if params["symbol"] != symbol or params["interval"] != interval:
        raise ValueError("RECEIPT_SYMBOL_OR_INTERVAL")
    start, end, limit = (int(params[k]) for k in ("startTime", "endTime", "limit"))
    if start < 0 or end < start or not 1 <= limit <= 1000:
        raise ValueError("RECEIPT_REQUEST_RANGE")
    return start, end


def _raw_bar(raw, start, end, received, now, step_ms=300000):
    if not isinstance(raw, list) or len(raw) != 12 or any(isinstance(v, bool) for v in raw):
        raise ValueError("INVALID_RAW_KLINE")
    try:
        numbers = tuple(Decimal(str(v)) for v in raw)
    except InvalidOperation:
        raise ValueError("INVALID_RAW_KLINE_NUMBER") from None
    if not all(v.is_finite() for v in numbers):
        raise ValueError("NONFINITE_RAW_KLINE")
    if any(numbers[i] != numbers[i].to_integral_value() for i in (0, 6, 8)):
        raise ValueError("NONINTEGER_RAW_KLINE")
    opened = int(numbers[0])
    if opened % step_ms or int(numbers[6]) != opened + step_ms-1 or not start <= opened <= end:
        raise ValueError("RAW_KLINE_TIME_RANGE")
    closed = pd.to_datetime(opened + step_ms, unit="ms", utc=True)
    if closed > received or closed > now:
        raise ValueError("RAW_KLINE_NOT_CLOSED")
    if min(numbers[1:5]) <= 0 or numbers[2] < max(numbers[1], numbers[3], numbers[4]) or numbers[3] > min(numbers[1], numbers[2], numbers[4]):
        raise ValueError("INVALID_RAW_OHLC")
    if any(numbers[i] < 0 for i in (5, 7, 8, 9, 10)):
        raise ValueError("NEGATIVE_RAW_FLOW")
    return opened, numbers


def _receipts(path, symbol, now, needed_opens, kind="futures", interval="5m"):
    if not path.is_file():
        raise ValueError("RECEIPTS_DATABASE_MISSING")
    observations = {opened: [] for opened in needed_opens}
    spec = REVISION_SPECS[(kind, interval)]
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id,url,requested_at,received_at,sha256,http_status,raw,time_basis,clock_evidence_sha256 "
                            "FROM receipts WHERE http_status=200 AND url LIKE ? AND url LIKE ? AND url LIKE ? ORDER BY id DESC",
                            (f"%symbol={symbol}%", f"%interval={interval}%", f"%{spec['endpoint']}?%"))
        for source in rows:
            source = dict(source)
            start, end = _query(source["url"], symbol, kind, interval)
            raw_bytes = bytes(source["raw"])
            if hashlib.sha256(raw_bytes).hexdigest() != source["sha256"]:
                raise ValueError("RAW_RECEIPT_HASH_MISMATCH")
            requested, received = _time(source["requested_at"]), _time(source["received_at"])
            if received < requested or received > now or requested > now:
                raise ValueError("RECEIPT_FUTURE_OR_TIME_ORDER")
            basis = source["time_basis"] or "SYSTEM_UTC"
            if basis not in ("SYSTEM_UTC", "CALIBRATED_UTC_UPPER_BOUND"):
                raise ValueError("UNSUPPORTED_RECEIPT_CLOCK")
            evidence = source["clock_evidence_sha256"]
            if basis == "CALIBRATED_UTC_UPPER_BOUND" and (not isinstance(evidence, str) or len(evidence) != 64 or any(c not in "0123456789abcdef" for c in evidence)):
                raise ValueError("CALIBRATED_RECEIPT_EVIDENCE_MISSING")
            payload = json.loads(raw_bytes)
            if not isinstance(payload, list):
                raise ValueError("INVALID_RECEIPT_PAYLOAD")
            seen = set()
            for raw in payload:
                opened, numbers = _raw_bar(raw, start, end, received, now, spec["step_ms"])
                if opened in seen:
                    raise ValueError("DUPLICATE_RECEIPT_BAR")
                seen.add(opened)
                if opened in observations:
                    observations[opened].append({"id": source["id"], "url": source["url"],
                        "sha256": source["sha256"], "requested_at": requested.isoformat(),
                        "received_at": received.isoformat(), "time_basis": basis,
                        "clock_evidence_sha256": evidence, "raw_bar": raw, "numbers": numbers})
    for values in observations.values():
        values.sort(key=lambda r: (_time(r["received_at"]), r["id"]), reverse=True)
    return observations


def _public_receipt(receipt):
    return {k: v for k, v in receipt.items() if k != "numbers"}


def _plans(runtime, sources, now, *, raw_root_override=None, receipts_path_override=None):
    _, features, raw_root, _ = _paths(runtime)
    raw_root = Path(raw_root_override) if raw_root_override is not None else raw_root
    receipts_path = Path(receipts_path_override) if receipts_path_override is not None else features / "http_receipts.sqlite3"
    plans, pending = [], []
    for kind, symbol, interval in sources:
        spec = REVISION_SPECS[(kind, interval)]
        source_name = f"{kind}:{symbol}:{interval}"
        path = raw_root / spec["directory"] / f"{symbol}_{interval}_{kind}.parquet"
        old = _read_cache(path, spec["step_ms"])
        opens = (pd.to_datetime(old.date, utc=True).astype("int64") // 1_000_000).tolist()
        evidence = _receipts(receipts_path, symbol, now, opens, kind, interval)
        updated, changes = old.copy(deep=True), []
        seen = False
        for index, opened in zip(old.index, opens):
            receipts = evidence[opened]
            if not receipts:
                continue
            seen = True
            latest = receipts[0]
            differs = [field for raw_index, field in PRICE_FIELDS.items()
                       if float(old.at[index, field]) != float(latest["numbers"][raw_index])]
            if not differs:
                continue
            original = next((r for r in receipts if r["sha256"] == old.at[index, "receipt_sha256"]
                             and _time(r["received_at"]) == _time(old.at[index, "first_received_at"])), None)
            if original is None:
                raise ValueError("CACHE_ORIGINAL_RECEIPT_MISSING")
            if any(float(old.at[index, field]) != float(original["numbers"][raw_index])
                   for raw_index, field in PRICE_FIELDS.items()):
                raise ValueError("CACHE_ORIGINAL_RECEIPT_VALUE_MISMATCH")
            old_clock_hash = old.at[index, "clock_evidence_sha256"]
            old_clock_hash = None if pd.isna(old_clock_hash) else str(old_clock_hash)
            if (old.at[index, "receipt_time_basis"] != original["time_basis"]
                    or old_clock_hash != original["clock_evidence_sha256"]):
                raise ValueError("CACHE_ORIGINAL_RECEIPT_PROVENANCE_MISMATCH")
            confirm = None
            for earlier in receipts[1:]:
                if earlier["numbers"] != latest["numbers"]:
                    break
                if (earlier["id"] != latest["id"] and earlier["requested_at"] != latest["requested_at"]
                        and _time(latest["requested_at"]) >= _time(earlier["received_at"])
                        and (_time(latest["received_at"]) - _time(earlier["received_at"])).total_seconds() >= 60):
                    confirm = earlier
                    break
            if confirm is None:
                pending.append({"source": source_name, "bar_open": _time(old.at[index, "date"]).isoformat(),
                                "reason": "TWO_INDEPENDENT_MATCHING_RECEIPTS_60S_APART_REQUIRED"})
                continue
            before = {field: float(old.at[index, field]) for field in PRICE_FIELDS.values()}
            after = {field: float(latest["numbers"][i]) for i, field in PRICE_FIELDS.items()}
            for field, value in after.items():
                updated.at[index, field] = value
            updated.at[index, "first_received_at"] = _time(latest["received_at"])
            updated.at[index, "receipt_sha256"] = latest["sha256"]
            updated.at[index, "receipt_time_basis"] = latest["time_basis"]
            updated.at[index, "clock_evidence_sha256"] = latest["clock_evidence_sha256"]
            provenance_before = {field: (None if pd.isna(old.at[index, field]) else
                _time(old.at[index, field]).isoformat() if field == "first_received_at" else str(old.at[index, field])) for field in PROVENANCE}
            changes.append({"symbol": symbol, "kind": kind, "interval": interval,
                "source_endpoint": "https://fapi.binance.com" + spec["endpoint"],
                "bar_open": _time(old.at[index, "date"]).isoformat(),
                "changed_price_fields": differs, "before": before, "after": after,
                "provenance_before": provenance_before,
                "provenance_after": {"first_received_at": latest["received_at"], "receipt_sha256": latest["sha256"],
                    "receipt_time_basis": latest["time_basis"], "clock_evidence_sha256": latest["clock_evidence_sha256"]},
                "original_receipt": _public_receipt(original),
                "confirmation_receipts": [_public_receipt(confirm), _public_receipt(latest)]})
        if not seen:
            pending.append({"source": source_name, "reason": "NO_MATCHING_RECEIPT_BARS"})
        if changes:
            # Only existing rows and the explicit price/provenance fields may
            # differ. Dates, other columns, shape and ordering remain identical.
            if old.shape != updated.shape or list(old.columns) != list(updated.columns):
                raise ValueError("REVISION_SHAPE_CHANGED")
            allowed = set(PRICE_FIELDS.values()) | set(PROVENANCE)
            pd.testing.assert_frame_equal(old.drop(columns=list(allowed)), updated.drop(columns=list(allowed)))
            changed_dates = {_time(c["bar_open"]) for c in changes}
            untouched = ~pd.to_datetime(old.date, utc=True).isin(changed_dates)
            pd.testing.assert_frame_equal(old.loc[untouched], updated.loc[untouched])
            plans.append({"symbol": symbol, "kind": kind, "interval": interval,
                          "path": path, "relative_path": path.relative_to(raw_root).as_posix(),
                          "before_sha256": _sha(path), "frame": updated, "changes": changes})
    return plans, pending


def _backup_sqlite(source, target):
    if not source.is_file():
        raise ValueError("AUDIT_DATABASE_MISSING")
    # SQLite backup includes committed WAL contents; copying only the main file
    # could omit precisely the receipts/decisions this recovery must preserve.
    with target.open("xb"):
        pass
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
        with sqlite3.connect(target) as backup:
            original.backup(backup)
    return {"source_path": str(source), "snapshot_path": str(target), "sha256": _sha(target)}


def _disk_budget(runtime, raw_root, before_files, plans):
    raw_bytes = sum((raw_root / relative).stat().st_size for relative in before_files)
    database_bytes = 0
    for source in (runtime / "features/http_receipts.sqlite3", runtime / "observations.sqlite3"):
        if not source.is_file():
            raise ValueError("AUDIT_DATABASE_MISSING")
        physical = sum(p.stat().st_size for p in (source, Path(str(source)+"-wal"), Path(str(source)+"-shm")) if p.exists())
        with sqlite3.connect(source.as_uri()+"?mode=ro", uri=True) as conn:
            visible_pages = conn.execute("PRAGMA page_count").fetchone()[0] * conn.execute("PRAGMA page_size").fetchone()[0]
        database_bytes += max(physical, visible_pages)
    # Both immutable proposed files and replacement temporaries need space.
    # In-memory size is a conservative alternative to compressed file length.
    proposed_bytes = 2*sum(max(plan["path"].stat().st_size,
                              int(plan["frame"].memory_usage(index=True, deep=True).sum())) for plan in plans)
    reserve = 256*1024*1024
    needed = raw_bytes + database_bytes + proposed_bytes + reserve + 1024*1024
    return {"required_free_bytes": needed, "available_free_bytes": shutil.disk_usage(raw_root).free,
            "raw_snapshot_bytes": raw_bytes, "sqlite_backup_bytes": database_bytes,
            "proposed_and_temporary_bytes": proposed_bytes, "reserve_bytes": reserve}


def _audit_sqlite_backup(source, snapshot):
    """Read-only proof that an archived database is an intact live prefix."""
    if not source.is_file() or not snapshot.is_file() or source.is_symlink() or snapshot.is_symlink():
        raise ValueError("RECOVERY_SQLITE_BACKUP_MISSING_OR_SYMLINK")
    counts = {}
    with sqlite3.connect(snapshot.as_uri()+"?mode=ro", uri=True) as archived, \
            sqlite3.connect(source.as_uri()+"?mode=ro", uri=True) as current:
        if archived.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("RECOVERY_SQLITE_CORRUPT")
        schemas = dict(archived.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
        live_schemas = dict(current.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
        if not schemas or schemas != live_schemas:
            raise ValueError("RECOVERY_SQLITE_SCHEMA_MISMATCH")
        if snapshot.name == "receipts.sqlite3":
            columns = {row[1] for row in archived.execute("PRAGMA table_info(receipts)")}
            if not {"id", "url", "requested_at", "received_at", "sha256", "http_status", "raw",
                    "time_basis", "clock_evidence_sha256"}.issubset(columns):
                raise ValueError("RECOVERY_SQLITE_RECEIPT_SCHEMA")
        for name in schemas:
            # Only known append-only observation tables are recoverable. A
            # backoff table is mutable; validate its schema/integrity but never
            # treat it as an observation prefix or restore it into the live DB.
            if name == "backoff" and snapshot.name == "receipts.sqlite3":
                continue
            if name not in ("receipts", "snapshots", "decisions"):
                raise ValueError("RECOVERY_SQLITE_UNSUPPORTED_TABLE")
            primary = [row[1] for row in archived.execute('PRAGMA table_info("'+name+'")') if row[5]]
            if primary != ["id"]:
                raise ValueError("RECOVERY_SQLITE_PRIMARY_KEY")
            maximum = archived.execute('SELECT MAX(id) FROM "'+name+'"').fetchone()[0]
            old_rows = archived.execute('SELECT * FROM "'+name+'" ORDER BY id')
            live_rows = current.execute('SELECT * FROM "'+name+'" WHERE id<=? ORDER BY id', (maximum,))
            count = 0
            while True:
                old_part, live_part = old_rows.fetchmany(128), live_rows.fetchmany(128)
                if old_part != live_part:
                    raise ValueError("RECOVERY_SQLITE_OBSERVATION_PREFIX_MISMATCH")
                count += len(old_part)
                if not old_part:
                    break
            counts[name] = count
    return {"source_path": str(source), "snapshot_path": str(snapshot), "sha256": _sha(snapshot),
            "verified_append_only_prefix_rows": counts}


def _recovery_registry_receipt(registry_path, raw_root, plans, after_files, archive):
    """Reconstruct exactly the existing registry proposal without writing it."""
    from .mark_revision_registry import ENDPOINT, ROOT, SNAPSHOTS, _frame_metadata, verify_registry_archive
    selected = [p for p in plans if p["kind"] == "mark"]
    if not selected:
        if any((archive / name).exists() for name in SNAPSHOTS):
            raise ValueError("RECOVERY_UNEXPECTED_MARK_REGISTRY")
        return None
    if registry_path is None:
        raise ValueError("MARK_REGISTRY_REQUIRED")
    registry = Path(registry_path).resolve()
    if not registry.is_file() or registry.is_symlink() or registry.is_relative_to(archive):
        raise ValueError("RECOVERY_MARK_REGISTRY_PATH")
    before_path, proposed_path = [archive / name for name in SNAPSHOTS]
    if before_path.is_symlink() or proposed_path.is_symlink():
        raise ValueError("RECOVERY_MARK_REGISTRY_ARCHIVE_SYMLINK")
    before, proposed = [json.loads(path.read_text(encoding="utf-8")) for path in (before_path, proposed_path)]
    expected = deepcopy(before)
    entries = expected.get("research_references")
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        raise ValueError("RECOVERY_MARK_REGISTRY_SCHEMA")
    updates = []
    for plan in selected:
        symbol, relative = plan["symbol"], plan["relative_path"]
        target = (raw_root / relative).resolve()
        matches = [e for e in entries if isinstance(e.get("local_path"), str) and
                   (Path(e["local_path"]) if Path(e["local_path"]).is_absolute() else ROOT / e["local_path"]).resolve() == target]
        if len(matches) != 1:
            raise ValueError("RECOVERY_MARK_REGISTRY_IDENTITY")
        entry = matches[0]
        identifier = entry.get("dataset_id")
        if (not isinstance(identifier, str) or re.fullmatch("argon4_local_v4_online_mark_"+symbol+r"_\d{8}", identifier) is None
                or sum(e.get("dataset_id") == identifier for e in entries) != 1
                or entry.get("pair") != symbol[:-4]+"/USDT:USDT" or entry.get("timeframe") != "1h"
                or entry.get("candle_type") != "mark" or entry.get("source_endpoint") != ENDPOINT
                or entry.get("verification_status") != "VERIFIED_PERPETUAL_LABEL"
                or entry.get("installed") is not False or entry.get("research_only") is not True
                or entry.get("gate_effect") != "NONE" or entry.get("sha256") != plan["before_sha256"]):
            raise ValueError("RECOVERY_MARK_REGISTRY_IDENTITY")
        after = after_files[relative]
        entry.update(_frame_metadata(plan["frame"]), sha256=after)
        entry["revision_provenance"] = {"revision_id": archive.name, "manifest_path": str(archive / "manifest.json"),
            "before_sha256": plan["before_sha256"], "after_sha256": after, "source_endpoint": ENDPOINT,
            "future_cohorts_only": True, "confirmation_policy": "TWO_INDEPENDENT_MATCHING_RAW_RECEIPTS_AT_LEAST_60_SECONDS_APART"}
        updates.append({"dataset_id": identifier, "symbol": symbol, "relative_path": relative,
                        "before_sha256": plan["before_sha256"], "after_sha256": after})
    if expected != proposed or _sha(registry) != _sha(proposed_path):
        raise ValueError("RECOVERY_MARK_REGISTRY_NOT_EXACT_PROPOSAL")
    receipt = {"schema_version": 1, "registry_path": str(registry), "archive_path": str(archive),
               "before_sha256": _sha(before_path), "after_sha256": _sha(proposed_path),
               "before_snapshot": SNAPSHOTS[0], "proposed_snapshot": SNAPSHOTS[1],
               "updates": updates, "canonical_entries_modified": False}
    verify_registry_archive(archive, receipt)
    return receipt


def recover_prepared_price_revision(runtime, *, mark_registry_path=None, cutover_now=None, apply=False):
    """Finish only a fully applied, evidence-matched PREPARED cutover.

    The caller must stop the managed writer. Default operation is read-only.
    No raw file, registry, receipt, decision or order is replaced. Every partial
    cutover is refused; the original archive remains immutable. Applying needs
    a fresh calibrated-time callback invoked AFTER lengthy evidence checks.
    """
    if not _RECOVERY_LOCK.acquire(blocking=False):
        return {"status": "BLOCKED", "reason": "REVISION_RECOVERY_BUSY"}
    archive = None
    try:
        runtime, features, raw_root, revisions = _paths(runtime)
        marker_path = revisions / "view_state.json"
        marker_bytes = marker_path.read_bytes()
        marker = json.loads(marker_bytes)
        if marker.get("status") == "COMMITTED":
            return {**ensure_view_committed(runtime), "recovery_action": "NONE_ALREADY_COMMITTED"}
        if marker.get("status") != "PREPARED" or marker.get("schema_version") != 1:
            raise ValueError("RECOVERY_NOT_PREPARED")
        archive = (revisions / marker["revision_id"]).resolve()
        if archive.parent != revisions.resolve() or archive.is_symlink() or (archive / "manifest.json").exists():
            raise ValueError("RECOVERY_ARCHIVE_OR_MANIFEST_AMBIGUOUS")
        prepared_path = archive / "prepared.json"
        if prepared_path.is_symlink():
            raise ValueError("RECOVERY_PREPARED_SYMLINK")
        prepared_bytes = prepared_path.read_bytes()
        prepared = json.loads(prepared_bytes)
        if (prepared.get("status") != "PREPARED" or prepared.get("schema_version") != 1
                or prepared.get("revision_id") != marker["revision_id"]
                or prepared.get("validation_at") != marker.get("validation_at")
                or Path(prepared["data_root"]).resolve() != raw_root
                or Path(prepared["archive_path"]).resolve() != archive
                or prepared.get("current_cohort_must_be_skipped") is not True
                or prepared.get("next_cohort_requires_fresh_validation") is not True):
            raise ValueError("RECOVERY_PREPARED_IDENTITY")
        validation_at = _time(prepared["validation_at"])
        if _inventory(archive / "raw_data") != prepared["before_files"]:
            raise ValueError("RECOVERY_BEFORE_ARCHIVE_HASH_MISMATCH")
        changes = prepared["changes"]
        sources = _error_sources([{"source": f"{c['kind']}:{c['symbol']}:{c['interval']}",
                                   "error": "CONFLICTING_CLOSED_OBSERVATION"} for c in changes])
        if not sources:
            raise ValueError("RECOVERY_NO_SUPPORTED_CHANGES")
        backups = [_audit_sqlite_backup(features / "http_receipts.sqlite3", archive / "receipts.sqlite3"),
                   _audit_sqlite_backup(runtime / "observations.sqlite3", archive / "observations.sqlite3")]
        plans, pending = _plans(runtime, sources, validation_at, raw_root_override=archive / "raw_data",
                                receipts_path_override=archive / "receipts.sqlite3")
        if pending or [c for plan in plans for c in plan["changes"]] != changes:
            raise ValueError("RECOVERY_RECEIPT_PROOF_DOES_NOT_REPRODUCE_PLAN")
        proposed = _inventory(archive / "proposed_raw_data")
        if set(proposed) != {plan["relative_path"] for plan in plans}:
            raise ValueError("RECOVERY_PROPOSED_FILE_SET")
        after_files = dict(prepared["before_files"])
        for plan in plans:
            relative = plan["relative_path"]
            proposed_frame = pd.read_parquet(archive / "proposed_raw_data" / relative)
            pd.testing.assert_frame_equal(plan["frame"], proposed_frame, check_exact=True)
            fields = list(PRICE_FIELDS.values())
            if not np.array_equal(plan["frame"][fields].to_numpy(dtype=np.float64).view(np.uint64),
                                  proposed_frame[fields].to_numpy(dtype=np.float64).view(np.uint64)):
                raise ValueError("RECOVERY_PROPOSED_PRICE_BITS_DIFFER_FROM_RECEIPTS")
            after_files[relative] = proposed[relative]
        if _inventory(raw_root) != after_files:
            raise ValueError("RECOVERY_CURRENT_VIEW_NOT_FULLY_APPLIED_PROPOSAL")
        registry_update = _recovery_registry_receipt(mark_registry_path, raw_root, plans, after_files, archive)
        result = {"status": "VERIFIED_PREPARED_CUTOVER", "revision_id": archive.name,
                  "archive_path": str(archive), "prepared_sha256": hashlib.sha256(prepared_bytes).hexdigest(),
                  "after_files": after_files, "sqlite_backups": backups, "mark_registry_update": registry_update,
                  "current_cohort_must_be_skipped": True, "next_cohort_requires_fresh_validation": True,
                  "raw_registry_and_observations_modified": False}
        if not apply:
            return result
        if not callable(cutover_now):
            raise ValueError("RECOVERY_FRESH_CUTOVER_CLOCK_CALLBACK_REQUIRED")
        # Revalidate publication identities before obtaining fresh time; no
        # historical effective time is inferred from the interrupted attempt.
        if marker_path.read_bytes() != marker_bytes or prepared_path.read_bytes() != prepared_bytes:
            raise ValueError("RECOVERY_EVIDENCE_CHANGED")
        if _inventory(raw_root) != after_files:
            raise ValueError("RECOVERY_CURRENT_VIEW_CHANGED")
        if registry_update and _sha(Path(registry_update["registry_path"])) != registry_update["after_sha256"]:
            raise ValueError("RECOVERY_MARK_REGISTRY_CHANGED")
        effective_at = _now(cutover_now)
        if effective_at < validation_at:
            raise ValueError("CLOCK_REVERSED_DURING_RECOVERY")
        recovery_name = "recovery_validation_"+uuid.uuid4().hex+".json"
        _new_json(archive / recovery_name, {**result, "view_effective_at": effective_at.isoformat(),
                  "recovery_policy": "FINISH_EXACTLY_APPLIED_PROPOSAL_AT_FRESH_TIME_NO_RETROACTIVE_USE"})
        manifest = {**prepared, "status": "COMMITTED", "view_effective_at": effective_at.isoformat(),
                    "after_files": after_files, "sqlite_backups": backups, "mark_registry_update": registry_update,
                    "prepared_sha256": result["prepared_sha256"],
                    "cutover_clock_mode": "FRESH_CALIBRATED_PREPARED_RECOVERY_CALLBACK",
                    "working_view_for_future_cohorts_only": True,
                    "current_view_must_not_replay_as_of_before": effective_at.isoformat(),
                    "recovery_validation_file": recovery_name,
                    "recovery_validation_sha256": _sha(archive / recovery_name)}
        _new_json(archive / "manifest.json", manifest)
        committed = {"schema_version": 1, "status": "COMMITTED", "revision_id": archive.name,
                     "view_effective_at": effective_at.isoformat(), "manifest_sha256": _sha(archive / "manifest.json")}
        _atomic_json(marker_path, committed)
        return {**result, **committed, "manifest_path": str(archive / "manifest.json")}
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, AssertionError) as exc:
        return {"status": "BLOCKED", "reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                "archive_path": str(archive) if archive else None}
    finally:
        _RECOVERY_LOCK.release()


def reconcile_confirmed_price_revisions(runtime, errors, now=None, *, mark_registry_path=None,
                                       cutover_now=None):
    """All-or-nothing *validation*, followed by an audited, crash-gated cutover.

    Returns COMMITTED / WAITING_CONFIRMATION / NO_CHANGE / BLOCKED. PREPARED
    survives any failure after preparation; ensure_view_committed then prevents
    a partially replaced working view from being used. No automatic rollback,
    conflict tolerance, new bars, historical decision replay, or order mutation.
    """
    if not _RECOVERY_LOCK.acquire(blocking=False):
        return {"status": "BLOCKED", "reason": "REVISION_RECOVERY_BUSY"}
    archive = None
    try:
        prior_view = ensure_view_committed(runtime)
        sources = _error_sources(errors)
        if not sources:
            return {"status": "NO_CHANGE", "reason": "NO_SUPPORTED_CONFLICTS"}
        if any(kind == "mark" for kind, _, _ in sources) and mark_registry_path is None:
            raise ValueError("MARK_REGISTRY_REQUIRED")
        validation_at = _now(now)
        validation_monotonic = time.monotonic()
        runtime, features, raw_root, revisions = _paths(runtime)
        before_files = _inventory(raw_root)
        plans, pending = _plans(runtime, sources, validation_at)
        if pending:
            return {"status": "WAITING_CONFIRMATION", "pending": pending, "changes": [], "data_root": str(raw_root)}
        if not plans:
            return {"status": "NO_CHANGE", "reason": "LATEST_OBSERVED_VALUES_ALREADY_MATCH", "data_root": str(raw_root)}
        if not (runtime / "observations.sqlite3").is_file():
            raise ValueError("AUDIT_DATABASE_MISSING")
        disk_budget = _disk_budget(runtime, raw_root, before_files, plans)
        if disk_budget["available_free_bytes"] < disk_budget["required_free_bytes"]:
            return {"status": "BLOCKED", "reason": "INSUFFICIENT_SNAPSHOT_DISK_SPACE", "disk_budget": disk_budget}
        if _inventory(raw_root) != before_files or any(before_files[p["relative_path"]] != p["before_sha256"] for p in plans):
            raise ValueError("WORKING_VIEW_CHANGED_DURING_VALIDATION")
        revision_id = validation_at.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex
        revisions.mkdir(parents=True, exist_ok=True)
        archive = revisions / revision_id
        archive.mkdir(exist_ok=False)
        changes = [change for plan in plans for change in plan["changes"]]
        prepared = {"schema_version": 1, "status": "PREPARED", "revision_id": revision_id,
                    "validation_at": validation_at.isoformat(), "archive_path": str(archive),
                    "data_root": str(raw_root), "prior_view": prior_view, "changes": changes,
                    "before_files": before_files,
                    "disk_budget": disk_budget,
                    "snapshot_meaning": "PRE_REVISION_WORKING_VIEW_NOT_A_HISTORICAL_DECISION_TIME_SNAPSHOT",
                    "historical_replay_policy": "USE_OLD_ARCHIVED_VIEW_AND_ORIGINAL_DECISION_AS_OF_RECEIPTS_CLOCK_AND_MODEL_HASHES",
                    "model_weights_modified_by_this_operation": False,
                    "collector_model_strategy_source_modified": False,
                    "current_cohort_must_be_skipped": True, "next_cohort_requires_fresh_validation": True}
        _atomic_json(revisions / "view_state.json", {"schema_version": 1, "status": "PREPARED",
                      "revision_id": revision_id, "validation_at": validation_at.isoformat()})
        _new_json(archive / "prepared.json", prepared)
        for relative, digest in before_files.items():
            original, target = raw_root / relative, archive / "raw_data" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_new(original, target)
            if _sha(target) != digest:
                raise ValueError("SNAPSHOT_HASH_MISMATCH")
        backups = [_backup_sqlite(features / "http_receipts.sqlite3", archive / "receipts.sqlite3"),
                   _backup_sqlite(runtime / "observations.sqlite3", archive / "observations.sqlite3")]
        after_files = dict(before_files)
        for plan in plans:
            proposed = archive / "proposed_raw_data" / plan["relative_path"]
            proposed.parent.mkdir(parents=True, exist_ok=True)
            with proposed.open("xb") as handle:
                plan["frame"].to_parquet(handle, index=False)
                handle.flush()
                os.fsync(handle.fileno())
            after_files[plan["relative_path"]] = _sha(proposed)
        mark_registry_update = None
        if any(plan["kind"] == "mark" for plan in plans):
            from .mark_revision_registry import prepare_registry_update
            mark_registry_update = prepare_registry_update(mark_registry_path, raw_root, plans, after_files, archive)
        if _inventory(raw_root) != before_files:
            raise ValueError("WORKING_VIEW_CHANGED_BEFORE_CUTOVER")
        for plan in plans:
            target = plan["path"]
            temporary = target.with_name(target.name + "." + revision_id + ".tmp")
            _copy_new(archive / "proposed_raw_data" / plan["relative_path"], temporary)
            os.replace(temporary, target)
        if _inventory(raw_root) != after_files:
            raise ValueError("POST_CUTOVER_VIEW_MISMATCH")
        if mark_registry_update:
            from .mark_revision_registry import publish_registry_update
            publish_registry_update(mark_registry_update)
        # This is the actual view cutover time, not the time copying started.
        effective_at = (_now(cutover_now) if cutover_now is not None else _now(now) if callable(now) else
                        validation_at + pd.Timedelta(seconds=time.monotonic()-validation_monotonic))
        if effective_at < validation_at:
            raise ValueError("CLOCK_REVERSED_DURING_RECOVERY")
        manifest = {**prepared, "status": "COMMITTED", "view_effective_at": effective_at.isoformat(),
                    "after_files": after_files, "sqlite_backups": backups,
                    "mark_registry_update": mark_registry_update,
                    "prepared_sha256": _sha(archive / "prepared.json"),
                    "cutover_clock_mode": ("FRESH_CALIBRATED_CUTOVER_CALLBACK" if cutover_now is not None else
                                           "CALLABLE_CURRENT_UTC" if callable(now) else "PROVIDED_UTC_PLUS_MONOTONIC_ELAPSED"),
                    "working_view_for_future_cohorts_only": True,
                    "current_view_must_not_replay_as_of_before": effective_at.isoformat()}
        _new_json(archive / "manifest.json", manifest)
        marker = {"schema_version": 1, "status": "COMMITTED", "revision_id": revision_id,
                  "view_effective_at": effective_at.isoformat(), "manifest_sha256": _sha(archive / "manifest.json")}
        _atomic_json(revisions / "view_state.json", marker)
        return {**marker, "archive_path": str(archive), "manifest_path": str(archive / "manifest.json"),
                "data_root": str(raw_root), "changes": changes, "current_cohort_must_be_skipped": True,
                "next_cohort_requires_fresh_validation": True}
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, AssertionError) as exc:
        return {"status": "BLOCKED", "reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                "archive_path": str(archive) if archive else None}
    finally:
        _RECOVERY_LOCK.release()
