"""Evidence-backed versions of observed 5m futures and 1h mark working views.

No network, model changes, orders, funding corrections or synthetic bars. The
caller is the single managed input writer, holds OnlineInputs._lock, and has
finished refresh. Every recovery skips that decision cohort; the next cohort
must refresh and validate inputs again. Historical replay must select the old
archived view, never the current view before its recorded effective time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
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


def _plans(runtime, sources, now):
    _, features, raw_root, _ = _paths(runtime)
    plans, pending = [], []
    for kind, symbol, interval in sources:
        spec = REVISION_SPECS[(kind, interval)]
        source_name = f"{kind}:{symbol}:{interval}"
        path = raw_root / spec["directory"] / f"{symbol}_{interval}_{kind}.parquet"
        old = _read_cache(path, spec["step_ms"])
        opens = (pd.to_datetime(old.date, utc=True).astype("int64") // 1_000_000).tolist()
        evidence = _receipts(features / "http_receipts.sqlite3", symbol, now, opens, kind, interval)
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


def reconcile_confirmed_price_revisions(runtime, errors, now=None, *, mark_registry_path=None):
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
        effective_at = (_now(now) if callable(now) else
                        validation_at + pd.Timedelta(seconds=time.monotonic()-validation_monotonic))
        if effective_at < validation_at:
            raise ValueError("CLOCK_REVERSED_DURING_RECOVERY")
        manifest = {**prepared, "status": "COMMITTED", "view_effective_at": effective_at.isoformat(),
                    "after_files": after_files, "sqlite_backups": backups,
                    "mark_registry_update": mark_registry_update,
                    "prepared_sha256": _sha(archive / "prepared.json"),
                    "cutover_clock_mode": "CALLABLE_CURRENT_UTC" if callable(now) else "PROVIDED_UTC_PLUS_MONOTONIC_ELAPSED",
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
