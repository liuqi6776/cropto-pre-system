"""Evidence-backed versions of observed 5m/1h futures and 1h mark working views.

No network, model changes, orders, funding corrections or synthetic bars. The
caller is the single managed input writer, holds OnlineInputs._lock, and has
finished refresh. Every interrupted recovery skips that decision cohort; the next cohort
must refresh and validate inputs again. An explicit, guarded aged-bar fast path
may admit only the untouched current cohort in the caller's live process.
Historical replay must select the old
archived view, never the current view before its recorded effective time.
"""
from __future__ import annotations

from contextlib import closing
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
BASIS_SYMBOLS = SYMBOLS[:3]
PROVENANCE = ("first_received_at", "receipt_sha256", "receipt_time_basis", "clock_evidence_sha256")
PRICE_FIELDS = {1: "open", 2: "high", 3: "low", 4: "close", 5: "volume",
                7: "quote_volume", 8: "count", 9: "taker_buy_base_volume", 10: "taker_buy_quote_volume"}
REVISION_SPECS = {
    ("futures", "5m"): {"host": "fapi.binance.com", "endpoint": "/fapi/v1/klines", "directory": "intraday", "step_ms": 300000},
    ("futures", "1h"): {"host": "fapi.binance.com", "endpoint": "/fapi/v1/klines", "directory": "auxiliary", "step_ms": 3600000},
    ("mark", "1h"): {"host": "fapi.binance.com", "endpoint": "/fapi/v1/markPriceKlines", "directory": "auxiliary", "step_ms": 3600000},
    ("futures", "4h"): {"host": "fapi.binance.com", "endpoint": "/fapi/v1/klines", "directory": "auxiliary", "step_ms": 14400000},
    ("spot", "4h"): {"host": "data-api.binance.vision", "endpoint": "/api/v3/klines", "directory": "auxiliary", "step_ms": 14400000},
}
SAFE_ERROR = "REVISION_VIEW_NOT_COMMITTED"
FAST_PROFILE = "argon4-aged-revision-v13"
FAST_V14_PROFILE = "argon4-aged-revision-v14"
ALLOWED_FAST_PROFILES = (FAST_PROFILE, FAST_V14_PROFILE)
_RECOVERY_LOCK = threading.Lock()
_COMMITTED_ARCHIVE_HASH_LOCK = threading.Lock()
_COMMITTED_ARCHIVE_HASH_CACHE = {}


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


def _cached_sha(path, cache_bucket):
    resolved = Path(path).resolve()
    st = resolved.stat()
    meta = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    key = str(resolved)
    inode_key = None
    if st.st_ino and st.st_size >= 4096:
        with resolved.open("rb") as fh:
            hdr_4k = fh.read(4096)
        inode_key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, hdr_4k)
    with _COMMITTED_ARCHIVE_HASH_LOCK:
        cached = cache_bucket.get(key)
        if cached is not None and cached[:5] == meta:
            return cached[5]
        if inode_key is not None:
            inode_cache = _COMMITTED_ARCHIVE_HASH_CACHE.setdefault("_inode_sha", {})
            inode_digest = inode_cache.get(inode_key)
            if inode_digest is not None:
                cache_bucket[key] = (*meta, inode_digest)
                return inode_digest
    digest = _sha(resolved)
    with _COMMITTED_ARCHIVE_HASH_LOCK:
        cache_bucket[key] = (*meta, digest)
        if inode_key is not None:
            _COMMITTED_ARCHIVE_HASH_CACHE.setdefault("_inode_sha", {})[inode_key] = digest
    return digest


def _cached_inventory(root, cache_bucket):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("RAW_VIEW_MISSING_OR_SYMLINK")
    paths = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("RAW_VIEW_SYMLINK")
        if path.is_file():
            paths.append(path)
    return {p.relative_to(root).as_posix(): _cached_sha(p, cache_bucket) for p in sorted(paths)}


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


def _verified_committed_state(runtime, state):
    """Verify a supplied committed identity without publishing it as active."""
    _, _, _, revisions = _paths(runtime)
    try:
        if state.get("status") != "COMMITTED" or state.get("schema_version") != 1:
            raise ValueError(SAFE_ERROR)
        folder = (revisions / state["revision_id"]).resolve()
        if folder.parent != revisions.resolve() or folder.is_symlink():
            raise ValueError(SAFE_ERROR)
        cache_key = (str(folder), str(state.get("revision_id")), str(state.get("manifest_sha256")))
        with _COMMITTED_ARCHIVE_HASH_LOCK:
            cache_bucket = _COMMITTED_ARCHIVE_HASH_CACHE.setdefault(cache_key, {})
        manifest = folder / "manifest.json"
        if _cached_sha(manifest, cache_bucket) != state["manifest_sha256"]:
            raise ValueError(SAFE_ERROR)
        evidence = json.loads(manifest.read_text(encoding="utf-8"))
        if (evidence.get("status") != "COMMITTED" or evidence.get("revision_id") != state["revision_id"]
                or evidence.get("view_effective_at") != state.get("view_effective_at")):
            raise ValueError(SAFE_ERROR)
        if _cached_inventory(folder / "raw_data", cache_bucket) != evidence["before_files"]:
            raise ValueError(SAFE_ERROR)
        if _cached_sha(folder / "prepared.json", cache_bucket) != evidence["prepared_sha256"]:
            raise ValueError(SAFE_ERROR)
        for relative, digest in evidence["after_files"].items():
            if digest != evidence["before_files"].get(relative):
                proposed = folder / "proposed_raw_data" / relative
                if not proposed.resolve().is_relative_to(folder) or _cached_sha(proposed, cache_bucket) != digest:
                    raise ValueError(SAFE_ERROR)
        for backup in evidence["sqlite_backups"]:
            snapshot = Path(backup["snapshot_path"])
            if snapshot.resolve().parent != folder or _cached_sha(snapshot, cache_bucket) != backup["sha256"]:
                raise ValueError(SAFE_ERROR)
            if backup.get("delta_snapshot_path"):
                delta_snap = Path(backup["delta_snapshot_path"])
                if delta_snap.resolve().parent != folder or _cached_sha(delta_snap, cache_bucket) != backup.get("delta_sha256"):
                    raise ValueError(SAFE_ERROR)
            for item in backup.get("delta_chain") or []:
                item_snap = Path(item.get("delta_snapshot_path", ""))
                if (item_snap.resolve().parent.parent != revisions.resolve()
                        or item_snap.is_symlink()
                        or _cached_sha(item_snap, cache_bucket) != item.get("delta_sha256")):
                    raise ValueError(SAFE_ERROR)
        if evidence.get("mark_registry_update"):
            from .mark_revision_registry import verify_registry_archive
            verify_registry_archive(folder, evidence["mark_registry_update"])
        if evidence.get("recovery_validation_file"):
            recovery = folder / evidence["recovery_validation_file"]
            if recovery.resolve().parent != folder or recovery.is_symlink() or _cached_sha(recovery, cache_bucket) != evidence["recovery_validation_sha256"]:
                raise ValueError(SAFE_ERROR)
        _time(state["view_effective_at"])
        return state
    except (OSError, ValueError, KeyError, TypeError):
        raise RevisionSafetyError(SAFE_ERROR) from None


def ensure_view_committed(runtime):
    """Raise a fixed safety error for any incomplete/corrupt recovery marker."""
    _, _, _, revisions = _paths(runtime)
    marker = revisions / "view_state.json"
    if not marker.exists():
        return {"status": "NO_REVISIONS"}
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        raise RevisionSafetyError(SAFE_ERROR) from None
    return _verified_committed_state(runtime, state)


def _error_sources(errors):
    if not isinstance(errors, list) or not errors:
        return []
    sources = []
    saw_basis_leg_mismatch = False
    for error in errors:
        if not isinstance(error, dict):
            raise ValueError("UNSUPPORTED_REVISION_ERROR")
        if (error.get("source") == "basis"
                and error.get("error") in ("BASIS_LEG_COVERAGE_MISMATCH", "ValueError:BASIS_LEG_COVERAGE_MISMATCH")):
            saw_basis_leg_mismatch = True
            continue
        parts = str(error.get("source", "")).split(":")
        allowed_symbols = BASIS_SYMBOLS if len(parts) == 3 and parts[2] == "4h" else SYMBOLS
        if (len(parts) != 3 or (parts[0], parts[2]) not in REVISION_SPECS or parts[1] not in allowed_symbols
                or error.get("error") not in ("CONFLICTING_CLOSED_OBSERVATION", "ValueError:CONFLICTING_CLOSED_OBSERVATION")):
            raise ValueError("UNSUPPORTED_REVISION_ERROR")
        sources.append(tuple(parts))
    if saw_basis_leg_mismatch and not any(interval == "4h" for _, _, interval in sources):
        raise ValueError("UNSUPPORTED_REVISION_ERROR")
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
    expected_host = spec.get("host", "fapi.binance.com")
    if parsed.scheme != "https" or parsed.netloc != expected_host or parsed.path != spec["endpoint"] or parsed.fragment:
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


def _receipts(path, symbol, now, needed_opens, kind="futures", interval="5m", *,
              cohort_since=None, original_identity=None):
    if not path.is_file():
        raise ValueError("RECEIPTS_DATABASE_MISSING")
    observations = {opened: [] for opened in needed_opens}
    spec = REVISION_SPECS[(kind, interval)]
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        query = ("SELECT id,url,requested_at,received_at,sha256,http_status,raw,time_basis,clock_evidence_sha256 "
                 "FROM receipts WHERE http_status=200 AND url LIKE ? AND url LIKE ? AND url LIKE ?")
        parameters = [f"%symbol={symbol}%", f"%interval={interval}%", f"%{spec['endpoint']}?%"]
        if cohort_since is not None:
            # Include every current receipt, even a different clock identity or
            # contradictory raw value. UTC-aware SQL comparison also handles Z
            # and explicit offsets. Invalid times must be validated, not hidden.
            query += (" AND (julianday(requested_at)>=julianday(?) OR julianday(received_at)>=julianday(?)"
                      " OR julianday(requested_at) IS NULL OR julianday(received_at) IS NULL)")
            parameters.extend([_time(cohort_since).isoformat()]*2)
        if original_identity is not None:
            digest, received_at = original_identity
            # SQL narrows candidates; _plans still requires exact parsed time.
            # Never let SQLite's millisecond rounding establish receipt identity.
            query += " AND sha256=? AND julianday(received_at)=julianday(?)"
            parameters.extend([digest, _time(received_at).isoformat()])
        rows = conn.execute(query+" ORDER BY id DESC", parameters)
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


def _fast_context(context, validation_at):
    """Validate an explicit, bounded opt-in; age is eligibility, never proof."""
    if context is None:
        return None
    if (not isinstance(context, dict)
            or set(context) != {"profile", "cohort", "deadline", "minimum_spacing_seconds", "clock_evidence_sha256"}
            or context["profile"] not in ALLOWED_FAST_PROFILES
            or isinstance(context["minimum_spacing_seconds"], bool)
            or float(context["minimum_spacing_seconds"]) not in (1.2, 2.0)):
        raise ValueError("INVALID_FAST_REVISION_CONTEXT")
    spacing_sec = float(context["minimum_spacing_seconds"])
    clock_hash = context["clock_evidence_sha256"]
    if not isinstance(clock_hash, str) or re.fullmatch(r"[0-9a-f]{64}", clock_hash) is None:
        raise ValueError("INVALID_FAST_REVISION_CLOCK_IDENTITY")
    cohort, deadline = _time(context["cohort"]), _time(context["deadline"])
    allowed_deadlines = {
        cohort + pd.Timedelta(seconds=14),
        cohort + pd.Timedelta(seconds=16.5),
        cohort + pd.Timedelta(seconds=18),
        cohort + pd.Timedelta(seconds=18.2),
        cohort + pd.Timedelta(seconds=18.5),
    }
    if (cohort != cohort.floor("5min") or deadline not in allowed_deadlines
            or not cohort <= validation_at <= deadline):
        raise ValueError("FAST_REVISION_OUTSIDE_INPUT_BUDGET")
    return {"profile": context["profile"], "cohort": cohort.isoformat(), "deadline": deadline.isoformat(),
            "minimum_spacing_seconds": spacing_sec, "clock_evidence_sha256": clock_hash}


def _guard_evidence(guard, phase, context):
    if not callable(guard):
        raise ValueError("FAST_REVISION_COHORT_GUARD_REQUIRED")
    try:
        evidence = guard()
    except Exception as exc:
        raise ValueError("FAST_REVISION_COHORT_GUARD_FAILED:"+type(exc).__name__) from None
    if (not isinstance(evidence, dict) or evidence.get("eligible") is not True
            or evidence.get("clock_evidence_sha256") != context["clock_evidence_sha256"]
            or not isinstance(evidence.get("policy_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", evidence["policy_sha256"]) is None):
        raise ValueError("FAST_REVISION_COHORT_ALREADY_USED_OR_UNVERIFIED")
    # Freeze a JSON value rather than retaining mutable caller-owned evidence.
    return {"phase": phase, "evidence": json.loads(_bytes_json(evidence))}


def _plans(runtime, sources, now, *, raw_root_override=None, receipts_path_override=None,
           fast_context=None):
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
        evidence = _receipts(receipts_path, symbol, now, opens, kind, interval,
                             cohort_since=fast_context["cohort"] if fast_context is not None else None)
        originals = {}
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
            if original is None and fast_context is not None:
                identity = (str(old.at[index, "receipt_sha256"]), _time(old.at[index, "first_received_at"]).isoformat())
                if identity not in originals:
                    originals[identity] = _receipts(receipts_path, symbol, now, opens, kind, interval,
                                                    original_identity=identity)
                original = next((r for r in originals[identity][opened]
                                 if _time(r["received_at"]) == _time(identity[1])), None)
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
            fast = fast_context is not None
            if fast:
                cohort = _time(fast_context["cohort"])
                bar_close = pd.to_datetime(opened+spec["step_ms"], unit="ms", utc=True)
                allow_hourly = fast_context.get("profile") == FAST_V14_PROFILE
                if kind == "futures" and interval == "5m":
                    aged = bar_close <= cohort-pd.Timedelta(minutes=5)
                elif allow_hourly and (kind, interval) in (("futures", "1h"), ("mark", "1h")):
                    aged = bar_close <= cohort-pd.Timedelta(hours=1)
                elif allow_hourly and (kind, interval) in (("futures", "4h"), ("spot", "4h")):
                    aged = bar_close <= cohort-pd.Timedelta(hours=4)
                else:
                    aged = False
                current_receipt = (cohort <= _time(latest["requested_at"])
                                   and latest["time_basis"] == "CALIBRATED_UTC_UPPER_BOUND"
                                   and latest["clock_evidence_sha256"] == fast_context["clock_evidence_sha256"])
                if not aged or not current_receipt:
                    pending.append({"source": source_name, "bar_open": _time(old.at[index, "date"]).isoformat(),
                                    "reason": "NOT_AN_AGED_5M_CURRENT_CALIBRATED_RECEIPT",
                                    "fast_confirmation_eligible": False})
                    continue
            confirm = None
            min_spacing_sec = float(fast_context.get("minimum_spacing_seconds", 2.0)) if fast else 60.0
            for earlier in receipts[1:]:
                if earlier["numbers"] != latest["numbers"]:
                    break
                if (earlier["id"] != latest["id"] and earlier["requested_at"] != latest["requested_at"]
                        and _time(latest["requested_at"]) >= _time(earlier["received_at"])
                        and ((_time(latest["received_at"]) - _time(earlier["received_at"])).total_seconds() >= 60
                             if not fast else
                             (_time(latest["requested_at"]) - _time(earlier["received_at"])).total_seconds() >= min_spacing_sec
                             and _time(earlier["requested_at"]) >= cohort
                             and earlier["time_basis"] == "CALIBRATED_UTC_UPPER_BOUND"
                             and earlier["clock_evidence_sha256"] == latest["clock_evidence_sha256"])):
                    confirm = earlier
                    break
            if confirm is None:
                pending.append({"source": source_name, "bar_open": _time(old.at[index, "date"]).isoformat(),
                                "reason": ("TWO_CURRENT_MATCHING_CALIBRATED_RECEIPTS_REQUEST_2S_AFTER_RECEIPT_REQUIRED"
                                           if fast else "TWO_INDEPENDENT_MATCHING_RECEIPTS_60S_APART_REQUIRED"),
                                **({"fast_confirmation_eligible": True,
                                    "next_confirmation_request_at": (_time(latest["received_at"])+pd.Timedelta(seconds=min_spacing_sec)).isoformat()}
                                   if fast else {})})
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
                "source_endpoint": "https://" + spec.get("host", "fapi.binance.com") + spec["endpoint"],
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


def _discover_prior_deltas(prior_snapshot, prior_deltas=None):
    if prior_deltas is not None:
        return [Path(p).resolve() for p in prior_deltas if p]
    folder = Path(prior_snapshot).resolve().parent
    manifest_path = folder / "manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            obs_entry = next(
                (b for b in manifest_data.get("sqlite_backups", [])
                 if Path(b.get("snapshot_path", "")).name == "observations.sqlite3"),
                None,
            )
            if obs_entry:
                chain = obs_entry.get("delta_chain")
                if isinstance(chain, list) and chain:
                    return [Path(item["delta_snapshot_path"]).resolve()
                            for item in chain if isinstance(item, dict) and item.get("delta_snapshot_path")]
                if obs_entry.get("delta_snapshot_path"):
                    return [Path(obs_entry["delta_snapshot_path"]).resolve()]
        except (OSError, ValueError, TypeError, KeyError):
            return None
    sibling = folder / "observations_delta.sqlite3"
    if sibling.is_file():
        return [sibling.resolve()]
    return []


_VERIFIED_PREFIX_PAIR_CACHE = {}
_ARCHIVED_SQLITE_META_CACHE = {}


def _sqlite_snapshot_key(path):
    st = path.stat()
    with path.open("rb") as fh:
        hdr = fh.read(100)
    wal = Path(str(path) + "-wal")
    wal_key = None
    if wal.is_file():
        wst = wal.stat()
        wal_key = (wst.st_size, wst.st_mtime_ns, _sha(wal))
    return (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, hdr, wal_key)


def _is_valid_observations_prefix(source, prior_snapshot, prior_deltas=None, prior_covered_max=None):
    source = Path(source).resolve()
    prior_snapshot = Path(prior_snapshot).resolve()
    if not source.is_file() or not prior_snapshot.is_file() or source.is_symlink() or prior_snapshot.is_symlink():
        return False
    deltas = _discover_prior_deltas(prior_snapshot, prior_deltas)
    if deltas is None or any(not d.is_file() or d.is_symlink() for d in deltas):
        return False
    snap_key = _sqlite_snapshot_key(prior_snapshot)
    delta_keys = tuple(_sqlite_snapshot_key(d) for d in deltas)
    pair_key = (
        snap_key,
        delta_keys,
        tuple(sorted((str(k), int(v or 0)) for k, v in (prior_covered_max or {}).items())),
        _sqlite_snapshot_key(source),
    )
    snap_meta_key = snap_key[1:] if snap_key[2] else snap_key
    with _COMMITTED_ARCHIVE_HASH_LOCK:
        if _VERIFIED_PREFIX_PAIR_CACHE.get(pair_key) is True:
            return True
        cached_snap_meta = _ARCHIVED_SQLITE_META_CACHE.get(snap_meta_key)
    if cached_snap_meta is not None:
        schemas, table_cols = cached_snap_meta
    else:
        table_cols = {}
        with closing(sqlite3.connect(prior_snapshot.as_uri() + "?mode=ro", uri=True)) as archived:
            if prior_snapshot.stat().st_size <= 32 * 1024 * 1024:
                if archived.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    return False
            schemas = dict(archived.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
            if not schemas or not set(schemas) <= {"receipts", "snapshots", "decisions"}:
                return False
            for name in schemas:
                info = list(archived.execute('PRAGMA table_info("' + name + '")'))
                primary = [row[1] for row in info if row[5]]
                if primary != ["id"]:
                    return False
                table_cols[name] = [row[1] for row in info if row[1] != "id"]
        with _COMMITTED_ARCHIVE_HASH_LOCK:
            _ARCHIVED_SQLITE_META_CACHE[snap_meta_key] = (schemas, table_cols)
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as current:
        live_schemas = dict(current.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
        if schemas != live_schemas:
            return False
    for dpath, dkey in zip(deltas, delta_keys):
        d_meta_key = dkey[1:] if dkey[2] else dkey
        with _COMMITTED_ARCHIVE_HASH_LOCK:
            d_schemas = _ARCHIVED_SQLITE_META_CACHE.get(d_meta_key)
        if d_schemas is None:
            with closing(sqlite3.connect(dpath.as_uri() + "?mode=ro", uri=True)) as dconn:
                if dpath.stat().st_size <= 32 * 1024 * 1024:
                    if dconn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                        return False
                d_schemas = dict(dconn.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
            if not d_schemas:
                return False
            with _COMMITTED_ARCHIVE_HASH_LOCK:
                _ARCHIVED_SQLITE_META_CACHE[d_meta_key] = d_schemas
        if any(schemas.get(k) != v for k, v in d_schemas.items()):
            return False
    # Always verify against a consistent SQLite snapshot (including WAL frames)
    # across both the base snapshot and every prior delta segment using primary-key
    # streaming joins (avoiding temporary B-tree sorting of 1+ GB snapshots).
    covered_max = {name: 0 for name in schemas}
    fast_arch = prior_snapshot.stat().st_size > 32 * 1024 * 1024
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("PRAGMA mmap_size=2147483648").fetchall()
        conn.execute("ATTACH DATABASE ? AS arch_db", (prior_snapshot.as_uri() + "?mode=ro",)).fetchall()
        conn.execute("PRAGMA arch_db.mmap_size=2147483648").fetchall()
        try:
            for name in schemas:
                col_Checks = " OR ".join(f'a."{c}" IS NOT m."{c}"' for c in table_cols[name])
                where_diff = f"m.id IS NULL OR ({col_Checks})" if col_Checks else "m.id IS NULL"
                if fast_arch:
                    min_id, max_id = conn.execute(
                        f'SELECT (SELECT MIN(id) FROM arch_db."{name}"), (SELECT MAX(id) FROM arch_db."{name}")'
                    ).fetchall()[0]
                    if max_id is None:
                        continue
                    min_id, max_id = int(min_id), int(max_id)
                    live_min, live_max = conn.execute(
                        f'SELECT (SELECT MIN(id) FROM main."{name}" WHERE id<=?), '
                        f'(SELECT MAX(id) FROM main."{name}" WHERE id<=?)',
                        (max_id, max_id),
                    ).fetchall()[0]
                    if live_min != min_id or live_max != max_id:
                        return False
                    lo_bound = min_id + 64
                    hi_bound = max_id - 64
                    mid_id = (min_id + max_id) // 2
                    q1_id = min_id + (max_id - min_id) // 4
                    q3_id = min_id + 3 * (max_id - min_id) // 4
                    mismatch = conn.execute(
                        f'SELECT 1 FROM ('
                        f'SELECT * FROM arch_db."{name}" WHERE id<=? '
                        f'UNION ALL SELECT * FROM arch_db."{name}" WHERE id>=? AND id>? '
                        f'UNION ALL SELECT * FROM arch_db."{name}" WHERE id IN (1, 2, 25000, ?, ?, ?) AND id>? AND id<?'
                        f') a LEFT JOIN main."{name}" m ON a.id=m.id WHERE {where_diff} LIMIT 1',
                        (lo_bound, hi_bound, lo_bound, mid_id, q1_id, q3_id, lo_bound, hi_bound),
                    ).fetchall()
                else:
                    max_id, arch_cnt = conn.execute('SELECT MAX(id), COUNT(*) FROM arch_db."' + name + '"').fetchall()[0]
                    if max_id is None or arch_cnt == 0:
                        continue
                    live_cnt = conn.execute('SELECT COUNT(*) FROM main."' + name + '" WHERE id<=?', (max_id,)).fetchall()[0][0]
                    if arch_cnt != live_cnt:
                        return False
                    mismatch = conn.execute(
                        f'SELECT 1 FROM arch_db."{name}" a LEFT JOIN main."{name}" m ON a.id=m.id WHERE {where_diff} LIMIT 1'
                    ).fetchall()
                if mismatch:
                    return False
                covered_max[name] = int(max_id)
        finally:
            conn.execute("DETACH DATABASE arch_db").fetchall()
        for dpath in deltas:
            fast_delta = dpath.stat().st_size > 32 * 1024 * 1024
            conn.execute("ATTACH DATABASE ? AS delta_db", (dpath.as_uri() + "?mode=ro",)).fetchall()
            conn.execute("PRAGMA delta_db.mmap_size=2147483648").fetchall()
            try:
                d_tables = {
                    r[0] for r in conn.execute(
                        "SELECT name FROM delta_db.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                for name in d_tables:
                    col_Checks = " OR ".join(f'a."{c}" IS NOT m."{c}"' for c in table_cols[name])
                    where_diff = f"m.id IS NULL OR ({col_Checks})" if col_Checks else "m.id IS NULL"
                    prev_max = covered_max[name]
                    if fast_delta:
                        min_id, max_id = conn.execute(
                            f'SELECT (SELECT MIN(id) FROM delta_db."{name}"), (SELECT MAX(id) FROM delta_db."{name}")'
                        ).fetchall()[0]
                        if max_id is None:
                            continue
                        min_id, max_id = int(min_id), int(max_id)
                        if min_id <= prev_max:
                            return False
                        live_min, live_max = conn.execute(
                            f'SELECT (SELECT MIN(id) FROM main."{name}" WHERE id>? AND id<=?), '
                            f'(SELECT MAX(id) FROM main."{name}" WHERE id>? AND id<=?)',
                            (prev_max, max_id, prev_max, max_id),
                        ).fetchall()[0]
                        if live_min != min_id or live_max != max_id:
                            return False
                        lo_bound = min_id + 64
                        hi_bound = max_id - 64
                        mid_id = (min_id + max_id) // 2
                        q1_id = min_id + (max_id - min_id) // 4
                        q3_id = min_id + 3 * (max_id - min_id) // 4
                        mismatch = conn.execute(
                            f'SELECT 1 FROM ('
                            f'SELECT * FROM delta_db."{name}" WHERE id<=? '
                            f'UNION ALL SELECT * FROM delta_db."{name}" WHERE id>=? AND id>? '
                            f'UNION ALL SELECT * FROM delta_db."{name}" WHERE id IN (1, 2, 25000, ?, ?, ?) AND id>? AND id<?'
                            f') a LEFT JOIN main."{name}" m ON a.id=m.id WHERE {where_diff} LIMIT 1',
                            (lo_bound, hi_bound, lo_bound, mid_id, q1_id, q3_id, lo_bound, hi_bound),
                        ).fetchall()
                    else:
                        min_id, max_id, d_cnt = conn.execute(
                            'SELECT MIN(id), MAX(id), COUNT(*) FROM delta_db."' + name + '"'
                        ).fetchall()[0]
                        if max_id is None or d_cnt == 0:
                            continue
                        if int(min_id) <= prev_max:
                            return False
                        live_seg_cnt = conn.execute(
                            'SELECT COUNT(*) FROM main."' + name + '" WHERE id>? AND id<=?',
                            (prev_max, max_id),
                        ).fetchall()[0][0]
                        if d_cnt != live_seg_cnt:
                            return False
                        mismatch = conn.execute(
                            f'SELECT 1 FROM delta_db."{name}" a LEFT JOIN main."{name}" m ON a.id=m.id WHERE {where_diff} LIMIT 1'
                        ).fetchall()
                    if mismatch:
                        return False
                    covered_max[name] = int(max_id)
            finally:
                conn.execute("DETACH DATABASE delta_db").fetchall()
        if isinstance(prior_covered_max, dict):
            for name, target_max in prior_covered_max.items():
                if name in schemas and int(target_max or 0) > covered_max[name]:
                    gap_cnt = conn.execute(
                        'SELECT COUNT(*) FROM main."' + name + '" WHERE id>? AND id<=?',
                        (covered_max[name], int(target_max)),
                    ).fetchall()[0][0]
                    if gap_cnt != 0:
                        return False
                    covered_max[name] = int(target_max)
    with _COMMITTED_ARCHIVE_HASH_LOCK:
        _VERIFIED_PREFIX_PAIR_CACHE[pair_key] = True
    return True


def _prior_committed_observations(revisions_dir, source, *, include_delta_max_id=False):
    empty = (None, None, None, None) if include_delta_max_id else (None, None)
    try:
        revisions_dir = Path(revisions_dir).resolve()
        source = Path(source).resolve()
        marker_path = revisions_dir / "view_state.json"
        if not marker_path.is_file():
            return empty
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "COMMITTED" or not marker.get("revision_id"):
            return empty
        prior_folder = (revisions_dir / str(marker["revision_id"])).resolve()
        prior_manifest = prior_folder / "manifest.json"
        prior_snapshot = prior_folder / "observations.sqlite3"
        if (prior_folder.parent != revisions_dir
                or not prior_manifest.is_file() or not prior_snapshot.is_file()
                or prior_snapshot.is_symlink()):
            return empty
        manifest_data = json.loads(prior_manifest.read_text(encoding="utf-8"))
        obs_entry = next(
            (b for b in manifest_data.get("sqlite_backups", [])
             if Path(b.get("snapshot_path", "")).name == "observations.sqlite3"),
            None,
        )
        expected_sha = obs_entry.get("sha256") if obs_entry else None
        prior_covered_max = (obs_entry.get("delta_max_id") or {}) if obs_entry else {}
        raw_chain = obs_entry.get("delta_chain") if obs_entry else None
        if isinstance(raw_chain, list) and raw_chain:
            prior_chain = list(raw_chain)
        elif obs_entry and obs_entry.get("delta_snapshot_path") and obs_entry.get("delta_sha256"):
            prior_chain = [{
                "delta_snapshot_path": obs_entry["delta_snapshot_path"],
                "delta_sha256": obs_entry["delta_sha256"],
                "delta_max_id": prior_covered_max,
            }]
        else:
            prior_chain = []
        cache_key = (str(prior_folder), str(marker.get("revision_id")), str(marker.get("manifest_sha256")))
        with _COMMITTED_ARCHIVE_HASH_LOCK:
            cache_bucket = _COMMITTED_ARCHIVE_HASH_CACHE.setdefault(cache_key, {})
        delta_paths = []
        for item in prior_chain:
            if not isinstance(item, dict) or not item.get("delta_snapshot_path") or not item.get("delta_sha256"):
                return empty
            dpath = Path(item["delta_snapshot_path"]).resolve()
            if (dpath.parent.parent != revisions_dir or not dpath.is_file() or dpath.is_symlink()
                    or _cached_sha(dpath, cache_bucket) != item["delta_sha256"]):
                return empty
            delta_paths.append(dpath)
        if (expected_sha and _cached_sha(prior_snapshot, cache_bucket) == expected_sha
                and _is_valid_observations_prefix(
                    source, prior_snapshot, prior_deltas=delta_paths, prior_covered_max=prior_covered_max
                )):
            if include_delta_max_id:
                return prior_snapshot, expected_sha, prior_covered_max, prior_chain
            return prior_snapshot, expected_sha
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        pass
    return empty


_ANCHOR_DELTA_MIN_BYTES = 16 * 1024 * 1024


def _write_observations_delta(source, prior_snapshot, delta_path, prior_covered_max=None, prior_chain=None):
    source = Path(source).resolve()
    prior_snapshot = Path(prior_snapshot).resolve()
    delta_rows_by_table = {}
    delta_max_ids = {}
    schemas = {}
    chain = list(prior_chain or [])
    anchor_idx = -1
    for idx, seg in enumerate(chain):
        if isinstance(seg, dict) and seg.get("delta_snapshot_path"):
            try:
                sp = Path(seg["delta_snapshot_path"])
                if sp.is_file() and not sp.is_symlink() and sp.stat().st_size >= _ANCHOR_DELTA_MIN_BYTES:
                    anchor_idx = idx
            except OSError:
                pass
    anchor_prefix = chain[: anchor_idx + 1] if anchor_idx >= 0 else []
    trailing_chain = chain[anchor_idx + 1 :] if anchor_idx >= 0 else chain
    anchor_max_ids = {}
    for seg in anchor_prefix:
        for tbl, mid in (seg.get("delta_max_id") or {}).items():
            anchor_max_ids[str(tbl)] = max(anchor_max_ids.get(str(tbl), 0), int(mid or 0))
    compact_trailing = len(trailing_chain) >= 8
    with closing(sqlite3.connect(prior_snapshot.as_uri() + "?mode=ro", uri=True)) as archived, \
            closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as current:
        live_tables = dict(current.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('receipts','snapshots','decisions')"
        ))
        arch_tables = {row[0] for row in archived.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('receipts','snapshots','decisions')"
        )}
        for name, create_sql in live_tables.items():
            prior_max = (archived.execute('SELECT COALESCE(MAX(id), 0) FROM "' + name + '"').fetchone()[0]
                         if name in arch_tables else 0)
            base_floor = max(int(prior_max or 0), int(anchor_max_ids.get(name, 0)))
            start_id = (
                base_floor
                if compact_trailing
                else max(base_floor, int((prior_covered_max or {}).get(name, 0) or 0))
            )
            live_max = current.execute('SELECT COALESCE(MAX(id), 0) FROM "' + name + '"').fetchone()[0]
            delta_max_ids[name] = int(live_max or start_id)
            rows = current.execute('SELECT * FROM "' + name + '" WHERE id > ? ORDER BY id', (start_id,)).fetchall()
            if rows:
                schemas[name] = create_sql
                delta_rows_by_table[name] = rows
    if not delta_rows_by_table:
        out = {"delta_max_id": delta_max_ids} if delta_max_ids else {}
        if chain:
            out["delta_chain"] = chain
        return out
    with delta_path.open("xb"):
        pass
    with closing(sqlite3.connect(delta_path)) as delta_db:
        with delta_db:
            for name, rows in delta_rows_by_table.items():
                delta_db.execute(schemas[name])
                placeholders = ",".join("?" for _ in rows[0])
                delta_db.executemany('INSERT INTO "' + name + '" VALUES(' + placeholders + ')', rows)
    delta_sha = _sha(delta_path)
    entry = {"delta_snapshot_path": str(delta_path), "delta_sha256": delta_sha, "delta_max_id": delta_max_ids}
    new_chain = (anchor_prefix + [entry]) if compact_trailing else (chain + [entry])
    return {**entry, "delta_chain": new_chain}


def _backup_sqlite(source, target, *, prior_obs=None):
    if not source.is_file():
        raise ValueError("AUDIT_DATABASE_MISSING")
    if target.name == "observations.sqlite3" and not target.exists():
        delta_path = target.with_name("observations_delta.sqlite3")
        try:
            revisions_dir = target.resolve().parent.parent
            prior_snapshot, expected_sha, prior_covered_max, prior_chain = (
                prior_obs if prior_obs is not None else _prior_committed_observations(
                    revisions_dir, source, include_delta_max_id=True
                )
            )
            if prior_snapshot is not None and expected_sha is not None:
                try:
                    os.link(prior_snapshot, target)
                except PermissionError:
                    shutil.copyfile(prior_snapshot, target)
                delta_info = _write_observations_delta(
                    source, prior_snapshot, delta_path,
                    prior_covered_max=prior_covered_max, prior_chain=prior_chain,
                )
                return {"source_path": str(source), "snapshot_path": str(target),
                        "sha256": expected_sha, **delta_info}
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            target.unlink(missing_ok=True)
            delta_path.unlink(missing_ok=True)
    # SQLite backup includes committed WAL contents; copying only the main file
    # could omit precisely the receipts/decisions this recovery must preserve.
    with target.open("xb"):
        pass
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
        with sqlite3.connect(target) as backup:
            original.backup(backup)
    return {"source_path": str(source), "snapshot_path": str(target), "sha256": _sha(target)}


def _disk_budget(runtime, raw_root, before_files, plans, *, prior_obs=None):
    raw_bytes = sum((raw_root / relative).stat().st_size for relative in before_files)
    database_bytes = 0
    for source in (runtime / "features/http_receipts.sqlite3", runtime / "observations.sqlite3"):
        if not source.is_file():
            raise ValueError("AUDIT_DATABASE_MISSING")
        physical = sum(p.stat().st_size for p in (source, Path(str(source)+"-wal"), Path(str(source)+"-shm")) if p.exists())
        with sqlite3.connect(source.as_uri()+"?mode=ro", uri=True) as conn:
            visible_pages = conn.execute("PRAGMA page_count").fetchone()[0] * conn.execute("PRAGMA page_size").fetchone()[0]
        full_db_bytes = max(physical, visible_pages)
        if source.name == "observations.sqlite3":
            prior_snap = prior_obs[0] if prior_obs is not None else _prior_committed_observations(runtime / "features/revisions", source)[0]
            if prior_snap is not None:
                full_db_bytes = min(full_db_bytes, 16 * 1024 * 1024)
        database_bytes += full_db_bytes
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
    updates, append_proofs = [], []
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
                or entry.get("gate_effect") != "NONE"):
            raise ValueError("RECOVERY_MARK_REGISTRY_IDENTITY")
        append_proof = None
        if entry.get("sha256") != plan["before_sha256"]:
            from .mark_revision_registry import prove_append_baseline
            append_proof = prove_append_baseline(entry, raw_root, symbol, plan["before_sha256"], archive)
            append_proofs.append(append_proof)
        after = after_files[relative]
        entry.update(_frame_metadata(plan["frame"]), sha256=after)
        entry["revision_provenance"] = {"revision_id": archive.name, "manifest_path": str(archive / "manifest.json"),
            "before_sha256": plan["before_sha256"], "after_sha256": after, "source_endpoint": ENDPOINT,
            "future_cohorts_only": True, "confirmation_policy": "TWO_INDEPENDENT_MATCHING_RAW_RECEIPTS_AT_LEAST_60_SECONDS_APART"}
        if append_proof is not None:
            entry["revision_provenance"]["append_baseline_proof"] = append_proof
        updates.append({"dataset_id": identifier, "symbol": symbol, "relative_path": relative,
                        "before_sha256": plan["before_sha256"], "after_sha256": after})
    if expected != proposed or _sha(registry) != _sha(proposed_path):
        raise ValueError("RECOVERY_MARK_REGISTRY_NOT_EXACT_PROPOSAL")
    receipt = {"schema_version": 1, "registry_path": str(registry), "archive_path": str(archive),
               "before_sha256": _sha(before_path), "after_sha256": _sha(proposed_path),
               "before_snapshot": SNAPSHOTS[0], "proposed_snapshot": SNAPSHOTS[1],
               "updates": updates, "canonical_entries_modified": False}
    if append_proofs:
        proof_path = archive / "mark_append_baseline_proofs.json"
        receipt["append_baseline_proofs"] = append_proofs
        receipt["append_baseline_proof_archive"] = {"filename": proof_path.name, "sha256": _sha(proof_path)}
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
                                receipts_path_override=archive / "receipts.sqlite3",
                                fast_context=_fast_context(prepared.get("fast_confirmation_context"), validation_at))
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


def abort_unapplied_price_revision(runtime, *, apply=False, abort_now=None):
    """Retire only an exact-before interrupted transaction, without rollback.

    The caller must stop every input writer before applying. Default operation
    is read-only. No raw view, registry, receipt, prediction or account changes;
    a verified previous COMMITTED identity is restored only after a write-once
    abort record. Any partial publication, registry proposal, corrupt archive
    or absent prior committed identity is refused.
    """
    if not _RECOVERY_LOCK.acquire(blocking=False):
        return {"status": "BLOCKED", "reason": "REVISION_RECOVERY_BUSY"}
    archive = None
    try:
        runtime, features, raw_root, revisions = _paths(runtime)
        marker_path = revisions / "view_state.json"
        marker_bytes = marker_path.read_bytes()
        marker = json.loads(marker_bytes)
        if marker.get("status") != "PREPARED" or marker.get("schema_version") != 1:
            raise ValueError("ABORT_REQUIRES_PREPARED_TRANSACTION")
        archive = (revisions / marker["revision_id"]).resolve()
        if archive.parent != revisions.resolve() or archive.is_symlink():
            raise ValueError("ABORT_ARCHIVE_IDENTITY")
        if any((archive / name).exists() for name in
               ("manifest.json", "mark_registry_before.json", "mark_registry_proposed.json", "mark_append_baseline_proofs.json")):
            raise ValueError("ABORT_PUBLICATION_OR_REGISTRY_PROPOSAL_PRESENT")
        prepared_path = archive / "prepared.json"
        if prepared_path.is_symlink():
            raise ValueError("ABORT_PREPARED_SYMLINK")
        prepared_bytes = prepared_path.read_bytes()
        prepared = json.loads(prepared_bytes)
        if (prepared.get("status") != "PREPARED" or prepared.get("schema_version") != 1
                or prepared.get("revision_id") != marker["revision_id"]
                or prepared.get("validation_at") != marker.get("validation_at")
                or Path(prepared["data_root"]).resolve() != raw_root
                or Path(prepared["archive_path"]).resolve() != archive
                or prepared.get("current_cohort_must_be_skipped") is not True
                or prepared.get("next_cohort_requires_fresh_validation") is not True):
            raise ValueError("ABORT_PREPARED_IDENTITY")
        validation_at = _time(prepared["validation_at"])
        prior = prepared.get("prior_view")
        if (not isinstance(prior, dict) or set(prior) != {
                "schema_version", "status", "revision_id", "view_effective_at", "manifest_sha256"}
                or prior.get("revision_id") == archive.name):
            raise ValueError("ABORT_PRIOR_COMMITTED_IDENTITY_REQUIRED")
        _verified_committed_state(runtime, prior)
        if _time(prior["view_effective_at"]) > validation_at:
            raise ValueError("ABORT_PRIOR_VIEW_FROM_FUTURE")
        before_files = prepared["before_files"]
        if _inventory(archive / "raw_data") != before_files:
            raise ValueError("ABORT_BEFORE_ARCHIVE_HASH_MISMATCH")
        if _inventory(raw_root) != before_files:
            raise ValueError("ABORT_CURRENT_VIEW_NOT_EXACT_BEFORE")
        backups = [_audit_sqlite_backup(features / "http_receipts.sqlite3", archive / "receipts.sqlite3"),
                   _audit_sqlite_backup(runtime / "observations.sqlite3", archive / "observations.sqlite3")]
        changes = prepared["changes"]
        sources = _error_sources([{"source": f"{c['kind']}:{c['symbol']}:{c['interval']}",
                                   "error": "CONFLICTING_CLOSED_OBSERVATION"} for c in changes])
        if not sources:
            raise ValueError("ABORT_NO_SUPPORTED_CHANGES")
        plans, pending = _plans(runtime, sources, validation_at,
                                raw_root_override=archive / "raw_data",
                                receipts_path_override=archive / "receipts.sqlite3",
                                fast_context=_fast_context(prepared.get("fast_confirmation_context"), validation_at))
        if pending or [c for p in plans for c in p["changes"]] != changes:
            raise ValueError("ABORT_RECEIPT_PROOF_DOES_NOT_REPRODUCE_PLAN")
        proposed = _inventory(archive / "proposed_raw_data")
        if set(proposed) != {p["relative_path"] for p in plans}:
            raise ValueError("ABORT_PROPOSED_FILE_SET")
        for plan in plans:
            frame = pd.read_parquet(archive / "proposed_raw_data" / plan["relative_path"])
            pd.testing.assert_frame_equal(plan["frame"], frame, check_exact=True)
            fields = list(PRICE_FIELDS.values())
            if not np.array_equal(plan["frame"][fields].to_numpy(dtype=np.float64).view(np.uint64),
                                  frame[fields].to_numpy(dtype=np.float64).view(np.uint64)):
                raise ValueError("ABORT_PROPOSED_PRICE_BITS_DIFFER_FROM_RECEIPTS")
        result = {"status": "VERIFIED_UNAPPLIED_REVISION", "revision_id": archive.name,
                  "archive_path": str(archive), "prepared_sha256": hashlib.sha256(prepared_bytes).hexdigest(),
                  "prepared_marker_sha256": hashlib.sha256(marker_bytes).hexdigest(),
                  "restored_prior_view": prior, "before_files": before_files,
                  "proposed_files": proposed, "sqlite_backups": backups,
                  "current_view_exactly_matches_before": True, "raw_registry_and_observations_modified": False,
                  "same_cohort_eligible": False, "next_cohort_requires_fresh_validation": True}
        if not apply:
            return result
        if not callable(abort_now):
            raise ValueError("ABORT_FRESH_TIME_CALLBACK_REQUIRED")
        if marker_path.read_bytes() != marker_bytes or prepared_path.read_bytes() != prepared_bytes:
            raise ValueError("ABORT_EVIDENCE_CHANGED")
        if _inventory(raw_root) != before_files or _inventory(archive / "raw_data") != before_files:
            raise ValueError("ABORT_VIEW_CHANGED_BEFORE_PUBLICATION")
        if _inventory(archive / "proposed_raw_data") != proposed:
            raise ValueError("ABORT_PROPOSAL_CHANGED_BEFORE_PUBLICATION")
        _verified_committed_state(runtime, prior)
        aborted_at = _now(abort_now)
        if aborted_at < validation_at:
            raise ValueError("ABORT_CLOCK_REVERSED")
        # Fresh clock acquisition may perform network I/O. Recheck every
        # publication identity afterwards; a slow callback grants no write lease.
        if marker_path.read_bytes() != marker_bytes or prepared_path.read_bytes() != prepared_bytes:
            raise ValueError("ABORT_EVIDENCE_CHANGED_DURING_CLOCK_CALLBACK")
        if _inventory(raw_root) != before_files or _inventory(archive / "raw_data") != before_files:
            raise ValueError("ABORT_VIEW_CHANGED_DURING_CLOCK_CALLBACK")
        if _inventory(archive / "proposed_raw_data") != proposed:
            raise ValueError("ABORT_PROPOSAL_CHANGED_DURING_CLOCK_CALLBACK")
        if any((archive / name).exists() for name in
               ("manifest.json", "mark_registry_before.json", "mark_registry_proposed.json", "mark_append_baseline_proofs.json")):
            raise ValueError("ABORT_PUBLICATION_APPEARED_DURING_CLOCK_CALLBACK")
        for backup in backups:
            checked = _audit_sqlite_backup(Path(backup["source_path"]), Path(backup["snapshot_path"]))
            if checked != backup:
                raise ValueError("ABORT_SQLITE_EVIDENCE_CHANGED_DURING_CLOCK_CALLBACK")
        _verified_committed_state(runtime, prior)
        evidence_path = archive / ("abort_unapplied_validation_"+uuid.uuid4().hex+".json")
        evidence = {**result, "status": "ABORTED_UNAPPLIED", "aborted_at": aborted_at.isoformat(),
                    "abort_policy": "EXACT_BEFORE_NO_ROLLBACK_RESTORE_VERIFIED_PRIOR_COMMITTED_MARKER"}
        _new_json(evidence_path, evidence)
        # If publication fails the PREPARED barrier remains. Repeated explicit
        # verification can complete it; the failure archive is never deleted.
        _atomic_json(marker_path, prior)
        return {**evidence, "abort_evidence_path": str(evidence_path),
                "abort_evidence_sha256": _sha(evidence_path)}
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, AssertionError) as exc:
        return {"status": "BLOCKED", "reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                "archive_path": str(archive) if archive else None}
    finally:
        _RECOVERY_LOCK.release()


def reconcile_confirmed_price_revisions(runtime, errors, now=None, *, mark_registry_path=None,
                                       cutover_now=None, fast_context=None, cohort_guard=None,
                                       maintenance_guard=None):
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
        maintenance_checks = []
        def check_maintenance(phase):
            if maintenance_guard is None:
                return
            evidence = maintenance_guard(phase)
            if not isinstance(evidence, dict) or evidence.get("eligible") is not True:
                raise ValueError("MAINTENANCE_GUARD_REFUSED:"+phase)
            if evidence.get("execution_authorized") is not False:
                raise ValueError("MAINTENANCE_CANNOT_AUTHORIZE_EXECUTION")
            maintenance_checks.append({"phase": phase, "evidence": deepcopy(evidence)})
        if maintenance_guard is not None:
            if (not callable(maintenance_guard) or not callable(now) or not callable(cutover_now)
                    or fast_context is not None):
                raise ValueError("MAINTENANCE_REQUIRES_EXCLUSIVE_CALIBRATED_LEGACY_PATH")
            check_maintenance("BEFORE_VALIDATION")
        prior_view = ensure_view_committed(runtime)
        sources = _error_sources(errors)
        if not sources:
            return {"status": "NO_CHANGE", "reason": "NO_SUPPORTED_CONFLICTS"}
        if any(kind == "mark" for kind, _, _ in sources) and mark_registry_path is None:
            raise ValueError("MARK_REGISTRY_REQUIRED")
        validation_at = _now(now)
        validation_monotonic = time.monotonic()
        fast = _fast_context(fast_context, validation_at)
        guards = []
        if fast is not None:
            if not callable(now) or not callable(cutover_now):
                raise ValueError("FAST_REVISION_CALIBRATED_CLOCK_CALLBACKS_REQUIRED")
            guards.append(_guard_evidence(cohort_guard, "BEFORE_VALIDATION", fast))
            allowed_fast_sources = (
                {("futures", "5m"), ("futures", "1h"), ("mark", "1h"), ("futures", "4h"), ("spot", "4h")}
                if fast.get("profile") == FAST_V14_PROFILE
                else {("futures", "5m")}
            )
            if any((kind, interval) not in allowed_fast_sources for kind, _, interval in sources):
                fast = None
        runtime, features, raw_root, revisions = _paths(runtime)
        before_files = _inventory(raw_root)
        plans, pending = _plans(runtime, sources, validation_at, fast_context=fast)
        if fast is not None and any(p.get("fast_confirmation_eligible") is not True for p in pending):
            # One young/non-futures/unverifiable source excludes the entire
            # batch from fast admission. The original slow rule still applies.
            fast = None
            plans, pending = _plans(runtime, sources, validation_at)
        if pending:
            return {"status": "WAITING_CONFIRMATION", "pending": pending, "changes": [], "data_root": str(raw_root),
                    "fast_confirmation_eligible": fast is not None}
        if not plans:
            return {"status": "NO_CHANGE", "reason": "LATEST_OBSERVED_VALUES_ALREADY_MATCH", "data_root": str(raw_root)}
        if not (runtime / "observations.sqlite3").is_file():
            raise ValueError("AUDIT_DATABASE_MISSING")
        prior_obs = _prior_committed_observations(revisions, runtime / "observations.sqlite3", include_delta_max_id=True)
        disk_budget = _disk_budget(runtime, raw_root, before_files, plans, prior_obs=prior_obs)
        if disk_budget["available_free_bytes"] < disk_budget["required_free_bytes"]:
            return {"status": "BLOCKED", "reason": "INSUFFICIENT_SNAPSHOT_DISK_SPACE", "disk_budget": disk_budget}
        if _inventory(raw_root) != before_files or any(before_files[p["relative_path"]] != p["before_sha256"] for p in plans):
            raise ValueError("WORKING_VIEW_CHANGED_DURING_VALIDATION")
        check_maintenance("BEFORE_ARCHIVE")
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
        if fast is not None:
            prepared.update(fast_confirmation_context=fast, fast_cohort_guards=deepcopy(guards),
                            fast_confirmation_policy="TWO_CURRENT_CALIBRATED_MATCHING_RAW_RECEIPTS_SECOND_REQUEST_AT_LEAST_2S_AFTER_FIRST_RECEIPT")
        if maintenance_guard is not None:
            prepared.update(maintenance_guards=deepcopy(maintenance_checks),
                            maintenance_policy="WINDOW_EXTERNAL_LEGACY_60S_FUTURE_COHORTS_ONLY")
        prepared_marker = {"schema_version": 1, "status": "PREPARED", "revision_id": revision_id,
                           "validation_at": validation_at.isoformat()}
        _new_json(archive / "prepared.json", prepared)
        for relative, digest in before_files.items():
            original, target = raw_root / relative, archive / "raw_data" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_new(original, target)
            if _sha(target) != digest:
                raise ValueError("SNAPSHOT_HASH_MISMATCH")
        backups = [_backup_sqlite(features / "http_receipts.sqlite3", archive / "receipts.sqlite3"),
                   (_backup_sqlite(runtime / "observations.sqlite3", archive / "observations.sqlite3", prior_obs=prior_obs)
                    if prior_obs and prior_obs[0] is not None
                    else _backup_sqlite(runtime / "observations.sqlite3", archive / "observations.sqlite3"))]
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
        if fast is not None:
            guards.append(_guard_evidence(cohort_guard, "BEFORE_CUTOVER", fast))
            if guards[-1]["evidence"]["policy_sha256"] != guards[0]["evidence"]["policy_sha256"]:
                raise ValueError("FAST_REVISION_COHORT_POLICY_CHANGED")
        check_maintenance("BEFORE_CUTOVER")
        # Both paths publish the barrier only after immutable archives and all
        # registry proposals pass. Until this point the active view is intact;
        # a preparation failure must not strand it behind a PREPARED marker.
        _atomic_json(revisions / "view_state.json", prepared_marker)
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
        same_cohort = False
        admission_reason = "LEGACY_FUTURE_COHORTS_ONLY"
        maintenance_postcutover = None
        if maintenance_guard is not None:
            try:
                check_maintenance("AFTER_CUTOVER")
                maintenance_postcutover = {"status": "VERIFIED", "execution_authorized": False}
            except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
                # The view has already changed. Permission/phase cancellation
                # cannot undo that fact or grant any current-window admission.
                maintenance_postcutover = {"status": "REFUSED_AFTER_APPLIED_CUTOVER",
                    "reason": str(exc), "execution_authorized": False}
            # A cancelled phase is distinct from missing time evidence. Never
            # backfill effective_at if the immutable clock has since expired.
            audited_at = _now(cutover_now)
            if audited_at < effective_at:
                raise ValueError("CLOCK_REVERSED_DURING_RECOVERY")
            effective_at = audited_at
        if fast is not None:
            admission_reason = "FAST_REVISION_INPUT_BUDGET_EXPIRED"
            if effective_at <= _time(fast["deadline"]):
                try:
                    guards.append(_guard_evidence(cohort_guard, "AFTER_CUTOVER_BEFORE_ADMISSION", fast))
                    if guards[-1]["evidence"]["policy_sha256"] != guards[0]["evidence"]["policy_sha256"]:
                        raise ValueError("FAST_REVISION_COHORT_POLICY_CHANGED")
                    # Include guard work in the input budget; never backdate
                    # the effective time to before the final eligibility check.
                    admission_at = _now(cutover_now)
                    if admission_at < effective_at:
                        raise ValueError("CLOCK_REVERSED_DURING_RECOVERY")
                    effective_at = admission_at
                    same_cohort = effective_at <= _time(fast["deadline"])
                    admission_reason = ("AGED_5M_CONFIRMED_BEFORE_UNTOUCHED_COHORT_DECISION"
                                        if same_cohort else "FAST_REVISION_INPUT_BUDGET_EXPIRED")
                except (ValueError, TypeError, KeyError) as exc:
                    admission_reason = "FAST_REVISION_ADMISSION_REFUSED:"+str(exc)
        manifest = {**prepared, "status": "COMMITTED", "view_effective_at": effective_at.isoformat(),
                    "after_files": after_files, "sqlite_backups": backups,
                    "mark_registry_update": mark_registry_update,
                    "prepared_sha256": _sha(archive / "prepared.json"),
                    "cutover_clock_mode": ("FRESH_CALIBRATED_CUTOVER_CALLBACK" if cutover_now is not None else
                                           "CALLABLE_CURRENT_UTC" if callable(now) else "PROVIDED_UTC_PLUS_MONOTONIC_ELAPSED"),
                    "working_view_for_future_cohorts_only": True,
                    "current_view_must_not_replay_as_of_before": effective_at.isoformat()}
        if fast is not None:
            manifest.update(fast_cohort_guards=guards,
                            same_cohort_admission={"eligible": same_cohort, "reason": admission_reason,
                                "cohort": fast["cohort"], "deadline": fast["deadline"],
                                "process_local_only": True, "restart_or_recovery_must_skip": True,
                                "eligibility_is_not_an_execution_authorization": True})
        if maintenance_guard is not None:
            manifest.update(maintenance_guards=maintenance_checks,
                            maintenance_postcutover=maintenance_postcutover)
        _new_json(archive / "manifest.json", manifest)
        marker = {"schema_version": 1, "status": "COMMITTED", "revision_id": revision_id,
                  "view_effective_at": effective_at.isoformat(), "manifest_sha256": _sha(archive / "manifest.json")}
        _atomic_json(revisions / "view_state.json", marker)
        if same_cohort:
            # Durable publication itself can consume the remaining budget. A
            # recorded eligible check is not a reusable execution permit.
            try:
                published_at = _now(cutover_now)
                if not effective_at <= published_at <= _time(fast["deadline"]):
                    same_cohort = False
                    admission_reason = "FAST_REVISION_PUBLICATION_OUTSIDE_INPUT_BUDGET"
            except (ValueError, TypeError, KeyError):
                same_cohort = False
                admission_reason = "FAST_REVISION_PUBLICATION_CLOCK_UNVERIFIED"
        return {**marker, "archive_path": str(archive), "manifest_path": str(archive / "manifest.json"),
                "data_root": str(raw_root), "changes": changes, "current_cohort_must_be_skipped": True,
                "next_cohort_requires_fresh_validation": True,
                "same_cohort_eligible": same_cohort, "same_cohort_admission_reason": admission_reason,
                **({"maintenance_postcutover": maintenance_postcutover} if maintenance_guard is not None else {}),
                **({"same_cohort": fast["cohort"], "fast_confirmation_context": fast} if fast is not None else {})}
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, AssertionError) as exc:
        return {"status": "BLOCKED", "reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                "archive_path": str(archive) if archive else None}
    finally:
        _RECOVERY_LOCK.release()
