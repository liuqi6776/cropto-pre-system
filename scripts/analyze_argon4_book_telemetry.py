"""Read-only audit of retained public books at their actual receipt times.

Target offsets are grouping labels, not reconstructed point-in-time quotes.
Legacy samples are never relabelled or written back. Historical Gamma receipts
are optional inputs but are required to attest market identity and fee capacity;
the current market, current fee and current clock are never consulted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.argon4_forward.book_telemetry import (
    BOOK_MAX_AGE_SECONDS, OFFSETS, SAMPLE_TOLERANCE_SECONDS, fee_inclusive_capacity,
)
from tools.argon4_forward.clock_sync import bounds_from_samples, project_local, utc
from tools.argon4_forward.polymarket import STREAMS, validate_book, validate_market
from tools.argon4_forward.polymarket_transport import SOURCE, public_url

MAX_SAMPLES = 100_000
MAX_METADATA_BYTES = 64_000_000
MAX_ROW_BYTES = 4_000_000


class EvidenceError(ValueError):
    """Only fixed, non-sensitive reason codes are emitted in reports."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def parse(value):
    if not isinstance(value, str) or len(value.encode()) > MAX_ROW_BYTES:
        raise EvidenceError("EVIDENCE_SIZE_OR_TYPE_INVALID")
    def bad_constant(_):
        raise EvidenceError("NONFINITE_JSON")
    return json.loads(value, parse_constant=bad_constant)


def iso(value):
    return utc(value).isoformat().replace("+00:00", "Z")


def require(condition, code):
    if not condition:
        raise EvidenceError(code)


@contextmanager
def readonly(path):
    path = Path(path).resolve(strict=True)
    with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True, timeout=2) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        yield db


def historical_clock(row, receipt):
    clock = row["request_clock"]
    require(clock.get("status") == "CALIBRATED"
            and clock.get("source") == "POLYMARKET_CLOB_SERVER_TIME", "CLOCK_NOT_ATTESTED")
    samples = clock.get("receipt", {}).get("samples")
    require(isinstance(samples, list) and 3 <= len(samples) <= 8, "CLOCK_RAW_SAMPLES_MISSING")
    require(clock["receipt"].get("quantization_seconds") == 1, "CLOCK_QUANTIZATION_INVALID")
    require(sha(canonical(samples)) == clock.get("receipt_sha256"), "CLOCK_RECEIPT_HASH_MISMATCH")
    if "receipts" in clock:
        require(clock["receipts"] == samples, "CLOCK_RECEIPT_LIST_MISMATCH")
    for sample in samples:
        raw = sample.get("raw_text")
        require(isinstance(raw, str) and re.fullmatch(r"[0-9]{10}", raw.strip()) is not None
                and sha(raw) == sample.get("sha256")
                and int(raw.strip()) == sample.get("server_second")
                and sample.get("url") == public_url("time")
                and sample.get("http_status") == 200, "CLOCK_SAMPLE_PROOF_INVALID")
    bounds = bounds_from_samples(samples)
    for key, value in bounds.items():
        supplied = clock.get(key)
        require(not isinstance(supplied, bool) and isinstance(supplied, (int, float))
                and math.isfinite(supplied) and math.isclose(value, supplied, rel_tol=0, abs_tol=1e-12),
                "CLOCK_BOUNDS_MISMATCH")
    require(0 <= bounds["uncertainty_seconds"] <= .5, "CLOCK_UNCERTAINTY_INVALID")
    calibrated = utc(clock["calibrated_at_local"])
    require(all(utc(s["received_at_local"]) <= calibrated for s in samples), "CLOCK_CALIBRATION_ORDER_INVALID")
    requested, received = utc(receipt["requested_at"]), utc(receipt["received_at"])
    require(0 <= (requested-calibrated).total_seconds()
            <= (received-calibrated).total_seconds() <= 60, "HISTORICAL_CLOCK_EXPIRED_OR_REVERSED")
    elapsed = receipt.get("transport_timing", {}).get("request_elapsed_seconds")
    require(not isinstance(elapsed, bool) and isinstance(elapsed, (int, float))
            and math.isfinite(elapsed) and elapsed >= 0
            and abs((received-requested).total_seconds()-elapsed) <= .05, "REQUEST_CLOCK_ELAPSED_UNVERIFIABLE")
    require(utc(clock["observed_at"]) == project_local(calibrated, clock), "CLOCK_OBSERVED_AT_MISMATCH")
    return clock


def metadata_key(row):
    return row.get("metadata_sha256"), iso(row["metadata_received_at_local"])


def load_metadata(db, needed):
    """Stream only matching receipt links; never load the observation journal."""
    if db is None or not needed:
        return {}
    wanted = sorted({digest for digest, _ in needed if isinstance(digest, str)})
    result, total_bytes = {}, 0
    for start in range(0, len(wanted), 400):
        values = wanted[start:start+400]
        placeholders = ",".join("?" for _ in values)
        query = f"""SELECT j.value AS meta, r.received_at, r.url, r.sha256,
                           r.http_status, r.raw_text
                    FROM snapshots s, json_each(s.payload,'$.receipt_metadata') j
                    JOIN receipts r ON r.id=json_extract(j.value,'$.receipt_id')
                    WHERE s.kind IN ('polymarket','paper_market_receipt')
                      AND json_extract(j.value,'$.sha256') IN ({placeholders})"""
        for item in db.execute(query, values):
            meta = {key: value for key, value in parse(item["meta"]).items() if key != "receipt_id"}
            key = item["sha256"], iso(item["received_at"])
            if key not in needed:
                continue
            try:
                require(meta.get("sha256") == item["sha256"]
                        and iso(meta["received_at"]) == key[1]
                        and meta.get("url") == item["url"]
                        and meta.get("http_status") == item["http_status"] == 200
                        and meta.get("source") == SOURCE and meta.get("method") == "GET"
                        and meta.get("body_complete") is True
                        and isinstance(item["raw_text"], str)
                        and sha(item["raw_text"]) == key[0], "METADATA_RAW_RECEIPT_INVALID")
                raw = parse(item["raw_text"])
                proof = {"receipt": meta, "body": raw, "raw_text": item["raw_text"]}
            except (EvidenceError, KeyError, TypeError, ValueError):
                proof = {"error": "METADATA_RAW_RECEIPT_INVALID"}
            if key in result:
                if canonical(result[key]) != canonical(proof):
                    result[key] = {"error": "CONFLICTING_METADATA_RECEIPTS"}
            else:
                total_bytes += len(canonical(proof).encode())
                require(total_bytes <= MAX_METADATA_BYTES, "METADATA_BUDGET_EXCEEDED_USE_COHORT_FILTER")
                result[key] = proof
    return result


def verify_sample(row, metadata):
    result = {"status": "UNVERIFIABLE", "strict_slice": False,
              "actual_quote": False, "capacity_available": False}
    try:
        stream, outcome = row["stream"], row["outcome"]
        require(stream in STREAMS and outcome in ("UP", "DOWN"), "SAMPLE_IDENTITY_INVALID")
        cohort, target = utc(row["cohort"]), utc(row["target_at"])
        offset = row["target_offset_seconds"]
        require(type(offset) is int and offset in OFFSETS and cohort.timestamp() % 300 == 0
                and int(cohort.timestamp()) % STREAMS[stream][1] == 0
                and target == cohort+timedelta(seconds=offset), "TARGET_SCHEDULE_INVALID")
        require(row["sample_id"] == f"{int(cohort.timestamp())}:{stream}:{outcome}:{offset}", "SAMPLE_ID_MISMATCH")
        receipt = row.get("receipt")
        if not receipt:
            return {**result, "status": "NO_RECEIPT", "reason": "NO_RETAINED_RESPONSE"}
        require(receipt.get("source") == SOURCE and receipt.get("method") == "GET"
                and receipt.get("url") == public_url("book", row["token_id"])
                and receipt.get("http_status") == 200 and receipt.get("body_complete") is True,
                "COMPLETE_PUBLIC_BOOK_RECEIPT_REQUIRED")
        require(isinstance(receipt.get("raw_text"), str)
                and sha(receipt["raw_text"]) == receipt.get("sha256"), "BOOK_RECEIPT_HASH_MISMATCH")
        clock = historical_clock(row, receipt)
        requested_lower = project_local(receipt["requested_at"], clock, upper=False)
        requested_upper = project_local(receipt["requested_at"], clock)
        received_lower = project_local(receipt["received_at"], clock, upper=False)
        received_upper = project_local(receipt["received_at"], clock)
        for key, expected in (("requested_at_local", utc(receipt["requested_at"])),
                              ("received_at_local", utc(receipt["received_at"])),
                              ("requested_at_server_lower", requested_lower),
                              ("received_at_server_lower", received_lower),
                              ("received_at_server_upper", received_upper)):
            require(utc(row[key]) == expected, "STORED_RECEIPT_TIME_MISMATCH")
        if row.get("requested_at_server_upper"):
            require(utc(row["requested_at_server_upper"]) == requested_upper, "STORED_RECEIPT_TIME_MISMATCH")
        book = validate_book(parse(receipt["raw_text"]), row["token_id"], row["condition_id"], received_upper)
        require(canonical(book) == canonical(row["parsed_book"]), "NORMALIZED_BOOK_MISMATCH")
        source = utc(book["source_at"])
        require(book["status"] == "OBSERVED" and 0 <= (received_upper-source).total_seconds() <= BOOK_MAX_AGE_SECONDS,
                "BOOK_STALE_FUTURE_OR_INCOMPLETE")
        key = metadata_key(row)
        proof = metadata.get(key)
        require(proof is not None, "HISTORICAL_METADATA_RECEIPT_MISSING")
        require(not proof.get("error"), (proof or {}).get("error", "METADATA_RAW_RECEIPT_INVALID"))
        meta, body = proof["receipt"], proof["body"]
        require(sha(proof["raw_text"]) == key[0] and meta.get("sha256") == key[0]
                and canonical(parse(proof["raw_text"])) == canonical(body), "METADATA_RAW_RECEIPT_INVALID")
        require(meta["url"] == public_url("market", row["slug"])
                and utc(meta["requested_at"]) <= utc(meta["received_at"]) <= utc(receipt["requested_at"]),
                "METADATA_IDENTITY_OR_RECEIPT_ORDER_INVALID")
        metadata_age = (received_upper-project_local(meta["received_at"], clock, upper=False)).total_seconds()
        require(0 <= metadata_age <= 90, "HISTORICAL_METADATA_TOO_OLD")
        require(isinstance(body, list) and len(body) == 1, "GAMMA_RESPONSE_IDENTITY_INVALID")
        market = validate_market(body[0], stream, cohort, project_local(meta["received_at"], clock))
        require(market["market_id"] == row["market_id"] and market["condition_id"] == row["condition_id"]
                and market["outcome_ids"][outcome] == row["token_id"] and market["slug"] == row["slug"],
                "GAMMA_BOOK_IDENTITY_MISMATCH")
        require(canonical(market["fee"]) == canonical(row["fee"]), "STORED_FEE_DIFFERS_FROM_HISTORICAL_GAMMA")
        capacity = fee_inclusive_capacity(book, market["fee"])
        strict_timing = requested_lower >= target and received_upper <= target+timedelta(seconds=SAMPLE_TOLERANCE_SECONDS)
        # Verification may remove eligibility; it never promotes a historical
        # MISSED/AMBIGUOUS row to an originally OBSERVED target slice.
        result.update(status="VERIFIED_ACTUAL_QUOTE", actual_quote=True,
            strict_slice=row["status"] == "OBSERVED" and strict_timing,
            strict_timing_verified=strict_timing, original_status=row["status"],
            capacity_available=capacity["status"] == "OBSERVED_DEPTH",
            capacity_status=capacity["status"],
            capacity_usd=float(capacity["usd"]) if capacity["usd"] is not None else None,
            capacity_shares=float(capacity["shares"]) if capacity["shares"] is not None else None,
            receipt_lag_lower_seconds=(received_lower-target).total_seconds(),
            receipt_lag_upper_seconds=(received_upper-target).total_seconds(),
            receipt_lag_center_seconds=((received_lower+(received_upper-received_lower)/2)-target).total_seconds(),
            source_lag_seconds=(source-target).total_seconds(),
            uncertainty_seconds=clock["uncertainty_seconds"])
        return result
    except EvidenceError as exc:
        result["reason"] = str(exc)
    except (KeyError, TypeError, ValueError, OverflowError):
        result["reason"] = "MISSING_OR_INVALID_HISTORICAL_EVIDENCE"
    return result


def percentiles(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    def at(q):
        pos = q*(len(ordered)-1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)
    return {"count": len(ordered), "min": ordered[0], "p25": at(.25), "median": at(.5),
            "p75": at(.75), "p95": at(.95), "max": ordered[-1]}


def _rows(db, start, end):
    query, args = "SELECT * FROM samples", []
    filters = []
    if start is not None:
        filters.append("julianday(cohort)>=julianday(?)")
        args.append(iso(start))
    if end is not None:
        filters.append("julianday(cohort)<julianday(?)")
        args.append(iso(end))
    if filters:
        query += " WHERE "+" AND ".join(filters)
    query += " ORDER BY cohort,stream,outcome,target_offset,sample_id"
    for index, item in enumerate(db.execute(query, args), 1):
        require(index <= MAX_SAMPLES, "SAMPLE_BUDGET_EXCEEDED_USE_COHORT_FILTER")
        row = parse(item["payload_json"])
        for column, field in (("sample_id", "sample_id"), ("cohort", "cohort"), ("stream", "stream"),
                               ("outcome", "outcome"), ("target_offset", "target_offset_seconds"),
                               ("target_at", "target_at"), ("status", "status")):
            if item[column] != row.get(field):
                row["_sql_identity_mismatch"] = True
        yield row, item["payload_json"]


def analyze(database, observations_database=None, *, start=None, end=None):
    for boundary in (start, end):
        require(boundary is None or utc(boundary).timestamp() % 300 == 0, "FILTER_MUST_BE_UTC_COHORT")
    require(start is None or end is None or utc(start) < utc(end), "COHORT_FILTER_REVERSED")
    with readonly(database) as db:
        needed = set()
        for row, _ in _rows(db, start, end):
            if row.get("receipt"):
                try:
                    needed.add(metadata_key(row))
                except (KeyError, TypeError, ValueError):
                    pass
        if observations_database is None:
            metadata = {}
        else:
            with readonly(observations_database) as observations:
                metadata = load_metadata(observations, needed)
        groups, digest = {}, hashlib.sha256()
        for row, raw in _rows(db, start, end):
            digest.update(raw.encode()+b"\n")
            profile = row.get("sampling_profile", "legacy_unlabelled")
            if profile not in ("legacy_unlabelled", "legacy_v11", "clock_center_v12"):
                profile = "UNRECOGNIZED_PROFILE"
            offset = row.get("target_offset_seconds")
            group = groups.setdefault((str(profile), str(offset)), {
                "profile": profile, "target_offset_label_seconds": offset,
                "total_samples": 0, "original_status_counts": Counter(), "proof_status_counts": Counter(),
                "reason_counts": Counter(), "strict_slices": 0, "actual_quote_samples": 0,
                "fee_inclusive_depth_samples": 0, "non_strict_actual_depth_samples": 0,
                "_metrics": {key: [] for key in ("receipt_lag_lower_seconds", "receipt_lag_center_seconds",
                    "receipt_lag_upper_seconds", "source_lag_seconds", "uncertainty_seconds", "capacity_usd", "capacity_shares")}})
            group["total_samples"] += 1
            group["original_status_counts"][row.get("status", "MISSING")] += 1
            checked = ({"status": "UNVERIFIABLE", "reason": "SQL_PAYLOAD_IDENTITY_MISMATCH"}
                       if row.get("_sql_identity_mismatch") else verify_sample(row, metadata))
            group["proof_status_counts"][checked["status"]] += 1
            if checked.get("reason"):
                group["reason_counts"][checked["reason"]] += 1
            if checked.get("actual_quote"):
                group["actual_quote_samples"] += 1
                group["strict_slices"] += int(checked["strict_slice"])
                group["fee_inclusive_depth_samples"] += int(checked["capacity_available"])
                group["non_strict_actual_depth_samples"] += int(checked["capacity_available"] and not checked["strict_slice"])
                if not checked["capacity_available"]:
                    group["reason_counts"][checked["capacity_status"]] += 1
                for metric, values in group["_metrics"].items():
                    if checked.get(metric) is not None:
                        values.append(checked[metric])
        output = []
        for _, group in sorted(groups.items()):
            group["actual_receipt_distributions"] = {name: percentiles(values) for name, values in group.pop("_metrics").items()}
            output.append(group)
        counts = {key: sum(g[key] for g in output) for key in ("total_samples", "strict_slices", "actual_quote_samples",
            "fee_inclusive_depth_samples", "non_strict_actual_depth_samples")}
        counts["unverifiable_samples"] = sum(g["proof_status_counts"].get("UNVERIFIABLE", 0) for g in output)
        counts["no_receipt_samples"] = sum(g["proof_status_counts"].get("NO_RECEIPT", 0) for g in output)
        return {"schema_version": 1, "status": "COMPLETE_WITH_UNVERIFIABLE_EVIDENCE" if counts["unverifiable_samples"] else "COMPLETE",
            "generated_at": iso(datetime.now(timezone.utc)), "database_access": "SQLITE_MODE_RO_QUERY_ONLY",
            "sample_payload_sha256": digest.hexdigest(), "counts": counts, "groups": output,
            "cohort_filter": {"inclusive_start": iso(start) if start else None, "exclusive_end": iso(end) if end else None},
            "limit_price": .515, "limit_basis": "ASK_PRICE_WITH_FEES_ADDED_TO_CAPACITY_COST",
            "interpretation": "Target offsets are grouping labels. Distributions describe actual received observations, not exact T+X depth or fills.",
            "historical_status_modified": False, "live_orders": False, "current_clock_or_fee_used": False,
            "limitations": ["Only retained evidence is checked; missing observations are not reconstructed.",
                "Historical clock bounds are recomputed from stored time receipts and elapsed durations; independent OS clock history is unavailable.",
                "SHA256 attests retained byte consistency, not a cryptographic exchange signature.",
                "Stored profile absence is labelled legacy_unlabelled; it is not filled using the current configuration."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--observations-database", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON under this workspace's reports directory")
    parser.add_argument("--from-cohort", dest="start")
    parser.add_argument("--until-cohort", dest="end")
    args = parser.parse_args(argv)
    try:
        if args.output:
            output = args.output.resolve()
            require(output.is_relative_to((ROOT/"reports").resolve()) and output.suffix.lower() == ".json",
                    "OUTPUT_MUST_BE_LOCAL_REPORT_JSON")
            require(output not in {p.resolve() for p in (args.database, args.observations_database) if p is not None},
                    "OUTPUT_CANNOT_REPLACE_INPUT_DATABASE")
            require(not output.exists(), "OUTPUT_EXISTS_CHOOSE_NEW_REPORT")
        report = analyze(args.database, args.observations_database,
                         start=utc(args.start) if args.start else None, end=utc(args.end) if args.end else None)
        if args.output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        print(json.dumps({"status": report["status"], "counts": report["counts"], "output_written": bool(args.output)}, ensure_ascii=False))
        return 1 if report["counts"]["unverifiable_samples"] else 0
    except (EvidenceError, OSError, sqlite3.Error, ValueError, TypeError) as exc:
        code = str(exc) if isinstance(exc, EvidenceError) else type(exc).__name__
        print(json.dumps({"status": "FAILED", "reason": code, "output_written": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
