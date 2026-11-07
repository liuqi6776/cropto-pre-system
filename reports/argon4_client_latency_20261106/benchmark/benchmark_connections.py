"""Bounded public /book transport benchmark; no orders, SDK, or production writes."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics
import time
from urllib.parse import urlencode
import uuid

import requests
from requests.adapters import HTTPAdapter

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DB = ROOT / "paper_runtime/argon4_forward_20261105/observations.sqlite3"
ENDPOINT = "https://clob.polymarket.com/book"
MAX_BYTES = 2_000_000
PAIRS = 12
MIN_START_SPACING = 1.05


def utc():
    return datetime.now(timezone.utc)


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def parse(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("NAIVE_TIME")
    return value.astimezone(timezone.utc)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def select_market():
    now = utc()
    with sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True) as con:
        row = con.execute("SELECT id,saved_at,payload FROM snapshots WHERE kind='polymarket' ORDER BY id DESC LIMIT 1").fetchone()
        clock_row = con.execute("SELECT id,saved_at,payload FROM snapshots WHERE kind='execution_clock' ORDER BY id DESC LIMIT 1").fetchone()
        if row is None or not 0 <= (now-parse(row[1])).total_seconds() <= 120:
            raise ValueError("NO_RECENT_LOCAL_SNAPSHOT")
        payload = json.loads(row[2])
        state = payload["streams"]["BTC_15m"]
        selected = None
        for position in ("current", "next"):
            market = state.get(position, {})
            if (market.get("status") == "VERIFIED_MARKET" and market.get("active") is True
                    and market.get("closed") is False and market.get("accepting_orders") is True
                    and market.get("asset") == "BTC" and market.get("duration_seconds") == 900
                    and (parse(market["window_end"])-now).total_seconds() >= 120
                    and 0 <= (now-parse(market["metadata_received_at"])).total_seconds() <= 120
                    and market.get("books", {}).get("UP", {}).get("status") == "OBSERVED"):
                selected = dict(market)
                selected.pop("books", None)
                selected["selected_position"] = position
                break
        if selected is None:
            raise ValueError("NO_VALID_CURRENT_OR_NEXT_BTC15")
        token = selected["outcome_ids"]["UP"]
        if not re.fullmatch(r"[1-9][0-9]{0,77}", token):
            raise ValueError("INVALID_TOKEN")
        metadata_receipt = con.execute("SELECT received_at,url,http_status,raw_text FROM receipts WHERE sha256=? ORDER BY id DESC LIMIT 1", (selected["metadata_sha256"],)).fetchone()
    if metadata_receipt is None or metadata_receipt[2] != 200 or sha(metadata_receipt[3].encode("utf-8")) != selected["metadata_sha256"]:
        raise ValueError("LOCAL_METADATA_RECEIPT_UNVERIFIED")
    return {
        "selected_at_local": iso(now), "snapshot_id": row[0], "snapshot_saved_at": row[1],
        "snapshot_payload_sha256": sha(row[2].encode("utf-8")), "market": selected,
        "outcome": "UP", "token_id": token, "condition_id": selected["condition_id"],
        "metadata_receipt": dict(zip(("received_at", "url", "http_status", "raw_text"), metadata_receipt)),
        "clock_snapshot": None if clock_row is None else {
            "id": clock_row[0], "saved_at": clock_row[1], "payload": json.loads(clock_row[2]),
            "payload_sha256": sha(clock_row[2].encode("utf-8"))},
    }


def new_session():
    session = requests.Session()
    session.trust_env = False
    session.auth = None
    session.proxies.clear()
    session.cookies.clear()
    session.mount("https://", HTTPAdapter(max_retries=0, pool_connections=1, pool_maxsize=1))
    return session


def clock_age(received, mono, source, selection):
    result = {"local_source_age_seconds_uncalibrated": (received-source).total_seconds(),
              "time_basis": "LOCAL_UTC_UNCALIBRATED", "source_age_bounds_seconds": None,
              "execution_eligible": False}
    try:
        clock = selection["clock_snapshot"]["payload"]
        age = mono-float(clock["calibrated_monotonic"])
        expected = parse(clock["calibrated_at_local"]) + timedelta(seconds=age)
        if clock["status"] != "CALIBRATED" or not 0 <= age <= 60 or abs((received-expected).total_seconds()) > .05:
            return result
        offset, uncertainty = float(clock["server_minus_local_seconds"]), float(clock["uncertainty_seconds"])
        local_age = result["local_source_age_seconds_uncalibrated"]
        result.update(time_basis="EXISTING_CLOB_CALIBRATION_INTERVAL", clock_age_seconds=age,
                      clock_uncertainty_seconds=uncertainty,
                      source_age_bounds_seconds=[local_age+offset-uncertainty, local_age+offset+uncertainty])
    except (KeyError, TypeError, ValueError):
        pass
    return result


def inspect_book(raw, selection, received, mono):
    book = json.loads(raw)
    if book.get("market") != selection["condition_id"] or str(book.get("asset_id")) != selection["token_id"]:
        raise ValueError("BOOK_IDENTITY_MISMATCH")
    stamp = str(book.get("timestamp", ""))
    if not re.fullmatch(r"[0-9]{13}", stamp):
        raise ValueError("INVALID_BOOK_SOURCE_TIME")
    levels = {}
    for side in ("bids", "asks"):
        if not isinstance(book.get(side), list):
            raise ValueError("MISSING_LEVELS")
        clean = []
        for level in book[side]:
            price, size = Decimal(str(level["price"])), Decimal(str(level["size"]))
            if not price.is_finite() or not size.is_finite() or not 0 < price < 1 or size < 0:
                raise ValueError("INVALID_LEVEL")
            if size > 0:
                clean.append((price, size))
        levels[side] = clean
    bid = max((p for p, _ in levels["bids"]), default=None)
    ask = min((p for p, _ in levels["asks"]), default=None)
    if bid is not None and ask is not None and bid >= ask:
        raise ValueError("CROSSED_BOOK")
    source = datetime.fromtimestamp(int(stamp)/1000, timezone.utc)
    return {"source_at": iso(source), "source_timestamp": stamp, "book_hash": book.get("hash"),
            "condition_id": book["market"], "asset_id": str(book["asset_id"]),
            "bid_levels": len(book["bids"]), "ask_levels": len(book["asks"]),
            "best_bid": None if bid is None else str(bid), "best_ask": None if ask is None else str(ask),
            "full_l2_valid": True, **clock_age(received, mono, source, selection)}


def one_request(session, selection, variant, pair, order, previous_start, run):
    delay = MIN_START_SPACING-(time.monotonic()-previous_start)
    if delay > 0:
        time.sleep(delay)
    session.trust_env = False
    session.auth = None
    session.proxies.clear()
    session.cookies.clear()
    url = ENDPOINT + "?" + urlencode({"token_id": selection["token_id"]})
    start = time.monotonic()
    requested = utc()
    row = {"pair": pair, "order": order, "variant": variant, "method": "GET", "url": url,
           "requested_at_local": iso(requested), "request_monotonic": start,
           "http_status": None, "body_complete": False, "error": None}
    raw = bytearray()
    try:
        with session.get(url, timeout=(2, 3), allow_redirects=False, stream=True,
                         headers={"Accept": "application/json", "User-Agent": "argon4-forward-public/1.0"}) as response:
            row["http_status"] = response.status_code
            row["response_headers_monotonic"] = time.monotonic()
            row["response_headers"] = {key: response.headers[key] for key in
                ("Date", "Content-Type", "Content-Encoding", "Content-Length", "Cache-Control", "Age", "CF-Cache-Status") if key in response.headers}
            deadline = time.monotonic()+6
            for chunk in response.iter_content(32768):
                raw.extend(chunk)
                if len(raw) > MAX_BYTES or time.monotonic() > deadline:
                    raise ValueError("RESPONSE_LIMIT")
            row["body_complete"] = True
        if row["http_status"] != 200:
            row["error"] = "HTTP_" + str(row["http_status"])
    except (requests.RequestException, ValueError) as exc:
        row["error"] = type(exc).__name__
    finally:
        received, end = utc(), time.monotonic()
        session.cookies.clear()
    row.update(received_at_local=iso(received), receive_monotonic=end, elapsed_seconds=end-start,
               raw_sha256=sha(raw), raw_bytes=len(raw),
               raw_encoding="HTTP_CONTENT_DECODED_BYTES_AS_RETURNED_BY_REQUESTS_ITER_CONTENT")
    filename = f"{pair:02d}_{order}_{variant}.body"
    (run/filename).write_bytes(raw)
    row["raw_file"] = filename
    if row["http_status"] == 200 and row["body_complete"]:
        try:
            row.update(inspect_book(raw, selection, received, end))
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            row["error"] = "BOOK_VALIDATION_" + type(exc).__name__
    write_json(run/f"{pair:02d}_{order}_{variant}.receipt.json", row)
    return row, start


def stats(rows):
    successful = [row["elapsed_seconds"] for row in rows if row["error"] is None]
    values = sorted(successful)
    def percentile95():
        position = (len(values)-1)*.95
        low = int(position)
        return values[low]+(values[min(low+1,len(values)-1)]-values[low])*(position-low)
    return {"attempts": len(rows), "successes": len(values), "errors": len(rows)-len(values),
            "median_seconds": statistics.median(values) if values else None,
            "p95_seconds_linear_interpolation": percentile95() if values else None,
            "max_seconds": max(values) if values else None}


def main():
    registry = json.loads((ROOT/"archive/catalog/dataset_registry.json").read_text(encoding="utf-8"))
    assert any(item.get("id") == "argon4_public_connection_latency_20261106" and item.get("convertible") is False
               for item in registry["datasets"]), "DATASET_NOT_REGISTERED"
    selection = select_market()
    run = OUT / (utc().strftime("run_%Y%m%dT%H%M%SZ_")+uuid.uuid4().hex[:8])
    run.mkdir(parents=False, exist_ok=False)
    write_json(run/"selection.json", selection)
    write_json(run/"protocol.json", {
        "pairs": PAIRS, "max_book_gets": 24, "pair_orders": "AB, BA alternating; A=new Session, B=persistent Session",
        "min_start_spacing_seconds": MIN_START_SPACING, "timeout_connect_read_seconds": [2, 3],
        "body_deadline_seconds_after_headers": 6, "max_response_bytes": MAX_BYTES,
        "allowed_endpoint": ENDPOINT, "sdk_executed": False, "network_auth": False,
        "trust_env": False, "redirects": False, "retries": 0, "cookies_cleared_each_request": True,
        "first_persistent_request_is_cold": True, "timing_excludes_parse_and_disk_write": True,
        "limitations": ["Sequential observations are not simultaneous or identical market states.",
            "Source time ages are non-executable; calibrated intervals only if copied calibration remains valid.",
            "Benchmark compares HTTP request latency, not model signals, execution fills, or profit.",
            "Cold means a new HTTP session; OS DNS and upstream caches are not flushed."],
        "script_sha256": sha(Path(__file__).read_bytes()), "requests_version": requests.__version__})
    rows, previous_start, stop_reason = [], float("-inf"), "COMPLETED_24_REQUESTS"
    with new_session() as persistent:
        for pair in range(1, PAIRS+1):
            variants = ("A", "B") if pair % 2 else ("B", "A")
            for order, variant in enumerate(variants, 1):
                if (parse(selection["market"]["window_end"])-utc()).total_seconds() < 10:
                    stop_reason = "MARKET_WINDOW_ENDING"
                    break
                if variant == "A":
                    with new_session() as session:
                        row, previous_start = one_request(session, selection, variant, pair, order, previous_start, run)
                else:
                    row, previous_start = one_request(persistent, selection, variant, pair, order, previous_start, run)
                rows.append(row)
                if row["http_status"] in (418, 429):
                    stop_reason = "RATE_LIMIT_STOP_NO_RETRY"
                    break
                if row["error"] is not None:
                    stop_reason = "ERROR_STOP_NO_RETRY"
                    break
            if stop_reason != "COMPLETED_24_REQUESTS":
                break
    groups = {variant: [row for row in rows if row["variant"] == variant] for variant in ("A", "B")}
    pair_deltas = []
    for pair in range(1, PAIRS+1):
        same = {row["variant"]: row for row in rows if row["pair"] == pair and row["error"] is None}
        if len(same) == 2:
            pair_deltas.append({"pair": pair, "A_minus_B_seconds": same["A"]["elapsed_seconds"]-same["B"]["elapsed_seconds"],
                                "same_raw_sha256": same["A"]["raw_sha256"] == same["B"]["raw_sha256"],
                                "same_source_timestamp": same["A"].get("source_timestamp") == same["B"].get("source_timestamp")})
    summary = {"status": stop_reason, "finished_at_local": iso(utc()), "requests": len(rows),
               "all_samples": {key: stats(value) for key, value in groups.items()},
               "persistent_excluding_first": stats(groups["B"][1:]),
               "both_variants_excluding_first_pair": {key: stats([r for r in value if r["pair"] > 1]) for key, value in groups.items()},
               "paired_differences": pair_deltas,
               "median_paired_A_minus_B_seconds": statistics.median([r["A_minus_B_seconds"] for r in pair_deltas]) if pair_deltas else None,
               "minimum_observed_start_spacing_seconds": min([rows[i]["request_monotonic"]-rows[i-1]["request_monotonic"] for i in range(1,len(rows))], default=None),
               "source_age_time_bases": {basis: sum(row.get("time_basis")==basis for row in rows) for basis in sorted({row.get("time_basis", "UNAVAILABLE") for row in rows})},
               "raw_hash_check": all(sha((run/row["raw_file"]).read_bytes()) == row["raw_sha256"] for row in rows),
               "production_modified": False, "orders_sent": 0, "sdk_executed": False}
    write_json(run/"requests.json", rows)
    write_json(run/"summary.json", summary)
    files = sorted(path for path in run.iterdir() if path.is_file())
    write_json(run/"manifest.json", {"files": [{"path": path.name, "bytes": path.stat().st_size,
                                               "sha256": sha(path.read_bytes())} for path in files]})
    print(json.dumps({"report_path": str(run), "summary": summary}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
