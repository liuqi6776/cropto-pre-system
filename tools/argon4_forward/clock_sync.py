"""Read-only CLOB clock calibration; preserve local timestamps and uncertainty.

The public clock has whole-second resolution. Intersect request/response bounds
instead of claiming a millisecond-accurate offset from one rounded timestamp.
No system clock changes, credentials, or exchange order endpoints are used.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import time

import requests

SOURCE = "POLYMARKET_CLOB_SERVER_TIME"
URL = "https://clob.polymarket.com/time"


def utc(value=None):
    value = value or datetime.now(timezone.utc)
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone required")
    return result.astimezone(timezone.utc)


def bounds_from_samples(samples):
    if len(samples) < 3:
        raise ValueError("at least three clock observations required")
    low, high = -math.inf, math.inf
    for sample in samples:
        start, end = utc(sample["requested_at_local"]).timestamp(), utc(sample["received_at_local"]).timestamp()
        server = sample["server_second"]
        elapsed = sample["monotonic_elapsed_seconds"]
        if isinstance(server, bool) or not isinstance(server, int) or not 0 <= end-start <= 2:
            raise ValueError("invalid clock observation")
        if not math.isfinite(elapsed) or abs(end-start-elapsed) > .05:
            raise ValueError("local clock stepped during request")
        low, high = max(low, server-end), min(high, server+1-start)
    if high < low:
        raise ValueError("inconsistent server clock observations")
    return {"server_minus_local_seconds": (low+high)/2,
            "uncertainty_seconds": (high-low)/2+.002,
            "offset_lower_seconds": low-.002, "offset_upper_seconds": high+.002}


def project_local(value, clock, *, upper=True):
    offset = clock["server_minus_local_seconds"] + (1 if upper else -1)*clock["uncertainty_seconds"]
    return utc(value) + timedelta(seconds=offset)


def calibrated_now(clock, *, upper=True):
    if clock.get("status") != "CALIBRATED":
        raise ValueError("clock not calibrated")
    age = time.monotonic()-clock["calibrated_monotonic"]
    if not 0 <= age <= 60:
        raise ValueError("clock calibration expired")
    local = datetime.now(timezone.utc)
    expected = utc(clock["calibrated_at_local"]) + timedelta(seconds=age)
    if abs((local-expected).total_seconds()) > .05:
        raise ValueError("local clock changed after calibration")
    return project_local(local, clock, upper=upper)


def calibrate(samples=6, *, transport=None):
    if not 3 <= samples <= 8:
        raise ValueError("clock sample count out of range")
    receipts = []
    result = {"status": "UNCALIBRATED", "source": SOURCE, "receipts": receipts}
    try:
        if transport is not None:
            for _ in range(samples):
                data, receipt, error = transport.get("time")
                if error or not receipt.get("body_complete") or receipt.get("http_status") != 200:
                    result["failed_receipt"] = receipt
                    raise ValueError("clock response unavailable")
                raw = receipt["raw_text"].encode("ascii")
                body = raw.decode("ascii").strip()
                if len(raw) >= 64 or not body.isdigit() or len(body) != 10:
                    raise ValueError("unexpected server clock format")
                if isinstance(data, bool) or data != int(body):
                    raise ValueError("unexpected server clock value")
                receipts.append({"url": URL, "requested_at_local": receipt["requested_at"],
                                 "received_at_local": receipt["received_at"], "received_at": receipt["received_at"],
                                 "monotonic_elapsed_seconds": receipt["transport_timing"]["request_elapsed_seconds"],
                                 "server_second": int(body), "raw_text": raw.decode("ascii"), "http_status": 200,
                                 "sha256": receipt["sha256"]})
        else:
            _collect_samples(samples, receipts)
        result.update(bounds_from_samples(receipts))
        result["receipt_sha256"] = hashlib.sha256(json.dumps(receipts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result["receipt"] = {"samples": receipts, "quantization_seconds": 1}
        stamp = datetime.now(timezone.utc)
        result.update(calibrated_at_local=stamp.isoformat(), calibrated_monotonic=time.monotonic(),
                      observed_at=project_local(stamp, result).isoformat())
        result["status"] = "CALIBRATED" if result["uncertainty_seconds"] <= .5 else "UNCERTAINTY_TOO_LARGE"
    except (requests.RequestException, ValueError, UnicodeError) as exc:
        result["error"] = type(exc).__name__
    return result


def _collect_samples(samples, receipts):
    """Standalone compatibility path; callers with a lifecycle inject transport."""
    with requests.Session() as session:
        session.trust_env = False
        for _ in range(samples):
            requested = datetime.now(timezone.utc)
            begin = time.monotonic()
            with session.get(URL, timeout=(2, 2), allow_redirects=False, stream=True,
                             headers={"Cache-Control": "no-cache", "User-Agent": "argon4-paper-clock/1"}) as response:
                raw = response.raw.read(64)
                received = datetime.now(timezone.utc)
                elapsed = time.monotonic()-begin
                if response.status_code != 200 or len(raw) >= 64:
                    raise ValueError("clock response unavailable")
            body = raw.decode("ascii").strip()
            if not body.isdigit() or len(body) != 10:
                raise ValueError("unexpected server clock format")
            receipts.append({"url": URL, "requested_at_local": requested.isoformat(),
                             "received_at_local": received.isoformat(), "received_at": received.isoformat(),
                             "monotonic_elapsed_seconds": elapsed, "server_second": int(body),
                             "raw_text": raw.decode("ascii"), "http_status": 200,
                             "sha256": hashlib.sha256(raw).hexdigest()})
