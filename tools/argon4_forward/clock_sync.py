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


def _check_budget(budget_monotonic, epoch=None):
    if budget_monotonic is not None and time.monotonic() >= budget_monotonic:
        raise ValueError("CLOCK_DEADLINE_EXCEEDED")
    if epoch is not None:
        elapsed = time.monotonic()-epoch[1]
        if elapsed < 0 or abs((datetime.now(timezone.utc)-epoch[0]).total_seconds()-elapsed) > .05:
            raise ValueError("LOCAL_CLOCK_CHANGED_DURING_CALIBRATION")


def _transport_samples(count, receipts, result, transport, budget_monotonic, epoch=None):
    for _ in range(count):
        _check_budget(budget_monotonic, epoch)
        if budget_monotonic is None:
            data, receipt, error = transport.get("time")
        else:
            data, receipt, error = transport.get("time", deadline_monotonic=budget_monotonic)
        try:
            _check_budget(budget_monotonic, epoch)
            if error or not receipt.get("body_complete") or receipt.get("http_status") != 200:
                raise ValueError("clock response unavailable")
            if budget_monotonic is not None and (receipt.get("url") != URL
                    or receipt.get("method") != "GET" or receipt.get("source") != "POLYMARKET_PUBLIC_READ_ONLY"):
                raise ValueError("clock receipt source mismatch")
            raw = receipt["raw_text"].encode("ascii")
            body = raw.decode("ascii").strip()
            if len(raw) >= 64 or not body.isdigit() or len(body) != 10:
                raise ValueError("unexpected server clock format")
            if isinstance(data, bool) or data != int(body):
                raise ValueError("unexpected server clock value")
            if hashlib.sha256(raw).hexdigest() != receipt.get("sha256"):
                raise ValueError("clock receipt hash mismatch")
            receipts.append({"url": URL, "requested_at_local": receipt["requested_at"],
                             "received_at_local": receipt["received_at"], "received_at": receipt["received_at"],
                             "monotonic_elapsed_seconds": receipt["transport_timing"]["request_elapsed_seconds"],
                             "server_second": int(body), "raw_text": raw.decode("ascii"), "http_status": 200,
                             "sha256": receipt["sha256"]})
        except (requests.RequestException, ValueError, UnicodeError):
            result["failed_receipt"] = receipt
            raise


def calibrate(samples=6, *, transport=None, retry_borderline=False, budget_monotonic=None):
    """Keep the 0.5s acceptance bound; optionally try two more real samples.

    Defaults retain the original six observations. Extra sampling requires an
    explicit finite budget, uses at most eight observations, and never accepts
    a seven-sample partial retry after an error or deadline.
    """
    if not 3 <= samples <= 8:
        raise ValueError("clock sample count out of range")
    if not isinstance(retry_borderline, bool) or (retry_borderline and samples != 6):
        raise ValueError("borderline retry requires the six-sample baseline")
    if budget_monotonic is not None and (isinstance(budget_monotonic, bool)
            or not isinstance(budget_monotonic, (int, float)) or not math.isfinite(budget_monotonic)):
        raise ValueError("invalid clock budget")
    if transport is None and budget_monotonic is not None:
        # The old standalone collection path remains unchanged without a
        # budget. Budgeted calls share the deadline-aware public transport.
        from .polymarket_transport import PublicPolymarketTransport
        with PublicPolymarketTransport() as owned:
            result = calibrate(samples, transport=owned, retry_borderline=retry_borderline,
                               budget_monotonic=budget_monotonic)
        if time.monotonic() >= budget_monotonic:
            result.update(status="UNCALIBRATED", error="ValueError", error_code="CLOCK_DEADLINE_EXCEEDED")
            if result.get("borderline_retry", {}).get("status") == "COMPLETED":
                result["borderline_retry"]["status"] = "FAILED"
            for field in ("calibrated_at_local", "calibrated_monotonic", "observed_at"):
                result.pop(field, None)
        return result
    receipts = []
    result = {"status": "UNCALIBRATED", "source": SOURCE, "receipts": receipts}
    retry = None
    if retry_borderline:
        retry = {"enabled": True, "base_samples": 6, "maximum_extra_samples": 2,
                 "extra_samples_collected": 0, "status": "NOT_NEEDED"}
        result["borderline_retry"] = retry
    # Extra observations may cross a clock adjustment between GETs. The old
    # per-request elapsed checks alone cannot establish a common clock epoch.
    epoch = (datetime.now(timezone.utc), time.monotonic()) if budget_monotonic is not None or retry_borderline else None
    try:
        _check_budget(budget_monotonic, epoch)
        if transport is not None:
            _transport_samples(samples, receipts, result, transport, budget_monotonic, epoch)
        else:
            _collect_samples(samples, receipts)
        bounds = bounds_from_samples(receipts)
        if retry is not None:
            retry["initial_uncertainty_seconds"] = bounds["uncertainty_seconds"]
            if .5 < bounds["uncertainty_seconds"] <= .65:
                if budget_monotonic is None:
                    retry["status"] = "NO_RETRY_BUDGET"
                else:
                    retry["status"] = "COLLECTING"
                    _transport_samples(2, receipts, result, transport, budget_monotonic, epoch)
                    bounds = bounds_from_samples(receipts)
                    retry["status"] = "COMPLETED"
            elif bounds["uncertainty_seconds"] > .65:
                retry["status"] = "OUTSIDE_RETRY_RANGE"
        _check_budget(budget_monotonic, epoch)
        result.update(bounds)
        result["receipt_sha256"] = hashlib.sha256(json.dumps(receipts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result["receipt"] = {"samples": receipts, "quantization_seconds": 1}
        stamp = datetime.now(timezone.utc)
        _check_budget(budget_monotonic, epoch)
        result.update(calibrated_at_local=stamp.isoformat(), calibrated_monotonic=time.monotonic(),
                      observed_at=project_local(stamp, result).isoformat())
        _check_budget(budget_monotonic, epoch)
        result["status"] = "CALIBRATED" if result["uncertainty_seconds"] <= .5 else "UNCERTAINTY_TOO_LARGE"
    except (requests.RequestException, ValueError, UnicodeError) as exc:
        result["error"] = type(exc).__name__
        if retry is not None and retry["status"] in ("COLLECTING", "COMPLETED"):
            retry["status"] = "FAILED"
        if isinstance(exc, ValueError) and exc.args in (("CLOCK_DEADLINE_EXCEEDED",),
                                                       ("LOCAL_CLOCK_CHANGED_DURING_CALIBRATION",)):
            result["error_code"] = exc.args[0]
        # A failed or expired calibration never advertises a fresh accepted
        # clock even if six or seven earlier observations had narrow bounds.
        for field in ("calibrated_at_local", "calibrated_monotonic", "observed_at"):
            result.pop(field, None)
    if retry is not None:
        retry["extra_samples_collected"] = max(0, len(receipts)-samples)
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
