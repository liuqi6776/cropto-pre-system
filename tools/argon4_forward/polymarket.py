"""Public, read-only discovery for the seven ARGON short-duration markets.

No wallet, authentication, order methods, model, simulated fills or settlement
inference are present. A slug is only a lookup candidate: metadata, event times,
asset, duration, outcome mapping and the actual resolution rule must also match.

API references checked 2026-11-05:
https://docs.polymarket.com/market-data/discover-markets
https://docs.polymarket.com/market-data/market-details
https://docs.polymarket.com/market-data/prices-order-books
https://docs.polymarket.com/trading/fees
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import re

from .polymarket_transport import (ENDPOINTS, MAX_RESPONSE_BYTES, SOURCE,
                                   PublicPolymarketTransport, public_url)

STREAMS = {
    "BTC_15m": ("btc", 900, "Bitcoin"),
    "ETH_15m": ("eth", 900, "Ethereum"),
    "BTC_5m": ("btc", 300, "Bitcoin"),
    "ETH_5m": ("eth", 300, "Ethereum"),
    "SOL_15m": ("sol", 900, "Solana"),
    "SOL_5m": ("sol", 300, "Solana"),
    "BNB_15m": ("bnb", 900, "BNB"),
}
MAX_BOOK_AGE_SECONDS = 90
MAX_CLOCK_LEAD_SECONDS = 10


def _utc(value=None):
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        dt = datetime.fromtimestamp(value, timezone.utc)
    else:
        raise ValueError("UTC datetime, ISO timestamp or epoch seconds required")
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("naive timestamp is not accepted")
    return dt.astimezone(timezone.utc)


def _iso(value):
    return _utc(value).isoformat().replace("+00:00", "Z")


def _number(value):
    if isinstance(value, bool) or value is None:
        raise ValueError("number is missing or boolean")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid number") from exc
    if not result.is_finite():
        raise ValueError("nonfinite number")
    return result


def _array(value):
    result = json.loads(value) if isinstance(value, str) else value
    if not isinstance(result, list):
        raise ValueError("array required")
    return result


def _url(kind, value=None):
    return public_url(kind, value)


def _get_public(kind, value=None, *, transport=None):
    """One bounded GET; return (JSON or None, receipt, error code or None)."""
    _url(kind, value)
    if transport is not None:
        return transport.get(kind, value)
    with PublicPolymarketTransport() as owned:
        return owned.get(kind, value)


def parse_fee(market):
    """Report the market's schedule; absent fee information never means zero."""
    enabled = market.get("feesEnabled")
    schedule = market.get("feeSchedule")
    if enabled is False:
        return {"status": "KNOWN_DISABLED", "fees_enabled": False,
                "rate": 0.0, "source_field": "Gamma.feesEnabled"}
    try:
        if enabled is not True or not isinstance(schedule, dict):
            raise ValueError("missing schedule")
        rate = _number(schedule.get("rate"))
        exponent = _number(schedule.get("exponent"))
        rebate = _number(schedule.get("rebateRate"))
        if not (0 <= rate <= 1 and 0 <= exponent <= 10 and 0 <= rebate <= 1):
            raise ValueError("invalid schedule")
        if not isinstance(schedule.get("takerOnly"), bool):
            raise ValueError("missing taker flag")
        return {"status": "KNOWN_SCHEDULE", "fees_enabled": True,
                "rate": float(rate), "exponent": float(exponent),
                "taker_only": schedule["takerOnly"], "rebate_rate": float(rebate),
                "source_field": "Gamma.feeSchedule",
                "calculation_status": "NOT_APPLIED_NO_ORDERS"}
    except (ValueError, TypeError):
        return {"status": "UNKNOWN", "fees_enabled": enabled,
                "rate": None, "source_field": "Gamma.feeSchedule"}


def validate_market(market, stream, expected_start, now):
    """Validate an actual Gamma response. Never derive missing times from a slug."""
    asset, duration, name = STREAMS[stream]
    expected_start, now = _utc(expected_start), _utc(now)
    slug = f"{asset}-updown-{duration // 60}m-{int(expected_start.timestamp())}"
    if not isinstance(market, dict) or market.get("slug") != slug:
        raise ValueError("MARKET_IDENTITY_MISMATCH")
    if market.get("version") not in ("v1", "v2"):
        raise ValueError("UNSUPPORTED_PROTOCOL")
    config = market.get("cryptoMarketConfig")
    if not isinstance(config, dict) or config.get("asset") != asset or config.get("duration") != f"{duration // 60}m":
        raise ValueError("ASSET_OR_DURATION_MISMATCH")
    if not str(market.get("question", "")).startswith(f"{name} Up or Down - "):
        raise ValueError("QUESTION_MISMATCH")
    # startDate is often the day the market was created, NOT the prediction start.
    start = _utc(market.get("eventStartTime")) if market.get("eventStartTime") else None
    end = _utc(market.get("endDate")) if market.get("endDate") else None
    if start is None or end is None or start != expected_start or (end - start).total_seconds() != duration:
        raise ValueError("WINDOW_MISMATCH_OR_MISSING")
    if end <= now:
        raise ValueError("EXPIRED_WINDOW")
    current_start = int(now.timestamp()) // duration * duration
    if int(start.timestamp()) not in (current_start, current_start + duration):
        raise ValueError("NOT_CURRENT_OR_NEXT_WINDOW")
    events = market.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("EVENT_EVIDENCE_MISSING")
    matching_events = [e for e in events if e.get("slug") == slug]
    if len(matching_events) != 1:
        raise ValueError("EVENT_IDENTITY_MISMATCH")
    event = matching_events[0]
    if not event.get("startTime") or _utc(event["startTime"]) != start or not event.get("endDate") or _utc(event["endDate"]) != end:
        raise ValueError("EVENT_WINDOW_MISMATCH")
    if any(market.get(key) is not value for key, value in {
            "active": True, "closed": False, "archived": False,
            "acceptingOrders": True, "enableOrderBook": True}.items()):
        raise ValueError("MARKET_NOT_ACCEPTING_ORDERS")
    if market.get("negRisk") is not False:
        raise ValueError("UNSUPPORTED_NEGATIVE_RISK")
    lookback = config.get("twapLookbackSeconds")
    if config.get("twapEnabled") is not True or type(lookback) is not int or not 0 < lookback <= duration:
        raise ValueError("UNSUPPORTED_RESOLUTION_RULE")
    resolution = market.get("resolutionSource", "")
    expected_source = f"https://data.chain.link/streams/{asset}-usd-twap-{lookback}s-streams"
    if resolution != expected_source:
        raise ValueError("RESOLUTION_SOURCE_MISMATCH")
    description = str(market.get("description", ""))
    if not all(part in description for part in ('resolve to "Up"', 'greater than or equal to',
                                                'resolve to "Down"', 'Chainlink', f'{asset.upper()}/USD')):
        raise ValueError("RESOLUTION_RULE_MISMATCH")
    labels = _array(market.get("outcomes"))
    id_field = "positionIds" if market["version"] == "v2" else "clobTokenIds"
    ids = _array(market.get(id_field))
    if len(labels) != 2 or set(labels) != {"Up", "Down"} or len(ids) != 2 or len(set(map(str, ids))) != 2:
        raise ValueError("OUTCOME_MAPPING_INVALID")
    for value in ids:
        if not re.fullmatch(r"[1-9][0-9]{0,77}", str(value)):
            raise ValueError("OUTCOME_IDENTIFIER_INVALID")
    condition = market.get("conditionId")
    if not isinstance(condition, str) or not re.fullmatch(r"0x[0-9a-fA-F]{64}", condition):
        raise ValueError("CONDITION_ID_INVALID")
    if not str(market.get("id", "")).isdigit():
        raise ValueError("MARKET_ID_INVALID")
    return {"status": "VERIFIED_MARKET", "market_id": str(market["id"]),
            "condition_id": condition, "slug": slug, "question": market["question"],
            "url": f"https://polymarket.com/event/{slug}", "protocol": market["version"],
            "asset": asset.upper(), "duration_seconds": duration,
            "window_start": _iso(start), "window_end": _iso(end),
            "time_source_fields": ["eventStartTime", "events.startTime", "endDate"],
            "outcome_ids": {label.upper(): str(value) for label, value in zip(labels, ids)},
            "outcome_id_kind": id_field, "active": True, "closed": False,
            "accepting_orders": True, "geo_restricted": market.get("restricted"),
            "resolution": {"source": resolution, "type": "CHAINLINK_TWAP",
                           "lookback_seconds": lookback, "up_includes_equal": True,
                           "description": description},
            "fee": parse_fee(market), "books": {}, "errors": [],
            "execution_enabled": False, "model_status": "WAITING_MODEL"}


def validate_book(book, outcome_id, condition_id, received_at):
    if not isinstance(book, dict) or str(book.get("asset_id")) != outcome_id or book.get("market") != condition_id:
        raise ValueError("BOOK_IDENTITY_MISMATCH")
    raw_time = str(book.get("timestamp", ""))
    if not re.fullmatch(r"[0-9]{13}", raw_time):
        raise ValueError("BOOK_TIMESTAMP_MISSING_OR_INVALID")
    source_time = _utc(int(raw_time) / 1000)
    age = (_utc(received_at) - source_time).total_seconds()
    if age < -MAX_CLOCK_LEAD_SECONDS:
        raise ValueError("BOOK_CLOCK_AHEAD")
    levels = {}
    for side in ("bids", "asks"):
        rows = book.get(side)
        if not isinstance(rows, list):
            raise ValueError("BOOK_LEVELS_MISSING")
        clean = []
        for row in rows:
            price, size = _number(row.get("price")), _number(row.get("size"))
            if not (0 < price < 1) or size < 0:
                raise ValueError("BOOK_LEVEL_INVALID")
            if size > 0:
                clean.append({"price": float(price), "size": float(size)})
        levels[side] = sorted(clean, key=lambda level: level["price"], reverse=side == "bids")
    best_bid = levels["bids"][0]["price"] if levels["bids"] else None
    best_ask = levels["asks"][0]["price"] if levels["asks"] else None
    if best_bid is not None and best_ask is not None and best_bid >= best_ask:
        raise ValueError("BOOK_CROSSED")
    status = "STALE" if age > MAX_BOOK_AGE_SECONDS else ("OBSERVED" if best_ask is not None and best_bid is not None else "EMPTY_OR_ONE_SIDED")
    return {"status": status, "asset_id": outcome_id, "condition_id": condition_id,
            "source_at": _iso(source_time), "received_at": _iso(received_at),
            "age_seconds": age, "source_clock_calibrated": False,
            "best_bid": best_bid, "best_ask": best_ask, **levels,
            "source_hash": book.get("hash"), "execution_enabled": False}


def collect_market_snapshot(now=None, *, transport=None, getter=None):
    """Collect current/next windows, bounded to 14 Gamma + 28 book GETs.

    `now` is a timezone-aware datetime, UTC ISO string, or epoch seconds. It
    controls discovery only; response receipt times always use the real clock.
    No files are written. The caller owns persistence of raw response receipts.
    """
    if transport is not None and getter is not None:
        raise ValueError("choose one public transport or getter")
    get_public = (getter if getter is not None else
                  (lambda kind, value: _get_public(kind, value, transport=transport))
                  if transport is not None else _get_public)
    now = _utc(now)
    snapshot = {"requested_at": _iso(now), "observed_at": None, "source": SOURCE,
                "status": "WAITING_MODEL", "market_data_status": "UNAVAILABLE",
                "streams": {}, "errors": [], "receipts": [],
                "execution_enabled": False}
    jobs = []
    for stream, (asset, duration, _) in STREAMS.items():
        floor = int(now.timestamp()) // duration * duration
        snapshot["streams"][stream] = {"status": "WAITING_MODEL", "market_status": "NOT_FOUND",
                                        "current": None, "next": None, "errors": []}
        for slot, offset in (("current", 0), ("next", duration)):
            start = _utc(floor + offset)
            slug = f"{asset}-updown-{duration // 60}m-{floor + offset}"
            jobs.append((stream, slot, start, slug))

    def discover(job):
        stream, slot, start, slug = job
        data, receipt, error = get_public("market", slug)
        if error:
            return job, None, receipt, error
        try:
            if not isinstance(data, list):
                raise ValueError("MARKET_RESPONSE_NOT_ARRAY")
            if not data:
                return job, None, receipt, "MARKET_NOT_FOUND"
            if len(data) != 1:
                raise ValueError("MARKET_RESPONSE_AMBIGUOUS")
            market = validate_market(data[0], stream, start, now)
            market["metadata_received_at"] = receipt["received_at"]
            market["metadata_sha256"] = receipt["sha256"]
            return job, market, receipt, None
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return job, None, receipt, f"VALIDATION:{str(exc)[:100]}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        for job, market, receipt, error in pool.map(discover, jobs):
            stream, slot, _, slug = job
            snapshot["receipts"].append(receipt)
            if error:
                detail = {"stream": stream, "window": slot, "slug": slug, "code": error}
                snapshot["errors"].append(detail)
                snapshot["streams"][stream]["errors"].append(detail)
            snapshot["streams"][stream][slot] = market

    books = []
    for stream, state in snapshot["streams"].items():
        for slot in ("current", "next"):
            market = state[slot]
            if market:
                for side, outcome_id in market["outcome_ids"].items():
                    books.append((stream, slot, side, outcome_id, market["condition_id"]))

    def fetch_book(job):
        stream, slot, side, outcome_id, condition_id = job
        data, receipt, error = get_public("book", outcome_id)
        if error:
            return job, None, receipt, error
        try:
            parsed = validate_book(data, outcome_id, condition_id, receipt["received_at"])
            return job, parsed, receipt, None
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return job, None, receipt, f"VALIDATION:{str(exc)[:100]}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        for job, book, receipt, error in pool.map(fetch_book, books):
            stream, slot, side, _, _ = job
            snapshot["receipts"].append(receipt)
            market = snapshot["streams"][stream][slot]
            market["books"][side] = book
            if error:
                detail = {"stream": stream, "window": slot, "outcome": side, "code": error}
                snapshot["errors"].append(detail)
                market["errors"].append(detail)

    complete = 0
    quality_complete = True
    latest_receipt = max([now] + [_utc(r["received_at"]) for r in snapshot["receipts"] if r.get("received_at")])
    for state in snapshot["streams"].values():
        markets = [state[slot] for slot in ("current", "next") if state[slot]]
        if markets:
            state["market_status"] = "VERIFIED_MARKETS" if len(markets) == 2 else "PARTIAL"
            complete += len(markets)
            for market in markets:
                if _utc(market["window_end"]) <= latest_receipt:
                    market["status"] = "EXPIRED_DURING_COLLECTION"
                    state["market_status"] = "PARTIAL"
                    quality_complete = False
                if market["fee"]["status"] == "UNKNOWN" or len(market["books"]) != 2:
                    quality_complete = False
                if any(book is None or book["status"] != "OBSERVED" for book in market["books"].values()):
                    quality_complete = False
    snapshot["market_data_status"] = "COMPLETE" if complete == 14 and not snapshot["errors"] and quality_complete else ("PARTIAL" if complete else "UNAVAILABLE")
    snapshot["observed_at"] = _iso(None)
    return snapshot
