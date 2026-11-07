"""Offline behavioral checks: no credentials, network or orders are involved."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from tools.argon4_forward import polymarket as pm

NOW = datetime(2026, 11, 5, 15, 17, tzinfo=timezone.utc)
START = NOW.replace(minute=15)
CONDITION = "0x" + "a" * 64


def market_fixture():
    slug = f"btc-updown-15m-{int(START.timestamp())}"
    return {"id": "6678947", "slug": slug, "version": "v1", "conditionId": CONDITION,
            "question": "Bitcoin Up or Down - November 5, 10:15AM-10:30AM ET",
            "startDate": "2026-11-04T15:23:45Z", "eventStartTime": pm._iso(START),
            "endDate": pm._iso(START + timedelta(minutes=15)),
            "events": [{"slug": slug, "startTime": pm._iso(START),
                        "endDate": pm._iso(START + timedelta(minutes=15))}],
            "active": True, "closed": False, "archived": False,
            "acceptingOrders": True, "enableOrderBook": True, "negRisk": False,
            "cryptoMarketConfig": {"asset": "btc", "duration": "15m", "twapEnabled": True,
                                   "twapLookbackSeconds": 60},
            "resolutionSource": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
            "description": 'This market will resolve to "Up" if the Chainlink BTC/USD TWAP is greater than or equal to the initial price. Otherwise, it will resolve to "Down".',
            "outcomes": '["Down", "Up"]', "clobTokenIds": '["222", "111"]',
            "feesEnabled": True, "feeSchedule": {"rate": 0.07, "exponent": 1,
                                                   "takerOnly": True, "rebateRate": 0.2}}


def book_fixture(outcome="111"):
    return {"market": CONDITION, "asset_id": outcome,
            "timestamp": str(int(NOW.timestamp() * 1000)),
            "bids": [{"price": "0.01", "size": "10"}, {"price": "0.48", "size": "5"}],
            "asks": [{"price": "0.99", "size": "10"}, {"price": "0.52", "size": "20"}],
            "hash": "fixture"}


def test_market_uses_actual_window_and_aligned_outcome_labels():
    parsed = pm.validate_market(market_fixture(), "BTC_15m", START, NOW)
    assert parsed["window_start"] == "2026-11-05T15:15:00Z"
    assert parsed["outcome_ids"] == {"DOWN": "222", "UP": "111"}
    assert parsed["fee"]["rate"] == 0.07
    assert parsed["resolution"]["lookback_seconds"] == 60
    assert parsed["execution_enabled"] is False


@pytest.mark.parametrize("change", [
    {"cryptoMarketConfig": {"asset": "eth", "duration": "15m"}},
    {"question": "Ethereum Up or Down - November 5"},
    {"eventStartTime": "2026-11-05T15:00:00Z"},
    {"eventStartTime": None},
    {"endDate": "2026-11-05T15:35:00Z"},
    {"acceptingOrders": False}, {"closed": True}, {"active": "true"},
    {"resolutionSource": "https://example.com/btc-usd"},
    {"description": "Bitcoin goes down when price rises"},
    {"outcomes": '["Yes", "No"]'}, {"clobTokenIds": '["111", "111"]'},
    {"version": "v3"}, {"events": []},
])
def test_wrong_or_incomplete_market_rejected(change):
    market = market_fixture()
    market.update(change)
    with pytest.raises(ValueError):
        pm.validate_market(market, "BTC_15m", START, NOW)


def test_expired_window_rejected():
    with pytest.raises(ValueError, match="EXPIRED"):
        pm.validate_market(market_fixture(), "BTC_15m", START, START + timedelta(minutes=15))


def test_v2_ids_are_selected_by_protocol_not_field_presence():
    market = market_fixture()
    market.update(version="v2", positionIds=["444", "333"])
    parsed = pm.validate_market(market, "BTC_15m", START, NOW)
    assert parsed["outcome_ids"] == {"DOWN": "444", "UP": "333"}


@pytest.mark.parametrize("fee", [{}, {"feesEnabled": True}, {"feesEnabled": True, "feeSchedule": {"rate": "NaN"}}])
def test_unknown_fee_never_defaults_to_zero(fee):
    parsed = pm.parse_fee(fee)
    assert parsed["status"] == "UNKNOWN" and parsed["rate"] is None


def test_explicit_fee_disabled_is_distinct_from_unknown():
    assert pm.parse_fee({"feesEnabled": False})["status"] == "KNOWN_DISABLED"


def test_book_sorts_best_prices_and_preserves_source_and_receipt_times():
    parsed = pm.validate_book(book_fixture(), "111", CONDITION, NOW + timedelta(seconds=2))
    assert parsed["best_bid"] == 0.48 and parsed["best_ask"] == 0.52
    assert parsed["age_seconds"] == 2 and parsed["status"] == "OBSERVED"
    assert parsed["source_at"] != parsed["received_at"]


@pytest.mark.parametrize("change", [
    {"asset_id": "222"}, {"market": "0x" + "b" * 64}, {"timestamp": None},
    {"asks": [{"price": "1", "size": "10"}]},
    {"asks": [{"price": "0.5", "size": "-1"}]},
    {"asks": [{"price": "NaN", "size": "1"}]},
    {"asks": [{"price": "0.1", "size": "1"}]},
])
def test_bad_book_is_not_used(change):
    book = book_fixture()
    book.update(change)
    with pytest.raises(ValueError):
        pm.validate_book(book, "111", CONDITION, NOW)


def test_stale_book_not_presented_as_fresh():
    parsed = pm.validate_book(book_fixture(), "111", CONDITION, NOW + timedelta(minutes=3))
    assert parsed["status"] == "STALE"
    with pytest.raises(ValueError, match="CLOCK"):
        pm.validate_book(book_fixture(), "111", CONDITION, NOW - timedelta(minutes=1))


def test_missing_markets_and_failures_return_status_without_fabrication(monkeypatch):
    calls = []
    def fake_get(kind, value):
        calls.append((kind, value))
        return [], {"received_at": pm._iso(NOW), "sha256": "fixture", "raw_text": "[]"}, None
    monkeypatch.setattr(pm, "_get_public", fake_get)
    result = pm.collect_market_snapshot(NOW)
    assert len(calls) == 14 and all(kind == "market" for kind, _ in calls)
    assert result["market_data_status"] == "UNAVAILABLE"
    assert len(result["receipts"]) == 14
    assert all(row["current"] is None and row["status"] == "WAITING_MODEL" for row in result["streams"].values())


def test_partial_collection_only_reads_tokens_from_validated_market(monkeypatch):
    market = market_fixture()
    calls = []
    def fake_get(kind, value):
        calls.append((kind, value))
        payload = ([deepcopy(market)] if value == market["slug"] else []) if kind == "market" else book_fixture(value)
        return payload, {"received_at": pm._iso(NOW), "sha256": "fixture", "raw_text": json.dumps(payload)}, None
    monkeypatch.setattr(pm, "_get_public", fake_get)
    result = pm.collect_market_snapshot(NOW)
    assert {value for kind, value in calls if kind == "book"} == {"111", "222"}
    state = result["streams"]["BTC_15m"]
    assert state["current"]["books"]["UP"]["asset_id"] == "111"
    assert result["market_data_status"] == "PARTIAL"
    assert result["execution_enabled"] is False


def test_only_constant_public_get_routes_are_constructible():
    with pytest.raises(ValueError):
        pm._url("order", "111")
    with pytest.raises(ValueError):
        pm._url("book", "https://evil.invalid/?token=111")
    with pytest.raises(ValueError):
        pm._url("market", "../../credentials")
    assert pm._url("book", "111") == "https://clob.polymarket.com/book?token_id=111"


def test_naive_now_rejected():
    with pytest.raises(ValueError, match="naive"):
        pm.collect_market_snapshot(datetime(2026, 11, 5))
