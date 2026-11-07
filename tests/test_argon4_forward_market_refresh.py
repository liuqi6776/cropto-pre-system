"""Optional exact-market refresh stays behind signal/rank/deadline guards."""
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import requests

from tools.argon4_forward.account import Account
from tools.argon4_forward.execution import PaperExecutor
from test_argon4_forward_execution import BOUNDARY, NOW, clock, fixture_inputs, no_signal, v5_policy


ACTIVE = ["ETH_5m", "BTC_15m", "BTC_5m"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("network forbidden"))


def prepared(tmp_path, boundary=BOUNDARY):
    policy = v5_policy()
    policy["order"] = ACTIVE + [s for s in policy["order"] if s not in ACTIVE]
    policy["execution_stage"] = {"id": "refresh-fixture", "status": "CONFIRMED",
        "enabled_streams": ACTIVE, "effective_from": BOUNDARY.isoformat()}
    account = Account(tmp_path / "isolated.sqlite3", started_at=BOUNDARY-timedelta(seconds=1))
    decisions, markets = fixture_inputs(capacity=600, boundary=boundary)
    return SimpleNamespace(account=account, executor=PaperExecutor(account, policy),
        decisions=decisions, markets=markets, boundary=boundary, now=boundary+timedelta(seconds=1))


def execute(case, **kwargs):
    return case.executor.process_cohort(case.boundary.isoformat(), case.decisions, case.markets,
        now=kwargs.pop("now", case.now), clock=kwargs.pop("clock", clock(case.now)), **kwargs)


def test_missing_first_signal_refreshes_exact_market_once_then_follows_budget(tmp_path):
    f = prepared(tmp_path)
    fresh = f.markets.pop("ETH_5m")
    calls, books = [], []
    def market_refresh(stream, boundary, existing):
        calls.append((stream, boundary, existing))
        return fresh
    def book_refresh(market, side):
        books.append((market["market_id"], side))
        return market["books"][side]
    result = execute(f, market_refresh=market_refresh, book_refresh=book_refresh)
    assert calls == [("ETH_5m", BOUNDARY, None)]
    assert [x["cost_usd"] for x in result["fills"]] == [150, 70]
    assert [x[0] for x in books] == [fresh["market_id"], f.markets["BTC_15m"]["market_id"]]
    assert result["stream_results"]["BTC_5m"]["reason"] == "COHORT_BUDGET_EXHAUSTED"
    assert "ETH_5m" not in f.markets
    assert all(x["live_enabled"] is False for x in result["fills"])


def test_stale_metadata_refresh_does_not_change_cached_timestamp(tmp_path):
    f = prepared(tmp_path)
    fresh = deepcopy(f.markets["ETH_5m"])
    f.markets["ETH_5m"]["metadata_received_at"] = (NOW-timedelta(seconds=91)).isoformat()
    original = deepcopy(f.markets)
    calls = []
    def refresh(stream, boundary, existing):
        calls.append(deepcopy(existing))
        existing["metadata_received_at"] = NOW.isoformat()
        return existing
    result = execute(f, market_refresh=refresh)
    assert len(calls) == 1 and calls[0]["metadata_received_at"] == original["ETH_5m"]["metadata_received_at"]
    assert result["fills"][0]["market_id"] == fresh["market_id"]
    assert f.markets == original


@pytest.mark.parametrize("age", [0, 89.999, 90])
def test_valid_metadata_never_calls_refresh(tmp_path, age):
    f = prepared(tmp_path)
    f.markets["ETH_5m"]["metadata_received_at"] = (NOW-timedelta(seconds=age)).isoformat()
    result = execute(f, market_refresh=lambda *a: pytest.fail("unnecessary refresh"))
    assert [x["cost_usd"] for x in result["fills"]] == [150, 70]


def test_no_callback_keeps_missing_contract_blocked(tmp_path):
    f = prepared(tmp_path)
    f.markets.pop("ETH_5m")
    result = execute(f)
    assert result["fills"] == [] and result["blocked"] == ["market is missing"]
    assert result["stream_results"]["BTC_15m"]["status"] == "UNREACHED"


def test_no_signal_not_due_disabled_and_exhausted_tiers_are_not_refreshed(tmp_path):
    f = prepared(tmp_path, BOUNDARY+timedelta(minutes=5))
    f.decisions["ETH_5m"] = no_signal("ETH_5m", f.boundary)
    f.decisions["BTC_5m"] = no_signal("BTC_5m", f.boundary)
    f.markets = {}
    result = execute(f, market_refresh=lambda *a: pytest.fail("skip must not refresh"))
    assert result["status"] == "PROCESSED" and not result["fills"]
    assert result["stream_results"]["BTC_15m"]["status"] == "NOT_DUE"
    assert all(result["stream_results"][s]["status"] == "DISABLED_BY_STAGE"
               for s in set(f.executor.policy["order"])-set(ACTIVE))


@pytest.mark.parametrize("bad", ["stream", "underlying", "outcome", "hash", "validation", "missing_id",
    "future_decision", "future_feature", "expired", "wrong_start", "wrong_end", "before_account"])
def test_invalid_signal_never_triggers_market_lookup(tmp_path, bad):
    f = prepared(tmp_path)
    f.markets = {}
    signal = f.decisions["ETH_5m"]["signal"]
    if bad == "stream": signal["stream"] = "BTC_5m"
    elif bad == "underlying": signal["underlying"] = "BTC"
    elif bad == "outcome": signal["outcome"] = "SIDEWAYS"
    elif bad == "hash": signal["model_sha256"] = "invalid"
    elif bad == "validation": signal["validation_status"] = "UNKNOWN"
    elif bad == "missing_id": signal.pop("signal_id")
    elif bad == "future_decision": signal["decision_at"] = (NOW+timedelta(seconds=1)).isoformat()
    elif bad == "future_feature": signal["feature_available_at"] = NOW.isoformat()
    elif bad == "expired": signal["valid_until"] = NOW.isoformat()
    elif bad == "wrong_start": signal["window_start"] = (BOUNDARY-timedelta(minutes=5)).isoformat()
    elif bad == "wrong_end": signal["window_end"] = (BOUNDARY+timedelta(minutes=15)).isoformat()
    else: signal["decision_at"] = (BOUNDARY-timedelta(seconds=2)).isoformat()
    result = execute(f, market_refresh=lambda *a: pytest.fail("invalid signal caused I/O"))
    assert result["status"] == "BLOCKED" and not result["fills"]
    assert f.account.overview()["order_count"] == 0
    assert result["stream_results"]["BTC_15m"]["status"] == "UNREACHED"


@pytest.mark.parametrize("kind", ["cash", "global", "underlying"])
def test_existing_risk_exhaustion_prevents_refresh(tmp_path, monkeypatch, kind):
    f = prepared(tmp_path)
    summary = f.account.overview(now=NOW)
    if kind == "cash": summary["cash"] = 0
    elif kind == "global": summary["locked_cost"] = 1000
    else: summary["positions"] = [{"underlying": "ETH", "cost_usd": 500, "fee_usd": 0}]
    monkeypatch.setattr(f.account, "overview", lambda **kwargs: summary)
    f.markets = {}
    result = execute(f, market_refresh=lambda *a: pytest.fail("no risk budget"))
    assert result["blocked"] == ["persistent budget or existing position limit exhausted"]


@pytest.mark.parametrize("bad", ["none", "wrong_start", "wrong_end", "wrong_asset", "wrong_duration",
    "unverified", "closed", "stale", "future", "unknown_fee", "wrong_token"])
def test_refresh_result_passes_original_market_fee_and_book_guards(tmp_path, bad):
    f = prepared(tmp_path)
    candidate = f.markets.pop("ETH_5m")
    if bad == "none": candidate = None
    elif bad == "wrong_start": candidate["window_start"] = (BOUNDARY+timedelta(minutes=5)).isoformat()
    elif bad == "wrong_end": candidate["window_end"] = (BOUNDARY+timedelta(minutes=15)).isoformat()
    elif bad == "wrong_asset": candidate["asset"] = "BTC"
    elif bad == "wrong_duration": candidate["duration_seconds"] = 900
    elif bad == "unverified": candidate["status"] = "UNKNOWN"
    elif bad == "closed": candidate["closed"] = True
    elif bad == "stale": candidate["metadata_received_at"] = (NOW-timedelta(seconds=91)).isoformat()
    elif bad == "future": candidate["metadata_received_at"] = (NOW+timedelta(seconds=1)).isoformat()
    elif bad == "unknown_fee": candidate["fee"] = {"status": "UNKNOWN"}
    else: candidate["books"]["UP"]["asset_id"] = "999999"
    calls = []
    def refresh(*args):
        calls.append(args[0])
        return candidate
    result = execute(f, market_refresh=refresh)
    assert calls == ["ETH_5m"] and not result["fills"]
    assert result["status"] == "BLOCKED"
    assert result["stream_results"]["BTC_15m"]["status"] == "UNREACHED"
    assert f.account.overview()["cash"] == 10000


def test_stale_existing_contract_cannot_be_retargeted_by_callback(tmp_path):
    f = prepared(tmp_path)
    f.markets["ETH_5m"]["metadata_received_at"] = (NOW-timedelta(seconds=91)).isoformat()
    def refresh(stream, boundary, existing):
        existing["metadata_received_at"] = NOW.isoformat()
        existing["market_id"] = "999999"
        return existing
    result = execute(f, market_refresh=refresh)
    assert result["blocked"] == ["refreshed market contract identity mismatch"]
    assert not result["fills"] and f.markets["ETH_5m"]["market_id"] != "999999"


def test_refresh_exception_blocks_lower_rank_without_retry(tmp_path):
    f = prepared(tmp_path)
    f.markets = {}
    calls = []
    def refresh(stream, boundary, existing):
        calls.append(stream)
        raise RuntimeError("synthetic transport failure")
    result = execute(f, market_refresh=refresh)
    assert calls == ["ETH_5m"] and result["blocked"] == ["RuntimeError"]
    assert not result["fills"] and result["stream_results"]["BTC_15m"]["status"] == "UNREACHED"


@pytest.mark.parametrize("seconds", [20, 21])
def test_refresh_finishing_at_or_after_deadline_never_fetches_book(tmp_path, seconds):
    f = prepared(tmp_path)
    fresh = f.markets.pop("ETH_5m")
    current = [NOW]
    def refresh(*args):
        current[0] = BOUNDARY+timedelta(seconds=seconds)
        return fresh
    result = execute(f, now=lambda: current[0], market_refresh=refresh,
        book_refresh=lambda *a: pytest.fail("book after deadline"))
    assert result["blocked"] == ["decision cohort expired after market refresh"]
    assert not result["fills"] and f.account.overview()["cash"] == 10000


def test_delay_before_callback_rechecks_clock_before_any_refresh(tmp_path, monkeypatch):
    f = prepared(tmp_path)
    f.markets = {}
    current = [NOW]
    original = f.account.overview
    def delayed_overview(**kwargs):
        result = original(**kwargs)
        current[0] = BOUNDARY+timedelta(seconds=20)
        return result
    monkeypatch.setattr(f.account, "overview", delayed_overview)
    result = execute(f, now=lambda: current[0], market_refresh=lambda *a: pytest.fail("refresh after deadline"))
    assert result["blocked"] == ["decision cohort expired before market refresh"]
    assert not result["fills"]


def test_refresh_cannot_proceed_with_invalidated_clock(tmp_path):
    f = prepared(tmp_path)
    fresh = f.markets.pop("ETH_5m")
    calibration = clock()
    def refresh(*args):
        calibration["status"] = "UNCALIBRATED"
        return fresh
    result = execute(f, clock=calibration, market_refresh=refresh,
        book_refresh=lambda *a: pytest.fail("invalid clock caused book request"))
    assert result["blocked"] == ["CLOCK_NOT_CALIBRATED"] and not result["fills"]


def test_full_signal_validation_runs_again_after_callback_time_advances(tmp_path):
    f = prepared(tmp_path)
    fresh = f.markets.pop("ETH_5m")
    current = [NOW]
    def refresh(*args):
        current[0] = BOUNDARY+timedelta(seconds=16)
        return fresh
    result = execute(f, now=lambda: current[0], market_refresh=refresh,
        book_refresh=lambda *a: pytest.fail("expired signal caused book request"))
    assert result["blocked"] == ["signal expired or expiry invalid"]
    assert not result["fills"]


def test_refreshed_market_conflict_with_signal_identifier_still_blocks(tmp_path):
    f = prepared(tmp_path)
    fresh = f.markets.pop("ETH_5m")
    f.decisions["ETH_5m"]["signal"]["market_id"] = "another-contract"
    result = execute(f, market_refresh=lambda *a: fresh)
    assert result["blocked"] == ["signal market conflicts with official contract"]


def test_existing_market_signal_conflict_blocks_before_refresh(tmp_path):
    f = prepared(tmp_path)
    f.markets["ETH_5m"]["metadata_received_at"] = (NOW-timedelta(seconds=91)).isoformat()
    f.decisions["ETH_5m"]["signal"]["market_id"] = "another-contract"
    result = execute(f, market_refresh=lambda *a: pytest.fail("conflicting signal caused refresh"))
    assert result["blocked"] == ["signal market conflicts with official contract"] and not result["fills"]


def test_price_limit_failure_after_refresh_skips_with_full_evidence(tmp_path):
    f = prepared(tmp_path)
    fresh = f.markets.pop("ETH_5m")
    fresh["books"]["UP"]["asks"] = [{"price": .6, "size": 1000}]
    result = execute(f, market_refresh=lambda *a: fresh)
    assert result["stream_results"]["ETH_5m"]["status"] == "PRICE_LIMIT_NOT_MET"
    assert [x["underlying"] for x in result["fills"]] == ["BTC", "BTC"]
    assert result["cohort"]["streams"][0]["receipt"]["fee_quote"]["market_id"] == fresh["market_id"]


def test_repeat_completed_cohort_does_not_refresh_or_refill(tmp_path):
    f = prepared(tmp_path)
    first = execute(f)
    assert len(first["fills"]) == 2
    f.markets = {}
    again = execute(f, market_refresh=lambda *a: pytest.fail("completed or exhausted rank refreshed"))
    assert again["fills"] == [] and f.account.overview()["order_count"] == 2
