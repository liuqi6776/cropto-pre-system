from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import pytest

from tools.argon4_forward import (
    account,
    execution,
    model,
    model_v14,
    online,
    revisions,
    risk_view,
    runtime_profile,
)
from test_argon4_forward_revisions import (
    runtime,
    no_network,
    _add_receipt,
    _row,
    NOW,
    OLD,
    NEW,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_module1_spot_endpoints_and_dynamic_page_limits(tmp_path):
    assert online.ENDPOINTS["spot_clock"][0] == "https://data-api.binance.vision"
    assert online.ENDPOINTS["spot"][0] == "https://data-api.binance.vision"

    calls = []
    cutoff = pd.Timestamp("2026-11-07T12:00:05Z")

    class FakeClient:
        clock_evidence_sha256 = "a" * 64

        def get(self, kind, params=None):
            calls.append((kind, dict(params or {})))
            url = online.ENDPOINTS[kind][0] + online.ENDPOINTS[kind][1]
            if kind in ("clock", "spot_clock"):
                return {"serverTime": int(cutoff.timestamp() * 1000)}, {
                    "requested_at": cutoff.isoformat(),
                    "received_at": cutoff.isoformat(),
                    "url": url,
                    "sha256": _sha(kind),
                    "time_basis": "CALIBRATED_UTC_UPPER_BOUND",
                    "clock_evidence_sha256": "a" * 64,
                }
            if kind == "funding":
                row_ms = int((cutoff - pd.Timedelta(hours=4)).timestamp() * 1000)
                return [{"fundingTime": row_ms, "fundingRate": "0.0001"}], {
                    "requested_at": cutoff.isoformat(),
                    "received_at": cutoff.isoformat(),
                    "url": url,
                    "sha256": _sha("funding"),
                    "time_basis": "CALIBRATED_UTC_UPPER_BOUND",
                    "clock_evidence_sha256": "a" * 64,
                }
            step = pd.Timedelta(minutes=5) if params.get("interval") == "5m" else pd.Timedelta(hours=1)
            open_ts = cutoff.floor(step) - step
            open_ms = int(open_ts.timestamp() * 1000)
            close_ms = int((open_ts + step - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
            return [[
                open_ms, "100", "101", "99", "100.5", "10", close_ms, "1000", 10, "5", "500", "0"
            ]], {
                "requested_at": cutoff.isoformat(),
                "received_at": cutoff.isoformat(),
                "url": url,
                "sha256": _sha(f"{kind}:{params}"),
                "time_basis": "CALIBRATED_UTC_UPPER_BOUND",
                "clock_evidence_sha256": "a" * 64,
            }

    inp = online.OnlineInputs(
        tmp_path,
        symbols=("BTCUSDT",),
        client=FakeClient(),
        clock=lambda: cutoff,
        time_basis="CALIBRATED_UTC_UPPER_BOUND",
        clock_evidence_sha256="a" * 64,
    )
    try:
        step = pd.Timedelta(minutes=5)
        dates = pd.date_range(cutoff.floor(step) - 4321 * step, cutoff.floor(step) - 2 * step, freq="5min", tz="UTC")
        existing = pd.DataFrame({
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1000.0,
            "count": 10.0,
            "taker_buy_base_volume": 5.0,
            "taker_buy_quote_volume": 500.0,
            "first_received_at": cutoff - pd.Timedelta(minutes=5),
            "receipt_sha256": "b" * 64,
            "receipt_time_basis": "CALIBRATED_UTC_UPPER_BOUND",
            "clock_evidence_sha256": "a" * 64,
        })
        path_5m = inp._path("futures", "BTCUSDT", "5m")
        path_5m.parent.mkdir(parents=True, exist_ok=True)
        existing.to_parquet(path_5m, index=False)

        inp._active_generation = 1
        ctx = (1, None, "2026-11-07T12:00:00+00:00", inp.clock, inp.time_basis, inp.clock_evidence_sha256)
        inp._run_job(ctx, inp._candles, ("futures", "BTCUSDT", "5m", cutoff))
        futures_calls = [p for k, p in calls if k == "futures" and p.get("interval") == "5m"]
        assert len(futures_calls) == 1
        assert futures_calls[0]["limit"] == 10  # Dynamic small limit (Weight 1), not 1000!

        # Same-window 5m retry short-circuit: now latest closed bar has first_received_at >= cutoff.floor(5m)
        calls.clear()
        inp._run_job(ctx, inp._candles, ("futures", "BTCUSDT", "5m", cutoff))
        assert [p for k, p in calls if k == "futures" and p.get("interval") == "5m"] == []
    finally:
        inp.close()


def test_module1_cached_clock_preserved_across_refresh(tmp_path):
    cutoff = pd.Timestamp("2026-11-07T12:00:05Z")
    clock_calls = []

    class FakeClient:
        clock_evidence_sha256 = "a" * 64

        def get(self, kind, params=None):
            url = online.ENDPOINTS[kind][0] + online.ENDPOINTS[kind][1]
            if kind in ("clock", "spot_clock"):
                clock_calls.append(kind)
                return {"serverTime": int(cutoff.timestamp() * 1000)}, {
                    "requested_at": cutoff.isoformat(),
                    "received_at": cutoff.isoformat(),
                    "url": url,
                    "sha256": _sha(kind),
                    "time_basis": "CALIBRATED_UTC_UPPER_BOUND",
                    "clock_evidence_sha256": "a" * 64,
                }
            return [], {
                "requested_at": cutoff.isoformat(),
                "received_at": cutoff.isoformat(),
                "url": url,
                "sha256": _sha(f"{kind}:{params}"),
                "time_basis": "CALIBRATED_UTC_UPPER_BOUND",
                "clock_evidence_sha256": "a" * 64,
            }

    inp = online.OnlineInputs(
        tmp_path,
        symbols=("BTCUSDT",),
        client=FakeClient(),
        clock=lambda: cutoff,
        time_basis="CALIBRATED_UTC_UPPER_BOUND",
        clock_evidence_sha256="a" * 64,
    )
    try:
        inp.refresh(now=cutoff, cohort_id="2026-11-07T12:00:00+00:00")
        assert inp._cached_clock is not None
        assert inp._cached_clock.get("status") == "VERIFIED"
        assert len(clock_calls) == 1

        # Second refresh within 30s reuses _cached_clock without hitting clock endpoints again
        inp.refresh(now=cutoff + pd.Timedelta(seconds=1), cohort_id="2026-11-07T12:00:00+00:00")
        assert len(clock_calls) == 1
    finally:
        inp.close()


def test_module2_fastpath_v14_supports_1h_futures_and_1h_mark(runtime):
    policy = {
        "engineering": dict(runtime_profile.V12_ENGINEERING),
        "input_revision_policy": {
            "aged_futures_fastpath": {
                "profile": "argon4-aged-revision-v14",
                "enabled": True,
                "minimum_bar_age_seconds": 300,
                "minimum_receipt_spacing_seconds": 2.0,
                "same_cohort_before_prediction_only": True,
            }
        },
    }
    prof = runtime_profile.revision_fastpath_profile(policy)
    assert prof["enabled"] is True
    assert prof["profile"] == runtime_profile.V14_AGED_REVISION_PROFILE

    open_1h = pd.Timestamp("2026-11-06T00:00:00Z")
    old_1h = deepcopy(OLD)
    old_1h[0] = int(open_1h.timestamp() * 1000)
    old_1h[6] = int((open_1h + pd.Timedelta(hours=1)).timestamp() * 1000) - 1
    prev_1h = deepcopy(old_1h)
    prev_1h[0] -= 3_600_000
    prev_1h[6] -= 3_600_000
    new_1h = deepcopy(NEW)
    new_1h[0] = old_1h[0]
    new_1h[6] = old_1h[6]

    orig_digest = _add_receipt(
        runtime,
        "2026-11-06T01:30:03Z",
        old_1h,
        symbol="BTCUSDT",
        include_new_bar=False,
        params_extra={"interval": "1h"},
    )
    row_prev = _row(prev_1h)
    row_old = _row(old_1h)
    row_old["receipt_sha256"] = orig_digest
    row_old["clock_evidence_sha256"] = "c" * 64
    path_1h = runtime / "features/raw_data/auxiliary/BTCUSDT_1h_futures.parquet"
    pd.DataFrame([row_prev, row_old]).to_parquet(path_1h, index=False)

    _add_receipt(runtime, NOW + pd.Timedelta(seconds=4), new_1h, include_new_bar=False, params_extra={"interval": "1h"})
    _add_receipt(runtime, NOW + pd.Timedelta(seconds=6.4), new_1h, include_new_bar=False, params_extra={"interval": "1h"})
    fast_ctx = {
        "profile": revisions.FAST_HOURLY_PROFILE,
        "cohort": NOW.isoformat(),
        "deadline": (NOW + pd.Timedelta(seconds=14)).isoformat(),
        "minimum_spacing_seconds": 2.0,
        "clock_evidence_sha256": "c" * 64,
    }
    guard = lambda: {
        "eligible": True,
        "clock_evidence_sha256": "c" * 64,
        "policy_sha256": "d" * 64,
        "prediction_count": 0,
        "cohort_order_count": 0,
    }
    result = revisions.reconcile_confirmed_price_revisions(
        runtime,
        [{"source": "futures:BTCUSDT:1h", "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"}],
        now=lambda: NOW + pd.Timedelta(seconds=7),
        cutover_now=lambda: NOW + pd.Timedelta(seconds=8),
        fast_context=fast_ctx,
        cohort_guard=guard,
    )
    assert result["status"] == "COMMITTED"
    assert result["same_cohort_eligible"] is True
    assert result["same_cohort"] == NOW.isoformat()


def test_module2_clock_borderline_retry_relaxes_to_060s():
    stamp = datetime(2026, 11, 7, 12, 0, 5, tzinfo=timezone.utc)
    valid_borderline_clock = {
        "status": "CALIBRATED",
        "source": "POLYMARKET_CLOB_SERVER_TIME",
        "uncertainty_seconds": 0.55,
        "server_minus_local_seconds": 0.0,
        "observed_at": stamp.isoformat(),
        "calibrated_at_local": stamp.isoformat(),
        "receipt_sha256": "e" * 64,
        "calibrated_monotonic": time.monotonic(),
        "borderline_retry": {"status": "COMPLETED", "samples_added": 2},
        "receipts": [{"sha256": str(i) * 64} for i in range(8)],
    }
    execution.PaperExecutor._clock(valid_borderline_clock, stamp)
    bounds, err = risk_view._bounds(valid_borderline_clock, stamp)
    assert err is None
    assert bounds is not None


def test_module3_and_4_best_of_8_and_partial_fill_cascade_and_0520_cap(tmp_path):
    order_8 = [
        "BTC_15m",
        "ETH_5m",
        "SOL_15m",
        "SOL_5m",
        "ETH_15m",
        "BNB_15m",
        "BTC_5m",
        "XRP_15m",
    ]
    policy = {
        "status": "CONFIRMED",
        "version": "argon4-local311-independent65-20261107-v14",
        "ranking_source": "frozen v14 test policy",
        "order": order_8,
        "initial_capital_usd": 10000,
        "single_stream_equity_fraction": 0.015,
        "tick_equity_fraction": 0.022,
        "max_tick_budget_usd": 1200,
        "max_order_usd": 200,
        "max_open_risk_usd": 1000,
        "max_underlying_open_risk_usd": 500,
        "max_buy_price": 0.515,
        "daily_loss_fraction": 0.055,
        "paper_trading_active": True,
        "live_enabled": False,
        "waterfall_semantics": "ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1",
        "execution": {
            "multi_look_ioc": True,
            "max_price_rechecks": 3,
            "price_recheck_interval_seconds": 2.5,
            "minimum_recheck_remaining_seconds": 3.0,
            "last_recheck_start_after_boundary_seconds": 18.5,
            "partial_fill_remainder_cascade": True,
            "high_confidence_max_buy_price": 0.520,
        },
    }
    boundary = datetime(2026, 11, 7, 12, 0, 0, tzinfo=timezone.utc)
    now_dt = boundary + timedelta(seconds=6)
    acc = account.Account(tmp_path / "paper.sqlite3", started_at=boundary - timedelta(seconds=10))
    clock = {
        "status": "CALIBRATED",
        "source": "POLYMARKET_CLOB_SERVER_TIME",
        "uncertainty_seconds": 0.1,
        "observed_at": now_dt.isoformat(),
    }

    streams_obs = {}
    books = {}
    decisions = {}
    for idx, s in enumerate(order_8, start=1):
        asset, duration = account.ALL_STREAMS[s]
        start = datetime.fromtimestamp(int(boundary.timestamp()) // duration * duration, timezone.utc)
        end = start + timedelta(seconds=duration)
        m = {
            "status": "VERIFIED_MARKET",
            "market_id": str(1000 + idx),
            "condition_id": "0x" + f"{idx:02x}" * 32,
            "protocol": "v2",
            "asset": asset,
            "duration_seconds": duration,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "active": True,
            "closed": False,
            "accepting_orders": True,
            "outcome_ids": {"UP": str(idx * 10 + 1), "DOWN": str(idx * 10 + 2)},
            "metadata_received_at": now_dt.isoformat(),
            "metadata_sha256": _sha(s),
            "resolution": {"source": f"https://data.chain.link/streams/{asset.lower()}-usd-twap-60s-streams"},
            "fee": {"status": "KNOWN_DISABLED", "fees_enabled": False},
            "books": {},
        }
        streams_obs[s] = m
        if s == "BTC_15m":
            books[m["market_id"]] = {
                "status": "OBSERVED",
                "asset_id": m["outcome_ids"]["UP"],
                "condition_id": m["condition_id"],
                "source_at": now_dt.isoformat(),
                "received_at": now_dt.isoformat(),
                "asks": [{"price": 0.50, "size": 100.0}],  # $50 partial fill (approved is $150)
                "bids": [{"price": 0.49, "size": 500.0}],
            }
            decisions[s] = {
                "status": "SIGNAL",
                "limit_price": 0.515,
                "high_confidence": False,
                "signal": {
                    "signal_id": f"model:{s}:{int(boundary.timestamp())}",
                    "stream": s,
                    "underlying": asset,
                    "outcome": "UP",
                    "model_id": "frozen-test-model",
                    "model_sha256": "b" * 64,
                    "validation_status": "INPUTS_VALIDATED",
                    "feature_available_at": boundary.isoformat(),
                    "decision_at": (boundary + timedelta(milliseconds=500)).isoformat(),
                    "valid_until": (boundary + timedelta(seconds=15)).isoformat(),
                    "window_start": start.isoformat(),
                    "window_end": end.isoformat(),
                },
            }
        elif s == "XRP_15m":
            books[m["market_id"]] = {
                "status": "OBSERVED",
                "asset_id": m["outcome_ids"]["UP"],
                "condition_id": m["condition_id"],
                "source_at": now_dt.isoformat(),
                "received_at": now_dt.isoformat(),
                "asks": [{"price": 0.518, "size": 400.0}],  # Eligible under 0.520 high-confidence cap!
                "bids": [{"price": 0.51, "size": 500.0}],
            }
            decisions[s] = {
                "status": "SIGNAL",
                "limit_price": 0.520,
                "high_confidence": True,
                "signal": {
                    "signal_id": f"model:{s}:{int(boundary.timestamp())}",
                    "stream": s,
                    "underlying": asset,
                    "outcome": "UP",
                    "model_id": "frozen-test-model",
                    "model_sha256": "b" * 64,
                    "validation_status": "INPUTS_VALIDATED",
                    "feature_available_at": boundary.isoformat(),
                    "decision_at": (boundary + timedelta(milliseconds=500)).isoformat(),
                    "valid_until": (boundary + timedelta(seconds=15)).isoformat(),
                    "window_start": start.isoformat(),
                    "window_end": end.isoformat(),
                },
            }
        else:
            decisions[s] = {
                "status": "NO_SIGNAL",
                "prediction": {
                    "stream": s,
                    "selected": False,
                    "z_v4": 0.2,
                    "threshold_frozen": 1.4,
                    "side": "UP",
                    "model_version": "frozen-test-model",
                    "model_sha256": "b" * 64,
                    "manifest_sha256": "c" * 64,
                    "feature_bar_open": (boundary - timedelta(minutes=5)).isoformat(),
                    "decision_time": (boundary + timedelta(milliseconds=500)).isoformat(),
                },
            }

    executor = execution.PaperExecutor(acc, policy)
    outcome = executor.process_cohort(
        boundary.isoformat(),
        decisions,
        streams_obs,
        now=lambda: now_dt,
        book_refresh=lambda market, side: books[market["market_id"]],
        clock=clock,
    )
    assert outcome["status"] == "PROCESSED", f"Unexpected blocked: {outcome.get('blocked')}"
    filled_underlyings = [f["underlying"] for f in outcome["fills"]]
    assert filled_underlyings == ["BTC", "XRP"]
    assert outcome["fills"][0]["cost_usd"] == pytest.approx(50.0)
    assert outcome["fills"][1]["cost_usd"] == pytest.approx(150.0)
    assert outcome["fills"][1]["allocation_authorization"]["max_buy_price"] == pytest.approx(0.52)
    assert float(outcome["fills"][1]["fills"][0]["price"]) == pytest.approx(0.518)

    # Also verify explicit PARTIAL state cascade when pre-existing consumed depth causes a PARTIAL fill on rank 1
    acc2 = account.Account(tmp_path / "paper2.sqlite3", started_at=boundary - timedelta(seconds=10))
    # Pre-consume 200 shares of BTC_15m book so 300 quoted shares ($150 capacity) only fills 100 shares ($50 PARTIAL)
    books_partial = deepcopy(books)
    books_partial["1001"]["asks"] = [{"price": 0.50, "size": 300.0}]
    adapted_btc, _, _ = execution.PaperExecutor._adapt(
        streams_obs["BTC_15m"], "UP", books_partial["1001"], now_dt, clock, allow_price_skip=True
    )
    btc_book_hash = hashlib.sha256(
        account._json({
            "market": adapted_btc["market_id"],
            "outcome": adapted_btc["outcome"],
            "source": adapted_btc["source"],
            "observed_at": adapted_btc["observed_at"],
            "asks": [["0.5", "300.0"]],
        }).encode()
    ).hexdigest()
    with acc2._connection() as con:
        con.execute("INSERT INTO consumed_book_depth VALUES(?,?,?)", (btc_book_hash, "0.5", "200"))

    executor2 = execution.PaperExecutor(acc2, policy)
    outcome2 = executor2.process_cohort(
        boundary.isoformat(),
        decisions,
        streams_obs,
        now=lambda: now_dt,
        book_refresh=lambda market, side: books_partial[market["market_id"]],
        clock=clock,
    )
    assert outcome2["status"] == "PROCESSED", f"Unexpected blocked: {outcome2.get('blocked')}"
    assert [f["underlying"] for f in outcome2["fills"]] == ["BTC", "XRP"]
    assert outcome2["fills"][0]["cohort_stream_state"] == "PARTIAL"
    assert outcome2["fills"][0]["cost_usd"] == pytest.approx(50.0)
    assert outcome2["fills"][1]["cohort_stream_state"] == "FILLED_BUDGET"
    assert outcome2["fills"][1]["cost_usd"] == pytest.approx(150.0)


def test_module5_production_policy_v14_and_all_8_bundles_loadable():
    policy_path = Path("config/argon4_forward_policy.json")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy["version"] == "argon4-local311-independent65-20261107-v14"
    assert policy["live_enabled"] is False
    assert policy["order"] == [
        "BTC_15m",
        "ETH_5m",
        "SOL_15m",
        "SOL_5m",
        "ETH_15m",
        "BNB_15m",
        "BTC_5m",
        "XRP_15m",
    ]
    assert policy["input_revision_policy"]["aged_futures_fastpath"]["profile"] == "argon4-aged-revision-v14"
    assert policy["execution"]["partial_fill_remainder_cascade"] is True
    assert policy["execution"]["high_confidence_max_buy_price"] == 0.52
    assert "capital_scaling_roadmap" in policy

    for stream in policy["order"]:
        expected_sha = policy["deployment_manifest_sha256"][stream]
        bundle = model_v14.load_model_bundle_v14(stream, expected_manifest_sha256=expected_sha)
        assert bundle["manifest_sha256"] == expected_sha
        assert stream in policy["stream_thresholds"]
        assert stream in policy["high_confidence_thresholds"]


def test_monitor_audit_logger_no_trade_reasons_and_stall_watchdog(tmp_path):
    from tools.argon4_forward.monitor_logger import MonitorAuditLogger

    logger = MonitorAuditLogger(tmp_path / "runtime", stall_threshold_seconds=0.05)
    policy = {"order": ["BTC_15m", "ETH_5m", "SOL_15m", "XRP_15m"]}
    decisions = {
        "BTC_15m": {
            "status": "NO_SIGNAL",
            "outcome": "UP",
            "prediction": {"z_v4": 0.42, "threshold_frozen": 0.72, "side": "UP"},
        },
        "ETH_5m": {
            "status": "SIGNAL",
            "outcome": "DOWN",
            "prediction": {"z_v4": -0.95, "threshold_frozen": 0.81, "side": "DOWN"},
        },
        "SOL_15m": {"status": "NOT_DUE"},
        "XRP_15m": {
            "status": "SIGNAL",
            "outcome": "UP",
            "prediction": {"z_v4": 0.88, "threshold_frozen": 0.74, "side": "UP"},
        },
    }
    outcome = {
        "status": "WAITING_PRICE_RECHECK",
        "fills": [],
        "blocked": [],
        "streams": {
            "BTC_15m": {"status": "NO_SIGNAL", "approved_usd": 0.0, "spent_usd": 0.0},
            "ETH_5m": {
                "status": "WAITING_PRICE_RECHECK",
                "best_ask": 0.54,
                "limit_price": 0.515,
                "attempt": 1,
                "next_retry_at": "2026-11-08T00:05:09+00:00",
                "approved_usd": 0.0,
                "spent_usd": 0.0,
            },
            "SOL_15m": {"status": "NOT_DUE", "approved_usd": 0.0, "spent_usd": 0.0},
        },
    }
    entry = logger.log_cohort_execution(
        "2026-11-08T00:05:00+00:00",
        policy,
        decisions,
        outcome,
        timing={"duration_seconds": 1.25},
    )
    assert entry["event"] == "COHORT_NO_TRADE_AUDIT"
    streams_audit = entry["details"]["streams"]
    assert "BELOW_THRESHOLD" in streams_audit["BTC_15m"]["no_trade_reason"]
    assert "WAITING_PRICE_RECHECK" in streams_audit["ETH_5m"]["no_trade_reason"]
    assert "NOT_DUE" in streams_audit["SOL_15m"]["no_trade_reason"]
    assert "DEFERRED_BY_HIGHER_PRIORITY_RECHECK" in streams_audit["XRP_15m"]["no_trade_reason"]

    # Test stall detection & recovery with live stack capture
    with logger.phase("pipeline", "refresh_decision_inputs", cohort="2026-11-08T00:05:00+00:00", slow_warn_seconds=10.0):
        time.sleep(0.07)
        stalls = logger.check_for_stalls()
        assert len(stalls) == 1
        assert stalls[0]["event"] == "THREAD_STALL_DETECTED"
        assert stalls[0]["details"]["phase"] == "refresh_decision_inputs"
        assert len(stalls[0]["details"]["stack_tail"]) > 0

    events = [e["event"] for e in logger.recent_events()]
    assert "THREAD_STALL_DETECTED" in events
    assert "STALL_RECOVERED" in events
    assert logger.text_log_path.is_file()
    assert logger.jsonl_log_path.is_file()

