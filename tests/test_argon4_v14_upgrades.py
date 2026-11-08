"""Tests for ARGON-4 v14r1 architecture upgrades and review corrections."""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import time

import pandas as pd
import pytest

from test_argon4_forward_revisions import NEW, NOW, _add_receipt, _error, no_network, runtime
from tools.argon4_forward import (
    account,
    allocation,
    live_preflight,
    model,
    model_v14,
    monitor_logger,
    online,
    revisions,
)


def _bar_row(ts_ms, close=100.0):
    return [
        ts_ms,
        str(close),
        str(close + 1.0),
        str(close - 1.0),
        str(close),
        "10.0",
        ts_ms + 300_000 - 1,
        "1000.0",
        20,
        "5.0",
        "500.0",
        "0",
    ]


class _FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._raw = json.dumps(payload).encode("utf-8")
        self.status_code = status_code
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size=65536):
        yield self._raw


def test_module1_dynamic_page_limit_and_same_cohort_5m_short_circuit(tmp_path, monkeypatch):
    """Module 1: incremental klines/funding use dynamic page_limit and short-circuit ready 5m symbols."""
    client = online.OnlineInputs(tmp_path)
    calls = []

    cutoff = pd.Timestamp("2026-11-07T12:00:00Z")
    prev_open = cutoff - pd.Timedelta(minutes=10)
    last_open = cutoff - pd.Timedelta(minutes=5)

    for sym in online.SYMBOLS:
        target = client._path("futures", sym, "5m")
        target.parent.mkdir(parents=True, exist_ok=True)
        # BTCUSDT, ETHUSDT, SOLUSDT already have the latest bar (last_open); BNBUSDT is 1 bar behind (prev_open)
        end_bar = prev_open if sym == "BNBUSDT" else last_open
        bars = pd.date_range(end_bar - pd.Timedelta(days=21, minutes=5), end_bar, freq="5min", tz="UTC").astype(
            "datetime64[ns, UTC]"
        )
        df = pd.DataFrame(
            {
                "date": bars,
                "open_time": bars,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "close_time": (bars + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)).astype(
                    "datetime64[ns, UTC]"
                ),
                "quote_volume": 1000.0,
                "count": 20.0,
                "taker_buy_base_volume": 5.0,
                "taker_buy_quote_volume": 500.0,
                "first_received_at": (bars + pd.Timedelta(minutes=5, seconds=1)).astype("datetime64[ns, UTC]"),
                "receipt_sha256": "a" * 64,
                "receipt_time_basis": "CALIBRATED_UTC_UPPER_BOUND",
                "clock_evidence_sha256": "b" * 64,
            }
        )
        df.to_parquet(target, index=False)

    def fake_get(kind, params=None):
        calls.append((kind, dict(params or {})))
        if kind == "futures":
            start_ms = params["startTime"]
            return [
                _bar_row(start_ms),
                _bar_row(online._ms(prev_open)),
                _bar_row(online._ms(last_open)),
            ], {
                "url": "https://fapi.binance.com/fapi/v1/klines",
                "received_at": (cutoff + pd.Timedelta(seconds=2)).isoformat(),
                "sha256": "c" * 64,
                "time_basis": "CALIBRATED_UTC_UPPER_BOUND",
                "clock_evidence_sha256": "d" * 64,
            }
        raise AssertionError(f"unexpected endpoint {kind}")

    monkeypatch.setattr(client, "_get", fake_get)
    monkeypatch.setattr(
        client,
        "_guard_generation",
        lambda: (1, None, cutoff.isoformat(), client.clock, client.time_basis, client.clock_evidence_sha256),
    )
    for sym in online.SYMBOLS:
        client._candles("futures", sym, "5m", cutoff)

    # Only BNBUSDT should trigger an HTTP request; the 3 ready symbols short-circuit!
    assert len(calls) == 1
    assert calls[0][0] == "futures"
    assert calls[0][1]["symbol"] == "BNBUSDT"
    assert calls[0][1]["limit"] == 10


def test_module1_cached_clock_preserved_across_formal_refresh(tmp_path, monkeypatch):
    """Module 1: _cached_clock remains valid for 30s even when refresh(auxiliary=False) runs."""
    cutoff = pd.Timestamp("2026-11-07T12:00:10Z")
    client = online.OnlineInputs(tmp_path, clock=lambda: cutoff)
    clock_calls = []

    def fake_check_clock():
        clock_calls.append(1)
        return {
            "status": "VERIFIED",
            "source": "https://fapi.binance.com/fapi/v1/time",
            "server_time": (cutoff - pd.Timedelta(seconds=1)).isoformat(),
            "received_at": (cutoff - pd.Timedelta(seconds=1)).isoformat(),
            "skew_seconds": 0.01,
            "round_trip_seconds": 0.05,
            "max_skew_seconds": 2.0,
            "fallback_errors": [],
        }

    monkeypatch.setattr(client, "_check_clock", fake_check_clock)
    monkeypatch.setattr(client, "_candles", lambda *a, **kw: "ok")
    monkeypatch.setattr(client, "_funding", lambda *a, **kw: "ok")
    monkeypatch.setattr(client, "_dvol", lambda *a, **kw: None)
    monkeypatch.setattr(client, "_basis", lambda *a, **kw: None)
    monkeypatch.setattr(client, "snapshot", lambda now=None: {"status": "READY", "clock": client._cached_clock})

    res1 = client.refresh(cutoff, cohort_id=cutoff.isoformat())
    assert res1["clock"]["status"] == "VERIFIED"
    assert client._cached_clock is not None
    assert len(clock_calls) == 1

    # Second refresh within 30s reuses _cached_clock without hitting /fapi/v1/time again
    res2 = client.refresh(cutoff, cohort_id=cutoff.isoformat())
    assert res2["clock"]["status"] == "VERIFIED"
    assert len(clock_calls) == 1


def test_correction4_weight_soft_backoff_auto_recovers(tmp_path, monkeypatch):
    """Correction 4: X-MBX-USED-WEIGHT-1M >= 1600 throttles to 1.0s and auto-recovers below 1200 or after 30s."""
    http = online.PublicInputsHTTP(tmp_path / "receipts.sqlite3", min_spacing=0.2)
    session = http._session_for_request()
    headers_queue = [
        {"X-MBX-USED-WEIGHT-1M": "1650"},
        {"X-MBX-USED-WEIGHT-1M": "800"},
        {"X-MBX-USED-WEIGHT-1M": "1700"},
    ]

    def fake_session_get(url, **kwargs):
        hdrs = headers_queue.pop(0) if headers_queue else {}
        return _FakeResponse({"serverTime": 1762516800000}, status_code=200, headers=hdrs)

    monkeypatch.setattr(session, "get", fake_session_get)
    monkeypatch.setattr(online.time, "sleep", lambda s: None)

    http.get("clock")
    assert http.min_spacing == 1.0
    assert http._weight_throttle_until > 0.0

    # Recovery via weight dropping below 1200
    http.get("clock")
    assert http.min_spacing == 0.2
    assert http._weight_throttle_until == 0.0

    # Throttle again, then expire the 30s window before next request
    http.get("clock")
    assert http.min_spacing == 1.0
    http._weight_throttle_until = time.monotonic() - 1.0
    http.get("clock")
    assert http.min_spacing == 0.2
    assert http._weight_throttle_until == 0.0


def test_correction1_model_v14_preserves_monkeypatch_signature_and_skips_missing_artifacts(monkeypatch):
    """Correction 1: model_v14 omits phase kwarg on default deployment_q4 and skips local bundle load if absent."""
    called = []

    def fake_load(stream, *, expected_manifest_sha256=None):
        called.append((stream, expected_manifest_sha256))
        return {"stream": stream, "sha": expected_manifest_sha256}

    monkeypatch.setattr(model, "load_model_bundle", fake_load)
    res = model_v14.load_model_bundle_v14("BTC_15m", expected_manifest_sha256="abc123")
    assert res == {"stream": "BTC_15m", "sha": "abc123"}
    assert called == [("BTC_15m", "abc123")]


def test_module5_production_policy_v14_and_all_8_bundles_loadable():
    """Correction 1B: check (model.MODEL_ROOT / 'deployment_q4').is_dir() and skip if absent in source-only checkout."""
    q4_dir = model.MODEL_ROOT / "deployment_q4"
    if not q4_dir.is_dir():
        pytest.skip("local model artifacts not present in source-only checkout")
    for stream in model_v14.ALL_STREAMS:
        manifest_path = q4_dir / f"{stream}.manifest.json"
        if not manifest_path.is_file():
            pytest.skip(f"local model manifest {stream} not present in source-only checkout")
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_data.get("sklearn_version") != model.sklearn.__version__:
            pytest.skip("local model manifest sklearn_version does not match current interpreter")
        bundle = model_v14.load_model_bundle_v14(stream)
        assert bundle["manifest"]["stream"] == stream


def test_correction2_dynamic_policy_risk_limits_for_1k_trial_and_10k_default(tmp_path):
    """Correction 2: _policy_risk_limits drives Account, PaperExecutor, plan_waterfall, and live_preflight."""
    policy_path = Path("config/argon4_forward_policy.json")
    default_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    default_limits = account._policy_risk_limits(default_policy)
    assert default_limits["max_order_usd"] == Decimal("200")
    assert default_limits["max_open_risk_usd"] == Decimal("1000")
    assert default_limits["max_underlying_open_risk_usd"] == Decimal("500")
    assert default_limits["max_tick_budget_usd"] == Decimal("1200")

    trial_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    trial_policy["risk"] = {
        "single_stream_equity_fraction": 0.015,
        "tick_equity_fraction": 0.022,
        "max_tick_budget_usd": 120,
        "max_order_usd": 25,
        "max_open_risk_usd": 100,
        "max_underlying_open_risk_usd": 50,
        "daily_loss_fraction": 0.055,
    }
    trial_limits = account._policy_risk_limits(trial_policy)
    assert trial_limits["max_order_usd"] == Decimal("25")
    assert trial_limits["max_open_risk_usd"] == Decimal("100")
    assert trial_limits["max_underlying_open_risk_usd"] == Decimal("50")
    assert trial_limits["max_tick_budget_usd"] == Decimal("120")

    # Verify Account uses trial_policy limits
    ledger = account.Account(tmp_path / "account.sqlite3", initial_capital=1000)
    overview = ledger.overview(policy=trial_policy)
    assert overview["risk_limits"]["max_order_usd"] == 25
    assert overview["risk_limits"]["max_total_open_cost_usd"] == 100
    assert overview["risk_limits"]["max_underlying_open_cost_usd"] == 50

    # Verify allocation.plan_waterfall enforces trial limits
    plan = allocation.plan_waterfall(
        trial_policy,
        [],
        1000.0,
        1000.0,
        0.0,
        "2026-11-08T00:00:05Z",
    )
    assert plan["limits"]["order_hard_cap_usd"] == 25.0
    assert plan["limits"]["tick_hard_cap_usd"] == 120.0
    assert plan["limits"]["open_risk_hard_cap_usd"] == 100.0

    prep = live_preflight.preparation_status(
        trial_policy,
        {
            "live_enabled": False,
            "mode": "disabled",
            "signer_connected": False,
            "order_transport_connected": False,
            "max_order_usd": 25.0,
            "max_open_risk_usd": 100.0,
        },
    )
    assert prep["risk_limits"]["max_order_usd"] == 25
    assert prep["risk_limits"]["max_open_risk_usd"] == 100
    assert prep["risk_limits_match"] is True


def test_correction3_revision_archive_hash_cache_and_observations_hardlink(runtime, monkeypatch):
    """Correction 3: ensure_view_committed caches verified archive hashes and _backup_sqlite hardlinks observations.sqlite3."""
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    res1 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert res1["status"] == "COMMITTED"
    archive1 = Path(res1["archive_path"])

    # First ensure_view_committed populates cache; second call hits cache without calling _sha
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"
    sha_calls = []
    orig_sha = revisions._sha

    def counting_sha(path):
        sha_calls.append(str(path))
        return orig_sha(path)

    monkeypatch.setattr(revisions, "_sha", counting_sha)
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"
    assert sha_calls == []
    monkeypatch.setattr(revisions, "_sha", orig_sha)

    # Trigger a second revision and verify observations.sqlite3 is hardlinked from archive1
    changed = list(NEW)
    changed[4], changed[8] = "104.5", 29
    _add_receipt(runtime, "2026-11-06T01:50:05Z", changed)
    _add_receipt(runtime, "2026-11-06T01:55:05Z", changed)
    res2 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW + pd.Timedelta(minutes=15))
    assert res2["status"] == "COMMITTED"
    archive2 = Path(res2["archive_path"])

    obs1 = (archive1 / "observations.sqlite3").stat()
    obs2 = (archive2 / "observations.sqlite3").stat()
    if obs1.st_nlink >= 2:
        assert obs1.st_ino == obs2.st_ino
    else:
        assert revisions._sha(archive1 / "observations.sqlite3") == revisions._sha(archive2 / "observations.sqlite3")


def test_correction5_and_6_production_policy_preserves_verified_v13r2_parameters():
    """Correction 5 & 6: production policy keeps verified v13r2 execution parameters and fastpath profile."""
    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    assert policy["max_buy_price"] == 0.515
    assert "high_confidence_max_buy_price" not in policy
    assert "high_confidence_thresholds" not in policy
    assert "partial_fill_remainder_cascade" not in policy
    assert "stream_thresholds" not in policy
    assert policy["order"] == [
        "BTC_15m",
        "ETH_5m",
        "SOL_15m",
        "SOL_5m",
        "BTC_5m",
        "ETH_15m",
        "BNB_15m",
    ]
    assert policy["execution_stage"]["enabled_streams"] == [
        "BTC_15m",
        "ETH_5m",
        "SOL_15m",
        "SOL_5m",
        "BTC_5m",
    ]
    from tools.argon4_forward import runtime_profile
    assert runtime_profile.revision_fastpath_profile(policy)["profile"] == "argon4-aged-revision-v13"


def test_monitor_audit_logger_records_no_trade_and_stall_events(tmp_path):
    """MonitorAuditLogger writes structured JSONL and text logs for no-trade reasons and stalls."""
    logger = monitor_logger.MonitorAuditLogger(tmp_path, stall_threshold_seconds=0.01)
    cohort = "2026-11-08T00:00:00+00:00"
    policy = {"order": ["BTC_15m"]}
    decisions = {
        "BTC_15m": {
            "status": "NO_SIGNAL",
            "outcome": "Up",
            "prediction": {"z_v4": 0.12, "threshold_frozen": 0.45, "side": "Up"},
        }
    }
    outcome = {"status": "PROCESSED", "streams": {}, "fills": [], "blocked": []}
    logger.log_cohort_execution(cohort, policy, decisions, outcome)
    logger.log_cohort_blocked(cohort, "PUBLIC_RATE_LIMIT_BACKOFF", "429 backoff")
    recent = logger.recent_events(limit=10)
    assert len(recent) == 2
    assert recent[0]["event"] == "COHORT_NO_TRADE_AUDIT"
    assert "BELOW_THRESHOLD" in recent[0]["details"]["streams"]["BTC_15m"]["no_trade_reason"]
    assert recent[1]["event"] == "COHORT_NO_TRADE_BLOCKED"


def test_v14r2_patch1_observations_delta_and_disk_budget_cap(runtime):
    """Patch 1: second revision writes observations_delta.sqlite3 for new rows and caps disk budget at 16 MB."""
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    res1 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert res1["status"] == "COMMITTED"

    # Insert a new observation row after the first revision
    with sqlite3.connect(runtime / "observations.sqlite3") as conn:
        conn.execute("INSERT INTO decisions(content) VALUES(?)", ("new decision between revisions",))

    # Check disk budget caps observations.sqlite3 at <= 16 MB when prior hardlink exists
    budget = revisions._disk_budget(runtime, runtime / "features/raw_data", {}, [])
    assert budget["sqlite_backup_bytes"] <= 17 * 1024 * 1024
    assert budget["required_free_bytes"] < 300 * 1024 * 1024

    changed = list(NEW)
    changed[4], changed[8] = "104.5", 29
    _add_receipt(runtime, "2026-11-06T01:50:05Z", changed)
    _add_receipt(runtime, "2026-11-06T01:55:05Z", changed)
    res2 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW + pd.Timedelta(minutes=15))
    assert res2["status"] == "COMMITTED"
    archive2 = Path(res2["archive_path"])
    delta_path = archive2 / "observations_delta.sqlite3"
    assert delta_path.is_file()
    with sqlite3.connect(delta_path) as conn:
        rows = conn.execute("SELECT content FROM decisions").fetchall()
    assert rows == [("new decision between revisions",)]
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"


def test_v14r2_patch2_active_window_risk_vs_awaiting_settlement_open_risk(tmp_path):
    """Patch 2: max_active_window_risk_usd separates unexpired active risk from expired awaiting-settlement open risk."""
    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    policy["risk"] = {
        "max_order_usd": 50,
        "max_open_risk_usd": 200,
        "max_active_window_risk_usd": 50,
        "max_underlying_open_risk_usd": 150,
        "max_underlying_active_risk_usd": 50,
    }
    limits = account._policy_risk_limits(policy)
    assert limits["max_open_risk_usd"] == Decimal("200")
    assert limits["max_active_window_risk_usd"] == Decimal("50")
    assert limits["max_underlying_open_risk_usd"] == Decimal("150")
    assert limits["max_underlying_active_risk_usd"] == Decimal("50")

    ledger = account.Account(tmp_path / "account.sqlite3", initial_capital=1000, started_at="2026-11-08T00:00:00Z")
    ov = ledger.overview(now="2026-11-08T00:00:05Z", policy=policy)
    assert ov["risk_limits"]["max_active_window_cost_usd"] == 50.0
    assert ov["risk_limits"]["max_total_open_cost_usd"] == 200.0
    assert ov["risk_limits"]["max_underlying_active_cost_usd"] == 50.0
    assert ov["risk_limits"]["max_underlying_open_cost_usd"] == 150.0


def test_v14r2_patch3_settlement_alignment_analyzer(tmp_path):
    """Patch 3: analyze_argon4_settlement_alignment compares official TWAP vs Binance 5m point and VWAP."""
    from scripts.analyze_argon4_settlement_alignment import (
        compute_settlement_alignment_row,
        summarize_settlement_alignments,
    )
    intraday = tmp_path / "intraday"
    intraday.mkdir(parents=True)
    t0 = pd.Timestamp("2026-11-08T00:00:00Z")
    df = pd.DataFrame([
        {
            "date": t0,
            "open": 100.0,
            "high": 100.5,
            "low": 99.9,
            "close": 100.015,  # +1.5 bps (1_to_3_bps), point=UP
            "volume": 10.0,
            "quote_volume": 999.5,  # vwap = 99.95 < 100.0 -> vwap=DOWN
        }
    ])
    df.to_parquet(intraday / "BTCUSDT_5m_futures.parquet", index=False)
    row = compute_settlement_alignment_row(
        tmp_path,
        stream="BTC_5m",
        window_start="2026-11-08T00:00:00Z",
        window_end="2026-11-08T00:05:00Z",
        official_outcome="DOWN",
        market_id="m1",
        prediction_side="UP",
    )
    assert row["status"] == "ALIGNED_CHECKED"
    assert row["return_bucket"] == "1_to_3_bps"
    assert row["binance_point_outcome"] == "UP"
    assert row["binance_vwap_outcome"] == "DOWN"
    assert row["point_aligned"] is False
    assert row["vwap_aligned"] is True
    assert row["official_win"] is False
    assert row["binance_point_win"] is True
    summary = summarize_settlement_alignments([row])
    assert summary["checked_records"] == 1
    assert summary["disagreement_count"] == 1
    assert summary["by_return_bucket"]["1_to_3_bps"]["count"] == 1


def test_v14r2_patch4_ex_ante_rv12_bps_and_volatility_regime_guard(tmp_path):
    """Patch 4: compute_ex_ante_rv12_bps and apply_volatility_regime_guard filter extreme volatility when enabled."""
    intraday = tmp_path / "intraday"
    intraday.mkdir(parents=True)
    times = pd.date_range("2026-11-08T00:00:00Z", periods=12, freq="5min")
    # 12 bars each with |close - open| / open = 0.0040 -> 40.0 bps
    df = pd.DataFrame({
        "date": times,
        "open_time": times,
        "open": [100.0] * 12,
        "close": [100.4] * 12,
    })
    df.to_parquet(intraday / "BTCUSDT_5m_futures.parquet", index=False)
    rv12 = model_v14.compute_ex_ante_rv12_bps(tmp_path, "BTCUSDT", as_of="2026-11-08T01:00:04Z")
    assert rv12 == pytest.approx(40.0)

    base_pred = {"stream": "BTC_15m", "z_v4": 0.65, "threshold_frozen": 0.45, "selected": True, "side": "UP"}
    # Disabled guard preserves selection
    res_off = model_v14.apply_volatility_regime_guard(
        "BTC_15m", base_pred, ex_ante_rv12_bps=rv12,
        guard_config={"enabled": False, "max_rv12_bps": {"BTC_15m": 28.0}, "action": "SKIP_OR_HALF_STAKE"},
    )
    assert res_off["selected"] is True
    assert res_off["ex_ante_rv12_bps"] == pytest.approx(40.0)

    # Enabled guard skips when rv12 (40.0) > max_rv12_bps (28.0)
    res_on = model_v14.apply_volatility_regime_guard(
        "BTC_15m", base_pred, ex_ante_rv12_bps=rv12,
        guard_config={"enabled": True, "max_rv12_bps": {"BTC_15m": 28.0}, "action": "SKIP_OR_HALF_STAKE"},
    )
    assert res_on["selected"] is False
    assert res_on["volatility_regime_guarded"] is True
    assert res_on["volatility_regime_reason"] == "ABOVE_MAX_RV12_BPS"

    # Enabled guard also skips ultra-low volatility when rv12 (1.8) < min_rv12_bps (3.0)
    res_low = model_v14.apply_volatility_regime_guard(
        "BTC_5m", {**base_pred, "stream": "BTC_5m"}, ex_ante_rv12_bps=1.8,
        guard_config={"enabled": True, "min_rv12_bps": {"BTC_5m": 3.0}, "action": "SKIP_OR_HALF_STAKE"},
    )
    assert res_low["selected"] is False
    assert res_low["volatility_regime_guarded"] is True
    assert res_low["volatility_regime_min_rv12_bps"] == pytest.approx(3.0)
    assert res_low["volatility_regime_reason"] == "BELOW_MIN_RV12_BPS"


def test_v14r3_min_buy_price_adverse_selection_guard_and_settlement_buckets(tmp_path):
    """v14r3: min_buy_price skips deep-discount adverse selection asks (<0.37) and analyzer buckets by ask & rv12."""
    from tools.argon4_forward.execution import PaperExecutor
    from scripts.analyze_argon4_settlement_alignment import (
        compute_settlement_alignment_row,
        summarize_settlement_alignments,
    )

    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    assert policy["volatility_regime_guard"]["min_rv12_bps"] == {"BTC_5m": 3.0, "ETH_5m": 3.0, "SOL_5m": 3.0}
    policy["min_buy_price"] = 0.37
    limits = account._policy_risk_limits(policy)
    assert limits["min_buy_price"] == Decimal("0.37")

    market = {
        "status": "VERIFIED_MARKET",
        "active": True,
        "closed": False,
        "accepting_orders": True,
        "market_id": "m-1",
        "condition_id": "c-1",
        "protocol": "polymarket_clob",
        "asset": "BTC",
        "duration_seconds": 300,
        "window_start": "2026-11-08T12:00:00+00:00",
        "window_end": "2026-11-08T12:05:00+00:00",
        "outcome_ids": {"UP": "tok-up", "DOWN": "tok-down"},
        "metadata_received_at": "2026-11-08T12:00:04+00:00",
        "metadata_sha256": "a" * 64,
        "fee": {"status": "KNOWN_DISABLED", "fees_enabled": False},
        "resolution": {"source": "chainlink"},
    }
    book_toxic = {
        "status": "OBSERVED",
        "asset_id": "tok-up",
        "condition_id": "c-1",
        "source_at": "2026-11-08T12:00:05+00:00",
        "received_at": "2026-11-08T12:00:05.100000+00:00",
        "asks": [{"price": "0.24", "size": "500"}],
    }
    clock = {
        "server_minus_local_seconds": 0.0,
        "uncertainty_seconds": 0.05,
        "receipt_sha256": "b" * 64,
    }
    stamp = account._ts("2026-11-08T12:00:05.200000+00:00")
    adapted, fee, cap = PaperExecutor._adapt(
        market, "UP", book_toxic, stamp, clock, allow_price_skip=True, min_buy_price=Decimal("0.37")
    )
    assert cap == Decimal(0)

    # Verify analyzer ask_bucket, adverse_selection_zone, and combined_guards_simulation
    intraday = tmp_path / "intraday"
    intraday.mkdir(parents=True)
    times = pd.date_range("2026-11-08T11:00:00Z", periods=13, freq="5min")
    opens = [100.0] * 13
    closes = [100.05] * 13  # 5.0 bps per bar -> rv12 = 5.0 bps
    df = pd.DataFrame({"date": times, "open": opens, "close": closes, "volume": [10.0] * 13, "quote_volume": [1000.5] * 13})
    df.to_parquet(intraday / "BTCUSDT_5m_futures.parquet", index=False)

    r_good = compute_settlement_alignment_row(
        tmp_path,
        stream="BTC_5m",
        window_start="2026-11-08T12:00:00Z",
        window_end="2026-11-08T12:05:00Z",
        official_outcome="UP",
        prediction_side="UP",
        ask_price=0.45,
        effective_price=0.455,
        realized_pnl_usd=180.0,
    )
    r_toxic = compute_settlement_alignment_row(
        tmp_path,
        stream="BTC_5m",
        window_start="2026-11-08T12:00:00Z",
        window_end="2026-11-08T12:05:00Z",
        official_outcome="DOWN",
        prediction_side="UP",
        ask_price=0.24,
        effective_price=0.245,
        realized_pnl_usd=-150.0,
    )
    summary = summarize_settlement_alignments([r_good, r_toxic])
    assert summary["by_adverse_selection_zone"]["NORMAL_ASK (>=0.37)"]["count"] == 1
    assert summary["by_adverse_selection_zone"]["TOXIC_DISCOUNT (<=0.36)"]["count"] == 1
    assert summary["combined_guards_simulation"]["count"] == 1
    assert summary["combined_guards_simulation"]["official_win_rate"] == 1.0
    assert summary["combined_guards_simulation"]["realized_pnl_usd"] == pytest.approx(180.0)


def test_v14r2_patch5_early_window_book_ticker(tmp_path, monkeypatch):
    """Patch 5: OnlineInputs.capture_early_window_book_ticker validates params and records top-of-book snapshot."""
    online._parameters("book_ticker", {})
    online._parameters("book_ticker", {"symbol": "BTCUSDT"})
    with pytest.raises(ValueError):
        online._parameters("book_ticker", {"symbol": "INVALID"})

    collector = online.OnlineInputs(tmp_path / "runtime")
    try:
        def fake_get(kind, params=None, **kwargs):
            assert kind == "book_ticker"
            return (
                [
                    {"symbol": "BTCUSDT", "bidPrice": "90000.0", "bidQty": "3.0", "askPrice": "90001.0", "askQty": "1.0", "time": 1762560003500},
                    {"symbol": "ETHUSDT", "bidPrice": "3000.0", "bidQty": "5.0", "askPrice": "3000.2", "askQty": "5.0", "time": 1762560003500},
                ],
                {
                    "requested_at": "2026-11-08T00:00:03.400000+00:00",
                    "received_at": "2026-11-08T00:00:03.500000+00:00",
                    "sha256": "e" * 64,
                    "time_basis": "CALIBRATED_UTC_UPPER_BOUND",
                    "clock_evidence_sha256": "f" * 64,
                },
            )
        monkeypatch.setattr(collector.client, "get", fake_get)
        snap = collector.capture_early_window_book_ticker(
            now="2026-11-08T00:00:03.500000+00:00",
            cohort_id="2026-11-08T00:00:00+00:00",
        )
        assert snap["status"] == "OBSERVED"
        assert snap["receipt_sha256"] == "e" * 64
        assert snap["tickers"]["BTCUSDT"]["mid_price"] == pytest.approx(90000.5)
        assert snap["tickers"]["BTCUSDT"]["top_book_imbalance"] == pytest.approx(0.5)
    finally:
        collector.close()


def test_v14r4_snapshots_only_large_observations_hardlink_chained_delta_and_active_guards(runtime, tmp_path):
    """v14r4: >4MB snapshots-only observations.sqlite3 hardlinks across chained revisions and policy guards are active."""
    # Replace observations.sqlite3 with a snapshots-only table (matching live runtime schema)
    obs_path = runtime / "observations.sqlite3"
    with sqlite3.connect(obs_path) as conn:
        conn.execute("DROP TABLE IF EXISTS decisions")
        conn.execute(
            "CREATE TABLE snapshots(id INTEGER PRIMARY KEY, recorded_at TEXT, status TEXT, payload TEXT)"
        )
        # Pad payload to exceed 4 MB so _is_valid_observations_prefix takes the fast O(1) prefix branch
        big_payload = "x" * (2100 * 1024)
        conn.execute("INSERT INTO snapshots(recorded_at,status,payload) VALUES(?,?,?)", ("t1", "OK", big_payload))
        conn.execute("INSERT INTO snapshots(recorded_at,status,payload) VALUES(?,?,?)", ("t2", "OK", big_payload))
    assert obs_path.stat().st_size > 4 * 1024 * 1024

    # First revision: full backup
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    res1 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert res1["status"] == "COMMITTED"
    archive1 = Path(res1["archive_path"])

    # Add row #3 between revision 1 and revision 2
    with sqlite3.connect(obs_path) as conn:
        conn.execute("INSERT INTO snapshots(recorded_at,status,payload) VALUES(?,?,?)", ("t3", "OK", "delta_row_3"))

    changed2 = list(NEW)
    changed2[4], changed2[8] = "104.5", 29
    _add_receipt(runtime, "2026-11-06T01:50:05Z", changed2)
    _add_receipt(runtime, "2026-11-06T01:55:05Z", changed2)
    res2 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW + pd.Timedelta(minutes=15))
    assert res2["status"] == "COMMITTED"
    archive2 = Path(res2["archive_path"])
    if (archive1 / "observations.sqlite3").stat().st_nlink >= 2:
        assert (archive1 / "observations.sqlite3").stat().st_ino == (archive2 / "observations.sqlite3").stat().st_ino
    else:
        assert revisions._sha(archive1 / "observations.sqlite3") == revisions._sha(archive2 / "observations.sqlite3")
    with sqlite3.connect(archive2 / "observations_delta.sqlite3") as conn:
        assert conn.execute("SELECT id, payload FROM snapshots").fetchall() == [(3, "delta_row_3")]

    # Add row #4 between revision 2 and revision 3: chained delta must ONLY contain row #4
    with sqlite3.connect(obs_path) as conn:
        conn.execute("INSERT INTO snapshots(recorded_at,status,payload) VALUES(?,?,?)", ("t4", "OK", "delta_row_4"))

    changed3 = list(NEW)
    changed3[4], changed3[8] = "106.0", 35
    _add_receipt(runtime, "2026-11-06T02:05:05Z", changed3)
    _add_receipt(runtime, "2026-11-06T02:10:05Z", changed3)
    res3 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW + pd.Timedelta(minutes=30))
    assert res3["status"] == "COMMITTED"
    archive3 = Path(res3["archive_path"])
    if (archive1 / "observations.sqlite3").stat().st_nlink >= 2:
        assert (archive1 / "observations.sqlite3").stat().st_ino == (archive3 / "observations.sqlite3").stat().st_ino
    else:
        assert revisions._sha(archive1 / "observations.sqlite3") == revisions._sha(archive3 / "observations.sqlite3")
    with sqlite3.connect(archive3 / "observations_delta.sqlite3") as conn:
        assert conn.execute("SELECT id, payload FROM snapshots").fetchall() == [(4, "delta_row_4")]

    # Verify production policy has both empirical guards enabled and MonitorAuditLogger formats both
    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    assert policy["min_buy_price"] == 0.37
    assert policy["volatility_regime_guard"]["enabled"] is True
    assert policy["volatility_regime_guard"]["min_rv12_bps"]["BTC_5m"] == 3.0

    logger = monitor_logger.MonitorAuditLogger(tmp_path / "audit_logs")
    decisions = {
        "BTC_5m": {
            "status": "NO_SIGNAL",
            "outcome": "Up",
            "prediction": {
                "z_v4": 0.42,
                "threshold_frozen": 0.35,
                "side": "Up",
                "volatility_regime_guarded": True,
                "volatility_regime_reason": "BELOW_MIN_RV12_BPS",
                "ex_ante_rv12_bps": 1.85,
            },
        },
        "ETH_5m": {
            "status": "SIGNAL",
            "outcome": "Down",
            "prediction": {"z_v4": -0.55, "threshold_frozen": 0.35, "side": "Down"},
        },
    }
    outcome = {
        "status": "PROCESSED",
        "streams": {
            "ETH_5m": {
                "status": "PRICE_LIMIT_NOT_MET",
                "best_ask": 0.24,
                "limit_price": 0.515,
                "min_buy_price": 0.37,
            }
        },
        "fills": [],
        "blocked": [],
    }
    logger.log_cohort_execution("2026-11-08T12:00:00+00:00", {"order": ["BTC_5m", "ETH_5m"]}, decisions, outcome)
    ev = logger.recent_events(limit=1)[0]
    assert "VOLATILITY_REGIME_GUARD" in ev["details"]["streams"]["BTC_5m"]["no_trade_reason"]
    assert "ADVERSE_SELECTION_GUARD" in ev["details"]["streams"]["ETH_5m"]["no_trade_reason"]


def test_v14r5_multilevel_orderbook_adverse_selection_guard_and_immediate_skip_without_recheck(tmp_path):
    """v14r5: multi-level orderbook with best_ask=0.25 (<0.37) and resting asks at 0.42/0.50 zeroes capacity,
    never sweeps resting 0.42 asks, and skips immediately without WAITING_PRICE_RECHECK under multi_look_ioc."""
    from tools.argon4_forward.execution import PaperExecutor

    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    assert policy["min_buy_price"] == 0.37
    assert policy["execution"]["multi_look_ioc"] is True

    market_btc = {
        "status": "VERIFIED_MARKET",
        "active": True,
        "closed": False,
        "accepting_orders": True,
        "market_id": "m-btc",
        "condition_id": "c-btc",
        "protocol": "polymarket_clob",
        "asset": "BTC",
        "duration_seconds": 900,
        "window_start": "2026-11-08T12:00:00+00:00",
        "window_end": "2026-11-08T12:15:00+00:00",
        "outcome_ids": {"UP": "tok-btc-up", "DOWN": "tok-btc-down"},
        "metadata_received_at": "2026-11-08T12:00:04+00:00",
        "metadata_sha256": "a" * 64,
        "fee": {"status": "KNOWN_DISABLED", "fees_enabled": False},
        "resolution": {"source": "chainlink"},
    }
    market_eth = {
        "status": "VERIFIED_MARKET",
        "active": True,
        "closed": False,
        "accepting_orders": True,
        "market_id": "m-eth",
        "condition_id": "c-eth",
        "protocol": "polymarket_clob",
        "asset": "ETH",
        "duration_seconds": 300,
        "window_start": "2026-11-08T12:00:00+00:00",
        "window_end": "2026-11-08T12:05:00+00:00",
        "outcome_ids": {"UP": "tok-eth-up", "DOWN": "tok-eth-down"},
        "metadata_received_at": "2026-11-08T12:00:04+00:00",
        "metadata_sha256": "e" * 64,
        "fee": {"status": "KNOWN_DISABLED", "fees_enabled": False},
        "resolution": {"source": "chainlink"},
    }
    # Realistic multi-level Polymarket CLOB orderbook where best_ask=0.25 (<0.37) AND resting asks exist at 0.42 & 0.50
    crashed_multilevel_book = {
        "status": "OBSERVED",
        "asset_id": "tok-btc-up",
        "condition_id": "c-btc",
        "source_at": "2026-11-08T12:00:05+00:00",
        "received_at": "2026-11-08T12:00:05.100000+00:00",
        "asks": [
            {"price": "0.25", "size": "200"},
            {"price": "0.32", "size": "300"},
            {"price": "0.42", "size": "400"},
            {"price": "0.50", "size": "500"},
        ],
    }
    healthy_multilevel_book = {
        "status": "OBSERVED",
        "asset_id": "tok-eth-up",
        "condition_id": "c-eth",
        "source_at": "2026-11-08T12:00:05+00:00",
        "received_at": "2026-11-08T12:00:05.100000+00:00",
        "asks": [
            {"price": "0.45", "size": "400"},
            {"price": "0.48", "size": "500"},
            {"price": "0.60", "size": "800"},
        ],
    }
    clock = {
        "server_minus_local_seconds": 0.0,
        "uncertainty_seconds": 0.05,
        "receipt_sha256": "b" * 64,
    }
    stamp = account._ts("2026-11-08T12:00:05.200000+00:00")

    # 1. _adapt must return capacity=0 on the multi-level crashed orderbook despite resting asks at 0.42 and 0.50
    adapted, fee, cap = PaperExecutor._adapt(
        market_btc, "UP", crashed_multilevel_book, stamp, clock, allow_price_skip=True, min_buy_price=Decimal("0.37")
    )
    assert cap == Decimal(0)

    # 2. Full waterfall execution under multi_look_ioc:
    # Rank 1 (BTC_15m) has crashed multi-level book (best_ask=0.25 < 0.37): must skip immediately as PRICE_LIMIT_NOT_MET
    # WITHOUT entering WAITING_PRICE_RECHECK, and immediately unlock Rank 2 (ETH_5m) in the same call!
    from test_argon4_forward_execution import BOUNDARY, clock as make_clock, fixture_inputs, v5_policy

    test_policy = {
        **v5_policy(),
        "min_buy_price": 0.37,
        "execution": {
            "multi_look_ioc": True,
            "max_price_rechecks": 3,
            "price_recheck_interval_seconds": 2.5,
            "minimum_recheck_remaining_seconds": 3.0,
            "last_recheck_start_after_boundary_seconds": 18.5,
        },
    }
    ledger = account.Account(tmp_path / "account.sqlite3", initial_capital=10000, started_at=BOUNDARY - pd.Timedelta(seconds=1))
    executor = PaperExecutor(ledger, test_policy)
    decisions, markets = fixture_inputs(boundary=BOUNDARY)
    markets["BTC_15m"]["books"]["UP"]["asks"] = [
        {"price": "0.25", "size": "200"},
        {"price": "0.32", "size": "300"},
        {"price": "0.42", "size": "400"},
        {"price": "0.50", "size": "500"},
    ]
    markets["ETH_5m"]["books"]["UP"]["asks"] = [
        {"price": "0.45", "size": "400"},
        {"price": "0.48", "size": "500"},
    ]
    now_ts = BOUNDARY + pd.Timedelta(seconds=1)
    res = executor.process_cohort(BOUNDARY.isoformat(), decisions, markets, now=now_ts, clock=make_clock(now_ts))
    assert res.get("status") != "WAITING_PRICE_RECHECK"
    assert res["stream_results"]["BTC_15m"]["status"] == "PRICE_LIMIT_NOT_MET"
    assert res["stream_results"]["BTC_15m"]["best_ask"] == pytest.approx(0.25)
    assert res["stream_results"]["BTC_15m"]["min_buy_price"] == pytest.approx(0.37)
    assert res["stream_results"]["ETH_5m"]["status"] == "FILLED_BUDGET"
    assert any(f["signal_id"].startswith("model:ETH_5m:") for f in res["fills"])
    assert not any(f["signal_id"].startswith("model:BTC_15m:") for f in res["fills"])

    # 3. Fix 1: compute_ex_ante_rv12_bps ignores 1-row raw_frame and requires 12 closed 5m bars
    intraday = tmp_path / "rv12_root" / "intraday"
    intraday.mkdir(parents=True)
    times12 = pd.date_range("2026-11-08T11:00:00Z", periods=12, freq="5min")
    df12 = pd.DataFrame({"date": times12, "open_time": times12, "open": [100.0] * 12, "close": [100.2] * 12})
    df12.to_parquet(intraday / "BTCUSDT_5m_futures.parquet", index=False)
    one_row = df12.tail(1)
    assert model_v14.compute_ex_ante_rv12_bps(
        tmp_path / "rv12_root", "BTCUSDT", as_of="2026-11-08T12:00:04Z", raw_frame=one_row
    ) == pytest.approx(20.0)
    assert model_v14.compute_ex_ante_rv12_bps(None, "BTCUSDT", raw_frame=one_row) is None

    # 4. Fix 3: 5-day losing streak cooldown stays COOLDOWN_ACTIVE when losing_days < 5 and cooldown_until is active
    cd_policy = {**test_policy, "cooldown_until": (BOUNDARY + pd.Timedelta(days=3)).isoformat()}
    plan_cd = allocation.plan_waterfall(cd_policy, [], 10000.0, 10000.0, 0.0, BOUNDARY.isoformat(), losing_days=0)
    assert plan_cd["reasons"] == ["COOLDOWN_ACTIVE"]
    cohort_cd = ledger.ensure_cohort((BOUNDARY + pd.Timedelta(days=1)).isoformat(), cd_policy, now=BOUNDARY + pd.Timedelta(days=1, seconds=1))
    assert cohort_cd["blocked_reason"] == "COOLDOWN_ACTIVE"

    # 5. Fix 4: _is_valid_observations_prefix detects tampering of row id=1 on >4MB observations.sqlite3
    from contextlib import closing
    big_src = tmp_path / "big_live.sqlite3"
    big_snap = tmp_path / "big_snap.sqlite3"
    big_blob = "z" * (2100 * 1024)
    with closing(sqlite3.connect(big_src)) as conn, conn:
        conn.execute("CREATE TABLE snapshots(id INTEGER PRIMARY KEY, payload TEXT)")
        conn.execute("INSERT INTO snapshots(payload) VALUES(?)", (big_blob,))
        conn.execute("INSERT INTO snapshots(payload) VALUES(?)", (big_blob,))
    with closing(sqlite3.connect(big_src)) as src_c, closing(sqlite3.connect(big_snap)) as dst_c:
        src_c.backup(dst_c)
    assert big_snap.stat().st_size > 4 * 1024 * 1024
    assert revisions._is_valid_observations_prefix(big_src, big_snap) is True
    with closing(sqlite3.connect(big_src)) as conn, conn:
        conn.execute("UPDATE snapshots SET payload=? WHERE id=1", ("tampered" + big_blob[8:],))
    assert revisions._is_valid_observations_prefix(big_src, big_snap) is False

    # 6. Fix 5: early_window_book_ticker defaults to enabled=False in production policy
    assert policy["early_window_book_ticker"]["enabled"] is False


def test_v14r6_offline_day_preserves_5day_cooldown_and_50001_row_middle_tamper_detected(tmp_path):
    """v14r6:
    1. 5 consecutive losing days latch cooldown even if the system is powered off on Day 7 with 0 DB rows,
       and only unlocks after an explicit cooldown_until after the 5th losing day has expired.
    2. _is_valid_observations_prefix uses SQLite ATTACH + EXCEPT to verify 100% of rows even when >50,000 rows,
       detecting tampering at middle row id=25000.
    """
    from contextlib import closing
    from test_argon4_forward_execution import BOUNDARY, v5_policy

    base_day = account._ts("2026-11-01T00:00:00Z")
    ledger = account.Account(tmp_path / "cooldown_v14r6.sqlite3", initial_capital=10000, started_at=base_day - pd.Timedelta(seconds=1))
    pol = v5_policy()

    # Record 5 consecutive losing days (Nov 1 .. Nov 5) in events
    with ledger._connection() as con:
        for offset in range(5):
            day_ts = base_day + pd.Timedelta(days=offset, hours=12)
            ledger._event(
                con,
                "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER",
                {"market_id": f"m-{offset}", "realized_pnl_usd": -50.0},
                day_ts,
            )

    # Day 6 (Nov 6): triggers COOLDOWN_STATE_REQUIRED
    day6 = base_day + pd.Timedelta(days=5)
    c6 = ledger.ensure_cohort(day6.isoformat(), pol, now=day6 + pd.Timedelta(seconds=1))
    assert c6["blocked_reason"] == "COOLDOWN_STATE_REQUIRED"

    # Day 7 (Nov 7): machine is powered off all day (no cohorts or settlements on Nov 7)
    # Day 8 (Nov 8): machine restarts without cooldown_until -> MUST STILL be COOLDOWN_STATE_REQUIRED!
    day8 = base_day + pd.Timedelta(days=7)
    c8 = ledger.ensure_cohort(day8.isoformat(), pol, now=day8 + pd.Timedelta(seconds=1))
    assert c8["blocked_reason"] == "COOLDOWN_STATE_REQUIRED"

    # Day 8 + 5m: operator configures cooldown_until = Day 9 (Nov 9 00:00:00Z) -> COOLDOWN_ACTIVE on Day 8
    day9 = base_day + pd.Timedelta(days=8)
    pol_with_cd = {**pol, "cooldown_until": day9.isoformat()}
    c8_active = ledger.ensure_cohort((day8 + pd.Timedelta(minutes=5)).isoformat(), pol_with_cd, now=day8 + pd.Timedelta(minutes=5, seconds=1))
    assert c8_active["blocked_reason"] == "COOLDOWN_ACTIVE"

    # Day 9 (Nov 9 00:00:00Z, at/after cooldown_until): cooldown has been satisfied and unlocks!
    c9_unlocked = ledger.ensure_cohort(day9.isoformat(), pol_with_cd, now=day9 + pd.Timedelta(seconds=1))
    assert c9_unlocked["blocked_reason"] is None
    assert c9_unlocked["budget_usd"] > 0

    # Verify >50,000 rows (50,001 rows) middle-row tamper detection at id=25000 via SQLite ATTACH + EXCEPT
    src_50k = tmp_path / "obs_50001_live.sqlite3"
    snap_50k = tmp_path / "obs_50001_snap.sqlite3"
    with closing(sqlite3.connect(src_50k)) as conn, conn:
        conn.execute("CREATE TABLE snapshots(id INTEGER PRIMARY KEY, recorded_at TEXT, status TEXT, payload TEXT)")
        rows = [(i, "2026-11-08T00:00:00Z", "OK", f"row_payload_{i}") for i in range(1, 50002)]
        conn.executemany("INSERT INTO snapshots(id, recorded_at, status, payload) VALUES(?,?,?,?)", rows)
    with closing(sqlite3.connect(src_50k)) as s_conn, closing(sqlite3.connect(snap_50k)) as d_conn:
        s_conn.backup(d_conn)

    assert revisions._is_valid_observations_prefix(src_50k, snap_50k) is True
    # Append row 50002 to live DB -> still a valid prefix
    with closing(sqlite3.connect(src_50k)) as conn, conn:
        conn.execute("INSERT INTO snapshots(id, recorded_at, status, payload) VALUES(?,?,?,?)", (50002, "2026-11-08T00:05:00Z", "OK", "row_payload_50002"))
    assert revisions._is_valid_observations_prefix(src_50k, snap_50k) is True

    # Mutate middle row id=25000 in live DB -> must return False!
    with closing(sqlite3.connect(src_50k)) as conn, conn:
        conn.execute("UPDATE snapshots SET payload='TAMPERED_MIDDLE_ROW' WHERE id=25000")
    assert revisions._is_valid_observations_prefix(src_50k, snap_50k) is False


def test_v14r7_wal_tamper_chained_delta_tamper_and_sameday_6h_cooldown_unlock(runtime, tmp_path):
    """v14r7:
    1. WAL mode: modifying an old row in WAL (before checkpoint into main file) immediately invalidates
       _is_valid_observations_prefix without clearing any cache.
    2. Chained delta: when base has row 1 and Rev 2 delta has row 2, modifying row 2 and appending row 3
       before Rev 3 invalidates the prefix check so Rev 3 does not only save row 3.
    3. Same-day 6-hour cooldown: when 5 losing days trigger COOLDOWN_STATE_REQUIRED on Day 6 00:00Z and
       cooldown_until is configured for Day 6 06:00Z, trading unlocks at Day 6 06:05Z and stays unlocked on Day 7.
    """
    from contextlib import closing
    from test_argon4_forward_execution import v5_policy

    # 1. WAL mode uncheckpointed old-row modification immediately detected without clearing cache
    wal_src = tmp_path / "wal_live.sqlite3"
    wal_snap = tmp_path / "wal_snap.sqlite3"
    with closing(sqlite3.connect(wal_src)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE snapshots(id INTEGER PRIMARY KEY, recorded_at TEXT, status TEXT, payload TEXT)")
        conn.execute("INSERT INTO snapshots(id, recorded_at, status, payload) VALUES(1, 't1', 'OK', 'row1')")
        conn.execute("INSERT INTO snapshots(id, recorded_at, status, payload) VALUES(2, 't2', 'OK', 'row2')")
        conn.commit()
    with closing(sqlite3.connect(wal_src)) as s_conn, closing(sqlite3.connect(wal_snap)) as d_conn:
        s_conn.backup(d_conn)

    # First check returns True
    assert revisions._is_valid_observations_prefix(wal_src, wal_snap) is True

    # Keep a reader open so WAL frames cannot be auto-checkpointed into the main file on close
    reader = sqlite3.connect(wal_src)
    try:
        _ = reader.execute("SELECT COUNT(*) FROM snapshots").fetchall()
        main_mtime_before = wal_src.stat().st_mtime_ns
        main_size_before = wal_src.stat().st_size
        with closing(sqlite3.connect(wal_src)) as writer:
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("UPDATE snapshots SET payload='tampered_in_wal' WHERE id=1")
            writer.commit()
        assert Path(str(wal_src) + "-wal").is_file()
        assert wal_src.stat().st_size == main_size_before
        assert wal_src.stat().st_mtime_ns == main_mtime_before
        # Without clearing any cache, _is_valid_observations_prefix must detect the WAL tamper immediately!
        assert revisions._is_valid_observations_prefix(wal_src, wal_snap) is False
    finally:
        reader.close()

    # 2. Chained delta tamper detection (Base has row 1, Rev 2 delta has row 2 -> modify row 2 + append row 3 before Rev 3)
    obs_path = runtime / "observations.sqlite3"
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    res1 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert res1["status"] == "COMMITTED"

    # Append row 2 and commit Revision 2 (which writes row 2 into observations_delta.sqlite3)
    with closing(sqlite3.connect(obs_path)) as conn, conn:
        conn.execute("INSERT INTO decisions(content) VALUES('row_2_original')")
    changed2 = list(NEW)
    changed2[4], changed2[8] = "104.5", 29
    _add_receipt(runtime, "2026-11-06T01:50:05Z", changed2)
    _add_receipt(runtime, "2026-11-06T01:55:05Z", changed2)
    res2 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW + pd.Timedelta(minutes=15))
    assert res2["status"] == "COMMITTED"
    archive2 = Path(res2["archive_path"])
    assert (archive2 / "observations_delta.sqlite3").is_file()
    assert revisions._is_valid_observations_prefix(obs_path, archive2 / "observations.sqlite3") is True

    # Now mutate row 2 (which lives in Rev 2's delta) and append row 3 before Revision 3
    with closing(sqlite3.connect(obs_path)) as conn, conn:
        conn.execute("UPDATE decisions SET content='row_2_TAMPERED' WHERE id=2")
        conn.execute("INSERT INTO decisions(content) VALUES('row_3_new')")

    # Both direct prefix validation and _prior_committed_observations must reject the tampered delta row!
    assert revisions._is_valid_observations_prefix(obs_path, archive2 / "observations.sqlite3") is False
    assert revisions._prior_committed_observations(runtime / "features/revisions", obs_path)[0] is None

    # Revision 3 therefore falls back to a full snapshot containing all 3 rows rather than only saving row 3
    changed3 = list(NEW)
    changed3[4], changed3[8] = "106.0", 35
    _add_receipt(runtime, "2026-11-06T02:05:05Z", changed3)
    _add_receipt(runtime, "2026-11-06T02:10:05Z", changed3)
    res3 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW + pd.Timedelta(minutes=30))
    assert res3["status"] == "COMMITTED"
    archive3 = Path(res3["archive_path"])
    assert not (archive3 / "observations_delta.sqlite3").exists()
    with closing(sqlite3.connect(archive3 / "observations.sqlite3")) as conn:
        assert conn.execute("SELECT id, content FROM decisions ORDER BY id").fetchall() == [
            (1, "old decision uses original receipt, not a revised history"),
            (2, "row_2_TAMPERED"),
            (3, "row_3_new"),
        ]

    # 3. Same-day 6-hour cooldown unlock on Day 6 and Day 7
    base_day = account._ts("2026-11-01T00:00:00Z")
    ledger = account.Account(
        tmp_path / "cooldown_v14r7.sqlite3", initial_capital=10000, started_at=base_day - pd.Timedelta(seconds=1)
    )
    pol = v5_policy()
    with ledger._connection() as con:
        for offset in range(5):
            day_ts = base_day + pd.Timedelta(days=offset, hours=12)
            ledger._event(
                con,
                "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER",
                {"market_id": f"m-{offset}", "realized_pnl_usd": -50.0},
                day_ts,
            )

    # Day 6 00:00Z triggers COOLDOWN_STATE_REQUIRED
    day6 = base_day + pd.Timedelta(days=5)
    c6_init = ledger.ensure_cohort(day6.isoformat(), pol, now=day6 + pd.Timedelta(seconds=1))
    assert c6_init["blocked_reason"] == "COOLDOWN_STATE_REQUIRED"

    # Operator configures a 6-hour cooldown on Day 6 (until Day 6 06:00:00Z)
    pol_6h = {**pol, "cooldown_until": (day6 + pd.Timedelta(hours=6)).isoformat()}

    # At Day 6 01:00Z (during the 6-hour cooldown): COOLDOWN_ACTIVE
    c6_active = ledger.ensure_cohort(
        (day6 + pd.Timedelta(hours=1)).isoformat(), pol_6h, now=day6 + pd.Timedelta(hours=1, seconds=1)
    )
    assert c6_active["blocked_reason"] == "COOLDOWN_ACTIVE"

    # At Day 6 06:05Z (after the 6-hour cooldown expires on the same day): unlocks!
    c6_after = ledger.ensure_cohort(
        (day6 + pd.Timedelta(hours=6, minutes=5)).isoformat(), pol_6h, now=day6 + pd.Timedelta(hours=6, minutes=5, seconds=1)
    )
    assert c6_after["blocked_reason"] is None
    assert c6_after["budget_usd"] > 0

    # On Day 7 00:05Z (next day, even if cooldown_until is removed from policy.json): stays unlocked!
    day7 = base_day + pd.Timedelta(days=6)
    c7 = ledger.ensure_cohort((day7 + pd.Timedelta(minutes=5)).isoformat(), pol, now=day7 + pd.Timedelta(minutes=5, seconds=1))
    assert c7["blocked_reason"] is None
    assert c7["budget_usd"] > 0








