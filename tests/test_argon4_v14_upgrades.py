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
    fallback_limits = account._policy_risk_limits({})
    assert fallback_limits["max_order_usd"] == Decimal("200")
    assert fallback_limits["max_open_risk_usd"] == Decimal("1000")
    assert fallback_limits["max_underlying_open_risk_usd"] == Decimal("500")
    assert fallback_limits["max_tick_budget_usd"] == Decimal("1200")

    policy_path = Path("config/argon4_forward_policy.json")
    default_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    default_limits = account._policy_risk_limits(default_policy)
    assert default_policy["initial_capital_usd"] == 1269
    assert default_limits["single_stream_equity_fraction"] == Decimal("0.08")
    assert default_limits["tick_equity_fraction"] == Decimal("0.15")
    assert default_limits["max_order_usd"] == Decimal("240")
    assert default_limits["max_tick_budget_usd"] == Decimal("380")
    assert default_limits["max_active_window_risk_usd"] == Decimal("450")
    assert default_limits["max_open_risk_usd"] == Decimal("650")
    assert default_limits["max_underlying_active_risk_usd"] == Decimal("300")
    assert default_limits["max_underlying_open_risk_usd"] == Decimal("450")
    assert default_limits["daily_loss_fraction"] == Decimal("0.15")

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
        "ETH_15m",
        "BTC_5m",
        "SOL_15m",
        "SOL_5m",
        "BNB_15m",
    ]
    assert policy["execution_stage"]["enabled_streams"] == [
        "BTC_15m",
        "ETH_5m",
        "ETH_15m",
        "BTC_5m",
        "SOL_15m",
        "SOL_5m",
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
    assert policy["volatility_regime_guard"]["min_rv12_bps"].items() >= {"BTC_5m": 3.0, "ETH_5m": 3.0, "SOL_5m": 3.0}.items()
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
    assert policy["min_buy_price"] == 0.415
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
    assert policy["min_buy_price"] == 0.415
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


def test_v14r9_volatility_guards_daily_loss_room_cap_and_stream_min_buy_price(tmp_path):
    """v14r9:
    1. Production policy configures BTC_5m max_rv12_bps=28.0 and 15m stream min_rv12_bps thresholds.
    2. ensure_cohort caps budget_usd and single_usd by remaining distance to DAILY_LOSS_LIMIT (preventing last-order breach)
       and supports trailing_daily_loss_limit from intraday peak realized equity.
    3. stream_min_buy_price overrides global min_buy_price per stream in both PaperExecutor and Account.
    """
    from test_argon4_forward_execution import BOUNDARY, clock as make_clock, fixture_inputs, v5_policy
    from tools.argon4_forward.execution import PaperExecutor

    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    vguard = policy["volatility_regime_guard"]
    assert vguard["max_rv12_bps"]["BTC_5m"] == 28.0
    assert vguard["max_rv12_bps"]["BTC_15m"] == 28.0
    assert vguard["min_rv12_bps"]["BTC_15m"] == 2.5
    assert vguard["min_rv12_bps"]["SOL_15m"] == 3.0
    assert vguard["min_rv12_bps"]["ETH_15m"] == 2.5
    assert policy["storage_maintenance"]["enabled"] is True
    assert policy["storage_maintenance"]["scan_interval_days"] == 14

    # Test daily loss room cap: -$532 realized today on $10,000 day_start_equity leaves $18 before -$550 (-5.5%) limit
    day0 = account._ts("2026-11-08T10:00:00Z")
    ledger = account.Account(
        tmp_path / "daily_cap_v14r9.sqlite3", initial_capital=10000, started_at=day0 - pd.Timedelta(hours=2)
    )
    with ledger._connection() as con:
        con.execute("UPDATE account SET cash_atoms = cash_atoms - 532000000, realized_atoms = -532000000 WHERE id=1")
        ledger._event(
            con,
            "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER",
            {"market_id": "m-loss-1", "realized_pnl_usd": -532.0},
            day0 - pd.Timedelta(minutes=10),
        )
    pol = v5_policy()
    cohort = ledger.ensure_cohort(day0.isoformat(), pol, now=day0 + pd.Timedelta(seconds=1))
    assert cohort["blocked_reason"] is None
    assert cohort["budget_usd"] == pytest.approx(18.0)
    assert cohort["single_usd"] == pytest.approx(18.0)

    # Test trailing_daily_loss_limit: +$600 morning peak (ref_equity=$10,600, max drawdown=$583), then -$583 from peak
    ledger_trail = account.Account(
        tmp_path / "trailing_v14r9.sqlite3", initial_capital=10000, started_at=day0 - pd.Timedelta(hours=4)
    )
    with ledger_trail._connection() as con:
        con.execute("UPDATE account SET cash_atoms = cash_atoms + 17000000, realized_atoms = 17000000 WHERE id=1")
        ledger_trail._event(
            con,
            "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER",
            {"market_id": "m-win-1", "realized_pnl_usd": 600.0},
            day0 - pd.Timedelta(hours=2),
        )
        ledger_trail._event(
            con,
            "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER",
            {"market_id": "m-loss-2", "realized_pnl_usd": -583.0},
            day0 - pd.Timedelta(minutes=15),
        )
    pol_trail = {**pol, "trailing_daily_loss_limit": True}
    c_trail = ledger_trail.ensure_cohort(day0.isoformat(), pol_trail, now=day0 + pd.Timedelta(seconds=1))
    assert c_trail["blocked_reason"] == "DAILY_LOSS_LIMIT"

    # Test stream_min_buy_price: BTC_15m requires 0.42 while ETH_5m requires 0.37
    pol_stream_min = {
        **v5_policy(),
        "min_buy_price": 0.37,
        "stream_min_buy_price": {"BTC_15m": 0.42, "ETH_5m": 0.37},
    }
    ledger_sm = account.Account(
        tmp_path / "stream_min_v14r9.sqlite3", initial_capital=10000, started_at=BOUNDARY - pd.Timedelta(seconds=1)
    )
    executor = PaperExecutor(ledger_sm, pol_stream_min)
    decisions, markets = fixture_inputs(boundary=BOUNDARY)
    # BTC_15m has best_ask = 0.39 (>= global 0.37, but < BTC_15m's 0.42 stream_min_buy_price!)
    markets["BTC_15m"]["books"]["UP"]["asks"] = [{"price": "0.39", "size": "400"}, {"price": "0.48", "size": "500"}]
    markets["ETH_5m"]["books"]["UP"]["asks"] = [{"price": "0.39", "size": "400"}, {"price": "0.48", "size": "500"}]
    now_ts = BOUNDARY + pd.Timedelta(seconds=1)
    res = executor.process_cohort(BOUNDARY.isoformat(), decisions, markets, now=now_ts, clock=make_clock(now_ts))
    assert res["stream_results"]["BTC_15m"]["status"] == "PRICE_LIMIT_NOT_MET"
    assert res["stream_results"]["BTC_15m"]["min_buy_price"] == pytest.approx(0.42)
    assert res["stream_results"]["ETH_5m"]["status"] == "FILLED_BUDGET"


def test_v14r9_storage_janitor_biweekly_cold_archive_and_revision_integrity(runtime, tmp_path, no_network):
    """v14r9:
    StorageJanitor migrates >14-day unpinned revision directories, aged observations.sqlite3 rows,
    and aged monitor_audit.jsonl lines to D_drive/argon4_cold_archive with SHA256 verification,
    while preserving active COMMITTED revision integrity and subsequent revision commits.
    """
    from contextlib import closing
    from tools.argon4_forward.storage_janitor import StorageJanitor

    obs_path = runtime / "observations.sqlite3"
    with closing(sqlite3.connect(obs_path)) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY, kind TEXT, saved_at TEXT, payload TEXT)"
        )
        conn.execute(
            "INSERT INTO snapshots(kind, saved_at, payload) VALUES(?, ?, ?)",
            ("polymarket", "2026-10-15T00:00:00+00:00", "aged_snapshot_payload_" + ("x" * 2048)),
        )
        conn.execute(
            "INSERT INTO snapshots(kind, saved_at, payload) VALUES(?, ?, ?)",
            ("polymarket", "2026-11-05T00:00:00+00:00", "hot_snapshot_payload"),
        )

    # Revision 1 & Revision 2 (active COMMITTED revision)
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    res1 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert res1["status"] == "COMMITTED"

    # Also create an unpinned >14-day historical revision directory (from 2026-10-10)
    revs_dir = runtime / "features" / "revisions"
    old_rev_id = "20261010T000000000000Z_oldunpinned0001"
    old_rev_dir = revs_dir / old_rev_id
    old_rev_dir.mkdir(parents=True, exist_ok=True)
    (old_rev_dir / "manifest.json").write_text(
        json.dumps({"status": "COMMITTED", "revision_id": old_rev_id, "view_effective_at": "2026-10-10T00:00:00+00:00"}),
        encoding="utf-8",
    )
    (old_rev_dir / "dummy_archive.bin").write_bytes(b"cold_archive_payload_123456")
    expected_dummy_sha = revisions._sha(old_rev_dir / "dummy_archive.bin")

    # Create monitor_audit.jsonl with 1 aged line (2026-10-12) and 1 hot line (2026-11-06)
    logs_dir = runtime / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "monitor_audit.jsonl").write_text(
        json.dumps({"ts": "2026-10-12T00:00:00+00:00", "event": "OLD_EVENT"}) + "\n"
        + json.dumps({"ts": "2026-11-06T01:00:00+00:00", "event": "HOT_EVENT"}) + "\n",
        encoding="utf-8",
    )

    cold_root = tmp_path / "D_drive" / "argon4_cold_archive"
    janitor = StorageJanitor(runtime, cold_root=cold_root)
    maint = janitor.run_maintenance(now="2026-11-06T02:00:00+00:00")
    assert maint["status"] == "COMPLETED"
    assert old_rev_id in maint["revisions"]["migrated_revisions"]
    assert not old_rev_dir.exists()

    migrated_dir = cold_root / "revisions" / "2026-10" / old_rev_id
    assert migrated_dir.is_dir()
    assert revisions._sha(migrated_dir / "dummy_archive.bin") == expected_dummy_sha
    assert (cold_root / "revisions" / "migration_ledger.jsonl").is_file()

    # Verify aged observations.sqlite3 row was exported to cold SQLite and removed from hot DB
    assert maint["observations_sqlite"]["exported_rows"] >= 1
    cold_db = Path(maint["observations_sqlite"]["cold_archive_db"])
    assert cold_db.is_file()
    with closing(sqlite3.connect(obs_path)) as conn:
        remaining_kinds = conn.execute("SELECT saved_at FROM snapshots").fetchall()
        assert remaining_kinds == [("2026-11-05T00:00:00+00:00",)]

    # Verify aged log line was exported and hot log only retains hot line
    assert maint["logs"]["archived_lines"] == 1
    hot_log_lines = (logs_dir / "monitor_audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(hot_log_lines) == 1
    assert json.loads(hot_log_lines[0])["event"] == "HOT_EVENT"

    # Active revision remains 100% COMMITTED
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"

    # Subsequent price revision after janitor compaction still commits cleanly and hardlinks the refreshed baseline!
    assert maint["observations_sqlite"]["baseline_refreshed"] is True
    changed2 = list(NEW)
    changed2[4], changed2[8] = "105.5", 33
    _add_receipt(runtime, "2026-11-06T02:05:05Z", changed2)
    _add_receipt(runtime, "2026-11-06T02:10:05Z", changed2)
    res2 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW + pd.Timedelta(minutes=30))
    assert res2["status"] == "COMMITTED"
    rev1_obs = Path(res1["archive_path"]) / "observations.sqlite3"
    rev2_obs = Path(res2["archive_path"]) / "observations.sqlite3"
    assert rev1_obs.stat().st_ino == rev2_obs.stat().st_ino


def test_v14r10_async_janitor_trading_yield_min_buy_policy_and_active_risk_status(tmp_path):
    """v14r10:
    1. storage_maintenance configures 15GB/35GB thresholds and [95, 210] idle window;
       _yield_for_trading_window pauses during [T-15s, T+25s]; trigger_async_if_due runs non-blocking.
    2. Account rejects price-skip receipts whose min_buy_price mismatches cohort policy.
    3. Account and allocation report NO_RISK_BUDGET (not DAILY_LOSS_LIMIT) when active_risk temporarily
       consumes remaining daily loss room while realized drawdown is still above the -5.5% limit.
    """
    from test_argon4_forward_execution import BOUNDARY, clock as make_clock, fixture_inputs, v5_policy
    from tools.argon4_forward import allocation
    from tools.argon4_forward.execution import PaperExecutor
    from tools.argon4_forward.storage_janitor import StorageJanitor

    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    sm = policy["storage_maintenance"]
    assert sm["max_hot_observations_mb"] == 15000
    assert sm["max_hot_total_gb"] == 35.0
    assert sm["allowed_window_seconds_Range"] == [95, 210]

    # 1. Verify _yield_for_trading_window pauses during [T-15s, T+25s] and returns immediately at T+120s
    janitor = StorageJanitor(tmp_path / "rt_yield", cold_root=tmp_path / "cold_yield")
    fake_clock = [290.0]  # T - 10s (inside [285, 300))
    slept_steps = []

    def fake_time():
        return fake_clock[0]

    def fake_sleep(dt):
        slept_steps.append(dt)
        fake_clock[0] += dt

    total_slept = janitor._yield_for_trading_window(active=True, time_fn=fake_time, sleep_fn=fake_sleep)
    assert total_slept > 35.0
    assert 25.0 < (fake_clock[0] % 300.0) < 285.0
    assert janitor._yield_for_trading_window(active=True, time_fn=lambda: 120.0, sleep_fn=fake_sleep) == 0.0

    # 2. Verify Account rejects price-skip receipt with forged min_buy_price mismatching cohort policy
    pol_sm = {
        **v5_policy(),
        "min_buy_price": 0.37,
        "stream_min_buy_price": {"BTC_15m": 0.42, "ETH_5m": 0.37},
    }
    ledger_sm = account.Account(
        tmp_path / "v14r10_min_buy.sqlite3", initial_capital=10000, started_at=BOUNDARY - pd.Timedelta(seconds=1)
    )
    executor = PaperExecutor(ledger_sm, pol_sm)
    decisions, markets = fixture_inputs(boundary=BOUNDARY)
    markets["BTC_15m"]["books"]["UP"]["asks"] = [{"price": "0.39", "size": "400"}, {"price": "0.48", "size": "500"}]
    now_ts = BOUNDARY + pd.Timedelta(seconds=1)
    res = executor.process_cohort(BOUNDARY.isoformat(), decisions, markets, now=now_ts, clock=make_clock(now_ts))
    assert res["stream_results"]["BTC_15m"]["status"] == "PRICE_LIMIT_NOT_MET"

    # Attempt to record a price skip on a new cohort with forged min_buy_price=0.45 when policy is 0.42
    b2 = BOUNDARY + pd.Timedelta(minutes=15)
    ledger_sm2 = account.Account(
        tmp_path / "v14r10_min_buy_forged.sqlite3", initial_capital=10000, started_at=b2 - pd.Timedelta(seconds=1)
    )
    ledger_sm2.ensure_cohort(b2.isoformat(), pol_sm, now=b2 + pd.Timedelta(seconds=1))
    with ledger_sm._connection() as con:
        valid_row = con.execute(
            "SELECT receipt_json FROM cohort_streams WHERE stream='BTC_15m'"
        ).fetchone()
    forged_receipt = json.loads(valid_row[0])
    forged_receipt["min_buy_price"] = "0.45"
    with pytest.raises(account.AccountRejected, match="price skip min_buy_price does not match cohort policy"):
        ledger_sm.record_cohort_skip(
            BOUNDARY.isoformat(), "BTC_15m", "PRICE_LIMIT_NOT_MET", forged_receipt, now=now_ts
        )

    # 3. Verify NO_RISK_BUDGET vs DAILY_LOSS_LIMIT when active_risk consumes remaining daily loss room
    day0 = account._ts("2026-11-08T12:00:00Z")
    ledger_ar = account.Account(
        tmp_path / "v14r10_active_risk.sqlite3", initial_capital=10000, started_at=day0 - pd.Timedelta(hours=2)
    )
    with ledger_ar._connection() as con:
        # Realized loss today is -$300 (-3.0% on $10,000, well above -$550 limit), leaving $250 realized room
        con.execute("UPDATE account SET cash_atoms = cash_atoms - 560000000, realized_atoms = -300000000 WHERE id=1")
        ledger_ar._event(
            con,
            "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER",
            {"market_id": "m-loss-300", "realized_pnl_usd": -300.0},
            day0 - pd.Timedelta(minutes=20),
        )
        # Insert an unexpired active order of $260 (expires at day0 + 10m)
        con.execute(
            "INSERT INTO signals VALUES('sig-ar','{}',?)",
            (account._iso(day0 - pd.Timedelta(minutes=4)),),
        )
        con.execute(
            "INSERT INTO orders VALUES('ord-ar','sig-ar','m-ar','BTC','UP','520',260000000,0,'OPEN',?,?,?)",
            (
                account._iso(day0 - pd.Timedelta(minutes=4)),
                account._iso(day0 + pd.Timedelta(minutes=10)),
                json.dumps({"stream": "BTC_15m"}),
            ),
        )
    c_ar = ledger_ar.ensure_cohort(day0.isoformat(), v5_policy(), now=day0 + pd.Timedelta(seconds=1))
    assert c_ar["budget_usd"] == 0.0
    assert c_ar["blocked_reason"] == "NO_RISK_BUDGET"

    plan_ar = allocation.plan_waterfall(
        v5_policy(),
        [],
        equity=9700,
        cash=9440,
        open_risk=260,
        now=day0 + pd.Timedelta(seconds=1),
        day_pnl=-300,
        day_start_equity=10000,
    )
    assert plan_ar["initial_tick_budget_usd"] == 0.0
    assert plan_ar["reasons"] == ["NO_RISK_BUDGET"]


def test_v14r11_us_open_blackout_macro_trend_guard_and_stream_min_buy_0425(tmp_path):
    """v14r11:
    1. Production policy configures stream_min_buy_price (0.425 for 5m streams, 0.37 for 15m streams),
       us_open_blackout ([870, 960), weekdays_only, skip ETH_5m/BTC_5m/BTC_15m),
       and macro_trend_guard (max_btc_4h_abs_return_pct=2.0, skip_1h_momentum_chase_above_pct=0.3).
    2. apply_volatility_regime_guard skips ETH_5m/BTC_5m/BTC_15m during weekday 14:30-16:00 UTC while
       leaving SOL_15m/SOL_5m unguarded and leaving weekend 14:30-16:00 UTC unguarded.
    3. apply_volatility_regime_guard skips BTC/ETH when |btc_ret_4h_pct| > 2.0% and skips any stream
       chasing 1h BTC momentum (> +0.3% buying UP or < -0.3% buying DOWN) while keeping mean-reverting signals.
    """
    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    assert policy["stream_min_buy_price"] == {
        "BTC_5m": 0.425,
        "ETH_5m": 0.425,
        "SOL_5m": 0.425,
        "BTC_15m": 0.415,
        "ETH_15m": 0.415,
        "SOL_15m": 0.415,
    }
    vg = policy["volatility_regime_guard"]
    assert vg["us_open_blackout"]["enabled"] is True
    assert vg["us_open_blackout"]["timezone"] == "America/New_York"
    assert vg["us_open_blackout"]["local_start_minute"] == 570
    assert vg["us_open_blackout"]["local_end_minute"] == 660
    assert vg["us_open_blackout"]["weekdays_only"] is True
    assert set(vg["us_open_blackout"]["skip_streams"]) == {"ETH_5m", "BTC_5m", "BTC_15m", "ETH_15m"}
    assert vg["macro_trend_guard"]["enabled"] is True
    assert vg["macro_trend_guard"]["max_btc_4h_abs_return_pct"] == pytest.approx(2.0)
    assert vg["macro_trend_guard"]["skip_1h_momentum_chase_above_pct"] == pytest.approx(0.3)

    base_pred = {"selected": True, "side": "DOWN", "z_v4": -0.52}

    # 1. Weekday 15:10 UTC (Monday 2026-11-09 15:10:00Z, minute=910): ETH_5m & BTC_5m skipped, SOL_5m allowed!
    res_eth_open = model_v14.apply_volatility_regime_guard(
        "ETH_5m",
        base_pred,
        ex_ante_rv12_bps=10.0,
        guard_config=vg,
        cohort_time="2026-11-09T15:10:00+00:00",
        btc_ret_1h_pct=0.45,
        btc_ret_4h_pct=0.80,
    )
    assert res_eth_open["selected"] is False
    assert res_eth_open["volatility_regime_reason"] == "US_OPEN_BLACKOUT"

    res_eth15_open = model_v14.apply_volatility_regime_guard(
        "ETH_15m",
        base_pred,
        ex_ante_rv12_bps=10.0,
        guard_config=vg,
        cohort_time="2026-11-10T14:45:00+00:00",
        btc_ret_1h_pct=0.45,
        btc_ret_4h_pct=0.80,
    )
    assert res_eth15_open["selected"] is False
    assert res_eth15_open["volatility_regime_reason"] == "US_OPEN_BLACKOUT"

    res_sol_open = model_v14.apply_volatility_regime_guard(
        "SOL_5m",
        base_pred,
        ex_ante_rv12_bps=10.0,
        guard_config=vg,
        cohort_time="2026-11-09T15:10:00+00:00",
        btc_ret_1h_pct=0.45,  # BTC +0.45% and buying DOWN = mean reversion (not momentum chase)
        btc_ret_4h_pct=0.80,
    )
    assert res_sol_open["selected"] is True

    # Weekend 15:10 UTC (Sunday 2026-11-08 15:10:00Z): ETH_5m is NOT blocked by US_OPEN_BLACKOUT
    res_eth_sun = model_v14.apply_volatility_regime_guard(
        "ETH_5m",
        base_pred,
        ex_ante_rv12_bps=10.0,
        guard_config=vg,
        cohort_time="2026-11-08T15:10:00+00:00",
        btc_ret_1h_pct=0.45,
        btc_ret_4h_pct=0.80,
    )
    assert res_eth_sun["selected"] is True

    # Weekday 16:00 UTC (minute=960, end of 90m blackout): ETH_5m resumes!
    res_eth_after = model_v14.apply_volatility_regime_guard(
        "ETH_5m",
        base_pred,
        ex_ante_rv12_bps=10.0,
        guard_config=vg,
        cohort_time="2026-11-09T16:00:00+00:00",
        btc_ret_1h_pct=0.45,
        btc_ret_4h_pct=0.80,
    )
    assert res_eth_after["selected"] is True

    # 2. Macro 4h breakout (|btc_ret_4h_pct| = 2.45% > 2.0%): skips BTC_15m/ETH_5m, allows SOL_15m
    res_btc_4h = model_v14.apply_volatility_regime_guard(
        "BTC_15m",
        {"selected": True, "side": "UP", "z_v4": 0.60},
        ex_ante_rv12_bps=12.0,
        guard_config=vg,
        cohort_time="2026-11-09T10:00:00+00:00",
        btc_ret_1h_pct=-0.50,
        btc_ret_4h_pct=-2.45,
    )
    assert res_btc_4h["selected"] is False
    assert res_btc_4h["volatility_regime_reason"] == "BTC_4H_MACRO_BREAKOUT"

    res_sol_4h = model_v14.apply_volatility_regime_guard(
        "SOL_15m",
        {"selected": True, "side": "UP", "z_v4": 0.60},
        ex_ante_rv12_bps=12.0,
        guard_config=vg,
        cohort_time="2026-11-09T10:00:00+00:00",
        btc_ret_1h_pct=-0.50,
        btc_ret_4h_pct=-2.45,
    )
    assert res_sol_4h["selected"] is True

    # 3. 1h momentum chase (btc_ret_1h_pct = +0.42% > +0.3% and side == "UP"): skipped!
    res_chase_up = model_v14.apply_volatility_regime_guard(
        "SOL_15m",
        {"selected": True, "side": "UP", "z_v4": 0.60},
        ex_ante_rv12_bps=12.0,
        guard_config=vg,
        cohort_time="2026-11-09T10:00:00+00:00",
        btc_ret_1h_pct=0.42,
        btc_ret_4h_pct=0.90,
    )
    assert res_chase_up["selected"] is False
    assert res_chase_up["volatility_regime_reason"] == "BTC_1H_MOMENTUM_CHASE"

    # Compute causal 1h/4h BTC returns from 48 closed 5m candles
    intraday = tmp_path / "macro_root" / "intraday"
    intraday.mkdir(parents=True)
    times48 = pd.date_range("2026-11-09T06:00:00Z", periods=48, freq="5min")
    opens = [100.0] * 48
    closes = [100.0] * 48
    opens[0] = 100.0
    opens[36] = 101.0  # 1h ago (bar -12) open = 101.0
    closes[-1] = 102.01  # close = 102.01 -> 4h ret = +2.01%, 1h ret = +1.0%
    df48 = pd.DataFrame({"date": times48, "open_time": times48, "open": opens, "close": closes})
    df48.to_parquet(intraday / "BTCUSDT_5m_futures.parquet", index=False)
    macro_rets = model_v14.compute_ex_ante_btc_macro_returns_pct(
        tmp_path / "macro_root", "BTCUSDT", as_of="2026-11-09T10:00:04Z"
    )
    assert macro_rets["btc_ret_1h_pct"] == pytest.approx(1.0)
    assert macro_rets["btc_ret_4h_pct"] == pytest.approx(2.01)


def test_v14r12_audit_remediations_bleed_cap_stream_max_order_1h_retry_and_fast_revision(tmp_path):
    """v14r12:
    1. Issue 1: 15m stream_min_buy_price raised to 0.415 (blocking #16 BTC_15m UP @ 0.380).
    2. Issue 2: single_order_per_window_on_2h_bleed marks bleed_single_order_window_cap=True when
       |btc_ret_2h_pct| >= 0.50% and |btc_ema_gap_bps| >= 15.0 bps against signal direction, and
       PaperExecutor enforces at most 1 order per window (skipping second signal as BLEED_WINDOW_SINGLE_ORDER_CAP).
    3. Issue 3 & 4: engineering profile supports 3.8s min decision delay, 16.5s input deadline, 0.25s poll,
       0.35s retry, 25s prefetch lead, 8s prefetch budget, and 18.0s/1.2s fast-path revision context.
    4. Issue 5: Argon4ForwardPipeline.retryable_latest_candle allows top-of-hour 1h futures/mark latest candle lag.
    5. Issue 6: _resolve_5m_frame reads from snapshot._raw in-memory via _read_validated without disk fallback.
    6. Issue 7: stream_max_order_usd caps SOL_5m/SOL_15m at $45 and Step 1 $200 capital risk fractions (0.08/0.16) pass.
    """
    from datetime import datetime, timezone
    import hashlib
    from tools.argon4_forward.account import Account, _policy_risk_limits
    from tools.argon4_forward.execution import PaperExecutor
    from tools.argon4_forward.pipeline import ForwardPipeline
    from tools.argon4_forward.runtime_profile import engineering_profile

    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    assert policy["min_buy_price"] == pytest.approx(0.415)
    assert policy["stream_min_buy_price"]["BTC_15m"] == pytest.approx(0.415)
    assert policy["stream_min_buy_price"]["SOL_15m"] == pytest.approx(0.415)
    assert policy["stream_max_order_usd"] == {
        "BTC_15m": 240,
        "BTC_5m": 220,
        "ETH_15m": 180,
        "ETH_5m": 180,
        "SOL_15m": 50,
        "SOL_5m": 45,
    }
    eng = engineering_profile(policy)
    assert eng["enabled"] is True
    assert eng["window_poll_interval_seconds"] == pytest.approx(0.25)
    assert eng["input_deadline_after_boundary_seconds"] == pytest.approx(16.5)
    assert eng["input_ready_retry_interval_seconds"] == pytest.approx(0.35)
    assert eng["prefetch_lead_seconds"] == pytest.approx(25.0)
    assert eng["prefetch_budget_seconds"] == pytest.approx(8.0)

    # Issue 7: Step 1 ($200 capital, 0.08 single / 0.16 tick / 0.16 daily_loss) passes _policy_risk_limits
    step1_policy = {
        **policy,
        "initial_capital_usd": 200,
        "single_stream_equity_fraction": 0.08,
        "tick_equity_fraction": 0.16,
        "max_tick_budget_usd": 32,
        "max_order_usd": 16,
        "max_active_window_risk_usd": 48,
        "max_open_risk_usd": 64,
        "max_underlying_active_risk_usd": 32,
        "max_underlying_open_risk_usd": 48,
        "daily_loss_fraction": 0.16,
    }
    step1_limits = _policy_risk_limits(step1_policy)
    assert float(step1_limits["single_stream_equity_fraction"]) == pytest.approx(0.08)
    assert float(step1_limits["tick_equity_fraction"]) == pytest.approx(0.16)
    assert float(step1_limits["stream_max_order_usd"]["SOL_5m"]) == pytest.approx(45.0)

    # Issue 5: retryable_latest_candle allows 1h futures/mark lag at top-of-hour (minute == 0), rejects at minute != 0
    inputs_1h = {
        "status": "BLOCKED",
        "errors": [],
        "gaps": [
            {"source": "BTCUSDT:futures:5m", "error": "LATEST_CLOSED_BAR_MISSING"},
            {"source": "BTCUSDT:futures:1h", "error": "LATEST_CLOSED_BAR_MISSING"},
            {"source": "ETHUSDT:mark:1h", "error": "LATEST_CLOSED_BAR_MISSING"},
        ],
        "sources": {
            "BTCUSDT:futures:5m": {
                "latest_closed": "2026-11-10T03:55:00+00:00",
                "rows": 4500,
                "issues": ["LATEST_CLOSED_BAR_MISSING"],
            },
            "BTCUSDT:futures:1h": {
                "latest_closed": "2026-11-10T03:00:00+00:00",
                "rows": 400,
                "issues": ["LATEST_CLOSED_BAR_MISSING"],
            },
            "ETHUSDT:mark:1h": {
                "latest_closed": "2026-11-10T03:00:00+00:00",
                "rows": 400,
                "issues": ["LATEST_CLOSED_BAR_MISSING"],
            },
        },
    }
    assert ForwardPipeline.retryable_latest_candle(
        inputs_1h, datetime(2026, 11, 10, 4, 0, tzinfo=timezone.utc)
    ) is True
    assert ForwardPipeline.retryable_latest_candle(
        inputs_1h, datetime(2026, 11, 10, 4, 5, tzinfo=timezone.utc)
    ) is False

    # Issue 6: _resolve_5m_frame uses snapshot._raw + snapshot._read_validated without disk fallback
    class DummySnapshot:
        def __init__(self, frame):
            self._validated = {}
            self._raw = {Path("intraday/BTCUSDT_5m_futures.parquet"): b"in_memory_bytes"}
            self._frame = frame
            self.read_calls = 0

        def _read_validated(self, key):
            assert getattr(key, "name", "") == "BTCUSDT_5m_futures.parquet"
            self.read_calls += 1
            return self._frame, "sha", 0

    times48 = pd.date_range("2026-11-10T00:00:00Z", periods=48, freq="5min")
    # Construct a 2h downward bleed: open 2h ago = 100.0, close = 99.30 (-0.70%), EMA12 < EMA36 by > 15 bps
    prices = [100.0] * 24 + [100.0 - 0.03 * (i + 1) for i in range(24)]
    df_bleed = pd.DataFrame({"date": times48, "open_time": times48, "open": prices, "close": prices})
    snap = DummySnapshot(df_bleed)
    macro_bleed = model_v14.compute_ex_ante_btc_macro_returns_pct(
        tmp_path / "nonexistent_dir", "BTCUSDT", as_of="2026-11-10T04:00:04Z", snapshot=snap
    )
    assert snap.read_calls == 1
    assert macro_bleed["btc_ret_2h_pct"] < -0.50
    assert macro_bleed["btc_ema_gap_bps"] < -15.0

    # Issue 2: apply_volatility_regime_guard marks bleed_single_order_window_cap=True on counter-trend UP during 2h bleed
    pred_up = model_v14.apply_volatility_regime_guard(
        "ETH_5m",
        {"selected": True, "side": "UP", "z_v4": 0.55},
        ex_ante_rv12_bps=10.0,
        guard_config=policy["volatility_regime_guard"],
        cohort_time="2026-11-10T03:40:00+00:00",
        btc_ret_1h_pct=-0.35,
        btc_ret_2h_pct=macro_bleed["btc_ret_2h_pct"],
        btc_ret_4h_pct=-0.85,
        btc_ema_gap_bps=macro_bleed["btc_ema_gap_bps"],
    )
    assert pred_up["selected"] is True
    assert pred_up.get("bleed_single_order_window_cap") is True

    # Verify PaperExecutor fills Rank 1 (BTC_15m UP) and skips Rank 2 (ETH_5m UP) via BLEED_WINDOW_SINGLE_ORDER_CAP
    from test_argon4_forward_execution import BOUNDARY, clock as make_clock, fixture_inputs, v5_policy

    bleed_policy = {
        **v5_policy(),
        "min_buy_price": 0.415,
        "stream_min_buy_price": policy["stream_min_buy_price"],
        "stream_max_order_usd": policy["stream_max_order_usd"],
        "execution": policy["execution"],
    }
    ledger = Account(tmp_path / "bleed_account.sqlite3", initial_capital=10000, started_at=BOUNDARY - pd.Timedelta(seconds=1))
    exec_paper = PaperExecutor(ledger, bleed_policy)
    decisions, markets = fixture_inputs(boundary=BOUNDARY)
    for st_name in decisions:
        if decisions.get(st_name, {}).get("status") == "SIGNAL":
            decisions[st_name]["signal"]["bleed_single_order_window_cap"] = True
            decisions[st_name]["signal"]["max_orders_in_window"] = 1
    now_ts = BOUNDARY + pd.Timedelta(seconds=1)
    res_exec = exec_paper.process_cohort(BOUNDARY.isoformat(), decisions, markets, now=now_ts, clock=make_clock(now_ts))
    assert res_exec["status"] == "PROCESSED", res_exec
    assert len(res_exec["fills"]) == 1
    assert res_exec["stream_results"]["BTC_15m"]["status"] == "FILLED_BUDGET"
    assert res_exec["stream_results"]["ETH_5m"]["status"] == "BLEED_WINDOW_SINGLE_ORDER_CAP"


def test_v14r13_tiered_delta_compaction_and_same_cohort_post_completion_clock_refresh(tmp_path):
    """v14r13:
    1. _write_observations_delta preserves >=16MB anchor delta segments in prior_chain without re-copying them,
       compacting only trailing small delta segments once len(trailing_chain) >= 8.
    2. _same_cohort_view_admitted rejects clock changes while execution/recheck is pending, but preserves
       same-cohort admission after cohort execution has completed so routine clock refreshes later in the
       5m window do not overwrite PAPER_CYCLE_COMPLETE with REVISION_WAITING_FUTURE_WINDOW.
    """
    from contextlib import closing
    from datetime import datetime, timezone
    from tools.argon4_forward.pipeline import ForwardPipeline

    # 1. Verify tiered delta compaction preserves >=16MB anchor delta segment
    base_snap = tmp_path / "base_obs.sqlite3"
    live_src = tmp_path / "live_obs.sqlite3"
    anchor_delta = tmp_path / "anchor_delta.sqlite3"
    with closing(sqlite3.connect(base_snap)) as conn, conn:
        conn.execute("CREATE TABLE snapshots(id INTEGER PRIMARY KEY, kind TEXT, saved_at TEXT, payload TEXT)")
        conn.execute("INSERT INTO snapshots VALUES(1, 'init', '2026-11-10T00:00:00Z', 'row1')")
    with closing(sqlite3.connect(live_src)) as conn, conn:
        conn.execute("CREATE TABLE snapshots(id INTEGER PRIMARY KEY, kind TEXT, saved_at TEXT, payload TEXT)")
        for i in range(1, 12):
            conn.execute("INSERT INTO snapshots VALUES(?, 'k', '2026-11-10T00:00:00Z', ?)", (i, f"row_{i}"))

    # Create a >=16MB anchor delta file covering up to id=2
    anchor_delta.write_bytes(b"\x00" * (16 * 1024 * 1024 + 1024))
    chain_with_7_small = [
        {"delta_snapshot_path": str(anchor_delta), "delta_sha256": "a" * 64, "delta_max_id": {"snapshots": 2}},
        *[
            {
                "delta_snapshot_path": str(tmp_path / f"small_{idx}.sqlite3"),
                "delta_sha256": f"{idx}" * 64,
                "delta_max_id": {"snapshots": 2 + idx},
            }
            for idx in range(1, 8)
        ],
    ]
    # Total chain length is 8 (1 anchor + 7 small), so trailing_chain has 7 (<8): must NOT compact yet
    delta_out_1 = tmp_path / "new_delta_1.sqlite3"
    info_1 = revisions._write_observations_delta(
        live_src, base_snap, delta_out_1, prior_covered_max={"snapshots": 9}, prior_chain=chain_with_7_small
    )
    assert len(info_1["delta_chain"]) == 9
    with closing(sqlite3.connect(delta_out_1)) as conn:
        assert [r[0] for r in conn.execute("SELECT id FROM snapshots ORDER BY id")] == [10, 11]

    # Now trailing_chain has 8 small segments after the >=16MB anchor: compacts ONLY ids > 2 (never id=2 from anchor)
    chain_with_8_small = info_1["delta_chain"]
    with closing(sqlite3.connect(live_src)) as conn, conn:
        conn.execute("INSERT INTO snapshots VALUES(12, 'k', '2026-11-10T00:05:00Z', 'row_12')")
    delta_out_2 = tmp_path / "new_delta_2.sqlite3"
    info_2 = revisions._write_observations_delta(
        live_src, base_snap, delta_out_2, prior_covered_max={"snapshots": 11}, prior_chain=chain_with_8_small
    )
    assert len(info_2["delta_chain"]) == 2
    assert info_2["delta_chain"][0]["delta_snapshot_path"] == str(anchor_delta)
    with closing(sqlite3.connect(delta_out_2)) as conn:
        assert [r[0] for r in conn.execute("SELECT id FROM snapshots ORDER BY id")] == list(range(3, 13))

    # 2. Verify _same_cohort_view_admitted before vs after cohort completion when clock refreshes
    from tools.argon4_forward.clock_sync import calibrated_now
    from tools.argon4_forward.pipeline import cohort_at

    clock_1 = {
        "status": "CALIBRATED",
        "server_minus_local_seconds": 0.0,
        "uncertainty_seconds": 0.05,
        "calibrated_at_local": datetime.now(timezone.utc).isoformat(),
        "calibrated_monotonic": time.monotonic(),
        "receipt_sha256": "1" * 64,
    }
    cohort_dt = cohort_at(calibrated_now(clock_1, upper=False))
    clock_2 = {**clock_1, "receipt_sha256": "2" * 64}
    view = {"revision_id": "rev_1040", "manifest_sha256": "m" * 64}

    dummy = ForwardPipeline.__new__(ForwardPipeline)
    dummy.last_cohort = cohort_dt
    dummy.clock = clock_1
    dummy._pending_execution = {"cohort": cohort_dt}
    dummy._same_cohort_revision = {
        "cohort": cohort_dt,
        "revision_id": "rev_1040",
        "manifest_sha256": "m" * 64,
        "clock": clock_1,
    }
    assert dummy._same_cohort_view_admitted(view) is True

    # While _pending_execution is active (not completed), changing clock must reject admission
    dummy.clock = clock_2
    assert dummy._same_cohort_view_admitted(view) is False

    # Once cohort execution is completed (_pending_execution is None and completed=True),
    # a routine clock refresh later in the same 5m cohort preserves admission!
    dummy._pending_execution = None
    dummy._same_cohort_revision["completed"] = True
    assert dummy._same_cohort_view_admitted(view) is True


def test_v14r14_own_asset_guard_min_depth_recheck_and_tiered_buy_price(tmp_path):
    """v14r14 -> v14r15:
    1. Optimization 1: ETH_15m included in us_open_blackout.skip_streams.
    2. Production v14r15 policy disables confidence_tiered_max_buy_price (preventing reverse adverse selection)
       and tunes own_asset_crash_guard to 631-day verified black-swan thresholds (BTC 3.0%/200bp, ETH 3.5%/220bp, SOL 4.0%/250bp),
       while offline unit config still verifies tight waterfall & tiered-cap branches in apply_volatility_regime_guard.
    3. Optimization 3: min_executable_depth_usd ($25.00) rejects a $3.57 dust book on attempt=0
       as WAITING_PRICE_RECHECK instead of eating $3.57 and marking FILLED_BUDGET, then fills
       the full target budget on attempt=1 when depth replenishes.
    """
    from copy import deepcopy
    from datetime import timedelta
    from tools.argon4_forward.account import Account
    from tools.argon4_forward.execution import PaperExecutor
    from test_argon4_forward_execution import BOUNDARY, ORDER, clock as make_clock, fixture_inputs, v5_policy

    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    vg = policy["volatility_regime_guard"]
    assert "ETH_15m" in vg["us_open_blackout"]["skip_streams"]
    assert vg["own_asset_crash_guard"]["enabled"] is True
    assert vg["own_asset_crash_guard"]["max_own_1h_counter_return_pct"] == {
        "BTC_15m": 3.0,
        "BTC_5m": 3.0,
        "ETH_15m": 3.5,
        "ETH_5m": 3.5,
        "SOL_15m": 4.0,
        "SOL_5m": 4.0,
    }
    assert vg["own_asset_crash_guard"]["max_own_15m_counter_return_bps"] == {
        "BTC_15m": 200.0,
        "BTC_5m": 200.0,
        "ETH_15m": 220.0,
        "ETH_5m": 220.0,
        "SOL_15m": 250.0,
        "SOL_5m": 250.0,
    }
    assert vg["confidence_tiered_max_buy_price"]["enabled"] is False
    assert policy["execution"]["min_executable_depth_usd"] == pytest.approx(25.0)
    assert policy["execution"]["final_look_min_executable_depth_usd"] == pytest.approx(8.0)

    sol_pred = {
        "selected": True,
        "side": "UP",
        "z_v4": 1.5944,
        "threshold_frozen": 1.5204,
    }
    # Under production v14r15 black-swan thresholds, normal oversold bounce (-2.18% 1h / -163.2 bps 15m) is preserved
    res_sol_prod = model_v14.apply_volatility_regime_guard(
        "SOL_5m",
        sol_pred,
        ex_ante_rv12_bps=18.2,
        guard_config=vg,
        cohort_time="2026-11-10T15:40:00+00:00",
        btc_ret_1h_pct=-0.41,
        btc_ret_4h_pct=-1.50,
        own_ret_1h_pct=-2.18,
        own_ret_15m_bps=-163.2,
        own_streak_3bar=-3,
    )
    assert res_sol_prod["selected"] is True
    assert "effective_max_buy_price" not in res_sol_prod

    # True black-swan liquidation waterfall on SOL (-4.25% 1h or -260 bps 15m) is blocked by production vg
    res_sol_blackswan = model_v14.apply_volatility_regime_guard(
        "SOL_5m",
        sol_pred,
        ex_ante_rv12_bps=18.2,
        guard_config=vg,
        cohort_time="2026-11-10T15:40:00+00:00",
        btc_ret_1h_pct=-0.41,
        btc_ret_4h_pct=-1.50,
        own_ret_1h_pct=-4.25,
        own_ret_15m_bps=-260.0,
        own_streak_3bar=-3,
    )
    assert res_sol_blackswan["selected"] is False
    assert res_sol_blackswan["volatility_regime_reason"] == "OWN_ASSET_1H_WATERFALL_GUARD"

    # Explicit unit guard dictionary to test tight waterfall & confidence-tiered cap branches offline
    vg_unit = deepcopy(vg)
    vg_unit["own_asset_crash_guard"]["max_own_1h_counter_return_pct"]["SOL_5m"] = 1.8
    vg_unit["own_asset_crash_guard"]["max_own_15m_counter_return_bps"]["SOL_5m"] = 135.0
    vg_unit["confidence_tiered_max_buy_price"]["enabled"] = True

    res_sol_crash = model_v14.apply_volatility_regime_guard(
        "SOL_5m",
        sol_pred,
        ex_ante_rv12_bps=18.2,
        guard_config=vg_unit,
        cohort_time="2026-11-10T15:40:00+00:00",
        btc_ret_1h_pct=-0.41,
        btc_ret_4h_pct=-1.50,
        own_ret_1h_pct=-2.18,
        own_ret_15m_bps=-163.2,
        own_streak_3bar=-3,
    )
    assert res_sol_crash["selected"] is False
    assert res_sol_crash["volatility_regime_reason"] == "OWN_ASSET_1H_WATERFALL_GUARD"

    res_sol_15m_crash = model_v14.apply_volatility_regime_guard(
        "SOL_5m",
        sol_pred,
        ex_ante_rv12_bps=18.2,
        guard_config=vg_unit,
        cohort_time="2026-11-10T15:40:00+00:00",
        btc_ret_1h_pct=-0.41,
        btc_ret_4h_pct=-1.50,
        own_ret_1h_pct=-1.10,
        own_ret_15m_bps=-145.0,
        own_streak_3bar=-3,
    )
    assert res_sol_15m_crash["selected"] is False
    assert res_sol_15m_crash["volatility_regime_reason"] == "OWN_ASSET_15M_WATERFALL_GUARD"

    # Verify compute_ex_ante_symbol_returns on 12 closed 5m bars with a 3-bar positive streak
    times12 = pd.date_range("2026-11-10T17:00:00Z", periods=12, freq="5min")
    opens12 = [2000.0] * 9 + [2000.0, 2004.0, 2008.0]
    closes12 = [2000.0] * 9 + [2004.0, 2008.0, 2012.0]  # 3 positive bars, +60 bps over 15m, +0.60% over 1h
    df12 = pd.DataFrame({"date": times12, "open_time": times12, "open": opens12, "close": closes12}, index=times12)
    sym_rets = model_v14.compute_ex_ante_symbol_returns(raw_frame=df12)
    assert sym_rets["own_ret_15m_bps"] == pytest.approx(60.0)
    assert sym_rets["own_ret_1h_pct"] == pytest.approx(0.60)
    assert sym_rets["own_streak_3bar"] == 3

    eth_marginal = {
        "selected": True,
        "side": "DOWN",
        "z_v4": -1.8097,
        "threshold_frozen": 1.7207,
    }
    res_eth5_marginal = model_v14.apply_volatility_regime_guard(
        "ETH_5m",
        eth_marginal,
        ex_ante_rv12_bps=12.4,
        guard_config=vg_unit,
        cohort_time="2026-11-10T17:55:00+00:00",
        btc_ret_1h_pct=0.25,
        btc_ret_4h_pct=0.40,
        own_ret_1h_pct=0.53,
        own_ret_15m_bps=56.6,
        own_streak_3bar=3,
    )
    assert res_eth5_marginal["selected"] is True
    assert res_eth5_marginal["effective_max_buy_price"] == pytest.approx(0.475)
    assert res_eth5_marginal["price_tier_reason"] == "MARGINAL_AGAINST_15M_STREAK"

    eth_strong = {
        "selected": True,
        "side": "DOWN",
        "z_v4": -2.5071,
        "threshold_frozen": 1.7207,
    }
    res_eth4_strong = model_v14.apply_volatility_regime_guard(
        "ETH_5m",
        eth_strong,
        ex_ante_rv12_bps=11.0,
        guard_config=vg_unit,
        cohort_time="2026-11-10T17:35:00+00:00",
        btc_ret_1h_pct=0.10,
        btc_ret_4h_pct=0.20,
        own_ret_1h_pct=0.20,
        own_ret_15m_bps=25.0,
        own_streak_3bar=1,
    )
    assert res_eth4_strong["selected"] is True
    assert res_eth4_strong["effective_max_buy_price"] == pytest.approx(0.515)
    assert res_eth4_strong["price_tier_reason"] == "STRONG_SIGNAL_CAP"

    # 3. Verify PaperExecutor rejects $3.69 dust book on attempt=0 (WAITING_PRICE_RECHECK) and fills full budget on attempt=1
    exec_policy = {
        **v5_policy(),
        "version": "v14r14-min-depth-test",
        "single_stream_equity_fraction": policy["single_stream_equity_fraction"],
        "tick_equity_fraction": policy["tick_equity_fraction"],
        "min_buy_price": 0.415,
        "stream_min_buy_price": policy["stream_min_buy_price"],
        "stream_max_order_usd": policy["stream_max_order_usd"],
        "engineering": {"profile": "argon4-latency-v12"},
        "execution": deepcopy(policy["execution"]),
    }
    ledger = Account(tmp_path / "min_depth.sqlite3", initial_capital=1269, started_at=BOUNDARY - timedelta(seconds=1))
    executor = PaperExecutor(ledger, exec_policy)
    decisions, markets = fixture_inputs(capacity=200, boundary=BOUNDARY)
    for d_val in decisions.values():
        if d_val.get("signal"):
            d_val["signal"]["valid_until"] = (BOUNDARY + timedelta(seconds=20)).isoformat()

    now_cursor = BOUNDARY + timedelta(seconds=9)
    # On attempt 0, Rank 1 (BTC_15m) only has 7 shares @ 0.51 ($3.57 < $25.00 min_executable_depth_usd)
    attempt_sizes = [7, 400]

    def book_refresh(market, side):
        nonlocal now_cursor
        now_cursor += timedelta(seconds=0.05)
        b = deepcopy(market["books"][side])
        b.update(source_at=now_cursor.isoformat(), received_at=now_cursor.isoformat())
        if market["market_id"] == markets[ORDER[0]]["market_id"]:
            sz = attempt_sizes.pop(0) if attempt_sizes else 400
            b["asks"] = [{"price": 0.51, "size": sz}]
        else:
            b["asks"] = [{"price": 0.48, "size": 400}]
        return b

    res_look0 = executor.process_cohort(
        BOUNDARY, decisions, markets, now=lambda: now_cursor, clock=make_clock(BOUNDARY), book_refresh=book_refresh
    )
    assert res_look0["status"] == "WAITING_PRICE_RECHECK"
    assert res_look0["fills"] == []
    assert res_look0["stream_results"][ORDER[0]]["observed_executable_depth_usd"] == pytest.approx(3.57)
    assert res_look0["stream_results"][ORDER[0]]["min_executable_depth_usd"] == pytest.approx(25.0)

    # Advance to next_retry_at (+2.5s): now book has 400 shares @ 0.51 ($204 >= $25), so Rank 1 fills full budget!
    now_cursor = pd.Timestamp(res_look0["next_retry_at"]).to_pydatetime()
    res_look1 = executor.process_cohort(
        BOUNDARY, decisions, markets, now=lambda: now_cursor, clock=make_clock(BOUNDARY), book_refresh=book_refresh
    )
    assert res_look1["status"] == "PROCESSED"
    assert len(res_look1["fills"]) >= 1
    assert res_look1["fills"][0]["underlying"] == "BTC"
    assert res_look1["fills"][0]["cost_usd"] > 15.0


def test_update_clock_recovers_fast_after_uncertainty_too_large_and_monitor_logs_v14r14_guards(tmp_path, monkeypatch):
    """Verify:
    1. ForwardPipeline.update_clock() does NOT stay locked out for 40 seconds after
       calibrate() returns UNCERTAINTY_TOO_LARGE — it respects a 1.0s anti-spam cooldown
       and immediately re-calibrates on the next >=1.0s worker tick.
    2. MonitorAuditLogger.log_cohort_execution emits THIN_BOOK_DEPTH_GUARD and
       CONFIDENCE_TIERED_PRICE_CAP with exact numbers.
    """
    from types import SimpleNamespace
    from tools.argon4_forward import pipeline as pl_mod
    from tools.argon4_forward.monitor_logger import MonitorAuditLogger

    mono = {"now": 200.0}
    calls = []
    saved = []
    p = pl_mod.ForwardPipeline(
        None,
        tmp_path,
        SimpleNamespace(save=lambda kind, data: saved.append((kind, data))),
    )
    p._revision_fastpath = {"enabled": True}
    monkeypatch.setattr(pl_mod.time, "monotonic", lambda: mono["now"])

    def fake_calibrate(*, transport, **kwargs):
        calls.append((mono["now"], kwargs))
        if len(calls) == 1:
            return {"status": "UNCERTAINTY_TOO_LARGE", "uncertainty_seconds": 0.56, "calibrated_monotonic": mono["now"]}
        return {"status": "CALIBRATED", "uncertainty_seconds": 0.12, "calibrated_monotonic": mono["now"]}

    monkeypatch.setattr(pl_mod, "calibrate", fake_calibrate)
    try:
        with pytest.raises(ValueError, match="clock uncertainty"):
            p.update_clock()
        assert len(calls) == 1

        # Within < 1.0s (e.g. 0.4s later), does not spam /time
        mono["now"] = 200.4
        with pytest.raises(ValueError, match="clock uncertainty"):
            p.update_clock()
        assert len(calls) == 1

        # At >= 1.0s (e.g. 1.5s worker tick), immediately re-calibrates instead of waiting 40s!
        mono["now"] = 201.5
        recovered = p.update_clock()
        assert len(calls) == 2
        assert recovered["status"] == "CALIBRATED"
        assert recovered["refresh_reason"] == "UNCALIBRATED_RETRY"
    finally:
        p.close()

    # Verify MonitorAuditLogger tags THIN_BOOK_DEPTH_GUARD and CONFIDENCE_TIERED_PRICE_CAP
    mlog = MonitorAuditLogger(tmp_path / "rt")
    rec = mlog.log_cohort_execution(
        "2026-11-11T00:05:00+00:00",
        {"order": ["BTC_15m", "ETH_5m"]},
        {
            "BTC_15m": {"status": "SIGNAL", "outcome": "UP", "prediction": {"z_v4": 1.95, "threshold_frozen": 1.8, "side": "UP"}},
            "ETH_5m": {"status": "SIGNAL", "outcome": "DOWN", "prediction": {"z_v4": -1.78, "threshold_frozen": 1.72, "side": "DOWN"}},
        },
        {
            "status": "PROCESSED",
            "fills": [],
            "stream_results": {
                "BTC_15m": {
                    "status": "PRICE_LIMIT_NOT_MET",
                    "best_ask": 0.48,
                    "limit_price": 0.515,
                    "min_buy_price": 0.415,
                    "min_executable_depth_usd": 25.0,
                    "observed_executable_depth_usd": 3.57,
                },
                "ETH_5m": {
                    "status": "PRICE_LIMIT_NOT_MET",
                    "best_ask": 0.504,
                    "limit_price": 0.49,
                    "min_buy_price": 0.425,
                },
            },
        },
        {"duration_seconds": 4.8},
    )
    streams_info = rec["details"]["streams"]
    assert "INSUFFICIENT_EXECUTABLE_DEPTH" in streams_info["BTC_15m"]["no_trade_reason"]
    assert "CONFIDENCE_TIERED_PRICE_CAP" in streams_info["ETH_5m"]["no_trade_reason"]


def test_v14r15_final_look_depth_fill_low_cap_no_starve_and_capital_sync(tmp_path):
    """v14r15:
    1. Final-look depth floor (`final_look_min_executable_depth_usd = 8.0`):
       When no rechecks remain (`can_retry_depth == False`, e.g., at T+16.5s where T+16.5s + 2.5s > T+18.5s),
       an $18.50 book (>= $8.00, < $25.00) fills profitably (`FILLED_BUDGET`) and unlocks remaining budget
       for Rank 2 (`ETH_5m`), whereas a < $8.00 dust book ($3.57) still skips cleanly (`PRICE_LIMIT_NOT_MET`).
    2. Custom low `limit_price` (< 0.500) facing normal `best_ask` (0.505 > 0.490) skips immediately on
       Attempt 0 (`next_retry_at is None`) without starving Rank 2 (`ETH_5m`) for 7.5s.
    3. `Account.sync_initial_capital` explicitly adjusts `initial_atoms` and `cash_atoms` with a
       `CAPITAL_ADJUSTED` audit event while normal `Account(...)` restarts never reset capital.
    """
    from copy import deepcopy
    from datetime import timedelta
    from tools.argon4_forward.account import Account
    from tools.argon4_forward.execution import PaperExecutor
    from tools.argon4_forward.monitor_logger import MonitorAuditLogger
    from test_argon4_forward_execution import BOUNDARY, ORDER, clock as make_clock, fixture_inputs, v5_policy

    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    exec_policy = {
        **v5_policy(),
        "version": "v14r15-final-look-and-low-cap-test",
        "single_stream_equity_fraction": policy["single_stream_equity_fraction"],
        "tick_equity_fraction": policy["tick_equity_fraction"],
        "min_buy_price": 0.415,
        "stream_min_buy_price": policy["stream_min_buy_price"],
        "stream_max_order_usd": policy["stream_max_order_usd"],
        "engineering": {"profile": "argon4-latency-v12"},
        "execution": deepcopy(policy["execution"]),
    }

    # --- 1A. Final look ($18.50 >= $8.00) fills Rank 1 AND unlocks Rank 2 in the same cohort ---
    ledger_final = Account(tmp_path / "final_look.sqlite3", initial_capital=1269, started_at=BOUNDARY - timedelta(seconds=1))
    executor_final = PaperExecutor(ledger_final, exec_policy)
    decisions_a, markets_a = fixture_inputs(capacity=200, boundary=BOUNDARY)
    for d_val in decisions_a.values():
        if d_val.get("signal"):
            d_val["signal"]["valid_until"] = (BOUNDARY + timedelta(seconds=20)).isoformat()

    # T+16.5s => proposed = T+19.0s > T+18.5s => can_retry_depth is False (final look!)
    now_a = BOUNDARY + timedelta(seconds=16.5)

    def book_refresh_a(market, side):
        nonlocal now_a
        now_a += timedelta(seconds=0.05)
        b = deepcopy(market["books"][side])
        b.update(source_at=now_a.isoformat(), received_at=now_a.isoformat())
        if market["market_id"] == markets_a[ORDER[0]]["market_id"]:
            # 37 shares @ 0.50 = $18.50 (>= $8.00 final floor, < $25.00 initial floor)
            b["asks"] = [{"price": 0.50, "size": 37}]
        else:
            b["asks"] = [{"price": 0.48, "size": 400}]
        return b

    res_final_fill = executor_final.process_cohort(
        BOUNDARY, decisions_a, markets_a, now=lambda: now_a, clock=make_clock(BOUNDARY), book_refresh=book_refresh_a
    )
    assert res_final_fill["status"] == "PROCESSED"
    assert res_final_fill["stream_results"][ORDER[0]]["status"] == "FILLED_BUDGET"
    assert res_final_fill["stream_results"][ORDER[0]]["spent_usd"] == pytest.approx(18.50, abs=0.5)
    # Remaining budget unlocked for Rank 2!
    assert len(res_final_fill["fills"]) >= 2
    assert res_final_fill["stream_results"][ORDER[1]]["status"] == "FILLED_BUDGET"
    assert res_final_fill["fills"][0]["market_id"] == markets_a[ORDER[0]]["market_id"]
    assert res_final_fill["fills"][1]["market_id"] == markets_a[ORDER[1]]["market_id"]

    # --- 1B. Final look dust (< $8.00, e.g. $3.57) still skips cleanly and unlocks Rank 2 ---
    b2 = BOUNDARY + timedelta(minutes=15)
    decisions_b, markets_b = fixture_inputs(capacity=200, boundary=b2)
    for d_val in decisions_b.values():
        if d_val.get("signal"):
            d_val["signal"]["valid_until"] = (b2 + timedelta(seconds=20)).isoformat()
    now_b = b2 + timedelta(seconds=16.5)

    def book_refresh_b(market, side):
        nonlocal now_b
        now_b += timedelta(seconds=0.05)
        b = deepcopy(market["books"][side])
        b.update(source_at=now_b.isoformat(), received_at=now_b.isoformat())
        if market["market_id"] == markets_b[ORDER[0]]["market_id"]:
            b["asks"] = [{"price": 0.51, "size": 7}]  # $3.57 < $8.00
        else:
            b["asks"] = [{"price": 0.48, "size": 400}]
        return b

    res_final_skip = executor_final.process_cohort(
        b2, decisions_b, markets_b, now=lambda: now_b, clock=make_clock(b2), book_refresh=book_refresh_b
    )
    assert res_final_skip["status"] == "PROCESSED"
    assert res_final_skip["stream_results"][ORDER[0]]["status"] == "PRICE_LIMIT_NOT_MET"
    assert res_final_skip["stream_results"][ORDER[0]]["min_executable_depth_usd"] == pytest.approx(8.0)
    assert len(res_final_skip["fills"]) >= 1
    assert res_final_skip["stream_results"][ORDER[1]]["status"] == "FILLED_BUDGET"
    assert res_final_skip["fills"][0]["market_id"] == markets_b[ORDER[1]]["market_id"]

    # --- 2. Custom low cap (< 0.500) at ask=0.505 skips on Attempt 0 without starving Rank 2 ---
    ledger_low_cap = Account(tmp_path / "low_cap.sqlite3", initial_capital=1269, started_at=BOUNDARY - timedelta(seconds=1))
    executor_low_cap = PaperExecutor(ledger_low_cap, exec_policy)
    b3 = BOUNDARY + timedelta(minutes=30)
    decisions_c, markets_c = fixture_inputs(capacity=200, boundary=b3)
    for d_val in decisions_c.values():
        if d_val.get("signal"):
            d_val["signal"]["valid_until"] = (b3 + timedelta(seconds=20)).isoformat()
    decisions_c[ORDER[0]]["effective_max_buy_price"] = 0.490
    now_c = b3 + timedelta(seconds=9.0)

    def book_refresh_c(market, side):
        nonlocal now_c
        now_c += timedelta(seconds=0.05)
        b = deepcopy(market["books"][side])
        b.update(source_at=now_c.isoformat(), received_at=now_c.isoformat())
        if market["market_id"] == markets_c[ORDER[0]]["market_id"]:
            b["asks"] = [{"price": 0.505, "size": 400}]
        else:
            b["asks"] = [{"price": 0.48, "size": 400}]
        return b

    res_low_cap = executor_low_cap.process_cohort(
        b3, decisions_c, markets_c, now=lambda: now_c, clock=make_clock(b3), book_refresh=book_refresh_c
    )
    assert res_low_cap["status"] == "PROCESSED"
    assert res_low_cap["stream_results"][ORDER[0]]["status"] == "PRICE_LIMIT_NOT_MET"
    assert len(res_low_cap["fills"]) >= 1
    assert res_low_cap["stream_results"][ORDER[1]]["status"] == "FILLED_BUDGET"
    assert res_low_cap["fills"][0]["market_id"] == markets_c[ORDER[1]]["market_id"]

    # Verify WAITING_DEPTH_RECHECK log tag in MonitorAuditLogger
    mlog = MonitorAuditLogger(tmp_path / "rt_v15")
    rec_wait = mlog.log_cohort_execution(
        b3.isoformat(),
        {"order": ["BTC_15m"]},
        {"BTC_15m": {"status": "SIGNAL", "outcome": "UP", "prediction": {"z_v4": 2.1, "threshold_frozen": 1.8, "side": "UP"}}},
        {
            "status": "WAITING_PRICE_RECHECK",
            "fills": [],
            "stream_results": {
                "BTC_15m": {
                    "status": "WAITING_PRICE_RECHECK",
                    "attempt": 0,
                    "best_ask": 0.49,
                    "limit_price": 0.515,
                    "min_buy_price": 0.415,
                    "min_executable_depth_usd": 25.0,
                    "observed_executable_depth_usd": 6.50,
                    "next_retry_at": (b3 + timedelta(seconds=11.5)).isoformat(),
                }
            },
        },
        {"duration_seconds": 4.2},
    )
    assert "WAITING_DEPTH_RECHECK" in rec_wait["details"]["streams"]["BTC_15m"]["no_trade_reason"]

    # --- 3. Account.sync_initial_capital scales capital cleanly while normal restart never resets ---
    cap_db = tmp_path / "capital_sync.sqlite3"
    acc1 = Account(cap_db, initial_capital=1269, started_at=BOUNDARY - timedelta(seconds=1))
    assert acc1.overview(now=BOUNDARY)["initial_capital"] == pytest.approx(1269.0)
    # Normal restart with different initial_capital parameter never overwrites persisted DB
    acc_restart = Account(cap_db, initial_capital=1400, started_at=BOUNDARY)
    assert acc_restart.overview(now=BOUNDARY)["initial_capital"] == pytest.approx(1269.0)
    # Explicit sync_initial_capital adjusts initial_atoms & cash_atoms by +$131 and records CAPITAL_ADJUSTED
    changed = acc_restart.sync_initial_capital(1400, now=BOUNDARY + timedelta(seconds=1))
    assert changed is True
    ov_after = acc_restart.overview(now=BOUNDARY + timedelta(seconds=1))
    assert ov_after["initial_capital"] == pytest.approx(1400.0)
    assert ov_after["cash"] == pytest.approx(1400.0)
    # Idempotent when already at target
    assert acc_restart.sync_initial_capital(1400, now=BOUNDARY + timedelta(seconds=2)) is False
    with acc_restart._connection() as con:
        ev = con.execute("SELECT event_type, payload FROM events WHERE event_type='CAPITAL_ADJUSTED'").fetchone()
    assert ev is not None
    ev_payload = json.loads(ev["payload"])
    assert ev_payload["previous_initial_capital_usd"] == pytest.approx(1269.0)
    assert ev_payload["new_initial_capital_usd"] == pytest.approx(1400.0)
    assert ev_payload["delta_usd"] == pytest.approx(131.0)


def test_v14r16_time_segmented_playbook_us_close_sunday_reopen_weekend_rv12_and_dst():
    """Verify full 7x24h time-segmented trading playbook & Dec 6 23/5 US equity schedule:
    1. America/New_York DST-aligned US Close MOC Blackout (`15:00-16:00 ET`, Mon-Fri) blocks all 6 streams
       and immediately unlocks at `16:00 ET` (`05:00 BJT` EST / `04:00 BJT` EDT) golden hour.
    2. America/New_York DST-aligned US Open Blackout (`09:30-11:00 ET`, Mon-Fri) works in both EST and EDT,
       and legacy `utc_start_minute`/`utc_end_minute` configs remain backward-compatible.
    3. Sunday Reopen Blackout (`Sunday 18:00-20:00 ET` CME Futures Open + `Sunday 21:00-22:00 ET` 23/5 Equity
       Week Open) blocks all 6 streams across the UTC midnight rollover (`Mon 07:00-09:00 BJT` & `10:00-11:00 BJT`),
       allows the `20:00-21:00 ET` (`09:00-10:00 BJT`) digestion hour, and allows Tue-Fri `21:00-22:00 ET` night sessions.
    4. Weekend `weekend_min_rv12_bps` raises `5m` streams (`BTC_5m/ETH_5m/SOL_5m`) to `4.0 bps` on Saturday/Sunday
       while keeping `15m` streams (`BTC_15m/ETH_15m/SOL_15m`) at `2.5/3.0 bps` and weekday `5m` at `3.0 bps`.
    """
    policy = json.loads(Path("config/argon4_forward_policy.json").read_text(encoding="utf-8"))
    vg = policy["volatility_regime_guard"]
    six_streams = ["BTC_15m", "ETH_5m", "ETH_15m", "BTC_5m", "SOL_15m", "SOL_5m"]

    assert vg["weekend_min_rv12_bps"] == {"BTC_5m": 4.0, "ETH_5m": 4.0, "SOL_5m": 4.0}
    assert vg["us_close_blackout"]["enabled"] is True
    assert vg["us_close_blackout"]["timezone"] == "America/New_York"
    assert vg["us_close_blackout"]["local_start_minute"] == 900
    assert vg["us_close_blackout"]["local_end_minute"] == 960
    assert set(vg["us_close_blackout"]["skip_streams"]) == set(six_streams)
    assert vg["sunday_reopen_blackout"]["enabled"] is True
    assert vg["sunday_reopen_blackout"]["timezone"] == "America/New_York"
    assert vg["sunday_reopen_blackout"]["windows_local_minutes"] == [[1080, 1200], [1260, 1320]]
    assert set(vg["sunday_reopen_blackout"]["skip_streams"]) == set(six_streams)

    pred = {"selected": True, "side": "DOWN", "z_v4": -2.05, "threshold_frozen": 1.72}

    # --- 1. US Close MOC Blackout (15:00-16:00 ET) in Winter EST (20:00-21:00 UTC) & Summer EDT (19:00-20:00 UTC) ---
    for st in six_streams:
        r_est = model_v14.apply_volatility_regime_guard(
            st, pred, ex_ante_rv12_bps=12.0, guard_config=vg, cohort_time="2026-11-11T20:15:00+00:00"
        )
        assert r_est["selected"] is False
        assert r_est["volatility_regime_reason"] == "US_CLOSE_BLACKOUT"
        assert r_est["us_close_blackout_minute_local"] == 915

        r_edt = model_v14.apply_volatility_regime_guard(
            st, pred, ex_ante_rv12_bps=12.0, guard_config=vg, cohort_time="2026-07-15T19:30:00+00:00"
        )
        assert r_edt["selected"] is False
        assert r_edt["volatility_regime_reason"] == "US_CLOSE_BLACKOUT"

    # 16:00 ET sharp (21:00 UTC in EST / 20:00 UTC in EDT) unlocks the 61.61% post-close golden hour!
    r_unlock_est = model_v14.apply_volatility_regime_guard(
        "BTC_15m", pred, ex_ante_rv12_bps=12.0, guard_config=vg, cohort_time="2026-11-11T21:00:00+00:00"
    )
    assert r_unlock_est["selected"] is True
    r_unlock_edt = model_v14.apply_volatility_regime_guard(
        "BTC_15m", pred, ex_ante_rv12_bps=12.0, guard_config=vg, cohort_time="2026-07-15T20:00:00+00:00"
    )
    assert r_unlock_edt["selected"] is True

    # Weekend 15:15 ET (Sunday 2026-11-15T20:15:00Z) is NOT blocked by US_CLOSE_BLACKOUT
    r_sun_afternoon = model_v14.apply_volatility_regime_guard(
        "BTC_15m", pred, ex_ante_rv12_bps=12.0, guard_config=vg, cohort_time="2026-11-15T20:15:00+00:00"
    )
    assert r_sun_afternoon["selected"] is True

    # --- 2. Summer EDT US Open Blackout (09:30-11:00 EDT = 13:30-15:00 UTC) + legacy UTC fallback ---
    r_open_edt = model_v14.apply_volatility_regime_guard(
        "ETH_5m", pred, ex_ante_rv12_bps=12.0, guard_config=vg, cohort_time="2026-07-15T13:45:00+00:00"
    )
    assert r_open_edt["selected"] is False
    assert r_open_edt["volatility_regime_reason"] == "US_OPEN_BLACKOUT"
    r_open_edt_sol = model_v14.apply_volatility_regime_guard(
        "SOL_5m", pred, ex_ante_rv12_bps=12.0, guard_config=vg, cohort_time="2026-07-15T13:45:00+00:00"
    )
    assert r_open_edt_sol["selected"] is True
    r_open_legacy = model_v14.apply_volatility_regime_guard(
        "ETH_5m",
        pred,
        ex_ante_rv12_bps=12.0,
        guard_config={"enabled": True, "us_open_blackout": {"enabled": True, "utc_start_minute": 870, "utc_end_minute": 960, "skip_streams": ["ETH_5m"]}},
        cohort_time="2026-11-11T14:40:00+00:00",
    )
    assert r_open_legacy["selected"] is False
    assert r_open_legacy["volatility_regime_reason"] == "US_OPEN_BLACKOUT"

    # --- 3. Sunday Reopen Blackout (Sunday 18:00-20:00 ET & 21:00-22:00 ET) ---
    # Window 1: Sunday 18:30 EST (2026-11-15T23:30:00Z = Mon 07:30 BJT) and Sunday 19:30 EST (2026-11-16T00:30:00Z = Mon 08:30 BJT)
    for ts_reopen in ("2026-11-15T23:30:00+00:00", "2026-11-16T00:30:00+00:00", "2026-11-16T02:15:00+00:00"):
        for st in six_streams:
            r_sun = model_v14.apply_volatility_regime_guard(
                st, pred, ex_ante_rv12_bps=12.0, guard_config=vg, cohort_time=ts_reopen
            )
            assert r_sun["selected"] is False
            assert r_sun["volatility_regime_reason"] == "SUNDAY_REOPEN_BLACKOUT"

    # Sunday 20:00-21:00 EST (2026-11-16T01:15:00Z = Mon 09:15 BJT) & Sunday 22:00 EST (Mon 11:00 BJT) are OPEN
    for ts_open in ("2026-11-16T01:15:00+00:00", "2026-11-16T03:00:00+00:00", "2026-11-17T02:15:00+00:00"):
        r_ok = model_v14.apply_volatility_regime_guard(
            "BTC_15m", pred, ex_ante_rv12_bps=12.0, guard_config=vg, cohort_time=ts_open
        )
        assert r_ok["selected"] is True

    # --- 4. Weekend 5m min_rv12_bps raised to 4.0 bps while 15m stays 2.5/3.0 bps ---
    for st_5m in ("BTC_5m", "ETH_5m", "SOL_5m"):
        r_wknd_dead = model_v14.apply_volatility_regime_guard(
            st_5m, pred, ex_ante_rv12_bps=3.5, guard_config=vg, cohort_time="2026-11-14T12:00:00+00:00"
        )
        assert r_wknd_dead["selected"] is False
        assert r_wknd_dead["volatility_regime_reason"] == "BELOW_MIN_RV12_BPS"
        assert r_wknd_dead["volatility_regime_min_rv12_bps"] == pytest.approx(4.0)

        r_wknd_ok = model_v14.apply_volatility_regime_guard(
            st_5m, pred, ex_ante_rv12_bps=4.2, guard_config=vg, cohort_time="2026-11-14T12:00:00+00:00"
        )
        assert r_wknd_ok["selected"] is True

        r_wkday_ok = model_v14.apply_volatility_regime_guard(
            st_5m, pred, ex_ante_rv12_bps=3.5, guard_config=vg, cohort_time="2026-11-11T12:00:00+00:00"
        )
        assert r_wkday_ok["selected"] is True

    for st_15m in ("BTC_15m", "ETH_15m", "SOL_15m"):
        r_wknd_15m = model_v14.apply_volatility_regime_guard(
            st_15m, pred, ex_ante_rv12_bps=3.5, guard_config=vg, cohort_time="2026-11-14T12:00:00+00:00"
        )
        assert r_wknd_15m["selected"] is True




