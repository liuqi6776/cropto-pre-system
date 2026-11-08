"""Tests for ARGON-4 v14r1 architecture upgrades and review corrections."""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
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
    assert obs1.st_ino == obs2.st_ino
    assert obs1.st_nlink >= 2


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
