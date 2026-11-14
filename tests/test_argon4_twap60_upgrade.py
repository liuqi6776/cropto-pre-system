"""Unit tests for ARGON-4 Chainlink TWAP60 + 270d decay trainer and runtime handicap guard."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tools.argon4_forward import model, model_v14, online, twap60_retrain


def test_frozen_model_py_sha256_preserved():
    """Frozen tools/argon4_forward/model.py must remain byte-for-byte untouched."""
    digest = twap60_retrain.verify_frozen_model_untouched()
    assert digest == "1d9629abf05c27eeec0ed6e63ae4b4fc65c4f2b904dca4de9549698cf1a009f7"


def test_exponential_sample_weights_half_life_270_days():
    end_ts = pd.Timestamp("2026-10-01T00:00:00Z")
    idx = pd.DatetimeIndex(
        [
            end_ts - pd.Timedelta(days=540),
            end_ts - pd.Timedelta(days=270),
            end_ts,
        ]
    )
    sw = twap60_retrain.compute_exponential_sample_weights(idx, end_ts, 270.0)
    np.testing.assert_allclose(sw, [0.25, 0.5, 1.0], rtol=1e-12, atol=1e-12)
    assert twap60_retrain.compute_exponential_sample_weights(idx, end_ts, None) is None


def test_build_twap60_labels_pure_and_blend_with_fallback():
    idx = pd.date_range("2026-10-01T00:00:00Z", periods=6, freq="5min", tz="UTC")
    raw = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        },
        index=idx,
    )
    # Provide spot_1m for rows 1..5 (leaving row 0 without spot_1m to test fallback)
    close_ts = idx + pd.Timedelta(minutes=5)
    s1 = pd.DataFrame(
        {
            "bound_ts": close_ts[1:],
            "spot_close": [100.0, 101.0, 100.0, 102.0, 103.0],
            "twap60": [100.5, 100.2, 100.8, 101.5, 102.5],
        }
    )
    tr_pure, ev_pure = twap60_retrain.build_twap60_labels(
        raw, 1, index=idx, mode="SPOT_TWAP60", spot_1m_frame=s1
    )
    # Row 0 lacks tw60[0], so it falls back to perp return c[1]/o[1] - 1 = 0.01
    assert np.isclose(tr_pure[0], 0.01)
    # Row 1 has tw60[2]/tw60[1] - 1 = 100.2 / 100.5 - 1 (< 0, even though spot_close rose 100 -> 101!)
    expected_twap_r1 = 100.2 / 100.5 - 1.0
    assert np.isclose(tr_pure[1], expected_twap_r1)
    assert np.isclose(ev_pure[1], expected_twap_r1)

    tr_blend, ev_blend = twap60_retrain.build_twap60_labels(
        raw, 1, index=idx, mode="SPOT_TWAP60_BLEND", spot_1m_frame=s1
    )
    expected_s2t_r1 = 100.2 / 100.0 - 1.0
    assert np.isclose(tr_blend[1], 0.5 * expected_twap_r1 + 0.5 * expected_s2t_r1)
    # Validation evaluation target must remain pure Chainlink TWAP60 return!
    assert np.isclose(ev_blend[1], expected_twap_r1)

    tr_s2t, ev_s2t = twap60_retrain.build_twap60_labels(
        raw, 1, index=idx, mode="SPOT_TO_TWAP60_PURE", spot_1m_frame=s1
    )
    assert np.isclose(tr_s2t[1], expected_s2t_r1)
    assert np.isclose(ev_s2t[1], expected_twap_r1)


def test_fit_family_weighted_produces_frozen_model_compatible_bundle():
    rng = np.random.default_rng(42)
    x_tr = rng.normal(size=(400, 12)).astype(np.float32)
    x_va = rng.normal(size=(80, 12)).astype(np.float32)
    y_tr = rng.normal(0.0, 0.004, size=400)
    sw = np.linspace(0.2, 1.0, 400)
    fam, z_va = twap60_retrain.fit_family_weighted(x_tr, y_tr, x_va, expert=False, sample_weight=sw)
    scored = model._score_family(fam, x_va)
    np.testing.assert_allclose(scored, z_va, rtol=1e-12, atol=1e-12)


def test_online_spot_1m_twap60_capture_and_parameter_allowlist(tmp_path):
    # 1m is allowed ONLY on spot, never on futures or mark
    online._parameters("spot", {"symbol": "BTCUSDT", "interval": "1m", "startTime": 1000, "endTime": 61000, "limit": 2})
    with pytest.raises(ValueError, match="Unsupported interval"):
        online._parameters("futures", {"symbol": "BTCUSDT", "interval": "1m", "startTime": 1000, "endTime": 61000, "limit": 2})

    now = pd.Timestamp("2026-11-12T12:00:02Z")
    bar_open = pd.Timestamp("2026-11-12T11:59:00Z")

    class Spot1mFixture:
        def get(self, kind, params=None):
            assert kind == "spot"
            assert params["interval"] == "1m"
            assert params["limit"] == 2
            row = [int(bar_open.timestamp() * 1000), "90100.0", "90120.0", "89980.0", "90000.0", "12.5", 0, "0", 10, "0", "0", "0"]
            raw = json.dumps([row]).encode()
            return [row], {
                "requested_at": now.isoformat(),
                "received_at": now.isoformat(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "http_status": 200,
            }

    inp = online.OnlineInputs(tmp_path, client=Spot1mFixture(), clock=lambda: now)
    obs = inp.capture_spot_1m_twap60("BTCUSDT", pd.Timestamp("2026-11-12T12:00:00Z"), cohort_id="2026-11-12T12:00:00+00:00")
    assert obs["status"] == "OBSERVED"
    assert np.isclose(obs["twap60"], 0.25 * (90100.0 + 90120.0 + 89980.0 + 90000.0))  # 90050.0
    assert np.isclose(obs["spot_close"], 90000.0)
    assert len(obs["receipt_sha256"]) == 64

    # Handicap for UP when twap60=90050 > spot=90000 is +5.5556 bps (> +1.0 bps -> blocked on 5m!)
    hcap_up = model_v14.compute_twap60_handicap_bps(obs["twap60"], obs["spot_close"], "UP")
    assert np.isclose(hcap_up, (50.0 / 90000.0) * 10000.0)

    guard_cfg = {
        "enabled": True,
        "action": "SKIP_OR_HALF_STAKE",
        "twap60_handicap_guard": {
            "enabled": True,
            "max_5m_handicap_bps": 1.0,
            "max_15m_handicap_bps": 2.0,
        },
    }
    pred_up = {"stream": "BTC_5m", "side": "UP", "selected": True, "z_v4": 1.95, "threshold_frozen": 1.70}
    res_blocked = model_v14.apply_volatility_regime_guard(
        "BTC_5m",
        pred_up,
        guard_config=guard_cfg,
        cohort_time="2026-11-12T12:00:00+00:00",
        spot_twap60_t=obs["twap60"],
        spot_close_t=obs["spot_close"],
    )
    assert res_blocked["selected"] is False
    assert res_blocked["volatility_regime_reason"] == "TWAP60_START_HANDICAP_GUARD"
    assert res_blocked["ex_ante_twap60_handicap_bps"] > 5.0

    # Same market state for DOWN has favorable negative handicap (-5.55 bps) -> passes!
    pred_dn = {"stream": "BTC_5m", "side": "DOWN", "selected": True, "z_v4": -1.95, "threshold_frozen": 1.70}
    res_kept = model_v14.apply_volatility_regime_guard(
        "BTC_5m",
        pred_dn,
        guard_config=guard_cfg,
        cohort_time="2026-11-12T12:00:00+00:00",
        spot_twap60_t=obs["twap60"],
        spot_close_t=obs["spot_close"],
    )
    assert res_kept["selected"] is True
    assert res_kept["ex_ante_twap60_handicap_bps"] < -5.0


def test_policy_twap60_handicap_guard_and_manifest_pins():
    policy_path = model.ROOT / "config" / "argon4_forward_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    hcap_cfg = policy["volatility_regime_guard"]["twap60_handicap_guard"]
    assert hcap_cfg["enabled"] is True
    assert hcap_cfg["max_5m_handicap_bps"] == 1.0
    assert hcap_cfg["max_15m_handicap_bps"] == 2.0
    assert hcap_cfg["on_unavailable"] == "FAIL_CLOSED"
    pins = policy["deployment_manifest_sha256"]
    assert len(pins) == 7

    q4 = model.MODEL_ROOT / "deployment_q4"
    if not (q4 / "BTC_15m.manifest.json").is_file():
        pytest.skip("source-only checkout without local model binaries")

    for stream, expected_sha in pins.items():
        bundle = model.load_model_bundle(stream, "deployment_q4", expected_manifest_sha256=expected_sha)
        assert bundle["manifest_sha256"] == expected_sha
        fold_meta = bundle["manifest"]["fold"]
        assert pd.Timestamp(fold_meta["valid_until"]) >= pd.Timestamp("2027-02-01T00:00:00Z")
        assert fold_meta["label_mode"] == "SPOT_TO_TWAP60_PURE"
        assert fold_meta["half_life_days"] == 270.0
        assert bundle["manifest"]["sample_weight_policy"] == "exponential_decay_half_life_270.0d"


def test_twap60_handicap_guard_fail_closed_on_unavailable_and_pipeline_pass2(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from tools.argon4_forward import account, pipeline

    guard_cfg = {
        "enabled": True,
        "action": "SKIP_OR_HALF_STAKE",
        "twap60_handicap_guard": {
            "enabled": True,
            "max_5m_handicap_bps": 1.0,
            "max_15m_handicap_bps": 2.0,
            "on_unavailable": "FAIL_CLOSED",
        },
    }
    pred = {"stream": "BTC_5m", "side": "UP", "selected": True, "z_v4": 1.95, "threshold_frozen": 1.70}

    # Pass 1 offline call (no Spot 1m args, twap60_unavailable=False) stays unblocked for later Pass-2 check
    res_pass1 = model_v14.apply_volatility_regime_guard(
        "BTC_5m",
        pred,
        guard_config=guard_cfg,
        cohort_time="2026-11-12T12:00:00+00:00",
    )
    assert res_pass1["selected"] is True

    # Explicit unavailability, partial None, or NaN Spot 1m inputs must FAIL_CLOSED
    for bad_kw in (
        {"twap60_unavailable": True},
        {"spot_twap60_t": 90000.0, "spot_close_t": None},
        {"spot_twap60_t": float("nan"), "spot_close_t": 90000.0},
        {"spot_twap60_t": 90000.0, "spot_close_t": -1.0},
    ):
        res_bad = model_v14.apply_volatility_regime_guard(
            "BTC_5m",
            pred,
            guard_config=guard_cfg,
            cohort_time="2026-11-12T12:00:00+00:00",
            **bad_kw,
        )
        assert res_bad["selected"] is False
        assert res_bad["volatility_regime_guarded"] is True
        assert res_bad["volatility_regime_reason"] == "TWAP60_SPOT_1M_UNAVAILABLE"

    # Pipeline Pass-2 check: missing capture_spot_1m_twap60 or non-OBSERVED statuses must block signal
    cohort = datetime(2026, 11, 12, 12, 0, 0, tzinfo=timezone.utc)
    now_t = cohort + pd.Timedelta(seconds=4)
    rt = tmp_path / "rt_twap_failclosed"
    rt.mkdir(parents=True, exist_ok=True)
    acc = account.Account(rt / "paper.sqlite3", initial_capital=1000, started_at=cohort - pd.Timedelta(minutes=5))
    obs_saved = {}
    obs = SimpleNamespace(save=lambda k, v: obs_saved.__setitem__(k, v), path=str(rt / "obs.sqlite3"))
    pipe = pipeline.ForwardPipeline(acc, rt, obs)
    try:
        pipe.clock = {
            "status": "CALIBRATED",
            "uncertainty_seconds": 0.1,
            "server_minus_local_seconds": 0.0,
            "calibrated_monotonic": 1e9,
            "observed_at": cohort.isoformat(),
            "receipt_sha256": "a" * 64,
        }
        monkeypatch.setattr(pipeline, "calibrated_now", lambda clock, upper=True: now_t)
        base_policy = {
            "order": ["BTC_5m"],
            "volatility_regime_guard": guard_cfg,
        }
        profile = {"disabled_streams": set()}
        idx_bar = pd.DatetimeIndex([cohort - pd.Timedelta(minutes=5)])
        fake_frames = {
            "BTCUSDT": {
                "base": pd.DataFrame({"open": [90000.0], "close": [90010.0]}, index=idx_bar),
                "expert": pd.DataFrame({"x": [1.0]}, index=idx_bar),
            }
        }
        pipe.bundles = {"BTC_5m": {"bundle": "stub"}}
        pred_fn = lambda s, b, e, dt, **kw: {
            "stream": s,
            "side": "UP",
            "selected": True,
            "z_v4": 2.0,
            "threshold_frozen": 1.7,
            "model_version": "m1",
            "model_sha256": "b" * 64,
        }

        # 1) Legacy inputs object lacking capture_spot_1m_twap60 -> FAIL_CLOSED
        pipe.inputs = SimpleNamespace()
        d_missing_fn = pipe._score_decisions(
            base_policy, {"data_root": None}, cohort, profile, now_t, now_t, dict(fake_frames), {}, None, None, pred_fn
        )
        assert d_missing_fn["BTC_5m"]["status"] == "NO_SIGNAL"
        assert d_missing_fn["BTC_5m"]["prediction"]["volatility_regime_reason"] == "TWAP60_SPOT_1M_UNAVAILABLE"

        # 2) capture_spot_1m_twap60 returns ERROR / MISSING_BAR / DEADLINE_SKIPPED -> FAIL_CLOSED
        for bad_obs in (
            {"status": "ERROR", "error": "timeout"},
            {"status": "MISSING_BAR"},
            {"status": "DEADLINE_SKIPPED"},
        ):
            pipe.inputs = SimpleNamespace(capture_spot_1m_twap60=lambda *a, _ret=bad_obs, **kw: dict(_ret))
            d_bad = pipe._score_decisions(
                base_policy, {"data_root": None}, cohort, profile, now_t, now_t, dict(fake_frames), {}, None, None, pred_fn
            )
            assert d_bad["BTC_5m"]["status"] == "NO_SIGNAL"
            assert d_bad["BTC_5m"]["prediction"]["volatility_regime_reason"] == "TWAP60_SPOT_1M_UNAVAILABLE"

        # 3) OBSERVED within <= 1.0 bps handicap -> produces SIGNAL
        pipe.inputs = SimpleNamespace(
            capture_spot_1m_twap60=lambda *a, **kw: {
                "status": "OBSERVED",
                "twap60": 90005.0,
                "spot_close": 90000.0,
                "receipt_sha256": "c" * 64,
            }
        )
        d_ok = pipe._score_decisions(
            base_policy, {"data_root": None}, cohort, profile, now_t, now_t, dict(fake_frames), {}, None, None, pred_fn
        )
        assert d_ok["BTC_5m"]["status"] == "SIGNAL"
        assert d_ok["BTC_5m"]["prediction"]["spot_1m_receipt_sha256"] == "c" * 64
    finally:
        pipe.close()


